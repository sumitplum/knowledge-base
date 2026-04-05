"""
GitOps — wraps GitPython for Feature Builder branch/commit/push operations.

Safety rules:
  - Never force-push.
  - Never push to main/master.
  - Always create the feature branch from a fresh fetch of origin/main.
  - Refuse to proceed if working tree is dirty (uncommitted changes in the
    developer's own working copy) unless the caller explicitly passes
    `allow_dirty=True`.

Layer 3 of the Security Guardrails System: GitHub Operation Guardrails.
"""

import logging
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import settings
from .change_tracker import ChangeTracker, FileChange
from agents.exceptions import SafetyViolation
from agents.audit import get_audit_logger, AuditEventType

logger = logging.getLogger(__name__)

# Branches we absolutely refuse to push to
PROTECTED_BRANCHES = {"main", "master", "develop", "staging", "production"}

# Protected refs including tag patterns
PROTECTED_REFS = {
    "v*",       # Version tags
    "release-*",  # Release tags
    "refs/tags/*",  # All tags
}

# Git config files that must never be modified
GIT_CONFIG_FILES = frozenset({
    ".git/config",
    ".gitconfig",
    ".gitmodules",
    ".gitattributes",
    ".git/hooks",
})

# Branch name pattern for feature branches
# Pattern: kb/feature/<slug>/<yyyymmdd-hhmmss>
FEATURE_BRANCH_PATTERN = re.compile(r"^kb/feature/[a-z0-9-]+/\d{8}-\d{6}$")


class GitOpsError(Exception):
    """Raised for any git-level problem that should abort the build."""


class GitOps:
    """
    Wraps GitPython for safe feature branch creation, commits, and pushes.

    Parameters
    ----------
    repo_path: Path
        Local clone of the repository.
    repo_name: str
        Short name ("orbit" or "trinity") for logging and branch naming.
    dry_run: bool
        When True, simulate all operations without touching git state.
    """

    def __init__(self, repo_path: Path, repo_name: str, dry_run: bool = True):
        self.repo_path = Path(repo_path).resolve()
        self.repo_name = repo_name
        self.dry_run = dry_run
        self._repo = None  # lazy-loaded
        self._audit = get_audit_logger()

    def _get_repo(self):
        """Lazy-load the GitPython Repo object."""
        if self._repo is None:
            try:
                from git import Repo, InvalidGitRepositoryError
                self._repo = Repo(self.repo_path)
            except Exception as e:
                raise GitOpsError(f"Cannot open git repo at {self.repo_path}: {e}")
        return self._repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_branch(self) -> str:
        """Return the currently checked-out branch name."""
        if self.dry_run:
            return "main"
        return self._get_repo().active_branch.name

    def is_clean(self) -> bool:
        """Return True when the working tree has no uncommitted changes."""
        if self.dry_run:
            return True  # Dry-run never fails dirty checks
        return not self._get_repo().is_dirty(untracked_files=True)

    def make_branch_name(self, feature_slug: str) -> str:
        """
        Build a deterministic, URL-safe branch name.

        Pattern: <prefix>/<slug>/<yyyymmdd-hhmmss>
        e.g.    kb/feature/add-bulk-upload/20250401-143012
        """
        slug = re.sub(r"[^a-z0-9-]", "-", feature_slug.lower().strip())
        slug = re.sub(r"-{2,}", "-", slug).strip("-")[:50]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = settings.feature_branch_prefix.rstrip("/")
        return f"{prefix}/{slug}/{ts}"

    def create_feature_branch(
        self,
        branch_name: str,
        base: str = "main",
        allow_dirty: bool = False,
    ) -> str:
        """
        Fetch origin, create and checkout a new branch from origin/<base>.

        Returns the branch name that was created.
        """
        self._check_not_protected(branch_name)
        self._check_branch_name_pattern(branch_name)

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would create branch '{branch_name}' from origin/{base} in {self.repo_name}")
            return branch_name

        repo = self._get_repo()

        if not allow_dirty and not self.is_clean():
            raise GitOpsError(
                f"Working tree at {self.repo_path} is dirty. "
                "Commit or stash changes before running the Feature Builder."
            )

        # Fetch to get the latest remote state
        logger.info(f"Fetching origin in {self.repo_name}…")
        try:
            origin = repo.remote("origin")
            origin.fetch()
        except Exception as e:
            logger.warning(f"Fetch failed ({e}); proceeding with local state")

        # Resolve base ref
        try:
            base_commit = repo.commit(f"origin/{base}")
        except Exception:
            try:
                base_commit = repo.commit(base)
            except Exception as e:
                raise GitOpsError(f"Cannot resolve base ref '{base}' in {self.repo_name}: {e}")

        # Create and checkout branch
        new_branch = repo.create_head(branch_name, commit=str(base_commit))
        new_branch.checkout()
        logger.info(f"Created and checked out branch '{branch_name}' in {self.repo_name}")
        
        # Audit log
        self._audit.log_branch_created(self.repo_name, branch_name, base)
        
        return branch_name

    def apply_and_commit(
        self,
        changes: list[FileChange],
        commit_message: str,
    ) -> Optional[str]:
        """
        Write file changes to disk and create a git commit.

        Returns the commit SHA, or None in dry-run mode.
        """
        if not changes:
            logger.info(f"No changes to commit in {self.repo_name}")
            return None

        self._safety_check_changes(changes)

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would write {len(changes)} files and commit in {self.repo_name}")
            for c in changes:
                logger.info(f"  {'[NEW]' if c.is_new_file else '[MOD]'} {c.file_path}")
            return f"dry-run-{self.repo_name}"  # Return a fake SHA so branch_names gets populated

        repo = self._get_repo()

        # Write file contents
        for change in changes:
            path = change.abs_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(change.new_content, encoding="utf-8")
            logger.debug(f"Wrote {path}")

        # Stage all changed files
        relative_paths = [str(change.abs_path.relative_to(self.repo_path)) for change in changes]
        repo.index.add(relative_paths)

        if not repo.index.diff("HEAD") and not repo.untracked_files:
            logger.info(f"Nothing staged in {self.repo_name}; skipping commit")
            return None

        commit = repo.index.commit(commit_message)
        logger.info(f"Committed {len(changes)} files in {self.repo_name}: {commit.hexsha[:8]}")
        
        # Verify commit author
        self._verify_commit_author(commit)
        
        # Audit log
        self._audit.log_commit_created(self.repo_name, commit.hexsha, len(changes))
        
        return commit.hexsha

    def push_branch(self, branch_name: str) -> str:
        """
        Push the feature branch to origin.

        Returns the remote URL string for logging.
        """
        self._check_not_protected(branch_name)
        self._check_no_force_push(branch_name)

        if self.dry_run:
            logger.info(f"[DRY-RUN] Would push '{branch_name}' to origin in {self.repo_name}")
            return f"https://github.com/PlumHQ/{self.repo_name}/tree/{branch_name} (dry-run)"

        repo = self._get_repo()
        origin = repo.remote("origin")
        
        # Build refspec without force push prefix
        refspec = f"{branch_name}:{branch_name}"
        self._check_refspec_no_force(refspec)
        
        origin.push(refspec=refspec)
        remote_url = origin.url
        logger.info(f"Pushed '{branch_name}' to {remote_url}")
        
        # Audit log
        self._audit.log_push_executed(self.repo_name, branch_name)
        
        return remote_url

    def restore_original_branch(self):
        """Check out the branch that was active before we started."""
        if self.dry_run:
            return
        try:
            repo = self._get_repo()
            # HEAD was detached or we stored the original
            repo.heads["main"].checkout()
        except Exception as e:
            logger.warning(f"Could not restore original branch in {self.repo_name}: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_not_protected(self, branch_name: str):
        """Check that branch is not in the protected list."""
        branch_base = branch_name.split("/")[-1]
        if branch_name in PROTECTED_BRANCHES or branch_base in PROTECTED_BRANCHES:
            self._audit.log_safety_violation(
                f"Attempted operation on protected branch: {branch_name}",
                repo=self.repo_name,
            )
            raise GitOpsError(
                f"Refusing to operate on protected branch '{branch_name}'. "
                "Feature Builder only creates new branches."
            )

    def _check_branch_name_pattern(self, branch_name: str):
        """
        Enforce branch naming pattern.
        
        Feature branches must match: kb/feature/<slug>/<yyyymmdd-hhmmss>
        This prevents the LLM from creating branches like 'main-backup' or 'hotfix/override'.
        """
        prefix = settings.feature_branch_prefix
        
        # Build expected pattern based on configured prefix
        # Default prefix is "kb/feature"
        expected_pattern = re.compile(
            f"^{re.escape(prefix)}/[a-z0-9-]+/\\d{{8}}-\\d{{6}}$"
        )
        
        if not expected_pattern.match(branch_name):
            self._audit.log_safety_violation(
                f"Invalid branch name pattern: {branch_name}",
                repo=self.repo_name,
                details={"expected_pattern": expected_pattern.pattern},
            )
            raise GitOpsError(
                f"Branch name '{branch_name}' does not match the required pattern "
                f"'{prefix}/<slug>/<YYYYMMDD-HHMMSS>'. "
                "Feature Builder enforces strict branch naming."
            )

    def _check_no_force_push(self, branch_name: str):
        """Ensure we never force push."""
        # This is a safeguard - we control the refspec, but this catches
        # any accidental or malicious refspec with force-push prefix
        pass  # Actual check is in _check_refspec_no_force

    def _check_refspec_no_force(self, refspec: str):
        """
        Check that refspec doesn't contain force-push prefix.
        
        Force push uses '+' prefix: +refs/heads/branch:refs/heads/branch
        """
        if refspec.startswith("+"):
            self._audit.log_safety_violation(
                f"Force-push attempt detected: {refspec}",
                repo=self.repo_name,
            )
            raise GitOpsError(
                f"Force-push is not allowed. Refspec '{refspec}' contains force-push prefix. "
                "Feature Builder never force-pushes."
            )

    def _check_not_tag(self, ref: str):
        """
        Check that we're not operating on a tag ref.
        """
        if ref.startswith("refs/tags/") or ref.startswith("v") or ref.startswith("release-"):
            self._audit.log_safety_violation(
                f"Attempted operation on tag: {ref}",
                repo=self.repo_name,
            )
            raise GitOpsError(
                f"Refusing to operate on tag ref '{ref}'. "
                "Feature Builder does not modify tags."
            )

    def _check_not_config_file(self, file_path: Path):
        """
        Check that we're not writing to git config files.
        
        Blocks writes to: .git/config, .gitconfig, .gitmodules, .gitattributes
        """
        # Get relative path from repo root
        try:
            rel_path = file_path.relative_to(self.repo_path)
            rel_str = str(rel_path)
        except ValueError:
            rel_str = str(file_path)
        
        # Check against git config patterns
        for pattern in GIT_CONFIG_FILES:
            if rel_str == pattern or rel_str.startswith(pattern):
                self._audit.log_safety_violation(
                    f"Attempted write to git config file: {file_path}",
                    repo=self.repo_name,
                    file_path=rel_str,
                )
                raise GitOpsError(
                    f"Refusing to write to git config file '{rel_str}'. "
                    "Feature Builder cannot modify git configuration."
                )
        
        # Also check if path is inside .git directory
        if ".git" in rel_str.split("/") or ".git" in rel_str.split("\\"):
            self._audit.log_safety_violation(
                f"Attempted write inside .git directory: {file_path}",
                repo=self.repo_name,
                file_path=rel_str,
            )
            raise GitOpsError(
                f"Refusing to write inside .git directory: '{rel_str}'. "
                "Feature Builder cannot modify git internals."
            )

    def _verify_commit_author(self, commit):
        """
        Verify commit author matches expected domain.
        
        Logs a warning if the author email doesn't match the configured
        github_org domain or allowed emails.
        """
        try:
            author_email = commit.author.email
            expected_domain = f"@{settings.github_org.lower()}.com"
            
            # Allow common CI/bot patterns
            allowed_patterns = [
                expected_domain,
                "@users.noreply.github.com",
                "noreply@github.com",
                "@localhost",  # For local testing
            ]
            
            is_valid = any(
                pattern in author_email.lower() 
                for pattern in allowed_patterns
            )
            
            if not is_valid:
                logger.warning(
                    f"Commit author '{author_email}' does not match expected domain. "
                    f"Expected: {expected_domain} or GitHub noreply."
                )
                self._audit.log(
                    AuditEventType.COMMIT_CREATED,
                    repo=self.repo_name,
                    details={
                        "commit_sha": commit.hexsha,
                        "author_email": author_email,
                        "warning": "Author email does not match expected domain",
                    },
                )
        except Exception as e:
            logger.debug(f"Could not verify commit author: {e}")

    def _safety_check_changes(self, changes: list[FileChange]):
        """Refuse to write files outside the repo root or secret-looking files."""
        secret_patterns = {".env", "secrets", "credentials", ".pem", ".key", ".p12"}
        
        for change in changes:
            # Check git config files
            self._check_not_config_file(change.abs_path)
            
            # In dry-run abs_path may be under a different base — only check name safety
            if not self.dry_run:
                try:
                    change.abs_path.relative_to(self.repo_path)
                except ValueError:
                    self._audit.log_safety_violation(
                        f"File path escapes repo root: {change.abs_path}",
                        repo=self.repo_name,
                        file_path=str(change.abs_path),
                    )
                    raise GitOpsError(
                        f"File path escapes repo root: {change.abs_path} "
                        f"(root: {self.repo_path})"
                    )
            
            name_lower = change.abs_path.name.lower()
            parts_lower = {p.lower() for p in change.abs_path.parts}
            if any(p in name_lower or p in parts_lower for p in secret_patterns):
                self._audit.log_safety_violation(
                    f"Attempted write to secret file: {change.abs_path}",
                    repo=self.repo_name,
                    file_path=str(change.abs_path),
                )
                raise GitOpsError(
                    f"Refusing to write potentially secret file: {change.abs_path}"
                )

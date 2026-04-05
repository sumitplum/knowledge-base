"""
PRCreator — opens GitHub Pull Requests via PyGitHub.

Creates one PR per repository that has changes.  The PR body is auto-generated
from the orchestrator plan and the ChangeTracker summary.

Upgraded with:
- LLM-generated PR titles and descriptions (Item 1)
- Idempotency check: update existing PR instead of creating duplicates (Item 10)

Layer 3 of the Security Guardrails System: GitHub Operation Guardrails.
- Base branch whitelist enforcement
- Repo slug whitelist assertion
- PR body length cap
- Secret stripping from PR body
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from .change_tracker import ChangeTracker
from agents.exceptions import SafetyViolation
from agents.audit import get_audit_logger, AuditEventType
from agents.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# GitHub's maximum PR body length
MAX_PR_BODY_LENGTH = 65000

# Allowed base branches for PRs
ALLOWED_BASE_BRANCHES = frozenset({"main", "develop"})

# Secret patterns to strip from PR body (subset of safety.py patterns)
PR_BODY_SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9]{20,}", "[REDACTED-OPENAI-KEY]"),
    (r"ghp_[A-Za-z0-9]{36,}", "[REDACTED-GITHUB-TOKEN]"),
    (r"gho_[A-Za-z0-9]{36,}", "[REDACTED-GITHUB-OAUTH]"),
    (r"AKIA[0-9A-Z]{16}", "[REDACTED-AWS-KEY]"),
    (r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}['\"]?", "[REDACTED-AWS-SECRET]"),
    (r"DefaultEndpointsProtocol=https;AccountName=\w+;AccountKey=[A-Za-z0-9+/=]{88}", "[REDACTED-AZURE-CONN]"),
    (r'"private_key":\s*"-----BEGIN[^"]+-----"', '"private_key": "[REDACTED-PRIVATE-KEY]"'),
    (r"sk_live_[A-Za-z0-9]{24,}", "[REDACTED-STRIPE-KEY]"),
    (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----[^-]+-----END[^-]+-----", "[REDACTED-PRIVATE-KEY-BLOCK]"),
    (r"(?i)(password|passwd|secret|token|api_key)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "[REDACTED-CREDENTIAL]"),
    (r"(?i)(postgres(ql)?|mysql|mongodb)://[^:]+:[^@]+@", "[REDACTED-DB-URL]://***:***@"),
]


@dataclass
class PRResult:
    repo_slug: str
    branch: str
    pr_url: Optional[str]
    pr_number: Optional[int]
    dry_run: bool
    error: Optional[str] = None
    was_updated: bool = False  # True if we updated an existing PR (Item 10)

    @property
    def success(self) -> bool:
        return self.error is None


class PRCreator:
    """
    Creates GitHub Pull Requests for feature branches.

    Parameters
    ----------
    dry_run: bool
        When True, log what would happen but create no PRs.
    llm: ChatOpenAI | None
        LLM instance for generating PR content. If None, uses default.
    """

    REPO_MAP = {
        "orbit": lambda: settings.orbit_repo_github,
        "trinity": lambda: settings.trinity_repo_github,
    }

    def __init__(self, dry_run: bool = True, llm: Optional[ChatOpenAI] = None):
        self.dry_run = dry_run
        self._gh = None  # lazy-loaded
        self._llm = llm or ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.3,  # Slight creativity for PR writing
        )
        self._audit = get_audit_logger()

    def _get_github(self):
        if self._gh is None:
            if not settings.github_token:
                raise RuntimeError(
                    "GITHUB_TOKEN is not set. Add it to .env to enable PR creation."
                )
            from github import Github
            self._gh = Github(settings.github_token)
        return self._gh

    def _get_github_repo(self, repo_slug: str):
        """Return a PyGitHub Repository object."""
        gh = self._get_github()
        return gh.get_repo(repo_slug)

    def _validate_base_branch(self, base_branch: str) -> None:
        """
        Validate that base_branch is in the allowed list.
        
        Raises SafetyViolation if not allowed.
        """
        # Check against settings first, fall back to hardcoded list
        try:
            allowed = set(getattr(settings, 'allowed_pr_base_branches', ALLOWED_BASE_BRANCHES))
        except Exception:
            allowed = ALLOWED_BASE_BRANCHES
        
        if base_branch not in allowed:
            self._audit.log_safety_violation(
                f"PR base branch '{base_branch}' not in allowed list: {allowed}",
                details={"base_branch": base_branch, "allowed": list(allowed)},
            )
            raise SafetyViolation(
                f"Base branch '{base_branch}' is not allowed. "
                f"PRs can only target: {', '.join(sorted(allowed))}.",
                details={"base_branch": base_branch, "allowed": list(allowed)},
            )

    def _validate_repo_slug(self, repo_slug: str, repo_name: str) -> None:
        """
        Validate that repo_slug matches expected configuration.
        
        Prevents PRs being opened against arbitrary repos.
        """
        allowed_slugs = {
            settings.orbit_repo_github,
            settings.trinity_repo_github,
        }
        
        if repo_slug not in allowed_slugs:
            self._audit.log_safety_violation(
                f"Repo slug '{repo_slug}' not in allowed list",
                repo=repo_name,
                details={"repo_slug": repo_slug, "allowed": list(allowed_slugs)},
            )
            raise SafetyViolation(
                f"Repository '{repo_slug}' is not in the allowed list. "
                f"PRs can only be created for: {', '.join(sorted(allowed_slugs))}.",
                details={"repo_slug": repo_slug, "allowed": list(allowed_slugs)},
            )

    def _strip_secrets_from_body(self, body: str) -> str:
        """
        Remove any accidental secrets from PR body before posting.
        
        Uses regex patterns to identify and redact potential secrets.
        """
        cleaned = body
        for pattern, replacement in PR_BODY_SECRET_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        if cleaned != body:
            logger.warning("Stripped potential secrets from PR body before posting")
            self._audit.log(
                AuditEventType.SAFETY_VIOLATION,
                error="Stripped secrets from PR body",
                details={"action": "secrets_redacted"},
            )
        
        return cleaned

    def _truncate_body(self, body: str) -> str:
        """
        Truncate PR body to GitHub's limit.
        
        GitHub's limit is 65536 characters. We use 65000 for safety margin.
        """
        if len(body) <= MAX_PR_BODY_LENGTH:
            return body
        
        truncation_notice = (
            "\n\n---\n"
            "*⚠️ This PR description was truncated due to length limits. "
            "See the linked implementation plan for full details.*"
        )
        
        # Truncate to leave room for notice
        max_content = MAX_PR_BODY_LENGTH - len(truncation_notice)
        truncated = body[:max_content] + truncation_notice
        
        logger.warning(f"PR body truncated from {len(body)} to {len(truncated)} characters")
        
        return truncated

    def create_prs(
        self,
        tracker: ChangeTracker,
        feature_title: str,
        feature_description: str,
        branch_names: dict[str, str],   # {repo_name: branch_name}
        plan_text: Optional[str] = None,
        base_branch: str = "main",
        orbit_analysis: Optional[Any] = None,
        trinity_analysis: Optional[Any] = None,
    ) -> list[PRResult]:
        """
        Create one PR for each repo that has a recorded feature branch.

        Returns a list of PRResult (one per repo).
        """
        # Validate base branch upfront
        self._validate_base_branch(base_branch)

        results = []

        # Only create PRs for repos that git_operations actually wrote a branch for.
        # Never fall back to tracker.repos_with_changes() — if git_operations didn't
        # create a branch, there is nothing to PR against.
        if not branch_names:
            logger.warning("create_prs called with empty branch_names — no PRs to create")
            return results

        repos_to_pr = list(branch_names.keys())

        for repo_name in repos_to_pr:
            branch = branch_names.get(repo_name)
            if not branch:
                logger.warning(f"No branch name found for repo '{repo_name}'; skipping PR")
                results.append(PRResult(
                    repo_slug=repo_name,
                    branch="",
                    pr_url=None,
                    pr_number=None,
                    dry_run=self.dry_run,
                    error="No branch name recorded",
                ))
                continue

            repo_slug_fn = self.REPO_MAP.get(repo_name)
            if not repo_slug_fn:
                logger.warning(f"Unknown repo '{repo_name}'; cannot determine GitHub slug")
                results.append(PRResult(
                    repo_slug=repo_name,
                    branch=branch,
                    pr_url=None,
                    pr_number=None,
                    dry_run=self.dry_run,
                    error=f"Unknown repo: {repo_name}",
                ))
                continue

            repo_slug = repo_slug_fn()
            
            # Validate repo slug
            try:
                self._validate_repo_slug(repo_slug, repo_name)
            except SafetyViolation as e:
                results.append(PRResult(
                    repo_slug=repo_slug,
                    branch=branch,
                    pr_url=None,
                    pr_number=None,
                    dry_run=self.dry_run,
                    error=str(e),
                ))
                continue
            
            # Get the appropriate analysis for this repo
            analysis = trinity_analysis if repo_name == "trinity" else orbit_analysis
            
            result = self._create_single_pr(
                repo_slug=repo_slug,
                repo_name=repo_name,
                branch=branch,
                base_branch=base_branch,
                feature_title=feature_title,
                feature_description=feature_description,
                tracker=tracker,
                plan_text=plan_text,
                analysis=analysis,
                orbit_analysis=orbit_analysis,
                trinity_analysis=trinity_analysis,
            )
            results.append(result)

        return results

    def _create_single_pr(
        self,
        repo_slug: str,
        repo_name: str,
        branch: str,
        base_branch: str,
        feature_title: str,
        feature_description: str,
        tracker: ChangeTracker,
        plan_text: Optional[str],
        analysis: Optional[Any] = None,
        orbit_analysis: Optional[Any] = None,
        trinity_analysis: Optional[Any] = None,
    ) -> PRResult:
        # Generate LLM-powered PR title and body (Item 1)
        pr_title = self._generate_pr_title(
            feature_description=feature_description,
            repo_name=repo_name,
            tracker=tracker,
        )
        
        body = self._generate_pr_body(
            feature_description=feature_description,
            repo_name=repo_name,
            tracker=tracker,
            plan_text=plan_text,
            orbit_analysis=orbit_analysis,
            trinity_analysis=trinity_analysis,
        )
        
        # Security: strip secrets and truncate
        body = self._strip_secrets_from_body(body)
        body = self._truncate_body(body)

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Would open PR in {repo_slug}:\n"
                f"  title : {pr_title}\n"
                f"  head  : {branch}\n"
                f"  base  : {base_branch}\n"
                f"  body  :\n{body[:500]}..."
            )
            return PRResult(
                repo_slug=repo_slug,
                branch=branch,
                pr_url=f"https://github.com/{repo_slug}/compare/{base_branch}...{branch} (dry-run)",
                pr_number=None,
                dry_run=True,
            )

        try:
            # Track GitHub API calls
            rate_limiter = get_rate_limiter()
            
            gh_repo = self._get_github_repo(repo_slug)
            
            # Safety check: if a PR already exists for this branch, skip creation.
            # We never silently update an existing PR — the caller should
            # explicitly resume from a checkpoint if they want to amend one.
            rate_limiter.check_github_api_call()
            existing_pr = self._find_existing_pr(gh_repo, branch, base_branch)
            rate_limiter.increment_github_api_calls()

            if existing_pr:
                logger.warning(
                    f"PR #{existing_pr.number} already exists for branch '{branch}' in {repo_slug}. "
                    "Skipping creation to avoid duplicates."
                )
                return PRResult(
                    repo_slug=repo_slug,
                    branch=branch,
                    pr_url=existing_pr.html_url,
                    pr_number=existing_pr.number,
                    dry_run=False,
                    was_updated=False,
                    error="PR already exists for this branch — skipped creation.",
                )
            
            # Create new PR
            rate_limiter.check_github_api_call()
            pr = gh_repo.create_pull(
                title=pr_title,
                body=body,
                head=branch,
                base=base_branch,
                draft=True,          # Always open as Draft for human review
            )
            rate_limiter.increment_github_api_calls()
            
            logger.info(f"Opened PR #{pr.number} in {repo_slug}: {pr.html_url}")
            
            self._audit.log_pr_created(repo_slug, pr.number, branch, base_branch)
            
            return PRResult(
                repo_slug=repo_slug,
                branch=branch,
                pr_url=pr.html_url,
                pr_number=pr.number,
                dry_run=False,
            )
        except Exception as e:
            logger.error(f"Failed to create PR in {repo_slug}: {e}")
            self._audit.log_safety_violation(
                f"PR creation failed: {e}",
                repo=repo_slug,
            )
            return PRResult(
                repo_slug=repo_slug,
                branch=branch,
                pr_url=None,
                pr_number=None,
                dry_run=False,
                error=str(e),
            )
    
    def _find_existing_pr(self, gh_repo, head_branch: str, base_branch: str):
        """
        Check for existing open PR from head_branch to base_branch (Item 10).
        
        Returns the PR object if found, None otherwise.
        """
        try:
            # PyGitHub's get_pulls returns PRs matching criteria
            pulls = gh_repo.get_pulls(state="open", head=head_branch, base=base_branch)
            for pr in pulls:
                return pr  # Return first match
        except Exception as e:
            logger.debug(f"Error checking for existing PRs: {e}")
        return None

    def _generate_pr_title(
        self,
        feature_description: str,
        repo_name: str,
        tracker: ChangeTracker,
    ) -> str:
        """
        Generate a conventional-commit-style PR title using LLM (Item 1).
        """
        from agents.prompts import PR_TITLE_PROMPT
        
        summary = tracker.get_summary().get(repo_name, {})
        files = summary.get("files", [])
        file_list = ", ".join(f["path"] for f in files[:10])
        
        prompt = PR_TITLE_PROMPT.format(
            feature_description=feature_description,
            repo_name=repo_name,
            file_list=file_list or "no files changed",
        )
        
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            # Handle if content is a list (shouldn't happen but be defensive)
            if isinstance(content, list):
                content = str(content[0]) if content else ""
            title = content.strip()
            # Ensure max 72 chars
            if len(title) > 72:
                title = title[:69] + "..."
            return title
        except Exception as e:
            logger.warning(f"LLM title generation failed: {e}, using fallback")
            return f"[KB] {feature_description[:50]} ({repo_name})"

    def _generate_pr_body(
        self,
        feature_description: str,
        repo_name: str,
        tracker: ChangeTracker,
        plan_text: Optional[str],
        orbit_analysis: Optional[Any] = None,
        trinity_analysis: Optional[Any] = None,
    ) -> str:
        """
        Generate a professional PR description using LLM (Item 1).
        """
        from agents.prompts import PR_DESCRIPTION_PROMPT
        
        summary = tracker.get_summary().get(repo_name, {})
        files = summary.get("files", [])
        file_list = "\n".join(
            f"- `{f['path']}` ({'new' if f['is_new'] else 'modified'}): +{f['lines_added']}/-{f['lines_removed']}"
            for f in files
        )
        
        # Extract analysis text
        orbit_text = ""
        if orbit_analysis:
            orbit_text = getattr(orbit_analysis, 'analysis', str(orbit_analysis))[:1000]
        
        trinity_text = ""
        if trinity_analysis:
            trinity_text = getattr(trinity_analysis, 'analysis', str(trinity_analysis))[:1000]
        
        prompt = PR_DESCRIPTION_PROMPT.format(
            feature_description=feature_description,
            repo_name=repo_name,
            trinity_analysis=trinity_text or "Not available",
            orbit_analysis=orbit_text or "Not available",
            file_list=file_list or "No files changed",
            plan_text=plan_text or "No implementation plan available",
        )
        
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            # Handle if content is a list
            if isinstance(content, list):
                content = str(content[0]) if content else ""
            body = content.strip()
            
            # Append footer
            body += "\n\n---\n*This PR was created automatically by the Knowledge Base Feature Builder.*\n*Always review AI-generated code carefully before merging.*"
            
            return body
        except Exception as e:
            logger.warning(f"LLM body generation failed: {e}, using fallback")
            return self._build_fallback_body(
                feature_description=feature_description,
                repo_name=repo_name,
                tracker=tracker,
                plan_text=plan_text,
            )
    
    def _build_fallback_body(
        self,
        feature_description: str,
        repo_name: str,
        tracker: ChangeTracker,
        plan_text: Optional[str],
    ) -> str:
        """Fallback to template-based PR body if LLM fails."""
        summary = tracker.get_summary().get(repo_name, {})
        files = summary.get("files", [])

        file_table_rows = "\n".join(
            f"| `{f['path']}` | {'New' if f['is_new'] else 'Modified'} "
            f"| +{f['lines_added']} / -{f['lines_removed']} | {f['description'][:80]} |"
            for f in files
        )

        file_table = (
            "| File | Status | Lines | Description |\n"
            "|------|--------|-------|-------------|\n"
            + file_table_rows
        ) if files else "_No files recorded._"

        plan_section = f"\n## Implementation Plan\n\n```\n{plan_text}\n```\n" if plan_text else ""

        return f"""## Feature Description

{feature_description}

## Changes ({repo_name})

{file_table}

**Summary:** {summary.get('files_modified', 0)} files modified, {summary.get('files_created', 0)} files created, \
+{summary.get('lines_added', 0)} / -{summary.get('lines_removed', 0)} lines.
{plan_section}
## Review Checklist

- [ ] Code logic is correct
- [ ] Tests added or updated
- [ ] No hardcoded secrets or credentials
- [ ] API contracts match the other repo (if cross-repo feature)

---
*This PR was created automatically by the Knowledge Base Feature Builder.*
*Always review AI-generated code carefully before merging.*
"""

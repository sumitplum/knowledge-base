"""
Safety guards for the Feature Builder.

All checks are centralised here so they can be called from both
the orchestrator nodes and the individual tools.

Layer 2 of the Security Guardrails System: Content Security Scanner.
"""

import logging
import math
import re
from pathlib import Path
from typing import Optional

from config import settings

# Import from the central exceptions module
from agents.exceptions import SafetyViolation

logger = logging.getLogger(__name__)

# ── Hardcoded constant lists ──────────────────────────────────────────────────

PROTECTED_BRANCHES = frozenset({"main", "master", "develop", "staging", "production", "release"})

SECRET_FILE_PATTERNS = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    "secrets",
    "credentials",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    "keystore",
    "truststore",
    "id_rsa",
    "id_ed25519",
})


def _get_max_diff_lines() -> int:
    """Get max diff lines from settings or use default."""
    try:
        return getattr(settings, 'max_diff_lines_hard_cap', 2000)
    except Exception:
        return 2000


# Maximum diff size (lines added + removed) per repo — hard fail beyond this
MAX_DIFF_LINES_PER_REPO = _get_max_diff_lines()


# ── Expanded Secret Patterns (Layer 2A) ───────────────────────────────────────

# Each tuple: (pattern, label, is_hard_fail)
# Hard fail patterns are blocked immediately; soft patterns log warnings
SECRET_PATTERNS = [
    # Original patterns (now hard-fail)
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI API key", True),
    (r"ghp_[A-Za-z0-9]{36,}", "GitHub token", True),
    (r"gho_[A-Za-z0-9]{36,}", "GitHub OAuth token", True),
    (r"ghs_[A-Za-z0-9]{36,}", "GitHub server token", True),
    (r"ghr_[A-Za-z0-9]{36,}", "GitHub refresh token", True),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID", True),
    (r"(?i)(aws_secret_access_key|aws_secret)\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}['\"]?", "AWS Secret Key", True),
    
    # Azure connection strings
    (r"DefaultEndpointsProtocol=https;AccountName=\w+;AccountKey=[A-Za-z0-9+/=]{88}", "Azure Storage connection string", True),
    (r"(?i)azure[_-]?(storage|account)?[_-]?key\s*[=:]\s*['\"]?[A-Za-z0-9+/=]{44,}", "Azure Storage Key", True),
    
    # GCP service account
    (r'"type"\s*:\s*"service_account"', "GCP service account JSON", True),
    (r'"private_key_id"\s*:\s*"[a-f0-9]{40}"', "GCP private key ID", True),
    
    # Stripe keys
    (r"sk_live_[A-Za-z0-9]{24,}", "Stripe live secret key", True),
    (r"rk_live_[A-Za-z0-9]{24,}", "Stripe live restricted key", True),
    (r"sk_test_[A-Za-z0-9]{24,}", "Stripe test secret key", False),  # Test keys are warnings
    
    # Twilio
    (r"AC[a-f0-9]{32}", "Twilio Account SID", True),
    (r"SK[a-f0-9]{32}", "Twilio API Key SID", True),
    
    # Private key blocks
    (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "Private key block", True),
    (r"-----BEGIN PGP PRIVATE KEY BLOCK-----", "PGP private key block", True),
    
    # JWT secrets (HS256/HS512 with long base64)
    (r"(?i)(jwt[_-]?secret|hs256|hs384|hs512)[_-]?(secret|key)?\s*[=:]\s*['\"]?[A-Za-z0-9+/=]{32,}['\"]?", "JWT secret", True),
    
    # Database URLs with credentials
    (r"(?i)(postgres(ql)?|mysql|mongodb(\+srv)?|redis)://[^:]+:[^@]{6,}@[^\s]+", "Database URL with credentials", True),
    (r"(?i)mongodb\+srv://[^:]+:[^@]+@", "MongoDB Atlas connection string", True),
    
    # Generic credential patterns (more specific than before)
    (r"(?i)(api[_-]?key|secret[_-]?key|private[_-]?key|auth[_-]?token)\s*[=:]\s*['\"][^'\"]{20,}['\"]", "Hardcoded API/secret key", True),
    (r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]", "Hardcoded password", True),
    
    # Slack tokens
    (r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}", "Slack token", True),
    
    # SendGrid
    (r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}", "SendGrid API key", True),
    
    # Mailchimp
    (r"[a-f0-9]{32}-us[0-9]{1,2}", "Mailchimp API key", True),
    
    # Square
    (r"sq0atp-[A-Za-z0-9_-]{22}", "Square access token", True),
    (r"sq0csp-[A-Za-z0-9_-]{43}", "Square OAuth secret", True),
]


class ContentScanner:
    """
    Content security scanner for generated code.
    
    Implements Layer 2 of the Security Guardrails System:
    - Expanded secret patterns with hard-fail
    - Shannon entropy scoring
    - Binary/compiled file detection
    - Symlink traversal prevention
    """
    
    # Entropy thresholds
    ENTROPY_WARNING_THRESHOLD = 4.5
    ENTROPY_HARD_FAIL_THRESHOLD = 5.5
    MIN_ENTROPY_STRING_LENGTH = 20
    
    # Patterns to exclude from entropy checking (UUIDs, URLs, hashes)
    ENTROPY_EXCLUSION_PATTERNS = [
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",  # UUID
        r"https?://[^\s'\"]+",  # URLs
        r"[0-9a-f]{32}",  # MD5 hash
        r"[0-9a-f]{40}",  # SHA1 hash
        r"[0-9a-f]{64}",  # SHA256 hash
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO timestamps
    ]
    
    @staticmethod
    def _entropy_score(s: str) -> float:
        """
        Calculate Shannon entropy of a string.
        
        Higher entropy = more random = more likely to be a secret.
        """
        if not s:
            return 0.0
        
        # Count character frequencies
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        length = len(s)
        entropy = 0.0
        for count in freq.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    @classmethod
    def _is_excluded_from_entropy(cls, s: str) -> bool:
        """Check if string matches exclusion patterns (UUID, URL, etc.)."""
        for pattern in cls.ENTROPY_EXCLUSION_PATTERNS:
            if re.fullmatch(pattern, s, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def _extract_string_literals(cls, content: str) -> list[str]:
        """Extract string literals from code content."""
        # Match single or double quoted strings
        pattern = r'["\']([^"\']{20,})["\']'
        return re.findall(pattern, content)
    
    @classmethod
    def check_secrets(cls, content: str, file_path: str) -> list[tuple[str, bool]]:
        """
        Scan content for secret patterns.
        
        Returns list of (message, is_hard_fail) tuples.
        """
        violations = []
        
        for pattern, label, is_hard_fail in SECRET_PATTERNS:
            if re.search(pattern, content):
                message = f"Detected {label} in '{file_path}'"
                violations.append((message, is_hard_fail))
        
        return violations
    
    @classmethod
    def check_entropy(cls, content: str, file_path: str) -> list[tuple[str, bool]]:
        """
        Check for high-entropy strings that may be secrets.
        
        Returns list of (message, is_hard_fail) tuples.
        """
        violations = []
        
        for literal in cls._extract_string_literals(content):
            # Skip if too short or matches exclusion patterns
            if len(literal) < cls.MIN_ENTROPY_STRING_LENGTH:
                continue
            if cls._is_excluded_from_entropy(literal):
                continue
            
            entropy = cls._entropy_score(literal)
            
            if entropy > cls.ENTROPY_HARD_FAIL_THRESHOLD:
                # Truncate the literal for logging
                truncated = literal[:20] + "..." if len(literal) > 20 else literal
                message = (
                    f"High-entropy string detected in '{file_path}': "
                    f"'{truncated}' (entropy: {entropy:.2f} > {cls.ENTROPY_HARD_FAIL_THRESHOLD})"
                )
                violations.append((message, True))
            elif entropy > cls.ENTROPY_WARNING_THRESHOLD:
                truncated = literal[:20] + "..." if len(literal) > 20 else literal
                message = (
                    f"Possible secret in '{file_path}': "
                    f"'{truncated}' (entropy: {entropy:.2f})"
                )
                violations.append((message, False))
        
        return violations
    
    @classmethod
    def check_binary_content(cls, content: str, file_path: str) -> Optional[tuple[str, bool]]:
        """
        Check if content contains binary/non-UTF-8 data.
        
        Returns (message, is_hard_fail) or None.
        """
        # Check for null bytes
        if '\x00' in content:
            return (
                f"Binary content detected in '{file_path}': contains null bytes. "
                "Feature Builder cannot write binary files.",
                True,
            )
        
        # Check for high concentration of control characters
        control_chars = sum(1 for c in content if ord(c) < 32 and c not in '\n\r\t')
        if len(content) > 0 and control_chars / len(content) > 0.1:
            return (
                f"Binary content detected in '{file_path}': high concentration of control characters.",
                True,
            )
        
        return None
    
    @classmethod
    def check_symlink_traversal(cls, abs_path: Path, repo_root: Path) -> Optional[tuple[str, bool]]:
        """
        Check for symlink traversal attacks.
        
        If the resolved path differs from the original (after resolving symlinks),
        the path may be attempting to escape the repo root.
        
        Returns (message, is_hard_fail) or None.
        """
        try:
            # Resolve the path following symlinks
            resolved = abs_path.resolve()
            
            # Check if resolved path is still within repo root
            try:
                resolved.relative_to(repo_root.resolve())
            except ValueError:
                return (
                    f"Symlink traversal detected: '{abs_path}' resolves to '{resolved}' "
                    f"which is outside the repo root '{repo_root}'.",
                    True,
                )
            
            # Additional check: if the path before resolution differs significantly
            # from the resolved path, it might be suspicious
            if abs_path.resolve() != abs_path.absolute():
                logger.debug(
                    f"Path '{abs_path}' resolves differently: '{resolved}'. "
                    "This may indicate symlinks in the path."
                )
        
        except OSError as e:
            # Path doesn't exist yet, which is fine for new files
            pass
        
        return None
    
    @classmethod
    def scan_content(
        cls,
        content: str,
        file_path: str,
        abs_path: Optional[Path] = None,
        repo_root: Optional[Path] = None,
        hard_fail: bool = True,
    ) -> None:
        """
        Perform all content security scans.
        
        Parameters
        ----------
        content : str
            The content to scan.
        file_path : str
            Relative file path for logging.
        abs_path : Path, optional
            Absolute file path for symlink checking.
        repo_root : Path, optional
            Repository root for symlink checking.
        hard_fail : bool
            If True, raise SafetyViolation on hard-fail patterns.
            If False, log warnings for all violations.
        
        Raises
        ------
        SafetyViolation
            If hard_fail is True and a hard-fail violation is found.
        """
        all_violations = []
        
        # Check for binary content first
        binary_result = cls.check_binary_content(content, file_path)
        if binary_result:
            all_violations.append(binary_result)
        
        # Check for secrets
        all_violations.extend(cls.check_secrets(content, file_path))
        
        # Check entropy
        all_violations.extend(cls.check_entropy(content, file_path))
        
        # Check symlink traversal
        if abs_path and repo_root:
            symlink_result = cls.check_symlink_traversal(abs_path, repo_root)
            if symlink_result:
                all_violations.append(symlink_result)
        
        # Process violations
        hard_fails = [(msg, hf) for msg, hf in all_violations if hf]
        warnings = [(msg, hf) for msg, hf in all_violations if not hf]
        
        # Log warnings
        for msg, _ in warnings:
            logger.warning(f"⚠️  {msg}")
        
        # Handle hard failures
        if hard_fails and hard_fail:
            messages = [msg for msg, _ in hard_fails]
            raise SafetyViolation(
                f"Content security scan failed:\n" + "\n".join(f"  - {m}" for m in messages),
                details={"violations": messages, "file_path": file_path},
            )
        elif hard_fails:
            for msg, _ in hard_fails:
                logger.error(f"🚫 {msg}")


def check_branch_name(branch_name: str) -> None:
    """Raise SafetyViolation if branch_name is or resolves to a protected branch."""
    base = branch_name.split("/")[-1]
    if branch_name in PROTECTED_BRANCHES or base in PROTECTED_BRANCHES:
        raise SafetyViolation(
            f"Refusing to operate on protected branch '{branch_name}'. "
            "Feature Builder only creates new feature branches."
        )


def check_file_path(abs_path: Path, repo_root: Path) -> None:
    """Raise SafetyViolation if the file is outside the repo or looks secret-like."""
    # Must be inside repo root
    try:
        abs_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        raise SafetyViolation(
            f"File path '{abs_path}' is outside the repo root '{repo_root}'. "
            "Feature Builder cannot write outside the repository."
        )
    
    # Symlink traversal check
    symlink_result = ContentScanner.check_symlink_traversal(abs_path, repo_root)
    if symlink_result:
        raise SafetyViolation(symlink_result[0])

    # Check name against secret patterns
    name_lower = abs_path.name.lower()
    parts_lower = {p.lower() for p in abs_path.parts}
    for pattern in SECRET_FILE_PATTERNS:
        if pattern in name_lower or pattern in parts_lower:
            raise SafetyViolation(
                f"Refusing to write potentially sensitive file: {abs_path}\n"
                f"Matched pattern: '{pattern}'"
            )


def check_file_count(repo: str, proposed_count: int) -> None:
    """Raise SafetyViolation if too many files would be modified."""
    limit = settings.max_files_per_repo
    if proposed_count > limit:
        raise SafetyViolation(
            f"Proposed {proposed_count} file changes in '{repo}' exceeds the "
            f"MAX_FILES_PER_REPO limit of {limit}. "
            "Reduce the scope or increase the limit in .env."
        )


def check_diff_size(repo: str, lines_added: int, lines_removed: int) -> None:
    """
    HARD FAIL when diff is very large.
    
    Changed from warn-only to hard-fail per security guardrails plan.
    """
    total = lines_added + lines_removed
    max_lines = _get_max_diff_lines()
    
    if total > max_lines:
        raise SafetyViolation(
            f"Diff size limit exceeded in '{repo}': +{lines_added} / -{lines_removed} lines "
            f"({total} total, limit {max_lines}). "
            "Split the feature into smaller PRs or increase max_diff_lines_hard_cap in settings."
        )


def check_no_secrets_in_content(content: str, file_path: str) -> None:
    """
    Scan for common secret patterns in generated content.
    
    Now uses ContentScanner for comprehensive detection with hard-fail.
    """
    ContentScanner.scan_content(content, file_path, hard_fail=True)


def run_all_checks(
    repo: str,
    file_path: str,
    abs_path: Path,
    repo_root: Path,
    content: str,
    proposed_file_count: Optional[int] = None,
) -> None:
    """
    Run all safety checks for a single file edit.
    Raises SafetyViolation on hard failures; logs warnings for soft ones.
    """
    check_file_path(abs_path, repo_root)
    if proposed_file_count is not None:
        check_file_count(repo, proposed_file_count)
    
    # Use the enhanced content scanner
    ContentScanner.scan_content(
        content=content,
        file_path=file_path,
        abs_path=abs_path,
        repo_root=repo_root,
        hard_fail=True,
    )


# Legacy alias for backward compatibility
# (SafetyViolation is now imported from agents.exceptions)
__all__ = [
    'SafetyViolation',
    'ContentScanner',
    'check_branch_name',
    'check_file_path',
    'check_file_count',
    'check_diff_size',
    'check_no_secrets_in_content',
    'run_all_checks',
    'PROTECTED_BRANCHES',
    'SECRET_FILE_PATTERNS',
    'SECRET_PATTERNS',
]

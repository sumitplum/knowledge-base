"""
Security exceptions hierarchy for the Knowledge Base system.

All security-related exceptions derive from KBBaseError, allowing
unified handling at the top level.
"""

from typing import Optional


class KBBaseError(Exception):
    """Base exception for all Knowledge Base errors."""
    pass


class SecurityViolation(KBBaseError):
    """
    Raised when IntentGuard blocks a query at the input layer.
    
    This is a query-level block before any execution begins.
    """
    
    CATEGORIES = frozenset({
        "destructive_git",
        "secret_exfiltration",
        "privilege_escalation",
        "social_engineering",
        "scope_explosion",
        "data_deletion",
        "repository_manipulation",
    })
    
    def __init__(
        self,
        message: str,
        category: Optional[str] = None,
        reasoning: Optional[str] = None,
        sanitized_query: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.reasoning = reasoning
        self.sanitized_query = sanitized_query
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.category:
            parts.append(f"[Category: {self.category}]")
        if self.reasoning:
            parts.append(f"Reason: {self.reasoning}")
        return " ".join(parts)
    
    def to_dict(self) -> dict:
        """Serialize for JSON logging and UI display."""
        return {
            "type": "SecurityViolation",
            "message": self.message,
            "category": self.category,
            "reasoning": self.reasoning,
            "sanitized_query": self.sanitized_query,
        }


class SafetyViolation(KBBaseError):
    """
    Raised when an operation-level safety check fails.
    
    Used by ContentScanner, GitOps, and PRCreator for:
    - Secret detection in generated code
    - Path traversal attempts
    - Protected branch operations
    - Diff size limits exceeded
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        return self.message
    
    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "type": "SafetyViolation",
            "message": self.message,
            "details": self.details,
        }


class RateLimitExceeded(KBBaseError):
    """
    Raised when a rate limit is exceeded during a session.
    
    Tracks which resource hit its limit and the current count.
    """
    
    def __init__(self, resource: str, limit: int, current: int):
        message = f"Rate limit exceeded for {resource}: {current}/{limit}"
        super().__init__(message)
        self.resource = resource
        self.limit = limit
        self.current = current
    
    def __str__(self) -> str:
        return f"Rate limit exceeded for {self.resource}: {self.current}/{self.limit}"
    
    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "type": "RateLimitExceeded",
            "resource": self.resource,
            "limit": self.limit,
            "current": self.current,
        }


class AuditError(KBBaseError):
    """
    Raised when the audit log cannot be written.
    
    This is a critical error that should not silently fail.
    """
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
    
    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message}: {self.original_error}"
        return self.message
    
    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "type": "AuditError",
            "message": self.message,
            "original_error": str(self.original_error) if self.original_error else None,
        }

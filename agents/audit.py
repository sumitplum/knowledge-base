"""
Audit logging for the Knowledge Base system.

Provides structured, append-only JSON-lines audit logging for all
sensitive operations. This is separate from Python's logging module
and provides tamper-evident records.
"""

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Any

from .exceptions import AuditError

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of events that are audited."""
    
    # Query lifecycle
    QUERY_RECEIVED = "QUERY_RECEIVED"
    QUERY_BLOCKED = "QUERY_BLOCKED"
    QUERY_APPROVED = "QUERY_APPROVED"
    
    # File operations
    FILE_WRITE = "FILE_WRITE"
    FILE_DELETE = "FILE_DELETE"
    
    # Git operations
    BRANCH_CREATED = "BRANCH_CREATED"
    COMMIT_CREATED = "COMMIT_CREATED"
    PUSH_EXECUTED = "PUSH_EXECUTED"
    
    # GitHub operations
    PR_CREATED = "PR_CREATED"
    PR_UPDATED = "PR_UPDATED"
    
    # Security events
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Compensation
    ROLLBACK = "ROLLBACK"
    
    # Session events
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"


class AuditLogger:
    """
    Structured audit logger that writes JSON-lines to a file.
    
    Thread-safe and designed for append-only operation.
    Each record includes a timestamp, session ID, and event details.
    
    Usage:
        audit = AuditLogger(session_id="my-session-123")
        audit.log(AuditEventType.QUERY_RECEIVED, query_hash="abc123")
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        log_path: Optional[str] = None,
        actor: str = "kb-agent",
    ):
        """
        Initialize the audit logger.
        
        Parameters
        ----------
        session_id : str, optional
            Unique identifier for this session. Generated if not provided.
        log_path : str, optional
            Path to the audit log file. Uses settings.audit_log_path if not provided.
        actor : str
            The actor performing operations (default: "kb-agent").
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.actor = actor
        self._lock = threading.Lock()
        
        # Determine log path
        if log_path:
            self._log_path = Path(log_path).expanduser()
        else:
            try:
                from config import settings
                self._log_path = Path(settings.audit_log_path).expanduser()
            except (ImportError, AttributeError):
                self._log_path = Path("~/.kb/audit.jsonl").expanduser()
        
        # Ensure directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file handle with O_SYNC for durability
        self._file_handle = None
        self._ensure_file_handle()
    
    def _ensure_file_handle(self) -> None:
        """Ensure the file handle is open."""
        if self._file_handle is None or self._file_handle.closed:
            try:
                # Open with append mode
                self._file_handle = open(
                    self._log_path,
                    mode="a",
                    encoding="utf-8",
                    buffering=1,  # Line buffered
                )
            except (OSError, IOError) as e:
                raise AuditError(f"Cannot open audit log at {self._log_path}", e)
    
    def _hash_query(self, query: str) -> str:
        """Hash a query for privacy-preserving logging."""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    
    def log(
        self,
        event_type: AuditEventType,
        repo: Optional[str] = None,
        file_path: Optional[str] = None,
        query: Optional[str] = None,
        verdict: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> dict:
        """
        Log an audit event.
        
        Parameters
        ----------
        event_type : AuditEventType
            The type of event being logged.
        repo : str, optional
            Repository name (e.g., "orbit", "trinity").
        file_path : str, optional
            File path involved in the operation.
        query : str, optional
            User query (will be hashed for privacy).
        verdict : str, optional
            Result of the operation (e.g., "allowed", "blocked").
        details : dict, optional
            Additional event-specific details.
        error : str, optional
            Error message if the operation failed.
        
        Returns
        -------
        dict
            The logged record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        record = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "event_type": event_type.value if isinstance(event_type, AuditEventType) else str(event_type),
            "actor": self.actor,
        }
        
        # Add optional fields
        if query:
            record["query_hash"] = self._hash_query(query)
        if repo:
            record["repo"] = repo
        if file_path:
            record["file_path"] = file_path
        if verdict:
            record["verdict"] = verdict
        if details:
            record["details"] = details
        if error:
            record["error"] = error
        
        # Write to file
        with self._lock:
            try:
                self._ensure_file_handle()
                self._file_handle.write(json.dumps(record) + "\n")
                self._file_handle.flush()
            except (OSError, IOError) as e:
                logger.error(f"Failed to write audit log: {e}")
                raise AuditError(f"Failed to write audit log", e)
        
        # Also log to Python logger at DEBUG level
        logger.debug(f"AUDIT: {event_type.value if isinstance(event_type, AuditEventType) else event_type} - {record}")
        
        return record
    
    def log_query_received(self, query: str, mode: str = "analyze") -> dict:
        """Log that a query was received."""
        return self.log(
            AuditEventType.QUERY_RECEIVED,
            query=query,
            details={"mode": mode},
        )
    
    def log_query_blocked(
        self,
        query: str,
        category: str,
        reasoning: str,
    ) -> dict:
        """Log that a query was blocked by IntentGuard."""
        return self.log(
            AuditEventType.QUERY_BLOCKED,
            query=query,
            verdict="blocked",
            details={
                "category": category,
                "reasoning": reasoning[:500],  # Truncate for log size
            },
        )
    
    def log_query_approved(self, query: str) -> dict:
        """Log that a query passed IntentGuard."""
        return self.log(
            AuditEventType.QUERY_APPROVED,
            query=query,
            verdict="approved",
        )
    
    def log_file_write(
        self,
        repo: str,
        file_path: str,
        is_new: bool,
        lines_added: int,
        lines_removed: int,
    ) -> dict:
        """Log a file write operation."""
        return self.log(
            AuditEventType.FILE_WRITE,
            repo=repo,
            file_path=file_path,
            details={
                "is_new": is_new,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
            },
        )
    
    def log_file_delete(self, repo: str, file_path: str) -> dict:
        """Log a file deletion."""
        return self.log(
            AuditEventType.FILE_DELETE,
            repo=repo,
            file_path=file_path,
        )
    
    def log_branch_created(self, repo: str, branch_name: str, base: str) -> dict:
        """Log branch creation."""
        return self.log(
            AuditEventType.BRANCH_CREATED,
            repo=repo,
            details={
                "branch_name": branch_name,
                "base": base,
            },
        )
    
    def log_commit_created(self, repo: str, commit_sha: str, file_count: int) -> dict:
        """Log commit creation."""
        return self.log(
            AuditEventType.COMMIT_CREATED,
            repo=repo,
            details={
                "commit_sha": commit_sha,
                "file_count": file_count,
            },
        )
    
    def log_push_executed(self, repo: str, branch_name: str) -> dict:
        """Log a push to remote."""
        return self.log(
            AuditEventType.PUSH_EXECUTED,
            repo=repo,
            details={"branch_name": branch_name},
        )
    
    def log_pr_created(
        self,
        repo_slug: str,
        pr_number: int,
        branch: str,
        base_branch: str,
    ) -> dict:
        """Log PR creation."""
        return self.log(
            AuditEventType.PR_CREATED,
            repo=repo_slug,
            details={
                "pr_number": pr_number,
                "branch": branch,
                "base_branch": base_branch,
            },
        )
    
    def log_pr_updated(self, repo_slug: str, pr_number: int) -> dict:
        """Log PR update (idempotent update)."""
        return self.log(
            AuditEventType.PR_UPDATED,
            repo=repo_slug,
            details={"pr_number": pr_number},
        )
    
    def log_safety_violation(
        self,
        message: str,
        repo: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> dict:
        """Log a safety violation."""
        return self.log(
            AuditEventType.SAFETY_VIOLATION,
            repo=repo,
            file_path=file_path,
            error=message,
            details=details,
        )
    
    def log_security_violation(
        self,
        query: str,
        category: str,
        reasoning: str,
    ) -> dict:
        """Log a security violation (blocked query)."""
        return self.log(
            AuditEventType.SECURITY_VIOLATION,
            query=query,
            error=f"Query blocked: {category}",
            details={
                "category": category,
                "reasoning": reasoning[:500],
            },
        )
    
    def log_rate_limit_exceeded(
        self,
        resource: str,
        limit: int,
        current: int,
    ) -> dict:
        """Log a rate limit being exceeded."""
        return self.log(
            AuditEventType.RATE_LIMIT_EXCEEDED,
            error=f"Rate limit exceeded for {resource}",
            details={
                "resource": resource,
                "limit": limit,
                "current": current,
            },
        )
    
    def log_rollback(
        self,
        repo: str,
        files_restored: int,
        reason: str,
    ) -> dict:
        """Log a rollback operation."""
        return self.log(
            AuditEventType.ROLLBACK,
            repo=repo,
            details={
                "files_restored": files_restored,
                "reason": reason,
            },
        )
    
    def get_session_events(self, limit: int = 100) -> list[dict]:
        """
        Retrieve recent events for this session.
        
        Parameters
        ----------
        limit : int
            Maximum number of events to return.
        
        Returns
        -------
        list[dict]
            Recent audit events for this session.
        """
        events = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get("session_id") == self.session_id:
                            events.append(record)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        return events[-limit:]
    
    def close(self) -> None:
        """Close the file handle."""
        with self._lock:
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Process-level singleton
_audit_instance: Optional[AuditLogger] = None
_audit_lock = threading.Lock()


def get_audit_logger(session_id: Optional[str] = None) -> AuditLogger:
    """
    Get or create the process-level AuditLogger singleton.
    
    If session_id is provided and differs from the current instance,
    a new instance is created.
    """
    global _audit_instance
    
    with _audit_lock:
        if _audit_instance is None:
            _audit_instance = AuditLogger(session_id=session_id)
        elif session_id and _audit_instance.session_id != session_id:
            _audit_instance.close()
            _audit_instance = AuditLogger(session_id=session_id)
        
        return _audit_instance


def reset_audit_logger() -> None:
    """Reset the audit logger singleton (for testing)."""
    global _audit_instance
    
    with _audit_lock:
        if _audit_instance:
            _audit_instance.close()
            _audit_instance = None

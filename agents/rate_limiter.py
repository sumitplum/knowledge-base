"""
Rate limiter for the Knowledge Base system.

Layer 5 of the Security Guardrails System: Rate Limiting and Cost Controls.

Provides per-session rate limiting for:
- LLM API calls
- GitHub API calls
- File write operations
- Build operations (hourly)
- Token budget tracking
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .exceptions import RateLimitExceeded
from .audit import get_audit_logger, AuditEventType

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limits."""
    max_llm_calls_per_session: int = 50
    max_github_api_calls_per_session: int = 20
    max_file_writes_per_session: int = 30
    max_builds_per_hour: int = 5
    max_tokens_per_session: int = 200_000
    
    @classmethod
    def from_settings(cls) -> "RateLimitConfig":
        """Create config from settings, with fallbacks."""
        try:
            from config import settings
            return cls(
                max_llm_calls_per_session=getattr(settings, 'max_llm_calls_per_session', 50),
                max_github_api_calls_per_session=getattr(settings, 'max_github_api_calls_per_session', 20),
                max_file_writes_per_session=getattr(settings, 'max_file_writes_per_session', 30),
                max_builds_per_hour=getattr(settings, 'max_builds_per_hour', 5),
                max_tokens_per_session=getattr(settings, 'max_tokens_per_session', 200_000),
            )
        except Exception:
            return cls()


@dataclass
class SessionCounters:
    """Thread-safe counters for a single session."""
    llm_calls: int = 0
    github_api_calls: int = 0
    file_writes: int = 0
    total_tokens: int = 0
    build_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        self._lock = threading.Lock()
    
    def increment_llm_calls(self, count: int = 1) -> int:
        with self._lock:
            self.llm_calls += count
            return self.llm_calls
    
    def increment_github_api_calls(self, count: int = 1) -> int:
        with self._lock:
            self.github_api_calls += count
            return self.github_api_calls
    
    def increment_file_writes(self, count: int = 1) -> int:
        with self._lock:
            self.file_writes += count
            return self.file_writes
    
    def add_tokens(self, tokens: int) -> int:
        with self._lock:
            self.total_tokens += tokens
            return self.total_tokens
    
    def record_build(self) -> int:
        """Record a build and return count in last hour."""
        with self._lock:
            now = time.time()
            self.build_timestamps.append(now)
            
            # Count builds in last hour
            hour_ago = now - 3600
            recent = [ts for ts in self.build_timestamps if ts > hour_ago]
            return len(recent)
    
    def builds_in_last_hour(self) -> int:
        """Count builds in the last hour."""
        with self._lock:
            now = time.time()
            hour_ago = now - 3600
            return sum(1 for ts in self.build_timestamps if ts > hour_ago)
    
    def reset(self):
        """Reset all counters."""
        with self._lock:
            self.llm_calls = 0
            self.github_api_calls = 0
            self.file_writes = 0
            self.total_tokens = 0
            self.build_timestamps.clear()


class SessionRateLimiter:
    """
    Process-singleton rate limiter for a session.
    
    Thread-safe and tracks multiple resource types.
    
    Usage:
        limiter = get_rate_limiter()
        limiter.check_llm_call()  # Raises RateLimitExceeded if over limit
        limiter.increment_llm_calls()
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.
        
        Parameters
        ----------
        config : RateLimitConfig, optional
            Rate limit configuration. Uses settings-based config if not provided.
        """
        self.config = config or RateLimitConfig.from_settings()
        self.counters = SessionCounters()
        self._lock = threading.Lock()
        self._audit = get_audit_logger()
    
    def _check_and_log(
        self,
        resource: str,
        current: int,
        limit: int,
    ) -> None:
        """Check limit and raise/log if exceeded."""
        if current >= limit:
            self._audit.log_rate_limit_exceeded(resource, limit, current)
            raise RateLimitExceeded(resource, limit, current)
    
    # ── LLM Calls ─────────────────────────────────────────────────────────────
    
    def check_llm_call(self) -> None:
        """
        Check if an LLM call is allowed.
        
        Raises RateLimitExceeded if limit is reached.
        """
        self._check_and_log(
            "llm_calls",
            self.counters.llm_calls,
            self.config.max_llm_calls_per_session,
        )
    
    def increment_llm_calls(self, count: int = 1) -> int:
        """
        Increment LLM call counter and return new value.
        
        Should be called after a successful LLM call.
        """
        new_count = self.counters.increment_llm_calls(count)
        logger.debug(f"LLM calls: {new_count}/{self.config.max_llm_calls_per_session}")
        return new_count
    
    def track_llm_call(self) -> int:
        """
        Check limit, increment, and return count.
        
        Convenience method that combines check and increment.
        """
        self.check_llm_call()
        return self.increment_llm_calls()
    
    # ── GitHub API Calls ──────────────────────────────────────────────────────
    
    def check_github_api_call(self) -> None:
        """
        Check if a GitHub API call is allowed.
        
        Raises RateLimitExceeded if limit is reached.
        """
        self._check_and_log(
            "github_api_calls",
            self.counters.github_api_calls,
            self.config.max_github_api_calls_per_session,
        )
    
    def increment_github_api_calls(self, count: int = 1) -> int:
        """Increment GitHub API call counter."""
        new_count = self.counters.increment_github_api_calls(count)
        logger.debug(f"GitHub API calls: {new_count}/{self.config.max_github_api_calls_per_session}")
        return new_count
    
    def track_github_api_call(self) -> int:
        """Check limit, increment, and return count."""
        self.check_github_api_call()
        return self.increment_github_api_calls()
    
    # ── File Writes ───────────────────────────────────────────────────────────
    
    def check_file_write(self) -> None:
        """
        Check if a file write is allowed.
        
        Raises RateLimitExceeded if limit is reached.
        """
        self._check_and_log(
            "file_writes",
            self.counters.file_writes,
            self.config.max_file_writes_per_session,
        )
    
    def increment_file_writes(self, count: int = 1) -> int:
        """Increment file write counter."""
        new_count = self.counters.increment_file_writes(count)
        logger.debug(f"File writes: {new_count}/{self.config.max_file_writes_per_session}")
        return new_count
    
    def track_file_write(self) -> int:
        """Check limit, increment, and return count."""
        self.check_file_write()
        return self.increment_file_writes()
    
    # ── Build Rate ────────────────────────────────────────────────────────────
    
    def check_build_rate(self) -> None:
        """
        Check if a build is allowed (hourly rate limit).
        
        Raises RateLimitExceeded if too many builds in the last hour.
        """
        builds = self.counters.builds_in_last_hour()
        self._check_and_log(
            "builds_per_hour",
            builds,
            self.config.max_builds_per_hour,
        )
    
    def record_build(self) -> int:
        """Record a build and return count in last hour."""
        return self.counters.record_build()
    
    def track_build(self) -> int:
        """Check rate limit, record build, and return count."""
        self.check_build_rate()
        return self.record_build()
    
    # ── Token Budget ──────────────────────────────────────────────────────────
    
    def check_token_budget(self, tokens_to_use: int = 0) -> None:
        """
        Check if token budget allows the operation.
        
        Parameters
        ----------
        tokens_to_use : int
            Number of tokens about to be used.
        
        Raises RateLimitExceeded if budget would be exceeded.
        """
        projected = self.counters.total_tokens + tokens_to_use
        if projected > self.config.max_tokens_per_session:
            self._audit.log_rate_limit_exceeded(
                "token_budget",
                self.config.max_tokens_per_session,
                projected,
            )
            raise RateLimitExceeded(
                "token_budget",
                self.config.max_tokens_per_session,
                projected,
            )
    
    def add_tokens(self, tokens: int) -> int:
        """
        Add tokens to the session total.
        
        Should be called after getting token usage from LLM response.
        """
        new_total = self.counters.add_tokens(tokens)
        logger.debug(f"Token usage: {new_total}/{self.config.max_tokens_per_session}")
        return new_total
    
    def track_tokens(self, tokens: int) -> int:
        """Check budget, add tokens, and return total."""
        self.check_token_budget(tokens)
        return self.add_tokens(tokens)
    
    # ── Utility Methods ───────────────────────────────────────────────────────
    
    def get_usage_summary(self) -> dict:
        """
        Get current usage summary.
        
        Returns a dict with current counts and limits.
        """
        return {
            "llm_calls": {
                "current": self.counters.llm_calls,
                "limit": self.config.max_llm_calls_per_session,
                "remaining": max(0, self.config.max_llm_calls_per_session - self.counters.llm_calls),
            },
            "github_api_calls": {
                "current": self.counters.github_api_calls,
                "limit": self.config.max_github_api_calls_per_session,
                "remaining": max(0, self.config.max_github_api_calls_per_session - self.counters.github_api_calls),
            },
            "file_writes": {
                "current": self.counters.file_writes,
                "limit": self.config.max_file_writes_per_session,
                "remaining": max(0, self.config.max_file_writes_per_session - self.counters.file_writes),
            },
            "builds_per_hour": {
                "current": self.counters.builds_in_last_hour(),
                "limit": self.config.max_builds_per_hour,
                "remaining": max(0, self.config.max_builds_per_hour - self.counters.builds_in_last_hour()),
            },
            "tokens": {
                "current": self.counters.total_tokens,
                "limit": self.config.max_tokens_per_session,
                "remaining": max(0, self.config.max_tokens_per_session - self.counters.total_tokens),
            },
        }
    
    def estimate_cost(self, model: str = "gpt-4o") -> float:
        """
        Estimate cost based on token usage.
        
        Parameters
        ----------
        model : str
            Model name for pricing lookup.
        
        Returns
        -------
        float
            Estimated cost in USD.
        """
        # Approximate pricing per 1M tokens (input + output averaged)
        pricing = {
            "gpt-4o": 7.50,  # ~$2.50 input + $10 output averaged
            "gpt-4o-mini": 0.30,
            "gpt-4-turbo": 20.00,
            "gpt-3.5-turbo": 1.00,
        }
        
        price_per_million = pricing.get(model, 10.0)  # Default to $10/M
        cost = (self.counters.total_tokens / 1_000_000) * price_per_million
        
        return round(cost, 4)
    
    def reset(self) -> None:
        """Reset all counters for a new session."""
        self.counters.reset()
        logger.info("Rate limiter counters reset")


# ── Process-level singleton ───────────────────────────────────────────────────

_limiter_instance: Optional[SessionRateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> SessionRateLimiter:
    """
    Get or create the process-level rate limiter singleton.
    
    Parameters
    ----------
    config : RateLimitConfig, optional
        Configuration for the limiter. Only used if creating new instance.
    """
    global _limiter_instance
    
    with _limiter_lock:
        if _limiter_instance is None:
            _limiter_instance = SessionRateLimiter(config=config)
        return _limiter_instance


def reset_rate_limiter() -> None:
    """Reset the rate limiter singleton (for testing or new sessions)."""
    global _limiter_instance
    
    with _limiter_lock:
        if _limiter_instance:
            _limiter_instance.reset()
        _limiter_instance = None

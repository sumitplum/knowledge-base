"""
Tests for SessionRateLimiter (Layer 5: Rate Limiting and Cost Controls).

Tests the per-session rate limiter for LLM calls, GitHub API, file writes, builds, and token budget.
"""

import pytest
import time
from unittest.mock import Mock, patch

from agents.rate_limiter import (
    SessionRateLimiter,
    RateLimitConfig,
    get_rate_limiter,
    reset_rate_limiter,
    SessionCounters,
)
from agents.exceptions import RateLimitExceeded


class TestRateLimitConfig:
    """Test RateLimitConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.max_llm_calls_per_session == 50
        assert config.max_github_api_calls_per_session == 20
        assert config.max_file_writes_per_session == 30
        assert config.max_builds_per_hour == 5
        assert config.max_tokens_per_session == 200_000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            max_llm_calls_per_session=100,
            max_github_api_calls_per_session=50,
            max_file_writes_per_session=60,
            max_builds_per_hour=10,
            max_tokens_per_session=500_000,
        )
        
        assert config.max_llm_calls_per_session == 100
        assert config.max_github_api_calls_per_session == 50
        assert config.max_file_writes_per_session == 60
        assert config.max_builds_per_hour == 10
        assert config.max_tokens_per_session == 500_000


class TestSessionCounters:
    """Test SessionCounters thread-safe counters."""
    
    def test_initial_state(self):
        """Test counters start at zero."""
        counters = SessionCounters()
        
        assert counters.llm_calls == 0
        assert counters.github_api_calls == 0
        assert counters.file_writes == 0
        assert counters.total_tokens == 0
        assert len(counters.build_timestamps) == 0
    
    def test_increment_llm_calls(self):
        """Test incrementing LLM call counter."""
        counters = SessionCounters()
        
        result = counters.increment_llm_calls()
        assert result == 1
        assert counters.llm_calls == 1
        
        result = counters.increment_llm_calls(5)
        assert result == 6
        assert counters.llm_calls == 6
    
    def test_increment_github_api_calls(self):
        """Test incrementing GitHub API call counter."""
        counters = SessionCounters()
        
        result = counters.increment_github_api_calls()
        assert result == 1
        
        result = counters.increment_github_api_calls(3)
        assert result == 4
    
    def test_increment_file_writes(self):
        """Test incrementing file write counter."""
        counters = SessionCounters()
        
        result = counters.increment_file_writes()
        assert result == 1
        
        result = counters.increment_file_writes(2)
        assert result == 3
    
    def test_add_tokens(self):
        """Test adding tokens to budget."""
        counters = SessionCounters()
        
        result = counters.add_tokens(1000)
        assert result == 1000
        
        result = counters.add_tokens(2500)
        assert result == 3500
    
    def test_record_build(self):
        """Test recording build operations."""
        counters = SessionCounters()
        
        # Record a build
        count = counters.record_build()
        assert count == 1
        
        # Record another
        count = counters.record_build()
        assert count == 2
    
    def test_builds_in_last_hour(self):
        """Test counting builds in the last hour."""
        counters = SessionCounters()
        
        # Record some builds
        counters.record_build()
        time.sleep(0.01)
        counters.record_build()
        
        assert counters.builds_in_last_hour() == 2
    
    def test_reset(self):
        """Test resetting all counters."""
        counters = SessionCounters()
        
        counters.increment_llm_calls(10)
        counters.increment_github_api_calls(5)
        counters.increment_file_writes(3)
        counters.add_tokens(10000)
        counters.record_build()
        
        counters.reset()
        
        assert counters.llm_calls == 0
        assert counters.github_api_calls == 0
        assert counters.file_writes == 0
        assert counters.total_tokens == 0
        assert len(counters.build_timestamps) == 0


class TestSessionRateLimiter:
    """Test SessionRateLimiter."""
    
    def setup_method(self):
        """Reset rate limiter before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Reset rate limiter after each test."""
        reset_rate_limiter()
    
    # ── LLM Calls ──
    
    def test_llm_call_within_limit(self):
        """Test LLM calls within limit are allowed."""
        config = RateLimitConfig(max_llm_calls_per_session=10)
        limiter = SessionRateLimiter(config)
        
        for i in range(10):
            limiter.check_llm_call()
            limiter.increment_llm_calls()
        
        # Should have made 10 calls successfully
        assert limiter.counters.llm_calls == 10
    
    def test_llm_call_exceeds_limit(self):
        """Test LLM calls exceeding limit raises exception."""
        config = RateLimitConfig(max_llm_calls_per_session=5)
        limiter = SessionRateLimiter(config)
        
        # Use up the limit
        for i in range(5):
            limiter.track_llm_call()
        
        # Next call should fail
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_llm_call()
        
        assert exc_info.value.resource == "llm_calls"
        assert exc_info.value.limit == 5
        assert exc_info.value.current == 5
    
    def test_track_llm_call_convenience(self):
        """Test track_llm_call combines check and increment."""
        config = RateLimitConfig(max_llm_calls_per_session=10)
        limiter = SessionRateLimiter(config)
        
        result = limiter.track_llm_call()
        assert result == 1
        assert limiter.counters.llm_calls == 1
    
    # ── GitHub API Calls ──
    
    def test_github_api_call_within_limit(self):
        """Test GitHub API calls within limit are allowed."""
        config = RateLimitConfig(max_github_api_calls_per_session=10)
        limiter = SessionRateLimiter(config)
        
        for i in range(10):
            limiter.track_github_api_call()
        
        assert limiter.counters.github_api_calls == 10
    
    def test_github_api_call_exceeds_limit(self):
        """Test GitHub API calls exceeding limit raises exception."""
        config = RateLimitConfig(max_github_api_calls_per_session=3)
        limiter = SessionRateLimiter(config)
        
        # Use up the limit
        for i in range(3):
            limiter.track_github_api_call()
        
        # Next call should fail
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_github_api_call()
        
        assert exc_info.value.resource == "github_api_calls"
        assert exc_info.value.limit == 3
    
    # ── File Writes ──
    
    def test_file_write_within_limit(self):
        """Test file writes within limit are allowed."""
        config = RateLimitConfig(max_file_writes_per_session=5)
        limiter = SessionRateLimiter(config)
        
        for i in range(5):
            limiter.track_file_write()
        
        assert limiter.counters.file_writes == 5
    
    def test_file_write_exceeds_limit(self):
        """Test file writes exceeding limit raises exception."""
        config = RateLimitConfig(max_file_writes_per_session=2)
        limiter = SessionRateLimiter(config)
        
        # Use up the limit
        limiter.track_file_write()
        limiter.track_file_write()
        
        # Next write should fail
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_file_write()
        
        assert exc_info.value.resource == "file_writes"
    
    # ── Build Rate (hourly) ──
    
    def test_build_rate_within_limit(self):
        """Test builds within hourly limit are allowed."""
        config = RateLimitConfig(max_builds_per_hour=3)
        limiter = SessionRateLimiter(config)
        
        for i in range(3):
            limiter.track_build()
        
        assert limiter.counters.builds_in_last_hour() == 3
    
    def test_build_rate_exceeds_limit(self):
        """Test builds exceeding hourly limit raises exception."""
        config = RateLimitConfig(max_builds_per_hour=2)
        limiter = SessionRateLimiter(config)
        
        # Use up the limit
        limiter.track_build()
        limiter.track_build()
        
        # Next build should fail
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_build_rate()
        
        assert exc_info.value.resource == "builds_per_hour"
        assert exc_info.value.limit == 2
    
    # ── Token Budget ──
    
    def test_token_budget_within_limit(self):
        """Test token usage within budget is allowed."""
        config = RateLimitConfig(max_tokens_per_session=10000)
        limiter = SessionRateLimiter(config)
        
        limiter.track_tokens(5000)
        limiter.track_tokens(3000)
        
        assert limiter.counters.total_tokens == 8000
    
    def test_token_budget_exceeds_limit(self):
        """Test token usage exceeding budget raises exception."""
        config = RateLimitConfig(max_tokens_per_session=10000)
        limiter = SessionRateLimiter(config)
        
        # Use up most of the budget
        limiter.add_tokens(9500)
        
        # Next large token use should fail
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_token_budget(1000)
        
        assert exc_info.value.resource == "token_budget"
        assert exc_info.value.limit == 10000
        assert exc_info.value.current == 10500  # projected
    
    # ── Usage Summary ──
    
    def test_get_usage_summary(self):
        """Test getting usage summary."""
        config = RateLimitConfig(
            max_llm_calls_per_session=50,
            max_github_api_calls_per_session=20,
            max_file_writes_per_session=30,
            max_builds_per_hour=5,
            max_tokens_per_session=200_000,
        )
        limiter = SessionRateLimiter(config)
        
        # Use some resources
        limiter.increment_llm_calls(10)
        limiter.increment_github_api_calls(5)
        limiter.increment_file_writes(3)
        limiter.add_tokens(50000)
        limiter.record_build()
        
        summary = limiter.get_usage_summary()
        
        assert summary["llm_calls"]["current"] == 10
        assert summary["llm_calls"]["limit"] == 50
        assert summary["llm_calls"]["remaining"] == 40
        
        assert summary["github_api_calls"]["current"] == 5
        assert summary["github_api_calls"]["limit"] == 20
        assert summary["github_api_calls"]["remaining"] == 15
        
        assert summary["file_writes"]["current"] == 3
        assert summary["file_writes"]["limit"] == 30
        assert summary["file_writes"]["remaining"] == 27
        
        assert summary["builds_per_hour"]["current"] == 1
        assert summary["builds_per_hour"]["limit"] == 5
        assert summary["builds_per_hour"]["remaining"] == 4
        
        assert summary["tokens"]["current"] == 50000
        assert summary["tokens"]["limit"] == 200_000
        assert summary["tokens"]["remaining"] == 150_000
    
    # ── Cost Estimation ──
    
    def test_estimate_cost_gpt4o(self):
        """Test cost estimation for GPT-4o."""
        limiter = SessionRateLimiter()
        limiter.add_tokens(100000)
        
        cost = limiter.estimate_cost("gpt-4o")
        # 100k tokens at ~$7.50/M = ~$0.75
        assert 0.7 <= cost <= 0.8
    
    def test_estimate_cost_gpt4o_mini(self):
        """Test cost estimation for GPT-4o mini."""
        limiter = SessionRateLimiter()
        limiter.add_tokens(100000)
        
        cost = limiter.estimate_cost("gpt-4o-mini")
        # 100k tokens at ~$0.30/M = ~$0.03
        assert 0.02 <= cost <= 0.04
    
    # ── Reset ──
    
    def test_reset_limiter(self):
        """Test resetting the limiter."""
        limiter = SessionRateLimiter()
        
        limiter.increment_llm_calls(10)
        limiter.increment_github_api_calls(5)
        limiter.add_tokens(10000)
        
        limiter.reset()
        
        assert limiter.counters.llm_calls == 0
        assert limiter.counters.github_api_calls == 0
        assert limiter.counters.total_tokens == 0
    
    # ── Singleton ──
    
    def test_get_rate_limiter_singleton(self):
        """Test get_rate_limiter returns singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2
    
    def test_reset_rate_limiter_singleton(self):
        """Test reset_rate_limiter clears singleton."""
        limiter1 = get_rate_limiter()
        limiter1.increment_llm_calls(5)
        
        reset_rate_limiter()
        
        limiter2 = get_rate_limiter()
        assert limiter1 is not limiter2
        assert limiter2.counters.llm_calls == 0
    
    # ── Audit Integration ──
    
    @patch('agents.rate_limiter.get_audit_logger')
    def test_rate_limit_exceeded_logs_audit(self, mock_audit):
        """Test rate limit exceeded logs to audit."""
        mock_logger = Mock()
        mock_audit.return_value = mock_logger
        
        config = RateLimitConfig(max_llm_calls_per_session=1)
        limiter = SessionRateLimiter(config)
        limiter._audit = mock_logger
        
        # Use up limit
        limiter.increment_llm_calls()
        
        # Try to exceed
        with pytest.raises(RateLimitExceeded):
            limiter.check_llm_call()
        
        # Check audit was called
        mock_logger.log_rate_limit_exceeded.assert_called_once()

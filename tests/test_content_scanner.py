"""
Tests for ContentScanner (Layer 2: Content Security Scanner).

Tests the hardened content scanning for secrets, binary files, and diff size limits.
"""

import pytest
from pathlib import Path
import tempfile

from codegen.safety import (
    ContentScanner,
    SafetyCheckResult,
    SECRET_PATTERNS,
)
from agents.exceptions import SafetyViolation


class TestContentScanner:
    """Test suite for ContentScanner."""
    
    # ── Secret Detection Tests ──
    
    def test_detect_openai_api_key(self):
        """Test detection of OpenAI API keys."""
        scanner = ContentScanner()
        
        content = "const apiKey = 'sk-proj-1234567890abcdefghijklmnopqrstuvwxyz';"
        
        result = scanner.scan_content(content, "test.ts")
        
        assert not result.is_safe
        assert "OpenAI API key" in result.violations[0]
    
    def test_detect_github_token(self):
        """Test detection of GitHub tokens."""
        scanner = ContentScanner()
        
        content = "GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyz1234"
        
        result = scanner.scan_content(content, "test.env")
        
        assert not result.is_safe
        assert "GitHub token" in result.violations[0]
    
    def test_detect_aws_keys(self):
        """Test detection of AWS keys."""
        scanner = ContentScanner()
        
        content_access_key = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan_content(content_access_key, "config.yaml")
        assert not result.is_safe
        
        content_secret = 'aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        result = scanner.scan_content(content_secret, "config.yaml")
        assert not result.is_safe
    
    def test_detect_private_keys(self):
        """Test detection of private keys."""
        scanner = ContentScanner()
        
        content = """
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890...
-----END RSA PRIVATE KEY-----
"""
        
        result = scanner.scan_content(content, "key.pem")
        
        assert not result.is_safe
        assert "Private key" in result.violations[0]
    
    def test_detect_database_urls(self):
        """Test detection of database connection strings."""
        scanner = ContentScanner()
        
        content = "DATABASE_URL=postgres://user:password@localhost:5432/db"
        
        result = scanner.scan_content(content, "config.py")
        
        assert not result.is_safe
        assert "Database URL" in result.violations[0]
    
    def test_detect_generic_secrets(self):
        """Test detection of generic password/token patterns."""
        scanner = ContentScanner()
        
        test_cases = [
            'password="SuperSecret123"',
            "api_key='my_secret_token_123'",
            'SECRET_TOKEN="abcdef12345"',
        ]
        
        for content in test_cases:
            result = scanner.scan_content(content, "test.py")
            assert not result.is_safe, f"Should detect secret in: {content}"
    
    def test_allow_safe_content(self):
        """Test that safe content passes."""
        scanner = ContentScanner()
        
        safe_content = """
def calculate_total(items):
    return sum(item.price for item in items)
"""
        
        result = scanner.scan_content(safe_content, "utils.py")
        
        assert result.is_safe
        assert len(result.violations) == 0
    
    # ── Entropy Detection Tests ──
    
    def test_high_entropy_detection(self):
        """Test detection of high-entropy strings (potential secrets)."""
        scanner = ContentScanner()
        
        # High entropy random string
        content = 'token = "aB3xQ9mK2pL7nV5wR8jF4hG6tY1uE0sD"'
        
        result = scanner.scan_content(content, "test.py")
        
        # Should flag high entropy
        assert not result.is_safe
        assert any("High entropy" in v for v in result.violations)
    
    def test_low_entropy_allowed(self):
        """Test that low entropy strings are allowed."""
        scanner = ContentScanner()
        
        # Low entropy, repeated patterns
        content = 'placeholder = "aaaabbbbccccdddd"'
        
        result = scanner.scan_content(content, "test.py")
        
        # Low entropy should pass
        assert result.is_safe
    
    # ── Binary File Detection Tests ──
    
    def test_detect_binary_content(self):
        """Test detection of binary content."""
        scanner = ContentScanner()
        
        # Binary-like content (null bytes)
        binary_content = b"Hello\x00World\x00\xff\xfe"
        
        result = scanner.scan_content(binary_content.decode('latin-1'), "test.bin")
        
        assert not result.is_safe
        assert any("Binary" in v or "null bytes" in v for v in result.violations)
    
    # ── Symlink Detection Tests ──
    
    def test_detect_symlink(self):
        """Test detection of symlinks."""
        scanner = ContentScanner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create a real file
            target_file = tmppath / "target.txt"
            target_file.write_text("real content")
            
            # Create a symlink
            symlink_file = tmppath / "link.txt"
            symlink_file.symlink_to(target_file)
            
            result = scanner.check_file(symlink_file)
            
            assert not result.is_safe
            assert any("symlink" in v.lower() for v in result.violations)
    
    def test_allow_regular_file(self):
        """Test that regular files are allowed."""
        scanner = ContentScanner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            regular_file = tmppath / "regular.txt"
            regular_file.write_text("def hello():\n    return 'world'")
            
            result = scanner.check_file(regular_file)
            
            # Should pass symlink check
            assert result.is_safe or all("symlink" not in v.lower() for v in result.violations)
    
    # ── Diff Size Limit Tests ──
    
    def test_scan_diff_within_limit(self):
        """Test diff scanning within size limit."""
        scanner = ContentScanner()
        
        diff_text = """
diff --git a/test.py b/test.py
+def new_function():
+    return 42
"""
        
        result = scanner.scan_diff(diff_text, max_lines=1000)
        
        assert result.is_safe
    
    def test_scan_diff_exceeds_hard_cap(self):
        """Test diff exceeding hard cap raises exception."""
        scanner = ContentScanner()
        
        # Create a large diff (2000+ lines)
        large_diff = "diff --git a/test.py b/test.py\n"
        large_diff += "\n".join([f"+line {i}" for i in range(2500)])
        
        with pytest.raises(SafetyViolation) as exc_info:
            scanner.scan_diff(large_diff, hard_cap=2000)
        
        assert "diff size" in str(exc_info.value).lower()
    
    def test_scan_diff_with_secrets(self):
        """Test diff containing secrets is rejected."""
        scanner = ContentScanner()
        
        diff_with_secret = """
diff --git a/config.py b/config.py
+API_KEY = "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz"
"""
        
        result = scanner.scan_diff(diff_with_secret)
        
        assert not result.is_safe
        assert any("API key" in v or "secret" in v.lower() for v in result.violations)
    
    # ── Integration Tests ──
    
    def test_scan_content_with_multiple_violations(self):
        """Test content with multiple security issues."""
        scanner = ContentScanner()
        
        bad_content = """
API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyz1234"
PASSWORD = "SuperSecret123"
"""
        
        result = scanner.scan_content(bad_content, "secrets.py")
        
        assert not result.is_safe
        # Should detect multiple secrets
        assert len(result.violations) >= 2
    
    def test_check_file_comprehensive(self):
        """Test comprehensive file checking."""
        scanner = ContentScanner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create file with secret
            secret_file = tmppath / "config.py"
            secret_file.write_text('API_KEY = "sk-proj-test123456789abcdefghijklmnopqrstuvw"')
            
            result = scanner.check_file(secret_file)
            
            assert not result.is_safe
            assert len(result.violations) > 0
    
    # ── SafetyCheckResult Tests ──
    
    def test_safety_check_result_safe(self):
        """Test SafetyCheckResult for safe content."""
        result = SafetyCheckResult(is_safe=True, violations=[])
        
        assert result.is_safe
        assert len(result.violations) == 0
    
    def test_safety_check_result_unsafe(self):
        """Test SafetyCheckResult for unsafe content."""
        violations = [
            "OpenAI API key detected",
            "High entropy string detected",
        ]
        result = SafetyCheckResult(is_safe=False, violations=violations)
        
        assert not result.is_safe
        assert len(result.violations) == 2
    
    # ── Secret Pattern Coverage Tests ──
    
    def test_all_secret_patterns_have_descriptions(self):
        """Test that all secret patterns have descriptions."""
        for pattern, description in SECRET_PATTERNS:
            assert description, f"Pattern {pattern} missing description"
            assert len(description) > 0
    
    def test_secret_pattern_count(self):
        """Test that we have comprehensive secret pattern coverage."""
        # Should have at least 30 patterns (as per the plan)
        assert len(SECRET_PATTERNS) >= 30, f"Only {len(SECRET_PATTERNS)} patterns defined"


class TestContentScannerEdgeCases:
    """Test edge cases for ContentScanner."""
    
    def test_empty_content(self):
        """Test scanning empty content."""
        scanner = ContentScanner()
        
        result = scanner.scan_content("", "empty.txt")
        
        assert result.is_safe
    
    def test_very_large_safe_content(self):
        """Test scanning very large but safe content."""
        scanner = ContentScanner()
        
        # Generate large safe content
        large_content = "\n".join([f"# Line {i}: Safe comment" for i in range(1000)])
        
        result = scanner.scan_content(large_content, "large.py")
        
        assert result.is_safe
    
    def test_unicode_content(self):
        """Test scanning content with unicode characters."""
        scanner = ContentScanner()
        
        unicode_content = """
def greet():
    return "Hello, 世界! 🌍"
"""
        
        result = scanner.scan_content(unicode_content, "unicode.py")
        
        assert result.is_safe
    
    def test_multiline_secrets(self):
        """Test detection of multiline secrets."""
        scanner = ContentScanner()
        
        multiline_secret = '''
private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"""
'''
        
        result = scanner.scan_content(multiline_secret, "keys.py")
        
        assert not result.is_safe

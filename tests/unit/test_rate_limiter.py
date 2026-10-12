"""
Tests for CapLimitDetector, Agy5HourCapError, and agy rate limiter utilities.
"""

from sequence_labelling.llm.agy_rate_limiter import CapLimitDetector, Agy5HourCapError


def test_cap_limit_detection_various_patterns():
    # 429 and RESOURCE_EXHAUSTED with underscore
    err1 = (
        'agy vision failed (exit 3): error: Individual quota reached. '
        'Please upgrade your subscription to increase your limits. Resets in 11m30s.\n'
        'AGY_ERROR: {"short_error":"RESOURCE_EXHAUSTED (code 429): Individual quota reached."}'
    )
    assert CapLimitDetector.is_cap_limit(3, stderr=err1)

    # Standard rate limit
    err2 = "rate limit exceeded, try again in 60s"
    assert CapLimitDetector.is_cap_limit(1, stderr=err2)

    # Standard 429
    err3 = "HTTP 429 Too Many Requests"
    assert CapLimitDetector.is_cap_limit(1, stderr=err3)

    # Return code 0 should not be detected as cap limit
    assert not CapLimitDetector.is_cap_limit(0, stderr=err1)


def test_parse_cooldown_seconds():
    assert CapLimitDetector.parse_cooldown_seconds("Resets in 11m30s.") == 690
    assert CapLimitDetector.parse_cooldown_seconds("Resets in 1h2m55s.") == 3775
    assert CapLimitDetector.parse_cooldown_seconds("resets in 4 hours 45 minutes") == 17100
    assert CapLimitDetector.parse_cooldown_seconds("try again in 5h") == 18000
    assert CapLimitDetector.parse_cooldown_seconds("try again in 280 minutes") == 16800
    assert CapLimitDetector.parse_cooldown_seconds("retry-after: 3600") == 3600

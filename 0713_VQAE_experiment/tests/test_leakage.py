"""Tests for label leakage validation."""

import pytest

from src.data.preprocessing import validate_no_label_leakage


def test_leakage_detection():
    with pytest.raises(ValueError, match="Label leakage"):
        validate_no_label_leakage(["coverage_ratio", "pass_fail_status"])


def test_clean_features_pass():
    validate_no_label_leakage(["coverage_ratio", "test_runtime_seconds"])

"""Tests for BugsInPy triggering-test parsing and revision aggregation."""

from __future__ import annotations

import pytest

from src.data.bugsinpy_features import aggregate_revision_features, build_revision_aggregate_row
from src.data.bugsinpy_runner import extract_test_id, single_test_target


def test_single_test_target_tox() -> None:
    command = "tox tests/test_generate_context.py::test_generate_context_decodes_non_ascii_chars"
    assert single_test_target(command) == "tests/test_generate_context.py::test_generate_context_decodes_non_ascii_chars"
    assert extract_test_id(command) == "test_generate_context_decodes_non_ascii_chars"


def test_single_test_target_pytest() -> None:
    command = "pytest tests/test_serialize_response_model.py::test_valid"
    assert single_test_target(command) == "tests/test_serialize_response_model.py::test_valid"
    assert extract_test_id(command) == "test_valid"


def test_single_test_target_unittest() -> None:
    command = "python -m unittest -q test.test_utils.TestUtil.test_match_str"
    assert single_test_target(command) == "test.test_utils.TestUtil.test_match_str"
    assert extract_test_id(command) == "test_match_str"


def test_aggregate_revision_features() -> None:
    rows = [
        {
            "test_id": "test_a",
            "test_runtime_seconds": 1.0,
            "coverage_ratio": 0.5,
            "covered_line_count": 100.0,
            "changed_line_coverage_ratio": 0.2,
            "meta_test_returncode": 0,
        },
        {
            "test_id": "test_b",
            "test_runtime_seconds": 3.0,
            "coverage_ratio": 0.7,
            "covered_line_count": 120.0,
            "changed_line_coverage_ratio": 0.4,
            "meta_test_returncode": 1,
        },
    ]
    aggregated = aggregate_revision_features(rows)
    assert aggregated["mean_test_runtime_seconds"] == 2.0
    assert aggregated["min_coverage_ratio"] == 0.5
    assert aggregated["max_coverage_ratio"] == 0.7
    assert aggregated["pass_rate"] == 0.5
    assert aggregated["n_triggering_tests"] == 2.0


def test_build_revision_aggregate_row() -> None:
    rows = [
        {
            "test_id": "test_a",
            "test_runtime_seconds": 1.0,
            "coverage_ratio": 0.5,
            "covered_line_count": 100.0,
            "changed_line_coverage_ratio": 0.2,
            "meta_test_returncode": 0,
        }
    ]
    row = build_revision_aggregate_row(
        project="fastapi",
        bug_id="3",
        revision="fixed",
        label=0,
        test_rows=rows,
    )
    assert row["granularity"] == "revision_aggregate"
    assert row["test_id"] == "__aggregate__"
    assert row["mean_test_runtime_seconds"] == 1.0

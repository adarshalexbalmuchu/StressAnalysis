"""
Tests for the LLM-vs-System Comparison engine.

Tests:
  - run_llm_baseline with stub generator
  - run_llm_reanalysis with stub generator
  - test_consistency scoring
  - compare_approaches metrics
"""

import pytest

from app.engine.comparison import (
    ComparisonResult,
    ConsistencyTest,
    compare_approaches,
    run_llm_baseline,
    run_llm_reanalysis,
    evaluate_consistency,
    _extract_json,
)
from app.engine.generator import StubGenerator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def stub_gen():
    return StubGenerator()


@pytest.fixture
def sample_llm_analysis(stub_gen):
    return run_llm_baseline(stub_gen, "Test policy on biometric authentication")


@pytest.fixture
def system_hypotheses():
    return [
        {"hid": "H001", "mechanism": "Authentication failures exclude elderly beneficiaries",
         "stakeholders": ["Elderly"], "time_horizon": "immediate"},
        {"hid": "H002", "mechanism": "Administrative discretion in exemption processing",
         "stakeholders": ["FPS dealers"], "time_horizon": "short_term"},
    ]


@pytest.fixture
def system_beliefs():
    return {"H001": 0.72, "H002": 0.45}


@pytest.fixture
def system_trajectories():
    return [
        {"name": "Exclusion cascade", "probability": 0.34, "active_hypotheses": ["H001"]},
        {"name": "Admin workaround", "probability": 0.28, "active_hypotheses": ["H002"]},
    ]


# ============================================================================
# Tests: _extract_json
# ============================================================================


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        result = _extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_embedded_json(self):
        result = _extract_json('Some text before {"key": "value"} and after')
        assert result == {"key": "value"}

    def test_invalid_json(self):
        result = _extract_json("Not JSON at all")
        assert "failure_modes" in result  # Falls back to default


# ============================================================================
# Tests: run_llm_baseline (stub)
# ============================================================================


class TestRunLlmBaseline:
    def test_returns_failure_modes(self, stub_gen):
        result = run_llm_baseline(stub_gen, "Policy text")
        assert "failure_modes" in result
        assert len(result["failure_modes"]) > 0

    def test_has_probabilities(self, stub_gen):
        result = run_llm_baseline(stub_gen, "Policy text")
        for mode in result["failure_modes"]:
            assert "probability" in mode
            assert 0 <= mode["probability"] <= 1

    def test_has_overall_risk(self, stub_gen):
        result = run_llm_baseline(stub_gen, "Policy text")
        assert "overall_risk" in result
        assert result["overall_risk"] in ("high", "medium", "low")

    def test_has_narrative(self, stub_gen):
        result = run_llm_baseline(stub_gen, "Policy text")
        assert "narrative" in result
        assert len(result["narrative"]) > 10


# ============================================================================
# Tests: run_llm_reanalysis (stub)
# ============================================================================


class TestRunLlmReanalysis:
    def test_returns_updated_modes(self, stub_gen, sample_llm_analysis):
        result = run_llm_reanalysis(
            stub_gen, "Policy text", sample_llm_analysis,
            [{"type": "field_report", "content": "New evidence"}],
        )
        assert "failure_modes" in result
        assert len(result["failure_modes"]) > 0

    def test_adds_new_mode(self, stub_gen, sample_llm_analysis):
        """Stub may add a new mode to show inconsistency (60% probability)."""
        # Run multiple times — at least one should show new mode addition
        found_new_mode = False
        for seed in range(10):
            result = run_llm_reanalysis(
                stub_gen, f"Policy text {seed}", sample_llm_analysis,
                [{"type": "field_report", "content": "New evidence"}],
            )
            if len(result["failure_modes"]) > len(sample_llm_analysis["failure_modes"]):
                found_new_mode = True
                break
        assert found_new_mode, "Expected at least one run (of 10) to add a new mode"

    def test_probabilities_shift(self, stub_gen, sample_llm_analysis):
        """Stub deliberately shifts probabilities."""
        result = run_llm_reanalysis(
            stub_gen, "Policy text", sample_llm_analysis,
            [{"type": "field_report", "content": "New evidence"}],
        )
        # At least one probability should differ
        initial_probs = [m["probability"] for m in sample_llm_analysis["failure_modes"]]
        updated_probs = [m["probability"] for m in result["failure_modes"][:len(initial_probs)]]
        assert initial_probs != updated_probs


# ============================================================================
# Tests: test_consistency
# ============================================================================


class TestConsistency:
    def test_detects_drift(self, stub_gen, sample_llm_analysis):
        updated = run_llm_reanalysis(
            stub_gen, "Policy text", sample_llm_analysis,
            [{"type": "field_report", "content": "Evidence"}],
        )
        result = evaluate_consistency(sample_llm_analysis, updated)
        assert isinstance(result, ConsistencyTest)
        assert result.probability_drift >= 0

    def test_detects_new_modes(self, stub_gen, sample_llm_analysis):
        # New mode appears ~60% of the time; run multiple times
        found_new = False
        for seed in range(10):
            updated = run_llm_reanalysis(
                stub_gen, f"Policy text {seed}", sample_llm_analysis,
                [{"type": "field_report", "content": "Evidence"}],
            )
            result = evaluate_consistency(sample_llm_analysis, updated)
            if result.new_modes_appeared > 0:
                found_new = True
                break
        assert found_new, "Expected new modes in at least one run"

    def test_has_explanations(self, stub_gen, sample_llm_analysis):
        updated = run_llm_reanalysis(
            stub_gen, "Policy text", sample_llm_analysis,
            [{"type": "field_report", "content": "Evidence"}],
        )
        result = evaluate_consistency(sample_llm_analysis, updated)
        assert len(result.explanation) > 0

    def test_consistent_analysis(self):
        """Identical analyses should show zero drift."""
        analysis = {
            "failure_modes": [
                {"description": "Mode A", "probability": 0.5},
                {"description": "Mode B", "probability": 0.3},
            ]
        }
        result = evaluate_consistency(analysis, analysis)
        assert result.probability_drift == 0.0
        assert result.ordering_preserved is True
        assert result.new_modes_appeared == 0
        assert result.modes_disappeared == 0


# ============================================================================
# Tests: compare_approaches
# ============================================================================


class TestCompareApproaches:
    def test_returns_comparison_result(
        self, sample_llm_analysis, system_hypotheses, system_beliefs, system_trajectories
    ):
        result = compare_approaches(
            llm_analysis=sample_llm_analysis,
            system_hypotheses=system_hypotheses,
            system_beliefs=system_beliefs,
            system_trajectories=system_trajectories,
        )
        assert isinstance(result, ComparisonResult)

    def test_scores_in_range(
        self, sample_llm_analysis, system_hypotheses, system_beliefs, system_trajectories
    ):
        result = compare_approaches(
            llm_analysis=sample_llm_analysis,
            system_hypotheses=system_hypotheses,
            system_beliefs=system_beliefs,
            system_trajectories=system_trajectories,
        )
        assert 0 <= result.consistency_score <= 1
        assert 0 <= result.coverage_score <= 1
        assert 0 <= result.structure_score <= 1

    def test_has_explanations(
        self, sample_llm_analysis, system_hypotheses, system_beliefs, system_trajectories
    ):
        result = compare_approaches(
            llm_analysis=sample_llm_analysis,
            system_hypotheses=system_hypotheses,
            system_beliefs=system_beliefs,
            system_trajectories=system_trajectories,
        )
        assert len(result.explanation) > 0

    def test_with_consistency_result(
        self, stub_gen, sample_llm_analysis,
        system_hypotheses, system_beliefs, system_trajectories,
    ):
        """Comparison with consistency test included."""
        updated = run_llm_reanalysis(
            stub_gen, "Policy text", sample_llm_analysis,
            [{"type": "field_report", "content": "Evidence"}],
        )
        consistency = evaluate_consistency(sample_llm_analysis, updated)

        result = compare_approaches(
            llm_analysis=sample_llm_analysis,
            system_hypotheses=system_hypotheses,
            system_beliefs=system_beliefs,
            system_trajectories=system_trajectories,
            llm_consistency=consistency,
        )
        # With consistency test, score should reflect drift
        assert isinstance(result.consistency_score, float)

    def test_system_data_preserved(
        self, sample_llm_analysis, system_hypotheses, system_beliefs, system_trajectories
    ):
        result = compare_approaches(
            llm_analysis=sample_llm_analysis,
            system_hypotheses=system_hypotheses,
            system_beliefs=system_beliefs,
            system_trajectories=system_trajectories,
        )
        assert result.system_beliefs == system_beliefs
        assert result.system_trajectory_count == 2
        assert result.system_top_trajectory == "Exclusion cascade"
        assert result.system_top_probability == 0.34

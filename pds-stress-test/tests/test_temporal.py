"""
Tests for the Temporal Belief Evolution engine.

Tests:
  - evolve_beliefs with single batch
  - evolve_beliefs with multiple batches
  - entropy computation
  - convergence tracking
  - most_changed_hid detection
"""

import math
from datetime import datetime, timezone
from uuid import uuid4

import networkx as nx
import pytest

from app.engine.temporal import compute_entropy, evolve_beliefs, TimeStep, BeliefEvolution
from app.schemas import HypothesisBase, Signal, SignalType


# ============================================================================
# Fixtures
# ============================================================================


def _make_hypotheses():
    """Create a minimal hypothesis set."""
    return [
        HypothesisBase(
            hid="H001",
            stakeholders=["Elderly"],
            triggers=["Biometric mandate"],
            mechanism="Authentication failures exclude elderly beneficiaries",
            primary_effects=["Food security loss"],
            time_horizon="immediate",
        ),
        HypothesisBase(
            hid="H002",
            stakeholders=["FPS dealers"],
            triggers=["Manual override ban"],
            mechanism="Dealers lose discretion to serve without biometrics",
            primary_effects=["Reduced service flexibility"],
            time_horizon="short_term",
        ),
        HypothesisBase(
            hid="H003",
            stakeholders=["Rural poor"],
            triggers=["Connectivity requirements"],
            mechanism="Network failures prevent authentication",
            primary_effects=["Denial of service"],
            time_horizon="immediate",
        ),
    ]


def _make_graph():
    """Create a minimal hypothesis graph."""
    G = nx.DiGraph()
    G.add_node("H001")
    G.add_node("H002")
    G.add_node("H003")
    G.add_edge("H001", "H003", edge_type="reinforces", weight=1.0)
    G.add_edge("H002", "H001", edge_type="depends_on", weight=1.0)
    return G


def _make_signal(affected_hids, strength=0.7):
    """Create a test signal."""
    return Signal(
        id=uuid4(),
        run_id=uuid4(),
        signal_type=SignalType.FIELD_REPORT,
        content="Test signal",
        source="test",
        date_observed=datetime.now(timezone.utc),
        affected_hids=affected_hids,
        strength=strength,
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


# ============================================================================
# Tests: compute_entropy
# ============================================================================


class TestComputeEntropy:
    def test_maximum_entropy(self):
        """Maximum entropy when all beliefs at 0.5."""
        beliefs = {"H001": 0.5, "H002": 0.5, "H003": 0.5}
        entropy = compute_entropy(beliefs)
        # Each hypothesis contributes -0.5*ln(0.5) - 0.5*ln(0.5) = ln(2)
        expected = 3 * math.log(2)
        assert abs(entropy - expected) < 1e-10

    def test_zero_entropy(self):
        """Zero entropy when all beliefs at 0 or 1."""
        beliefs = {"H001": 0.0, "H002": 1.0, "H003": 0.0}
        entropy = compute_entropy(beliefs)
        assert entropy == 0.0

    def test_partial_entropy(self):
        """Entropy between 0 and max for mixed beliefs."""
        beliefs = {"H001": 0.8, "H002": 0.2, "H003": 0.5}
        entropy = compute_entropy(beliefs)
        assert entropy > 0
        max_entropy = 3 * math.log(2)
        assert entropy < max_entropy

    def test_empty_beliefs(self):
        """Empty beliefs should have zero entropy."""
        assert compute_entropy({}) == 0.0


# ============================================================================
# Tests: evolve_beliefs
# ============================================================================


class TestEvolveBeliefs:
    def test_single_batch(self):
        """Evolution with one signal batch."""
        hypotheses = _make_hypotheses()
        graph = _make_graph()
        initial = {"H001": 0.5, "H002": 0.5, "H003": 0.5}

        signals = [_make_signal(["H001"], strength=0.8)]
        result = evolve_beliefs(
            hypotheses=hypotheses,
            graph=graph,
            initial_beliefs=initial,
            signal_batches=[signals],
            batch_labels=["Month 1"],
        )

        assert isinstance(result, BeliefEvolution)
        assert len(result.steps) == 2  # Initial + 1 batch
        assert result.steps[0].label == "Initial (priors)"
        assert result.steps[1].label == "Month 1"
        assert result.total_signals == 1

    def test_multiple_batches(self):
        """Evolution across three time periods."""
        hypotheses = _make_hypotheses()
        graph = _make_graph()
        initial = {"H001": 0.5, "H002": 0.5, "H003": 0.5}

        batches = [
            [_make_signal(["H001"], strength=0.6)],
            [_make_signal(["H002"], strength=0.7)],
            [_make_signal(["H003"], strength=0.8)],
        ]
        labels = ["Month 1", "Month 2", "Month 3"]

        result = evolve_beliefs(
            hypotheses=hypotheses,
            graph=graph,
            initial_beliefs=initial,
            signal_batches=batches,
            batch_labels=labels,
        )

        assert len(result.steps) == 4  # Initial + 3 batches
        assert result.total_signals == 3
        for i, label in enumerate(labels):
            assert result.steps[i + 1].label == label

    def test_auto_labels(self):
        """Labels default to 'Period N' when not provided."""
        hypotheses = _make_hypotheses()
        graph = _make_graph()
        initial = {"H001": 0.5, "H002": 0.5, "H003": 0.5}

        batches = [
            [_make_signal(["H001"])],
            [_make_signal(["H002"])],
        ]

        result = evolve_beliefs(
            hypotheses=hypotheses,
            graph=graph,
            initial_beliefs=initial,
            signal_batches=batches,
        )

        assert result.steps[1].label == "Period 1"
        assert result.steps[2].label == "Period 2"

    def test_entropy_decreases_with_strong_signals(self):
        """Entropy should generally decrease with strong confirming signals."""
        hypotheses = _make_hypotheses()
        graph = _make_graph()
        initial = {"H001": 0.5, "H002": 0.5, "H003": 0.5}

        # Strong signal pushing H001 higher
        batches = [
            [_make_signal(["H001"], strength=0.9)],
            [_make_signal(["H001"], strength=0.9)],
            [_make_signal(["H001"], strength=0.9)],
        ]

        result = evolve_beliefs(
            hypotheses=hypotheses,
            graph=graph,
            initial_beliefs=initial,
            signal_batches=batches,
        )

        # Entropy should have changed (convergence_delta != 0)
        assert result.convergence_delta != 0

    def test_most_changed_hid_tracked(self):
        """Should track which hypothesis changed the most."""
        hypotheses = _make_hypotheses()
        graph = _make_graph()
        initial = {"H001": 0.5, "H002": 0.5, "H003": 0.5}

        # Only signal H001 heavily
        batches = [
            [_make_signal(["H001"], strength=0.9)],
        ]

        result = evolve_beliefs(
            hypotheses=hypotheses,
            graph=graph,
            initial_beliefs=initial,
            signal_batches=batches,
        )

        assert result.most_changed_hid != ""
        assert result.most_changed_delta > 0

    def test_step_signals_applied_count(self):
        """Each step should report correct signal count."""
        hypotheses = _make_hypotheses()
        graph = _make_graph()
        initial = {"H001": 0.5, "H002": 0.5, "H003": 0.5}

        batches = [
            [_make_signal(["H001"]), _make_signal(["H002"])],
            [_make_signal(["H003"])],
        ]

        result = evolve_beliefs(
            hypotheses=hypotheses,
            graph=graph,
            initial_beliefs=initial,
            signal_batches=batches,
        )

        assert result.steps[0].signals_applied == 0  # Initial
        assert result.steps[1].signals_applied == 2
        assert result.steps[2].signals_applied == 1

    def test_initial_step_preserves_priors(self):
        """Step 0 should exactly match initial beliefs."""
        hypotheses = _make_hypotheses()
        graph = _make_graph()
        initial = {"H001": 0.3, "H002": 0.7, "H003": 0.5}

        result = evolve_beliefs(
            hypotheses=hypotheses,
            graph=graph,
            initial_beliefs=initial,
            signal_batches=[[_make_signal(["H001"])]],
        )

        assert result.steps[0].beliefs == initial

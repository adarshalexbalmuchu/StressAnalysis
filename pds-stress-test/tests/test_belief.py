"""
Tests for Belief Update Engine
"""

from datetime import datetime, timezone

import pytest

from app.engine import BeliefUpdateEngine
from app.schemas import Hypothesis, Signal, SignalType, TimeHorizon


@pytest.fixture
def sample_hypotheses():
    """Create sample hypotheses for testing."""
    return [
        Hypothesis(
            id="00000000-0000-0000-0000-000000000001",
            run_id="00000000-0000-0000-0000-000000000000",
            hid="H001",
            stakeholders=["Group A"],
            triggers=["Trigger 1"],
            mechanism="Mechanism 1",
            primary_effects=["Effect 1"],
            secondary_effects=[],
            time_horizon=TimeHorizon.SHORT_TERM,
            created_at="2026-01-31T00:00:00",
        ),
        Hypothesis(
            id="00000000-0000-0000-0000-000000000002",
            run_id="00000000-0000-0000-0000-000000000000",
            hid="H002",
            stakeholders=["Group B"],
            triggers=["Trigger 2"],
            mechanism="Mechanism 2",
            primary_effects=["Effect 2"],
            secondary_effects=[],
            time_horizon=TimeHorizon.IMMEDIATE,
            created_at="2026-01-31T00:00:00",
        ),
    ]


def test_belief_initialization(sample_hypotheses):
    """Test belief engine initialization."""
    engine = BeliefUpdateEngine(sample_hypotheses, prior_belief=0.6)
    
    assert engine.get_belief("H001") == 0.6
    assert engine.get_belief("H002") == 0.6
    assert len(engine.get_all_beliefs()) == 2


def test_set_belief(sample_hypotheses):
    """Test manually setting belief."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    engine.set_belief("H001", 0.8, "Test update")
    
    assert engine.get_belief("H001") == 0.8
    assert "Test update" in engine.get_explanation_log()[-1]


def test_set_belief_invalid_probability(sample_hypotheses):
    """Test setting invalid probability."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    with pytest.raises(ValueError, match="Probability must be in"):
        engine.set_belief("H001", 1.5, "Invalid")


def test_update_from_signal_supporting(sample_hypotheses):
    """Test belief update from supporting signal."""
    engine = BeliefUpdateEngine(sample_hypotheses, prior_belief=0.5)
    
    signal = Signal(
        id="00000000-0000-0000-0000-000000000010",
        run_id="00000000-0000-0000-0000-000000000000",
        signal_type=SignalType.FIELD_REPORT,
        content="Evidence supporting H001",
        source="Field observation",
        date_observed=datetime.now(timezone.utc),
        affected_hids=["H001"],
        strength=0.8,
        metadata={},
        created_at=datetime.now(timezone.utc),
    )
    
    updates = engine.update_from_signal(signal)
    
    assert "H001" in updates
    old_belief, new_belief = updates["H001"]
    assert old_belief == 0.5
    assert new_belief > 0.5  # Belief should increase


def test_update_from_multiple_signals(sample_hypotheses):
    """Test belief updates from multiple signals."""
    engine = BeliefUpdateEngine(sample_hypotheses, prior_belief=0.5)
    
    signals = [
        Signal(
            id="00000000-0000-0000-0000-000000000010",
            run_id="00000000-0000-0000-0000-000000000000",
            signal_type=SignalType.FIELD_REPORT,
            content="Evidence 1",
            source="Source 1",
            date_observed=datetime.now(timezone.utc),
            affected_hids=["H001"],
            strength=0.7,
            metadata={},
            created_at=datetime.now(timezone.utc),
        ),
        Signal(
            id="00000000-0000-0000-0000-000000000011",
            run_id="00000000-0000-0000-0000-000000000000",
            signal_type=SignalType.GRIEVANCE,
            content="Evidence 2",
            source="Source 2",
            date_observed=datetime.now(timezone.utc),
            affected_hids=["H001"],
            strength=0.6,
            metadata={},
            created_at=datetime.now(timezone.utc),
        ),
    ]
    
    all_updates = engine.update_from_signals(signals)
    
    assert "H001" in all_updates
    assert len(all_updates["H001"]) == 2  # Two updates
    
    # Belief should increase with each signal
    final_belief = engine.get_belief("H001")
    assert final_belief > 0.5


def test_get_high_confidence_hypotheses(sample_hypotheses):
    """Test getting high confidence hypotheses."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    engine.set_belief("H001", 0.8, "High confidence")
    engine.set_belief("H002", 0.3, "Low confidence")
    
    high_conf = engine.get_high_confidence_hypotheses(threshold=0.7)
    
    assert "H001" in high_conf
    assert "H002" not in high_conf


def test_get_low_confidence_hypotheses(sample_hypotheses):
    """Test getting low confidence hypotheses."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    engine.set_belief("H001", 0.8, "High confidence")
    engine.set_belief("H002", 0.2, "Low confidence")
    
    low_conf = engine.get_low_confidence_hypotheses(threshold=0.3)
    
    assert "H002" in low_conf
    assert "H001" not in low_conf


def test_get_uncertain_hypotheses(sample_hypotheses):
    """Test getting uncertain hypotheses."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    engine.set_belief("H001", 0.5, "Maximum uncertainty")
    engine.set_belief("H002", 0.9, "High certainty")
    
    uncertain = engine.get_uncertain_hypotheses(epsilon=0.1)
    
    assert "H001" in uncertain
    assert "H002" not in uncertain


def test_compute_entropy(sample_hypotheses):
    """Test entropy computation."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    # Maximum uncertainty (all beliefs at 0.5)
    entropy_max = engine.compute_entropy()
    
    # Reduce uncertainty
    engine.set_belief("H001", 0.9, "High confidence")
    engine.set_belief("H002", 0.1, "Low confidence")
    entropy_reduced = engine.compute_entropy()
    
    # Entropy should decrease when uncertainty decreases
    assert entropy_reduced < entropy_max


def test_apply_graph_constraints_reinforces(sample_hypotheses):
    """Test applying reinforcement constraints."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    # Set H001 to high belief
    engine.set_belief("H001", 0.8, "High belief")
    
    # H001 reinforces H002
    edges = [
        {
            "source_hid": "H001",
            "target_hid": "H002",
            "edge_type": "reinforces",
            "weight": 1.0,
        }
    ]
    
    updates = engine.apply_graph_constraints(edges, influence_strength=0.2)
    
    if "H002" in updates:
        old_belief, new_belief = updates["H002"]
        # H002 belief should increase due to reinforcement from H001
        assert new_belief > old_belief


def test_apply_graph_constraints_contradicts(sample_hypotheses):
    """Test applying contradiction constraints."""
    engine = BeliefUpdateEngine(sample_hypotheses)
    
    # Set H001 to high belief
    engine.set_belief("H001", 0.8, "High belief")
    engine.set_belief("H002", 0.6, "Moderate belief")
    
    # H001 contradicts H002
    edges = [
        {
            "source_hid": "H001",
            "target_hid": "H002",
            "edge_type": "contradicts",
            "weight": 1.0,
        }
    ]
    
    updates = engine.apply_graph_constraints(edges, influence_strength=0.2)
    
    if "H002" in updates:
        old_belief, new_belief = updates["H002"]
        # H002 belief should decrease due to contradiction from H001
        assert new_belief < old_belief


def test_to_dict(sample_hypotheses):
    """Test belief state serialization."""
    engine = BeliefUpdateEngine(sample_hypotheses, prior_belief=0.5)
    engine.set_belief("H001", 0.7, "Update 1")
    
    belief_dict = engine.to_dict()
    
    assert "beliefs" in belief_dict
    assert "explanation_log" in belief_dict
    assert "timestamp" in belief_dict
    assert "statistics" in belief_dict
    
    assert belief_dict["beliefs"]["H001"] == 0.7
    assert belief_dict["beliefs"]["H002"] == 0.5
    assert len(belief_dict["explanation_log"]) >= 2  # Init + update

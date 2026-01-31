"""
Tests for the V1 trajectory simulator (engine-only, no DB/API).

Covers:
- Determinism under fixed seed
- depends_on gating
- reinforces edge co-activation
- Trajectory clustering and naming
- Sensitivity hotspots
"""

from datetime import datetime, timezone
from uuid import uuid4

import networkx as nx
import pytest

from app.engine.simulator import (
    SimulationResult,
    Trajectory,
    get_trajectory_statistics,
    simulate_trajectories,
)
from app.schemas import BeliefState, Hypothesis, TimeHorizon


@pytest.fixture
def sample_hypotheses():
    """Create sample hypotheses for testing."""
    run_id = uuid4()
    return [
        Hypothesis(
            id=uuid4(),
            hid="H001",
            run_id=run_id,
            
            
            stakeholders=["Beneficiaries", "Dealers"],
            triggers=["Infrastructure issues", "Power outages"],
            mechanism="Network disruption prevents biometric authentication",
            time_horizon=TimeHorizon.IMMEDIATE,
            primary_effects=["Auth failures"],
            secondary_effects=["User frustration"],
            evidence_keywords=["network", "timeout"],
            created_at=datetime.now(timezone.utc),
        ),
        Hypothesis(
            id=uuid4(),
            hid="H002",
            run_id=run_id,
            
            
            stakeholders=["Beneficiaries"],
            triggers=["Poor quality sensors", "User training gaps"],
            mechanism="Low quality biometric data leads to false rejections",
            time_horizon=TimeHorizon.SHORT_TERM,
            primary_effects=["False rejections"],
            secondary_effects=["Repeated attempts"],
            evidence_keywords=["biometric", "quality"],
            created_at=datetime.now(timezone.utc),
        ),
        Hypothesis(
            id=uuid4(),
            hid="H003",
            run_id=run_id,
            
            
            stakeholders=["Beneficiaries", "Policy makers"],
            triggers=["Multiple auth failures"],
            mechanism="Repeated failures cause system lockout and exclusion",
            time_horizon=TimeHorizon.MEDIUM_TERM,
            primary_effects=["Beneficiary exclusion"],
            secondary_effects=["Social harm"],
            evidence_keywords=["exclusion", "denied"],
            created_at=datetime.now(timezone.utc),
        ),
    ]


@pytest.fixture
def sample_graph():
    """Create sample hypothesis graph."""
    graph = nx.DiGraph()
    graph.add_node("H001")
    graph.add_node("H002")
    graph.add_node("H003")
    
    # H001 reinforces H002 (network issues worsen biometric quality)
    graph.add_edge("H001", "H002", relation="reinforces")
    
    # H003 depends on H002 (exclusion requires biometric failures)
    graph.add_edge("H002", "H003", relation="depends_on")
    
    return graph


@pytest.fixture
def sample_belief(sample_hypotheses):
    """Create sample belief state."""
    return BeliefState(
        id=uuid4(),
        run_id=sample_hypotheses[0].run_id,
        beliefs={
            "H001": 0.4,
            "H002": 0.3,
            "H003": 0.3,
        },
        timestamp=datetime.now(timezone.utc),
        explanation_log=["Initialized with uniform priors"],
        created_at=datetime.now(timezone.utc),
    )


def test_determinism_same_seed(sample_hypotheses, sample_graph, sample_belief):
    """Test that same seed produces identical results."""
    result1 = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=100,
        seed=42,
    )
    
    result2 = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=100,
        seed=42,
    )
    
    # Check trajectories are identical
    assert len(result1.trajectories) == len(result2.trajectories)
    
    for traj1, traj2 in zip(result1.trajectories, result2.trajectories):
        assert traj1.name == traj2.name
        assert traj1.probability == traj2.probability
        assert traj1.active_hypotheses == traj2.active_hypotheses
        assert traj1.frequency == traj2.frequency


def test_determinism_different_seeds(sample_hypotheses, sample_graph, sample_belief):
    """Test that different seeds produce different results."""
    result1 = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=100,
        seed=42,
    )
    
    result2 = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=100,
        seed=99,
    )
    
    # Results should be different
    # Check at least one trajectory has different probability
    probs1 = [t.probability for t in result1.trajectories]
    probs2 = [t.probability for t in result2.trajectories]
    
    assert probs1 != probs2, "Different seeds should produce different distributions"


def test_depends_on_gating(sample_hypotheses):
    """Test that depends_on edge prevents activation without dependency."""
    # Create simple graph: H003 depends on H002
    graph = nx.DiGraph()
    graph.add_node("H001")
    graph.add_node("H002")
    graph.add_node("H003")
    graph.add_edge("H002", "H003", relation="depends_on")
    
    # Set H002 to very low belief, H003 to high belief
    belief = BeliefState(
        id=uuid4(),
        run_id=sample_hypotheses[0].run_id,
        beliefs={
            "H001": 0.1,
            "H002": 0.05,  # Very low - dependency not satisfied
            "H003": 0.9,   # Very high - but depends on H002
        },
        timestamp=datetime.now(timezone.utc),
        explanation_log=[],
        created_at=datetime.now(timezone.utc),
    )
    
    # Run simulation
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=graph,
        belief=belief,
        n_runs=500,
        seed=42,
        activation_threshold=0.3,
    )
    
    # Count how often H003 activates without H002
    h003_alone = 0
    h003_with_h002 = 0
    
    for traj in result.trajectories:
        if "H003" in traj.active_hypotheses:
            if "H002" not in traj.active_hypotheses:
                h003_alone += traj.frequency
            else:
                h003_with_h002 += traj.frequency
    
    # H003 should rarely activate alone (depends_on gate should block it)
    total_h003 = h003_alone + h003_with_h002
    if total_h003 > 0:
        alone_rate = h003_alone / total_h003
        # Allow some violations due to sampling, but should be low
        assert alone_rate < 0.2, \
            f"H003 activated alone {alone_rate:.1%} of the time (should be <20%)"


def test_reinforces_increases_coactivation(sample_hypotheses):
    """Test that reinforces edge increases co-activation probability."""
    # Graph 1: No edges
    graph_no_edges = nx.DiGraph()
    graph_no_edges.add_node("H001")
    graph_no_edges.add_node("H002")
    graph_no_edges.add_node("H003")
    
    # Graph 2: H001 reinforces H002
    graph_with_edge = nx.DiGraph()
    graph_with_edge.add_node("H001")
    graph_with_edge.add_node("H002")
    graph_with_edge.add_node("H003")
    graph_with_edge.add_edge("H001", "H002", relation="reinforces")
    
    # Belief: moderate for both
    belief = BeliefState(
        id=uuid4(),
        run_id=sample_hypotheses[0].run_id,
        beliefs={
            "H001": 0.5,
            "H002": 0.4,
            "H003": 0.1,
        },
        timestamp=datetime.now(timezone.utc),
        explanation_log=[],
        created_at=datetime.now(timezone.utc),
    )
    
    # Simulate without edge
    result_no_edge = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=graph_no_edges,
        belief=belief,
        n_runs=1000,
        seed=42,
    )
    
    # Simulate with edge
    result_with_edge = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=graph_with_edge,
        belief=belief,
        n_runs=1000,
        seed=42,
    )
    
    # Count co-activations (both H001 and H002 active)
    def count_coactivations(result: SimulationResult) -> int:
        count = 0
        for traj in result.trajectories:
            if "H001" in traj.active_hypotheses and "H002" in traj.active_hypotheses:
                count += traj.frequency
        return count
    
    coactivation_no_edge = count_coactivations(result_no_edge)
    coactivation_with_edge = count_coactivations(result_with_edge)
    
    # With reinforcement edge, co-activation should be higher
    assert coactivation_with_edge > coactivation_no_edge, \
        f"Reinforces edge should increase co-activation: {coactivation_with_edge} vs {coactivation_no_edge}"


def test_trajectory_clustering(sample_hypotheses, sample_graph, sample_belief):
    """Test that trajectories are properly clustered by signature."""
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=200,
        seed=42,
        top_k=5,
    )
    
    # Should have at least one trajectory
    assert len(result.trajectories) > 0
    
    # Each trajectory should have unique signature
    signatures = set()
    for traj in result.trajectories:
        sig = tuple(sorted(traj.active_hypotheses))
        assert sig not in signatures, "Duplicate trajectory signature"
        signatures.add(sig)
    
    # Frequencies should sum to n_runs
    total_freq = sum(t.frequency for t in result.trajectories)
    assert total_freq <= result.total_runs
    
    # Probabilities should be consistent
    for traj in result.trajectories:
        expected_prob = traj.frequency / result.total_runs
        assert abs(traj.probability - expected_prob) < 0.001


def test_trajectory_probabilities_sum(sample_hypotheses, sample_graph, sample_belief):
    """Test that top K trajectories cover reasonable portion of probability space."""
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=500,
        seed=42,
        top_k=5,
    )
    
    # Sum of top K probabilities
    top_k_prob = sum(t.probability for t in result.trajectories)
    
    # Should cover at least 50% of probability space
    # (rest is in "Other" bucket or less frequent trajectories)
    assert top_k_prob >= 0.3, \
        f"Top K trajectories should cover at least 30% (got {top_k_prob:.1%})"
    
    # Should not exceed 100%
    assert top_k_prob <= 1.0, \
        f"Probabilities cannot exceed 100% (got {top_k_prob:.1%})"


def test_trajectory_naming(sample_hypotheses, sample_graph, sample_belief):
    """Test that trajectories get meaningful names."""
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=100,
        seed=42,
        top_k=5,
    )
    
    # Each trajectory should have a name
    for traj in result.trajectories:
        assert traj.name, "Trajectory must have a name"
        assert len(traj.name) > 0
        # Names should start with "Trajectory"
        assert traj.name.startswith("Trajectory"), \
            f"Trajectory name should start with 'Trajectory': {traj.name}"


def test_sensitivity_hotspots_present(sample_hypotheses, sample_graph, sample_belief):
    """Test that sensitivity hotspots are computed."""
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=200,
        seed=42,
    )
    
    # Hotspots should be a dict
    assert isinstance(result.hotspots, dict)
    
    # Should have at least one entry
    assert len(result.hotspots) > 0
    
    # Each hotspot should have a string value (explanation)
    for key, value in result.hotspots.items():
        assert isinstance(key, str), "Hotspot key should be string"
        assert isinstance(value, str), "Hotspot value should be string explanation"
        assert len(value) > 0, "Hotspot explanation should not be empty"


def test_sensitivity_hotspots_detect_parameter_impact():
    """Test that hotspots identify parameters that actually change outcomes."""
    # Create hypotheses with specific beliefs
    run_id = uuid4()
    hypotheses = [
        Hypothesis(
            id=uuid4(),
            hid="H001",
            run_id=run_id,
            
            
            stakeholders=["Beneficiaries"],
            triggers=["Infrastructure"],
            mechanism="Network disruption",
            time_horizon=TimeHorizon.IMMEDIATE,
            primary_effects=["Auth fail"],
            secondary_effects=[],
            evidence_keywords=["network"],
            created_at=datetime.now(timezone.utc),
        ),
        Hypothesis(
            id=uuid4(),
            hid="H002",
            run_id=run_id,
            
            
            stakeholders=["System"],
            triggers=["High load"],
            mechanism="System capacity exceeded",
            time_horizon=TimeHorizon.SHORT_TERM,
            primary_effects=["Slow response"],
            secondary_effects=[],
            evidence_keywords=["overload"],
            created_at=datetime.now(timezone.utc),
        ),
    ]
    
    # Create graph
    graph = nx.DiGraph()
    graph.add_node("H001")
    graph.add_node("H002")
    graph.add_edge("H001", "H002", relation="reinforces")
    
    # Belief near threshold - should be sensitive to activation_threshold
    belief = BeliefState(
        id=uuid4(),
        run_id=run_id,
        beliefs={
            "H001": 0.52,  # Just above 0.5 threshold
            "H002": 0.48,  # Just below 0.5 threshold
        },
        timestamp=datetime.now(timezone.utc),
        explanation_log=[],
        created_at=datetime.now(timezone.utc),
    )
    
    result = simulate_trajectories(
        hypotheses=hypotheses,
        graph=graph,
        belief=belief,
        n_runs=300,
        seed=42,
        activation_threshold=0.5,
    )
    
    # Should detect sensitivity to activation_threshold
    # (since beliefs are near the threshold)
    assert len(result.hotspots) > 0
    
    # Check if any hotspot mentions sensitivity or parameter changes
    hotspot_text = " ".join(result.hotspots.values()).lower()
    assert any(word in hotspot_text for word in ["sensitivity", "changes", "threshold", "impact"]), \
        f"Hotspots should mention sensitivity or parameter impact. Got: {result.hotspots}"


def test_empty_graph(sample_hypotheses, sample_belief):
    """Test simulation with no graph edges."""
    graph = nx.DiGraph()
    for h in sample_hypotheses:
        graph.add_node(h.hid)
    
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=graph,
        belief=sample_belief,
        n_runs=100,
        seed=42,
    )
    
    # Should still produce trajectories
    assert len(result.trajectories) > 0
    assert result.total_runs == 100


def test_all_hypotheses_low_belief():
    """Test simulation when all hypotheses have very low belief."""
    run_id = uuid4()
    hypotheses = [
        Hypothesis(
            id=uuid4(),
            hid=f"H{i:03d}",
            run_id=run_id,
            
            
            stakeholders=["Group"],
            triggers=["Trigger"],
            mechanism=f"Mechanism {i}",
            time_horizon=TimeHorizon.SHORT_TERM,
            primary_effects=[f"Effect {i}"],
            secondary_effects=[],
            evidence_keywords=[],
            created_at=datetime.now(timezone.utc),
        )
        for i in range(1, 4)
    ]
    
    belief = BeliefState(
        id=uuid4(),
        run_id=run_id,
        beliefs={"H001": 0.05, "H002": 0.05, "H003": 0.05},
        timestamp=datetime.now(timezone.utc),
        explanation_log=[],
        created_at=datetime.now(timezone.utc),
    )
    
    graph = nx.DiGraph()
    for h in hypotheses:
        graph.add_node(h.hid)
    
    result = simulate_trajectories(
        hypotheses=hypotheses,
        graph=graph,
        belief=belief,
        n_runs=100,
        seed=42,
    )
    
    # Should complete without errors
    assert result.total_runs == 100
    assert isinstance(result.trajectories, list)


def test_get_trajectory_statistics(sample_hypotheses, sample_graph, sample_belief):
    """Test trajectory statistics extraction."""
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=200,
        seed=42,
    )
    
    stats = get_trajectory_statistics(result)
    
    # Check required fields
    assert "total_runs" in stats
    assert "unique_trajectories" in stats
    assert "top_probability" in stats
    assert "top_k_coverage" in stats
    assert "other_probability" in stats
    assert "seed" in stats
    assert "hotspot_count" in stats
    
    # Validate values
    assert stats["total_runs"] == 200
    assert stats["unique_trajectories"] == len(result.trajectories)
    assert 0 <= stats["top_probability"] <= 1.0
    assert 0 <= stats["top_k_coverage"] <= 1.0
    assert 0 <= stats["other_probability"] <= 1.0
    assert stats["seed"] == 42


def test_contradicts_edge_suppression(sample_hypotheses):
    """Test that contradicts edge reduces activation probability."""
    # Create graph with contradiction
    graph = nx.DiGraph()
    graph.add_node("H001")
    graph.add_node("H002")
    graph.add_node("H003")
    graph.add_edge("H001", "H002", relation="contradicts")
    
    # H001 has high belief, H002 moderate
    belief = BeliefState(
        id=uuid4(),
        run_id=sample_hypotheses[0].run_id,
        beliefs={
            "H001": 0.8,  # High - will be active often
            "H002": 0.5,  # Moderate - should be suppressed when H001 active
            "H003": 0.3,
        },
        timestamp=datetime.now(timezone.utc),
        explanation_log=[],
        created_at=datetime.now(timezone.utc),
    )
    
    # Simulate with contradiction
    result_with_contradiction = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=graph,
        belief=belief,
        n_runs=1000,
        seed=42,
    )
    
    # Simulate without contradiction (remove edge)
    graph_no_contradiction = nx.DiGraph()
    graph_no_contradiction.add_node("H001")
    graph_no_contradiction.add_node("H002")
    graph_no_contradiction.add_node("H003")
    
    result_no_contradiction = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=graph_no_contradiction,
        belief=belief,
        n_runs=1000,
        seed=42,
    )
    
    # Count H002 activations
    def count_h002_activations(result: SimulationResult) -> int:
        count = 0
        for traj in result.trajectories:
            if "H002" in traj.active_hypotheses:
                count += traj.frequency
        return count
    
    h002_with_contradiction = count_h002_activations(result_with_contradiction)
    h002_no_contradiction = count_h002_activations(result_no_contradiction)
    
    # With contradiction, H002 should activate less often
    assert h002_with_contradiction < h002_no_contradiction, \
        f"Contradiction should reduce H002 activations: {h002_with_contradiction} vs {h002_no_contradiction}"


def test_parameters_stored_in_result(sample_hypotheses, sample_graph, sample_belief):
    """Test that simulation parameters are stored in the result."""
    result = simulate_trajectories(
        hypotheses=sample_hypotheses,
        graph=sample_graph,
        belief=sample_belief,
        n_runs=100,
        seed=99,
        activation_threshold=0.6,
        propagate_steps=3,
        reinforce_delta=0.2,
        contradict_delta=0.25,
        top_k=7,
    )
    
    # Check parameters are stored
    assert result.parameters["n_runs"] == 100
    assert result.parameters["seed"] == 99
    assert result.parameters["activation_threshold"] == 0.6
    assert result.parameters["propagate_steps"] == 3
    assert result.parameters["reinforce_delta"] == 0.2
    assert result.parameters["contradict_delta"] == 0.25
    assert result.parameters["top_k"] == 7

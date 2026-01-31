"""
Tests for Hypothesis Graph Builder
"""

import pytest

from app.engine import HypothesisGraphBuilder
from app.schemas import EdgeType, Hypothesis, TimeHorizon


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
        Hypothesis(
            id="00000000-0000-0000-0000-000000000003",
            run_id="00000000-0000-0000-0000-000000000000",
            hid="H003",
            stakeholders=["Group C"],
            triggers=["Trigger 3"],
            mechanism="Mechanism 3",
            primary_effects=["Effect 3"],
            secondary_effects=["Secondary 3"],
            time_horizon=TimeHorizon.MEDIUM_TERM,
            created_at="2026-01-31T00:00:00",
        ),
    ]


def test_graph_builder_initialization(sample_hypotheses):
    """Test graph builder initialization."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    
    assert builder.graph.number_of_nodes() == 3
    assert builder.graph.number_of_edges() == 0
    assert "H001" in builder.hypotheses
    assert "H002" in builder.hypotheses
    assert "H003" in builder.hypotheses


def test_add_edge(sample_hypotheses):
    """Test adding edges to the graph."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    
    builder.add_edge("H001", "H002", EdgeType.REINFORCES, notes="Test edge")
    
    assert builder.graph.number_of_edges() == 1
    assert builder.graph.has_edge("H001", "H002")
    
    edge_data = builder.graph.get_edge_data("H001", "H002")
    assert edge_data["edge_type"] == EdgeType.REINFORCES.value
    assert edge_data["weight"] == 1.0  # V1 uses uniform weight
    assert edge_data["notes"] == "Test edge"


def test_add_edge_invalid_source(sample_hypotheses):
    """Test adding edge with invalid source."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    
    with pytest.raises(ValueError, match="Source HID .* not found"):
        builder.add_edge("H999", "H002", EdgeType.REINFORCES)


def test_add_edge_self_loop(sample_hypotheses):
    """Test that self-loops are not allowed."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    
    with pytest.raises(ValueError, match="Self-loops are not allowed"):
        builder.add_edge("H001", "H001", EdgeType.REINFORCES)


def test_validate_empty_graph(sample_hypotheses):
    """Test validation of empty graph."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    
    is_valid, warnings = builder.validate()
    assert is_valid
    assert len(warnings) == 0  # Empty graph is valid


def test_validate_with_edges(sample_hypotheses):
    """Test validation with edges."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    builder.add_edge("H001", "H002", EdgeType.REINFORCES)
    builder.add_edge("H002", "H003", EdgeType.CONTRADICTS)
    
    is_valid, warnings = builder.validate()
    assert is_valid
    # May have warnings about isolated nodes


def test_get_predecessors_successors(sample_hypotheses):
    """Test getting predecessors and successors."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    builder.add_edge("H001", "H002", EdgeType.REINFORCES)
    builder.add_edge("H001", "H003", EdgeType.REINFORCES)
    
    # H001 has no predecessors, two successors
    assert builder.get_predecessors("H001") == []
    assert set(builder.get_successors("H001")) == {"H002", "H003"}
    
    # H002 has one predecessor, no successors
    assert builder.get_predecessors("H002") == ["H001"]
    assert builder.get_successors("H002") == []


def test_get_reinforcement_chain(sample_hypotheses):
    """Test getting reinforcement chain."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    builder.add_edge("H001", "H002", EdgeType.REINFORCES)
    builder.add_edge("H002", "H003", EdgeType.REINFORCES)
    
    chain = builder.get_reinforcement_chain("H001", max_depth=2)
    assert chain == {"H001", "H002", "H003"}


def test_get_contradiction_pairs(sample_hypotheses):
    """Test getting contradiction pairs."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    builder.add_edge("H001", "H002", EdgeType.CONTRADICTS)
    builder.add_edge("H002", "H003", EdgeType.REINFORCES)
    
    pairs = builder.get_contradiction_pairs()
    assert len(pairs) == 1
    assert pairs[0] == ("H001", "H002")


def test_compute_centrality(sample_hypotheses):
    """Test centrality computation."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    builder.add_edge("H001", "H002", EdgeType.REINFORCES)
    builder.add_edge("H001", "H003", EdgeType.REINFORCES)
    
    centrality = builder.compute_centrality()
    
    # H001 should have highest centrality (most connections)
    assert centrality["H001"] > centrality["H002"]
    assert centrality["H001"] > centrality["H003"]


def test_to_dict(sample_hypotheses):
    """Test graph serialization to dict."""
    builder = HypothesisGraphBuilder(sample_hypotheses)
    builder.add_edge("H001", "H002", EdgeType.REINFORCES)
    
    graph_dict = builder.to_dict()
    
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert "metadata" in graph_dict
    assert len(graph_dict["nodes"]) == 3
    assert len(graph_dict["edges"]) == 1
    assert graph_dict["edges"][0]["source_hid"] == "H001"
    assert graph_dict["edges"][0]["target_hid"] == "H002"
    assert graph_dict["edges"][0]["edge_type"] == EdgeType.REINFORCES.value
    assert graph_dict["edges"][0]["weight"] == 1.0  # V1 uses uniform weight


def test_from_dict(sample_hypotheses):
    """Test graph reconstruction from dict."""
    # Create and serialize graph
    builder1 = HypothesisGraphBuilder(sample_hypotheses)
    builder1.add_edge("H001", "H002", EdgeType.REINFORCES)
    builder1.add_edge("H002", "H003", EdgeType.CONTRADICTS)
    graph_dict = builder1.to_dict()
    
    # Reconstruct from dict
    builder2 = HypothesisGraphBuilder.from_dict(sample_hypotheses, graph_dict)
    
    assert builder2.graph.number_of_nodes() == 3
    assert builder2.graph.number_of_edges() == 2
    assert builder2.graph.has_edge("H001", "H002")
    assert builder2.graph.has_edge("H002", "H003")

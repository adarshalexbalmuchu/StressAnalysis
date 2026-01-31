"""
Unit tests for hypothesis graph builder.

Tests graph construction, validation, and subgraph extraction.
"""

import pytest
import networkx as nx

from app.engine.graph_builder import (
    build_graph,
    validate_graph,
    extract_subgraph,
    get_graph_statistics,
)
from app.schemas import Edge, Hypothesis


# Test fixtures - using dicts for simplicity in tests
@pytest.fixture
def sample_hypotheses():
    """Sample hypotheses for testing."""
    return [
        {
            "hid": "H001",
            "stakeholders": ["beneficiaries"],
            "triggers": ["biometric failure"],
            "mechanism": "Authentication failures exclude eligible beneficiaries",
            "primary_effects": ["exclusion"],
            "time_horizon": "immediate",
        },
        {
            "hid": "H002",
            "stakeholders": ["beneficiaries", "fps_operators"],
            "triggers": ["network issues"],
            "mechanism": "Network connectivity prevents authentication",
            "primary_effects": ["service delay"],
            "time_horizon": "immediate",
        },
        {
            "hid": "H003",
            "stakeholders": ["elderly", "manual laborers"],
            "triggers": ["poor biometric quality"],
            "mechanism": "Low biometric quality causes auth failures",
            "primary_effects": ["exclusion"],
            "time_horizon": "short_term",
        },
    ]


@pytest.fixture
def sample_edges():
    """Sample edges for testing."""
    return [
        Edge(
            source_hid="H002",
            target_hid="H001",
            edge_type="reinforces",
            notes="Network failures contribute to exclusion",
        ),
        Edge(
            source_hid="H003",
            target_hid="H001",
            edge_type="reinforces",
            notes="Poor biometric quality leads to auth failures",
        ),
    ]


def test_build_graph_basic(sample_hypotheses, sample_edges):
    """Test basic graph construction."""
    graph = build_graph(sample_hypotheses, sample_edges)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

    # Check nodes have hypothesis data
    assert "H001" in graph.nodes
    assert graph.nodes["H001"]["mechanism"] == sample_hypotheses[0]["mechanism"]

    # Check edges have relation attribute and uniform weight
    edge_data = graph.get_edge_data("H002", "H001")
    assert edge_data["relation"] == "reinforces"
    assert edge_data["weight"] == 1.0  # V1: uniform weights


def test_build_graph_with_dicts():
    """Test graph construction with dict inputs."""
    hypotheses = [
        {
            "hid": "H001",
            "stakeholders": ["test"],
            "triggers": ["test"],
            "mechanism": "Test hypothesis",
            "primary_effects": ["test"],
            "time_horizon": "immediate",
        }
    ]
    edges = [
        {
            "source_hid": "H001",
            "target_hid": "H001",
            "edge_type": "reinforces",
            "notes": "",
        }
    ]

    # This should fail due to self-loop validation
    with pytest.raises(ValueError, match="self-loop"):
        build_graph(hypotheses, edges)


def test_validate_graph_success(sample_hypotheses, sample_edges):
    """Test validation passes for valid graph."""
    errors = validate_graph(sample_hypotheses, sample_edges)
    assert errors == []


def test_validate_duplicate_hids():
    """Test validation catches duplicate hypothesis IDs."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "First", "primary_effects": ["test"], "time_horizon": "immediate"},
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Duplicate", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = []

    errors = validate_graph(hypotheses, edges)
    assert any("Duplicate hypothesis IDs" in err for err in errors)
    assert any("H001" in err for err in errors)


def test_validate_missing_source_node():
    """Test validation catches edge with missing source node."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = [
        Edge(source_hid="H999", target_hid="H001", edge_type="reinforces", notes=""),
    ]

    errors = validate_graph(hypotheses, edges)
    assert any("H999" in err and "not found" in err for err in errors)


def test_validate_missing_target_node():
    """Test validation catches edge with missing target node."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = [
        Edge(source_hid="H001", target_hid="H999", edge_type="reinforces", notes=""),
    ]

    errors = validate_graph(hypotheses, edges)
    assert any("H999" in err and "not found" in err for err in errors)


def test_validate_self_loop():
    """Test validation catches self-loops."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = [
        Edge(source_hid="H001", target_hid="H001", edge_type="reinforces", notes=""),
    ]

    errors = validate_graph(hypotheses, edges)
    assert any("self-loop" in err for err in errors)


def test_validate_no_edges():
    """Test validation warns about graphs with no edges."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = []

    errors = validate_graph(hypotheses, edges)
    assert any("no edges" in err for err in errors)


def test_validate_no_hypotheses():
    """Test validation catches graphs with no hypotheses."""
    hypotheses = []
    edges = []

    errors = validate_graph(hypotheses, edges)
    assert any("at least 1 hypothesis" in err for err in errors)


def test_validate_invalid_relation_type():
    """Test validation catches invalid relation types."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 1", "primary_effects": ["test"], "time_horizon": "immediate"},
        {"hid": "H002", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 2", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = [
        {
            "source_hid": "H001",
            "target_hid": "H002",
            "edge_type": "invalid_relation",  # Invalid type
            "notes": "",
        }
    ]

    errors = validate_graph(hypotheses, edges)
    assert any("invalid relation" in err for err in errors)


def test_validate_depends_on_cycle():
    """Test validation catches dependency cycles."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 1", "primary_effects": ["test"], "time_horizon": "immediate"},
        {"hid": "H002", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 2", "primary_effects": ["test"], "time_horizon": "immediate"},
        {"hid": "H003", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 3", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = [
        Edge(source_hid="H001", target_hid="H002", edge_type="depends_on", notes=""),
        Edge(source_hid="H002", target_hid="H003", edge_type="depends_on", notes=""),
        Edge(source_hid="H003", target_hid="H001", edge_type="depends_on", notes=""),  # Cycle
    ]

    errors = validate_graph(hypotheses, edges)
    assert any("cycle" in err.lower() for err in errors)


def test_validate_reinforces_cycle_allowed():
    """Test that cycles in reinforces edges are allowed (only depends_on cycles are flagged)."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 1", "primary_effects": ["test"], "time_horizon": "immediate"},
        {"hid": "H002", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 2", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = [
        Edge(source_hid="H001", target_hid="H002", edge_type="reinforces", notes=""),
        Edge(source_hid="H002", target_hid="H001", edge_type="reinforces", notes=""),
    ]

    errors = validate_graph(hypotheses, edges)
    # Should have no errors (reinforces cycles are OK)
    assert errors == []


def test_extract_subgraph_basic(sample_hypotheses, sample_edges):
    """Test basic subgraph extraction."""
    graph = build_graph(sample_hypotheses, sample_edges)

    # Extract subgraph around H001 without neighbors
    subgraph = extract_subgraph(graph, ["H001"], include_neighbors=False)

    assert subgraph.number_of_nodes() == 1
    assert "H001" in subgraph.nodes


def test_extract_subgraph_with_neighbors(sample_hypotheses, sample_edges):
    """Test subgraph extraction with immediate neighbors."""
    graph = build_graph(sample_hypotheses, sample_edges)

    # Extract subgraph around H001 with neighbors
    # H001 has incoming edges from H002 and H003
    subgraph = extract_subgraph(graph, ["H001"], include_neighbors=True, depth=1)

    assert subgraph.number_of_nodes() == 3  # H001 + H002 + H003
    assert "H001" in subgraph.nodes
    assert "H002" in subgraph.nodes
    assert "H003" in subgraph.nodes


def test_extract_subgraph_multiple_seeds(sample_hypotheses, sample_edges):
    """Test subgraph extraction with multiple seed nodes."""
    graph = build_graph(sample_hypotheses, sample_edges)

    subgraph = extract_subgraph(graph, ["H001", "H002"], include_neighbors=False)

    assert subgraph.number_of_nodes() == 2
    assert "H001" in subgraph.nodes
    assert "H002" in subgraph.nodes


def test_extract_subgraph_missing_node(sample_hypotheses, sample_edges):
    """Test subgraph extraction fails with missing hypothesis ID."""
    graph = build_graph(sample_hypotheses, sample_edges)

    with pytest.raises(ValueError, match="not found in graph"):
        extract_subgraph(graph, ["H999"], include_neighbors=False)


def test_get_graph_statistics(sample_hypotheses, sample_edges):
    """Test graph statistics computation."""
    graph = build_graph(sample_hypotheses, sample_edges)

    stats = get_graph_statistics(graph)

    assert stats["node_count"] == 3
    assert stats["edge_count"] == 2
    assert "relation_distribution" in stats
    assert stats["relation_distribution"]["reinforces"] == 2
    assert "avg_in_degree" in stats
    assert "avg_out_degree" in stats


def test_build_graph_all_relation_types():
    """Test graph with all relation types."""
    hypotheses = [
        {"hid": "H001", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 1", "primary_effects": ["test"], "time_horizon": "immediate"},
        {"hid": "H002", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 2", "primary_effects": ["test"], "time_horizon": "immediate"},
        {"hid": "H003", "stakeholders": ["test"], "triggers": ["test"], "mechanism": "Test 3", "primary_effects": ["test"], "time_horizon": "immediate"},
    ]
    edges = [
        Edge(source_hid="H001", target_hid="H002", edge_type="reinforces", notes=""),
        Edge(source_hid="H002", target_hid="H003", edge_type="contradicts", notes=""),
        Edge(source_hid="H001", target_hid="H003", edge_type="depends_on", notes=""),
    ]

    graph = build_graph(hypotheses, edges)

    assert graph.number_of_edges() == 3

    # Check each edge has correct relation
    assert graph.get_edge_data("H001", "H002")["relation"] == "reinforces"
    assert graph.get_edge_data("H002", "H003")["relation"] == "contradicts"
    assert graph.get_edge_data("H001", "H003")["relation"] == "depends_on"

    # Check all edges have uniform weight
    for _, _, data in graph.edges(data=True):
        assert data["weight"] == 1.0

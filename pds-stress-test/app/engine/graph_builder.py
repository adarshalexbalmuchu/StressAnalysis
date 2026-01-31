"""
Hypothesis Graph Builder for Policy Stress-Testing Engine.

Builds and validates directed hypothesis graphs using NetworkX.
V1 uses uniform edge weights (no strength field).

NO web framework dependencies. NO database logic.
"""

from typing import Any

import networkx as nx

from app.schemas import Edge, Hypothesis

# Allowed relation types (from schema)
ALLOWED_RELATIONS = {"reinforces", "contradicts", "depends_on"}


def build_graph(
    hypotheses: list[Hypothesis] | list[dict[str, Any]],
    edges: list[Edge] | list[dict[str, Any]],
) -> nx.DiGraph:
    """
    Build a directed hypothesis graph from hypotheses and edges.

    Args:
        hypotheses: List of Hypothesis objects or dicts
        edges: List of Edge objects or dicts

    Returns:
        NetworkX DiGraph with hypotheses as nodes and relations as edges

    Raises:
        ValueError: If validation fails (use validate_graph first for detailed errors)
    """
    # Validate first
    errors = validate_graph(hypotheses, edges)
    if errors:
        raise ValueError(f"Graph validation failed: {errors}")

    # Create directed graph
    graph = nx.DiGraph()

    # Add hypothesis nodes
    for hyp in hypotheses:
        if isinstance(hyp, Hypothesis):
            hyp_dict = hyp.model_dump()
        else:
            hyp_dict = hyp

        hid = hyp_dict["hid"]
        graph.add_node(hid, **hyp_dict)

    # Add edges with relation attribute
    for edge in edges:
        if isinstance(edge, Edge):
            edge_dict = edge.model_dump()
        else:
            edge_dict = edge

        source = edge_dict["source_hid"]
        target = edge_dict["target_hid"]
        relation = edge_dict["edge_type"]

        # V1: uniform weight = 1.0 (no strength field)
        graph.add_edge(
            source,
            target,
            relation=relation,
            edge_type=relation,  # Alias for consistency
            weight=1.0,
            notes=edge_dict.get("notes", ""),
        )

    return graph


def validate_graph(
    hypotheses: list[Hypothesis] | list[dict[str, Any]],
    edges: list[Edge] | list[dict[str, Any]],
) -> list[str]:
    """
    Validate hypothesis graph structure and constraints.

    Returns:
        List of validation error messages (empty if valid)

    Validation rules:
    - No duplicate hypothesis IDs
    - At least 1 hypothesis
    - At least 1 edge (warning if empty)
    - All edges reference existing source/target hids
    - No self-loops
    - Valid relation types
    - No cycles in depends_on edges (logical inconsistency)
    """
    errors = []

    # Extract hids and check for duplicates
    hids = []
    for hyp in hypotheses:
        if isinstance(hyp, Hypothesis):
            hid = hyp.hid
        else:
            hid = hyp["hid"]
        hids.append(hid)

    # Check for duplicates
    if len(hids) != len(set(hids)):
        duplicates = [hid for hid in hids if hids.count(hid) > 1]
        errors.append(f"Duplicate hypothesis IDs found: {set(duplicates)}")

    # Check minimum node count
    if len(hids) == 0:
        errors.append("Graph must have at least 1 hypothesis")

    # Check minimum edge count
    if len(edges) == 0:
        errors.append("Graph has no edges (minimum 1 required)")

    # Create hid set for fast lookup
    hid_set = set(hids)

    # Validate edges
    depends_on_edges = []
    for idx, edge in enumerate(edges):
        if isinstance(edge, Edge):
            source = edge.source_hid
            target = edge.target_hid
            relation = edge.edge_type
        else:
            source = edge["source_hid"]
            target = edge["target_hid"]
            relation = edge["edge_type"]

        # Check source exists
        if source not in hid_set:
            errors.append(f"Edge {idx}: source_hid '{source}' not found in hypotheses")

        # Check target exists
        if target not in hid_set:
            errors.append(f"Edge {idx}: target_hid '{target}' not found in hypotheses")

        # Check self-loop (V1: disallow)
        if source == target:
            errors.append(f"Edge {idx}: self-loop detected (source == target == '{source}')")

        # Check valid relation type
        if relation not in ALLOWED_RELATIONS:
            errors.append(
                f"Edge {idx}: invalid relation '{relation}' (must be one of {ALLOWED_RELATIONS})"
            )

        # Track depends_on edges for cycle detection
        if relation == "depends_on":
            depends_on_edges.append((source, target))

    # Detect cycles in depends_on edges (logical inconsistency)
    if depends_on_edges and not errors:
        # Build dependency subgraph
        dep_graph = nx.DiGraph()
        dep_graph.add_edges_from(depends_on_edges)

        try:
            cycles = list(nx.simple_cycles(dep_graph))
            if cycles:
                cycle_strs = [" -> ".join(cycle + [cycle[0]]) for cycle in cycles]
                errors.append(f"Dependency cycles detected (logically inconsistent): {cycle_strs}")
        except Exception:
            # If cycle detection fails, skip this check
            pass

    return errors


def extract_subgraph(
    graph: nx.DiGraph,
    hypothesis_ids: list[str],
    include_neighbors: bool = True,
    depth: int = 1,
) -> nx.DiGraph:
    """
    Extract a subgraph focused on specific hypotheses.

    Useful for visualization and local analysis.

    Args:
        graph: Full hypothesis graph
        hypothesis_ids: Hypothesis IDs to include in subgraph
        include_neighbors: If True, include neighbors up to specified depth
        depth: How many hops to include (1 = immediate neighbors)

    Returns:
        NetworkX DiGraph subgraph

    Raises:
        ValueError: If any hypothesis_id not found in graph
    """
    # Validate all hypothesis_ids exist
    missing = [hid for hid in hypothesis_ids if hid not in graph.nodes]
    if missing:
        raise ValueError(f"Hypothesis IDs not found in graph: {missing}")

    # Start with seed nodes
    subgraph_nodes = set(hypothesis_ids)

    # Add neighbors if requested
    if include_neighbors and depth > 0:
        for _ in range(depth):
            new_nodes = set()
            for node in subgraph_nodes:
                # Add predecessors (inbound neighbors)
                new_nodes.update(graph.predecessors(node))
                # Add successors (outbound neighbors)
                new_nodes.update(graph.successors(node))
            subgraph_nodes.update(new_nodes)

    # Create subgraph
    subgraph = graph.subgraph(subgraph_nodes).copy()

    return subgraph


def get_graph_statistics(graph: nx.DiGraph) -> dict[str, Any]:
    """
    Compute basic graph statistics for analysis.

    Args:
        graph: Hypothesis graph

    Returns:
        Dictionary with graph metrics
    """
    stats = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "is_connected": nx.is_weakly_connected(graph),
        "is_dag": nx.is_directed_acyclic_graph(graph),
    }

    # Relation type distribution
    relation_counts = {}
    for _, _, data in graph.edges(data=True):
        relation = data.get("relation", "unknown")
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    stats["relation_distribution"] = relation_counts

    # Degree statistics
    if graph.number_of_nodes() > 0:
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]

        stats["avg_in_degree"] = sum(in_degrees) / len(in_degrees)
        stats["avg_out_degree"] = sum(out_degrees) / len(out_degrees)
        stats["max_in_degree"] = max(in_degrees) if in_degrees else 0
        stats["max_out_degree"] = max(out_degrees) if out_degrees else 0

    return stats

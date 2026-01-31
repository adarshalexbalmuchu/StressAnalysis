"""
Hypothesis Graph Builder

Constructs and validates directed graphs of hypothesis relationships.
This module must NOT import FastAPI - it's pure engine logic.
"""

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from app.schemas import EdgeType, GraphEdge, Hypothesis


class HypothesisGraphBuilder:
    """
    Builds and validates hypothesis graphs.
    
    The graph captures relationships between hypotheses:
    - reinforces: H1 strengthens H2
    - contradicts: H1 weakens H2
    - depends_on: H1 requires H2 to be true
    """

    def __init__(self, hypotheses: List[Hypothesis]):
        """
        Initialize builder with a set of hypotheses.
        
        Args:
            hypotheses: List of hypothesis objects
        """
        self.hypotheses = {h.hid: h for h in hypotheses}
        self.graph = nx.DiGraph()
        
        # Add all hypotheses as nodes
        for hid in self.hypotheses.keys():
            self.graph.add_node(hid)

    def add_edge(
        self,
        source_hid: str,
        target_hid: str,
        edge_type: EdgeType,
        notes: Optional[str] = None,
    ) -> None:
        """
        Add a directed edge between two hypotheses.
        
        V1 uses uniform edge weights (1.0) for all relationships.
        
        Args:
            source_hid: Source hypothesis HID
            target_hid: Target hypothesis HID
            edge_type: Type of relationship
            notes: Optional explanation
            
        Raises:
            ValueError: If HIDs don't exist or edge is invalid
        """
        if source_hid not in self.hypotheses:
            raise ValueError(f"Source HID {source_hid} not found in hypothesis set")
        if target_hid not in self.hypotheses:
            raise ValueError(f"Target HID {target_hid} not found in hypothesis set")
        if source_hid == target_hid:
            raise ValueError("Self-loops are not allowed")
        
        # V1 uses uniform edge weight of 1.0 for all relationships
        self.graph.add_edge(
            source_hid,
            target_hid,
            edge_type=edge_type.value,
            weight=1.0,
            notes=notes,
        )

    def add_edges(self, edges: List[GraphEdge]) -> None:
        """
        Add multiple edges to the graph.
        
        Args:
            edges: List of edge specifications
        """
        for edge in edges:
            self.add_edge(
                edge.source_hid,
                edge.target_hid,
                edge.edge_type,
                edge.notes,
            )

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the graph structure.
        
        Checks:
        - No isolated nodes (except if graph is empty)
        - Edges reference valid nodes
        - Logical consistency (no circular contradictions)
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings: List[str] = []
        
        # Check for isolated nodes
        if self.graph.number_of_edges() > 0:
            isolated = list(nx.isolates(self.graph))
            if isolated:
                warnings.append(
                    f"Found {len(isolated)} isolated hypotheses with no relationships: {isolated[:5]}"
                )
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                warnings.append(
                    f"Found {len(cycles)} cycles in graph. "
                    f"First cycle: {cycles[0] if cycles else None}"
                )
        except Exception:
            # NetworkX can throw if graph is too complex
            warnings.append("Could not check for cycles - graph may be very complex")
        
        # Check for contradictory relationship chains
        contradict_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == EdgeType.CONTRADICTS.value
        ]
        
        if len(contradict_edges) > len(self.graph.nodes()) * 0.5:
            warnings.append(
                f"High proportion of contradiction edges ({len(contradict_edges)}/{self.graph.number_of_edges()})"
            )
        
        # Graph is considered valid even with warnings
        return True, warnings

    def get_subgraph(self, hids: List[str]) -> nx.DiGraph:
        """
        Extract a subgraph containing only specified hypotheses.
        
        Args:
            hids: List of hypothesis IDs to include
            
        Returns:
            Subgraph as NetworkX DiGraph
        """
        return self.graph.subgraph(hids).copy()

    def get_predecessors(self, hid: str) -> List[str]:
        """
        Get all hypotheses that directly influence this hypothesis.
        
        Args:
            hid: Hypothesis ID
            
        Returns:
            List of predecessor HIDs
        """
        return list(self.graph.predecessors(hid))

    def get_successors(self, hid: str) -> List[str]:
        """
        Get all hypotheses directly influenced by this hypothesis.
        
        Args:
            hid: Hypothesis ID
            
        Returns:
            List of successor HIDs
        """
        return list(self.graph.successors(hid))

    def get_reinforcement_chain(self, hid: str, max_depth: int = 3) -> Set[str]:
        """
        Get all hypotheses in the reinforcement chain starting from hid.
        
        Args:
            hid: Starting hypothesis ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Set of HIDs in the reinforcement chain
        """
        chain: Set[str] = {hid}
        frontier = [hid]
        depth = 0
        
        while frontier and depth < max_depth:
            next_frontier = []
            for node in frontier:
                for successor in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, successor)
                    if edge_data and edge_data.get("edge_type") == EdgeType.REINFORCES.value:
                        if successor not in chain:
                            chain.add(successor)
                            next_frontier.append(successor)
            
            frontier = next_frontier
            depth += 1
        
        return chain

    def get_contradiction_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all pairs of hypotheses that contradict each other.
        
        Returns:
            List of (hid1, hid2) tuples
        """
        pairs = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == EdgeType.CONTRADICTS.value:
                pairs.append((u, v))
        return pairs

    def compute_centrality(self) -> Dict[str, float]:
        """
        Compute centrality scores for all hypotheses.
        
        Uses PageRank to identify influential hypotheses in the graph.
        
        Returns:
            Dict mapping HID to centrality score
        """
        if self.graph.number_of_edges() == 0:
            return {hid: 0.0 for hid in self.hypotheses.keys()}
        
        try:
            centrality = nx.pagerank(self.graph)
            return centrality
        except Exception:
            # Fallback to simple degree centrality
            return {
                hid: self.graph.degree(hid) / max(self.graph.number_of_nodes() - 1, 1)
                for hid in self.hypotheses.keys()
            }

    def get_connected_components(self) -> List[Set[str]]:
        """
        Get weakly connected components in the graph.
        
        Returns:
            List of sets, each containing HIDs in a connected component
        """
        return [set(component) for component in nx.weakly_connected_components(self.graph)]

    def to_dict(self) -> Dict:
        """
        Export graph structure as a dictionary for serialization.
        
        Returns:
            Dictionary with nodes, edges, and metadata
        """
        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            edges_data.append({
                "source_hid": u,
                "target_hid": v,
                "edge_type": data.get("edge_type"),
                "weight": data.get("weight", 1.0),
                "notes": data.get("notes"),
            })
        
        centrality = self.compute_centrality()
        components = self.get_connected_components()
        
        return {
            "nodes": list(self.graph.nodes()),
            "edges": edges_data,
            "metadata": {
                "node_count": self.graph.number_of_nodes(),
                "edge_count": self.graph.number_of_edges(),
                "centrality": centrality,
                "components": [list(comp) for comp in components],
                "density": nx.density(self.graph),
            },
        }

    @classmethod
    def from_dict(cls, hypotheses: List[Hypothesis], graph_dict: Dict) -> "HypothesisGraphBuilder":
        """
        Reconstruct a graph from serialized dictionary.
        
        Args:
            hypotheses: List of hypothesis objects
            graph_dict: Dictionary with nodes and edges
            
        Returns:
            New HypothesisGraphBuilder instance
        """
        builder = cls(hypotheses)
        
        for edge_data in graph_dict.get("edges", []):
            builder.add_edge(
                edge_data["source_hid"],
                edge_data["target_hid"],
                EdgeType(edge_data["edge_type"]),
                edge_data.get("notes"),
            )
        
        return builder

"""
Seed data loader script.

Loads hypothesis sets from JSON files into a run.
"""

import json
from pathlib import Path
from typing import Dict, List

from app.schemas import GraphEdgeCreate, HypothesisCreate, TimeHorizon


def load_seed_file(filepath: str) -> Dict:
    """
    Load seed data from JSON file.
    
    Args:
        filepath: Path to seed JSON file
        
    Returns:
        Parsed seed data dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)


def parse_hypotheses(seed_data: Dict) -> List[HypothesisCreate]:
    """
    Parse hypotheses from seed data into schema objects.
    
    Args:
        seed_data: Seed data dictionary
        
    Returns:
        List of HypothesisCreate objects
    """
    hypotheses = []
    
    for h_data in seed_data.get("hypotheses", []):
        hypothesis = HypothesisCreate(
            hid=h_data["hid"],
            stakeholders=h_data["stakeholders"],
            triggers=h_data["triggers"],
            mechanism=h_data["mechanism"],
            primary_effects=h_data["primary_effects"],
            secondary_effects=h_data.get("secondary_effects", []),
            time_horizon=TimeHorizon(h_data["time_horizon"]),
            confidence_notes=h_data.get("confidence_notes"),
        )
        hypotheses.append(hypothesis)
    
    return hypotheses


def parse_edges(seed_data: Dict) -> List[GraphEdgeCreate]:
    """
    Parse graph edges from seed data.
    
    Args:
        seed_data: Seed data dictionary
        
    Returns:
        List of GraphEdgeCreate objects
    """
    from app.schemas import EdgeType
    
    edges = []
    
    for e_data in seed_data.get("graph_edges", []):
        edge = GraphEdgeCreate(
            source_hid=e_data["source_hid"],
            target_hid=e_data["target_hid"],
            edge_type=EdgeType(e_data["edge_type"]),
            strength=e_data.get("strength"),
            notes=e_data.get("notes"),
        )
        edges.append(edge)
    
    return edges


def get_available_seeds() -> List[str]:
    """
    Get list of available seed files.
    
    Returns:
        List of seed file names
    """
    seeds_dir = Path(__file__).parent
    return [f.name for f in seeds_dir.glob("*.json")]


if __name__ == "__main__":
    # Example usage
    print("Available seed files:")
    for seed in get_available_seeds():
        print(f"  - {seed}")
    
    # Load and parse PDS seed
    seed_path = Path(__file__).parent / "pds_biometric_v1.json"
    seed_data = load_seed_file(str(seed_path))
    
    print(f"\nLoaded seed: {seed_data['policy_rule']}")
    print(f"Hypotheses: {len(seed_data['hypotheses'])}")
    print(f"Edges: {len(seed_data['graph_edges'])}")
    
    hypotheses = parse_hypotheses(seed_data)
    edges = parse_edges(seed_data)
    
    print(f"\nParsed {len(hypotheses)} hypotheses and {len(edges)} edges")

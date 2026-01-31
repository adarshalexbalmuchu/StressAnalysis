"""
Seed data loaders for hypothesis graphs.

Loads hypotheses and edges from JSON files in data/seeds/.
"""

import json
from pathlib import Path

from app.schemas import Edge, Hypothesis


def load_hypotheses(path: str) -> list[Hypothesis]:
    """
    Load hypotheses from JSON file.

    Args:
        path: Path to JSON file containing list of hypothesis objects

    Returns:
        List of validated Hypothesis objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or validation fails
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Hypothesis file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of hypotheses, got {type(data).__name__}")

    # Validate each hypothesis using Pydantic schema
    hypotheses = []
    for idx, hyp_dict in enumerate(data):
        try:
            hyp = Hypothesis(**hyp_dict)
            hypotheses.append(hyp)
        except Exception as e:
            raise ValueError(f"Hypothesis validation failed at index {idx}: {e}") from e

    return hypotheses


def load_edges(path: str) -> list[Edge]:
    """
    Load edges from JSON file.

    Args:
        path: Path to JSON file containing list of edge objects

    Returns:
        List of validated Edge objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or validation fails
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Edge file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of edges, got {type(data).__name__}")

    # Validate each edge using Pydantic schema
    edges = []
    for idx, edge_dict in enumerate(data):
        try:
            edge = Edge(**edge_dict)
            edges.append(edge)
        except Exception as e:
            raise ValueError(f"Edge validation failed at index {idx}: {e}") from e

    return edges


def save_hypotheses(hypotheses: list[Hypothesis], path: str) -> None:
    """
    Save hypotheses to JSON file.

    Args:
        hypotheses: List of Hypothesis objects
        path: Path to output JSON file
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = [hyp.model_dump() for hyp in hypotheses]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_edges(edges: list[Edge], path: str) -> None:
    """
    Save edges to JSON file.

    Args:
        edges: List of Edge objects
        path: Path to output JSON file
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = [edge.model_dump() for edge in edges]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

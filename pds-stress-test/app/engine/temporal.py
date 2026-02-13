"""
Temporal Belief Evolution Engine

Simulates how belief states evolve over multiple time periods as
signals accumulate. Demonstrates the key claim: "weak signals
accumulate meaningfully over time."

This module is pure engine logic — NO FastAPI, NO database imports.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import networkx as nx

from app.schemas import BeliefState, Hypothesis, Signal


@dataclass
class TimeStep:
    """A single point in the belief evolution timeline."""

    step: int
    label: str
    beliefs: dict[str, float]
    entropy: float
    signals_applied: int
    explanation: list[str]


@dataclass 
class BeliefEvolution:
    """Complete trajectory of belief states over time."""

    steps: list[TimeStep]
    total_signals: int
    initial_entropy: float
    final_entropy: float
    convergence_delta: float  # How much entropy changed
    most_changed_hid: str
    most_changed_delta: float


def compute_entropy(beliefs: dict[str, float]) -> float:
    """Compute Shannon entropy of a belief distribution."""
    total = 0.0
    for p in beliefs.values():
        if 0 < p < 1:
            total += -p * math.log(p) - (1 - p) * math.log(1 - p)
    return total


def evolve_beliefs(
    *,
    hypotheses: list[Hypothesis],
    graph: nx.DiGraph,
    initial_beliefs: dict[str, float],
    signal_batches: list[list[Signal]],
    batch_labels: list[str] | None = None,
    run_id: Any = None,
    influence_strength: float = 0.3,
) -> BeliefEvolution:
    """
    Run multi-step belief evolution over time.

    Each batch of signals represents a time period (e.g., Month 1, Month 2).
    Beliefs are updated sequentially, demonstrating accumulation.

    Args:
        hypotheses: The hypothesis set.
        graph: The hypothesis graph (nx.DiGraph).
        initial_beliefs: Starting belief state {hid: probability}.
        signal_batches: List of signal lists, one per time step.
        batch_labels: Optional labels for each step (e.g., ["Month 1", ...]).
        run_id: Run UUID for creating BeliefState objects.
        influence_strength: Graph propagation strength.

    Returns:
        BeliefEvolution with the full timeline.
    """
    from app.engine.belief import update_beliefs

    if not batch_labels:
        batch_labels = [f"Period {i+1}" for i in range(len(signal_batches))]

    _run_id = run_id or uuid4()

    steps: list[TimeStep] = []
    current_beliefs = initial_beliefs.copy()
    total_signals = 0

    # Record initial state
    initial_entropy = compute_entropy(current_beliefs)
    steps.append(TimeStep(
        step=0,
        label="Initial (priors)",
        beliefs=current_beliefs.copy(),
        entropy=initial_entropy,
        signals_applied=0,
        explanation=["Prior beliefs initialized"],
    ))

    # Evolve through each batch
    for i, (signals, label) in enumerate(zip(signal_batches, batch_labels)):
        total_signals += len(signals)

        # Build BeliefState for the update function
        state = BeliefState(
            id=uuid4(),
            run_id=_run_id,
            beliefs=current_beliefs.copy(),
            timestamp=datetime.now(timezone.utc),
            explanation_log=[],
        )

        # Apply update
        updated = update_beliefs(
            state=state,
            signals=signals,
            graph=graph,
            params={"influence_strength": influence_strength},
        )

        current_beliefs = updated.beliefs.copy()
        entropy = compute_entropy(current_beliefs)

        steps.append(TimeStep(
            step=i + 1,
            label=label,
            beliefs=current_beliefs.copy(),
            entropy=entropy,
            signals_applied=len(signals),
            explanation=updated.explanation_log,
        ))

    # Compute summary stats
    final_entropy = compute_entropy(current_beliefs)
    convergence_delta = initial_entropy - final_entropy

    # Find most changed hypothesis
    most_changed_hid = ""
    most_changed_delta = 0.0
    for hid in initial_beliefs:
        delta = abs(current_beliefs.get(hid, 0.0) - initial_beliefs[hid])
        if delta > most_changed_delta:
            most_changed_delta = delta
            most_changed_hid = hid

    return BeliefEvolution(
        steps=steps,
        total_signals=total_signals,
        initial_entropy=initial_entropy,
        final_entropy=final_entropy,
        convergence_delta=convergence_delta,
        most_changed_hid=most_changed_hid,
        most_changed_delta=most_changed_delta,
    )

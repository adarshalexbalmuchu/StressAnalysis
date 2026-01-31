"""
Monte Carlo trajectory simulator for policy stress-testing.

V1 implementation: deterministic sampling, simple clustering, sensitivity hotspots.
No ML, no LLMs, no external APIs, no database.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from app.schemas import BeliefState, Hypothesis
from app.schemas import BeliefState, Hypothesis


@dataclass
class Trajectory:
    """A single trajectory outcome from Monte Carlo simulation."""

    name: str
    probability: float
    active_hypotheses: list[str]  # Sorted list of activated HIDs
    frequency: int  # Number of runs that produced this trajectory


@dataclass
class SimulationResult:
    """Result of Monte Carlo trajectory simulation."""

    trajectories: list[Trajectory]  # Top K trajectories by frequency
    total_runs: int
    seed: int
    hotspots: dict[str, str]  # Sensitivity analysis results
    parameters: dict[str, Any]  # Parameters used in simulation


def simulate_trajectories(
    *,
    hypotheses: list[Hypothesis],
    graph: nx.DiGraph,
    belief: BeliefState,
    n_runs: int = 2000,
    seed: int = 42,
    activation_threshold: float = 0.5,
    propagate_steps: int = 2,
    reinforce_delta: float = 0.15,
    contradict_delta: float = 0.15,
    top_k: int = 5,
    params: dict | None = None,
    _skip_hotspots: bool = False,  # Internal flag to prevent infinite recursion
) -> SimulationResult:
    """
    Run Monte Carlo simulation to generate plausible trajectories.

    Args:
        hypotheses: List of hypotheses to simulate
        graph: Hypothesis relationship graph (edges have 'relation' attribute)
        belief: Current belief state (hid -> probability)
        n_runs: Number of Monte Carlo runs
        seed: Random seed for reproducibility
        activation_threshold: Minimum probability for activation consideration
        propagate_steps: Number of graph propagation steps
        reinforce_delta: How much 'reinforces' edge boosts activation probability
        contradict_delta: How much 'contradicts' edge reduces activation probability
        top_k: Number of top trajectories to return
        params: Additional parameters (for future use)

    Returns:
        SimulationResult with top trajectories and sensitivity hotspots
    """
    # Set random seed for determinism
    np.random.seed(seed)
    random.seed(seed)

    # Build HID to hypothesis map
    hid_map = {h.hid: h for h in hypotheses}

    # Store parameters
    sim_params = {
        "n_runs": n_runs,
        "seed": seed,
        "activation_threshold": activation_threshold,
        "propagate_steps": propagate_steps,
        "reinforce_delta": reinforce_delta,
        "contradict_delta": contradict_delta,
        "top_k": top_k,
    }
    if params:
        sim_params.update(params)

    # Run Monte Carlo simulations
    trajectory_signatures: list[tuple[str, ...]] = []

    for run_idx in range(n_runs):
        # Step 1: Sample initial activations based on belief probabilities
        activation_probs = {
            hid: belief.beliefs.get(hid, 0.0) for hid in hid_map.keys()
        }

        # Step 2: Propagate through graph for multiple steps
        for step in range(propagate_steps):
            new_probs = activation_probs.copy()

            for hid in hid_map.keys():
                # Check incoming edges
                if graph.has_node(hid):
                    for source in graph.predecessors(hid):
                        if source not in activation_probs:
                            continue

                        edge_data = graph.get_edge_data(source, hid)
                        relation = edge_data.get("relation", "")

                        source_prob = activation_probs[source]

                        if relation == "reinforces":
                            # If source likely active, boost target
                            boost = source_prob * reinforce_delta
                            new_probs[hid] = min(0.99, new_probs[hid] + boost)

                        elif relation == "contradicts":
                            # If source likely active, suppress target
                            suppress = source_prob * contradict_delta
                            new_probs[hid] = max(0.01, new_probs[hid] - suppress)

                        elif relation == "depends_on":
                            # Hard gate: target can't be higher than dependency
                            # If dependency is low, cap the target
                            if source_prob < 0.3:
                                new_probs[hid] = min(new_probs[hid], source_prob + 0.1)

            activation_probs = new_probs

        # Step 3: Sample final activations (Bernoulli per hypothesis)
        active_hids = []
        for hid, prob in activation_probs.items():
            # Apply activation threshold
            if prob >= activation_threshold:
                # Sample activation
                if np.random.random() < prob:
                    active_hids.append(hid)
            elif prob > 0.1:  # Allow some chance even below threshold
                if np.random.random() < prob * 0.5:
                    active_hids.append(hid)

        # Step 4: Enforce depends_on constraints (hard gate)
        # If a hypothesis depends on another, it cannot activate unless dependency is active
        final_active = set(active_hids)
        for hid in list(final_active):
            if graph.has_node(hid):
                for source in graph.predecessors(hid):
                    edge_data = graph.get_edge_data(source, hid)
                    if edge_data.get("relation") == "depends_on":
                        # Check if dependency is active
                        if source not in final_active:
                            # Dependency not active, cannot activate this hypothesis
                            final_active.discard(hid)
                            break

        # Create signature: sorted tuple of active HIDs
        signature = tuple(sorted(final_active))
        trajectory_signatures.append(signature)

    # Step 5: Cluster trajectories by signature
    signature_counts = Counter(trajectory_signatures)
    top_signatures = signature_counts.most_common(top_k)

    # Step 6: Name trajectories
    trajectories = []
    for idx, (signature, count) in enumerate(top_signatures, start=1):
        name = _name_trajectory(signature, hid_map, idx)
        traj = Trajectory(
            name=name,
            probability=count / n_runs,
            active_hypotheses=list(signature),
            frequency=count,
        )
        trajectories.append(traj)

    # Step 7: Compute sensitivity hotspots
    if _skip_hotspots:
        hotspots = {"status": "Hotspot calculation skipped"}
    else:
        hotspots = _compute_hotspots(
            hypotheses=hypotheses,
            graph=graph,
            belief=belief,
            base_trajectories=trajectories,
            base_params=sim_params,
            n_runs=min(500, n_runs),  # Use fewer runs for sensitivity
        )

    return SimulationResult(
        trajectories=trajectories,
        total_runs=n_runs,
        seed=seed,
        hotspots=hotspots,
        parameters=sim_params,
    )


def _name_trajectory(
    signature: tuple[str, ...], hid_map: dict[str, Hypothesis], idx: int
) -> str:
    """
    Generate a human-readable name for a trajectory.

    V1 uses simple rule-based naming based on active hypothesis patterns.
    """
    active_set = set(signature)

    # Rule-based naming
    # Check for specific patterns in the active hypotheses

    # Pattern 1: Empty trajectory
    if not active_set:
        return "Trajectory: No activation"

    # Pattern 2: Single hypothesis
    if len(active_set) == 1:
        hid = list(active_set)[0]
        if hid in hid_map:
            # Use mechanism from hypothesis
            mechanism = hid_map[hid].mechanism
            # Take first few words
            short_name = " ".join(mechanism.split()[:4]) if mechanism else hid
            return f"Trajectory: {short_name}"

    # Pattern 3: Check for common combinations
    # Look for keywords in hypothesis mechanisms and primary effects
    keywords_lower = []
    for hid in active_set:
        if hid in hid_map:
            h = hid_map[hid]
            keywords_lower.extend([w.lower() for w in h.mechanism.split()])
            keywords_lower.extend([e.lower() for e in h.primary_effects])

    # Check for exclusion-related patterns
    if any("exclusion" in word for word in keywords_lower):
        if any(word in keywords_lower for word in ["cascade", "multiple", "repeated"]):
            return "Trajectory: Exclusion cascade"
        if any(word in keywords_lower for word in ["dealer", "workaround", "alternative"]):
            return "Trajectory: Exclusion with workaround"
        return "Trajectory: Exclusion scenario"

    # Check for technical failure patterns
    if any(word in keywords_lower for word in ["network", "technical", "infrastructure"]):
        if any(word in keywords_lower for word in ["auth", "biometric", "authentication"]):
            return "Trajectory: Technical auth failure"
        return "Trajectory: Technical issues"

    # Check for policy/system patterns
    if any(word in keywords_lower for word in ["policy", "dealer", "system"]):
        return "Trajectory: Policy response"

    # Default: numbered trajectory
    return f"Trajectory {idx}"


def _compute_hotspots(
    *,
    hypotheses: list[Hypothesis],
    graph: nx.DiGraph,
    belief: BeliefState,
    base_trajectories: list[Trajectory],
    base_params: dict[str, Any],
    n_runs: int,
) -> dict[str, str]:
    """
    Compute sensitivity hotspots by testing parameter variations.

    Returns dict of parameter -> explanation of sensitivity.
    """
    hotspots: dict[str, str] = {}

    if not base_trajectories:
        return {"status": "No trajectories to analyze"}

    base_top_prob = base_trajectories[0].probability

    # Test 1: Activation threshold sensitivity
    threshold_variants = [0.3, 0.5, 0.7]
    threshold_impacts = []

    for threshold in threshold_variants:
        if threshold == base_params.get("activation_threshold"):
            continue

        result = simulate_trajectories(
            hypotheses=hypotheses,
            graph=graph,
            belief=belief,
            n_runs=n_runs,
            seed=base_params["seed"] + 1,
            activation_threshold=threshold,
            propagate_steps=base_params["propagate_steps"],
            reinforce_delta=base_params["reinforce_delta"],
            contradict_delta=base_params["contradict_delta"],
            top_k=1,
            _skip_hotspots=True,  # Prevent infinite recursion
        )

        if result.trajectories:
            new_top_prob = result.trajectories[0].probability
            impact = abs(new_top_prob - base_top_prob)
            threshold_impacts.append((threshold, impact))

    if threshold_impacts:
        max_threshold, max_impact = max(threshold_impacts, key=lambda x: x[1])
        if max_impact > 0.05:  # 5% change is significant
            hotspots["activation_threshold"] = (
                f"High sensitivity: changing threshold to {max_threshold:.2f} "
                f"shifts top trajectory probability by {max_impact:.1%}"
            )

    # Test 2: Reinforcement delta sensitivity
    reinforce_variants = [0.05, 0.15, 0.25]
    reinforce_impacts = []

    for delta in reinforce_variants:
        if delta == base_params.get("reinforce_delta"):
            continue

        result = simulate_trajectories(
            hypotheses=hypotheses,
            graph=graph,
            belief=belief,
            n_runs=n_runs,
            seed=base_params["seed"] + 2,
            activation_threshold=base_params["activation_threshold"],
            propagate_steps=base_params["propagate_steps"],
            reinforce_delta=delta,
            contradict_delta=base_params["contradict_delta"],
            top_k=1,
            _skip_hotspots=True,  # Prevent infinite recursion
        )

        if result.trajectories:
            new_top_prob = result.trajectories[0].probability
            impact = abs(new_top_prob - base_top_prob)
            reinforce_impacts.append((delta, impact))

    if reinforce_impacts:
        max_delta, max_impact = max(reinforce_impacts, key=lambda x: x[1])
        if max_impact > 0.03:  # 3% change is notable
            hotspots["reinforce_delta"] = (
                f"Moderate sensitivity: reinforcement strength {max_delta:.2f} "
                f"changes top trajectory by {max_impact:.1%}"
            )

    # Test 3: Contradiction delta sensitivity
    contradict_variants = [0.05, 0.15, 0.25]
    contradict_impacts = []

    for delta in contradict_variants:
        if delta == base_params.get("contradict_delta"):
            continue

        result = simulate_trajectories(
            hypotheses=hypotheses,
            graph=graph,
            belief=belief,
            n_runs=n_runs,
            seed=base_params["seed"] + 3,
            activation_threshold=base_params["activation_threshold"],
            propagate_steps=base_params["propagate_steps"],
            reinforce_delta=base_params["reinforce_delta"],
            contradict_delta=delta,
            top_k=1,
            _skip_hotspots=True,  # Prevent infinite recursion
        )

        if result.trajectories:
            new_top_prob = result.trajectories[0].probability
            impact = abs(new_top_prob - base_top_prob)
            contradict_impacts.append((delta, impact))

    if contradict_impacts:
        max_delta, max_impact = max(contradict_impacts, key=lambda x: x[1])
        if max_impact > 0.03:
            hotspots["contradict_delta"] = (
                f"Moderate sensitivity: contradiction strength {max_delta:.2f} "
                f"changes top trajectory by {max_impact:.1%}"
            )

    # Test 4: Propagation steps sensitivity
    steps_variants = [1, 2, 3]
    steps_impacts = []

    for steps in steps_variants:
        if steps == base_params.get("propagate_steps"):
            continue

        result = simulate_trajectories(
            hypotheses=hypotheses,
            graph=graph,
            belief=belief,
            n_runs=n_runs,
            seed=base_params["seed"] + 4,
            activation_threshold=base_params["activation_threshold"],
            propagate_steps=steps,
            reinforce_delta=base_params["reinforce_delta"],
            contradict_delta=base_params["contradict_delta"],
            top_k=1,
            _skip_hotspots=True,  # Prevent infinite recursion
        )

        if result.trajectories:
            new_top_prob = result.trajectories[0].probability
            impact = abs(new_top_prob - base_top_prob)
            steps_impacts.append((steps, impact))

    if steps_impacts:
        max_steps, max_impact = max(steps_impacts, key=lambda x: x[1])
        if max_impact > 0.04:
            hotspots["propagate_steps"] = (
                f"Propagation depth matters: {max_steps} steps changes "
                f"top trajectory by {max_impact:.1%}"
            )

    # Summary
    if not hotspots:
        hotspots["status"] = "Low sensitivity: results stable across parameter variations"

    return hotspots


def get_trajectory_statistics(result: SimulationResult) -> dict[str, Any]:
    """
    Extract summary statistics from simulation result.

    Useful for reporting and analysis.
    """
    if not result.trajectories:
        return {
            "total_runs": result.total_runs,
            "unique_trajectories": 0,
            "top_probability": 0.0,
            "coverage": 0.0,
        }

    top_k_prob = sum(t.probability for t in result.trajectories)

    return {
        "total_runs": result.total_runs,
        "unique_trajectories": len(result.trajectories),
        "top_probability": result.trajectories[0].probability,
        "top_k_coverage": top_k_prob,
        "other_probability": max(0.0, 1.0 - top_k_prob),
        "seed": result.seed,
        "hotspot_count": len(result.hotspots),
    }


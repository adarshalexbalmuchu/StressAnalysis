"""
Monte Carlo Trajectory Simulator

Simulates multiple plausible future trajectories based on hypothesis activations.
Produces a distribution over possible futures, not a single prediction.

NO LLMs - pure probabilistic simulation.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np

from app.schemas import Hypothesis, TimeHorizon, TrajectoryStatus


@dataclass
class SimulationParameters:
    """Configuration for Monte Carlo simulation."""

    n_trajectories: int = 1000  # Number of trajectories to simulate
    max_steps: int = 10  # Maximum steps per trajectory
    activation_threshold: float = 0.5  # Belief threshold for hypothesis activation
    propagation_strength: float = 0.3  # Strength of cascade effects
    random_seed: Optional[int] = None  # For reproducibility
    time_discount: Dict[TimeHorizon, float] = None  # Discount by time horizon

    def __post_init__(self) -> None:
        if self.time_discount is None:
            # Default: immediate effects happen faster
            self.time_discount = {
                TimeHorizon.IMMEDIATE: 1.0,
                TimeHorizon.SHORT_TERM: 0.7,
                TimeHorizon.MEDIUM_TERM: 0.4,
                TimeHorizon.LONG_TERM: 0.2,
            }


class MonteCarloSimulator:
    """
    Simulates multiple future trajectories based on hypothesis activations.
    
    Each trajectory represents a coherent sequence of hypothesis activations
    and their cascading effects.
    """

    def __init__(
        self,
        hypotheses: List[Hypothesis],
        beliefs: Dict[str, float],
        graph_edges: List[Dict],
        parameters: Optional[SimulationParameters] = None,
    ):
        """
        Initialize simulator.
        
        Args:
            hypotheses: List of all hypotheses
            beliefs: Current belief state (HID -> probability)
            graph_edges: List of graph edge dictionaries
            parameters: Simulation configuration
        """
        self.hypotheses = {h.hid: h for h in hypotheses}
        self.beliefs = beliefs.copy()
        self.graph_edges = graph_edges
        self.params = parameters or SimulationParameters()
        
        # Set random seed for reproducibility
        if self.params.random_seed is not None:
            random.seed(self.params.random_seed)
            np.random.seed(self.params.random_seed)
        
        # Build adjacency structures for efficient traversal
        self._build_graph_structures()

    def _build_graph_structures(self) -> None:
        """Build efficient graph lookup structures."""
        self.reinforces: Dict[str, List[str]] = defaultdict(list)
        self.contradicts: Dict[str, List[str]] = defaultdict(list)
        self.depends_on: Dict[str, List[str]] = defaultdict(list)
        
        for edge in self.graph_edges:
            source = edge["source_hid"]
            target = edge["target_hid"]
            edge_type = edge["edge_type"]
            
            if edge_type == "reinforces":
                self.reinforces[source].append(target)
            elif edge_type == "contradicts":
                self.contradicts[source].append(target)
            elif edge_type == "depends_on":
                self.depends_on[source].append(target)

    def _sample_initial_activations(self) -> Set[str]:
        """
        Sample initial hypothesis activations based on beliefs.
        
        Returns:
            Set of initially activated HIDs
        """
        activated: Set[str] = set()
        
        for hid, belief in self.beliefs.items():
            # Sample from belief probability, modulated by time horizon
            hypothesis = self.hypotheses[hid]
            time_discount = self.params.time_discount[hypothesis.time_horizon]
            adjusted_prob = belief * time_discount
            
            if random.random() < adjusted_prob:
                activated.add(hid)
        
        return activated

    def _propagate_activations(
        self,
        active: Set[str],
        step: int,
    ) -> Tuple[Set[str], Set[str]]:
        """
        Propagate activations through the graph for one step.
        
        Args:
            active: Currently active hypotheses
            step: Current step number (for time-based effects)
            
        Returns:
            Tuple of (newly_activated, newly_suppressed)
        """
        newly_activated: Set[str] = set()
        newly_suppressed: Set[str] = set()
        activation_scores: Dict[str, float] = defaultdict(float)
        
        # Compute activation scores based on graph relationships
        for source_hid in active:
            # Reinforcement: active hypotheses push successors toward activation
            for target_hid in self.reinforces.get(source_hid, []):
                if target_hid not in active:
                    activation_scores[target_hid] += self.params.propagation_strength
            
            # Contradiction: active hypotheses push contradicted ones toward suppression
            for target_hid in self.contradicts.get(source_hid, []):
                if target_hid in active:
                    # Already active but contradicted - chance of suppression
                    activation_scores[target_hid] -= self.params.propagation_strength * 0.5
                else:
                    activation_scores[target_hid] -= self.params.propagation_strength
            
            # Dependency: check if dependencies are satisfied
            for dependent_hid in [h for h, deps in self._get_reverse_deps().items() if source_hid in deps]:
                if dependent_hid not in active:
                    # All dependencies satisfied?
                    deps = self.depends_on.get(dependent_hid, [])
                    if all(d in active for d in deps):
                        activation_scores[dependent_hid] += self.params.propagation_strength * 0.8
        
        # Sample new activations/suppressions
        for hid, score in activation_scores.items():
            base_belief = self.beliefs.get(hid, 0.5)
            hypothesis = self.hypotheses[hid]
            
            # Time-based activation probability
            time_discount = self.params.time_discount[hypothesis.time_horizon]
            activation_prob = base_belief * time_discount + score
            activation_prob = max(0.0, min(1.0, activation_prob))
            
            if hid not in active and activation_prob > self.params.activation_threshold:
                if random.random() < activation_prob:
                    newly_activated.add(hid)
            elif hid in active and activation_prob < (1.0 - self.params.activation_threshold):
                if random.random() < (1.0 - activation_prob):
                    newly_suppressed.add(hid)
        
        return newly_activated, newly_suppressed

    def _get_reverse_deps(self) -> Dict[str, List[str]]:
        """Get reverse dependency mapping (hypothesis -> what it depends on)."""
        reverse: Dict[str, List[str]] = defaultdict(list)
        for source, targets in self.depends_on.items():
            for target in targets:
                reverse[source].append(target)
        return reverse

    def _collect_effects(self, active_hids: Set[str]) -> List[str]:
        """
        Collect all effects from active hypotheses.
        
        Args:
            active_hids: Set of active hypothesis IDs
            
        Returns:
            List of effect descriptions
        """
        effects: List[str] = []
        for hid in active_hids:
            hypothesis = self.hypotheses[hid]
            effects.extend(hypothesis.primary_effects)
            effects.extend(hypothesis.secondary_effects)
        return effects

    def simulate_single_trajectory(self, trajectory_id: str) -> Dict[str, Any]:
        """
        Simulate a single trajectory.
        
        Args:
            trajectory_id: Unique identifier for this trajectory
            
        Returns:
            Dictionary representing the trajectory
        """
        # Sample initial activations
        active = self._sample_initial_activations()
        
        if not active:
            # No hypotheses activated - null trajectory
            return {
                "trajectory_id": trajectory_id,
                "steps": [],
                "final_state": {
                    "active_count": 0,
                    "effect_count": 0,
                    "status": "failed",
                },
                "status": TrajectoryStatus.FAILED.value,
                "metadata": {"reason": "No initial activations"},
            }
        
        steps = []
        all_activated = active.copy()
        
        for step_num in range(self.params.max_steps):
            # Collect effects at this step
            effects = self._collect_effects(active)
            
            steps.append({
                "step": step_num,
                "activated_hids": list(active),
                "cumulative_effects": effects,
                "notes": f"Step {step_num}: {len(active)} active hypotheses",
            })
            
            # Propagate activations
            newly_activated, newly_suppressed = self._propagate_activations(active, step_num)
            
            # Update active set
            active.update(newly_activated)
            active -= newly_suppressed
            all_activated.update(newly_activated)
            
            # Check for steady state
            if not newly_activated and not newly_suppressed:
                break
        
        # Compute final state
        all_effects = self._collect_effects(all_activated)
        effect_counts = Counter(all_effects)
        
        return {
            "trajectory_id": trajectory_id,
            "steps": steps,
            "final_state": {
                "active_count": len(active),
                "total_activated_count": len(all_activated),
                "effect_count": len(all_effects),
                "unique_effects": len(effect_counts),
                "top_effects": [e for e, _ in effect_counts.most_common(5)],
                "status": "completed",
            },
            "status": TrajectoryStatus.COMPLETED.value,
            "metadata": {
                "steps_taken": len(steps),
                "activation_rate": len(all_activated) / len(self.hypotheses),
            },
        }

    def run_simulation(self, run_id: UUID) -> Dict[str, Any]:
        """
        Run full Monte Carlo simulation.
        
        Args:
            run_id: ID of the run this simulation belongs to
            
        Returns:
            Complete simulation results
        """
        trajectories = []
        
        # Generate trajectories
        for i in range(self.params.n_trajectories):
            trajectory_id = f"T{i:04d}"
            traj = self.simulate_single_trajectory(trajectory_id)
            traj["run_id"] = str(run_id)
            trajectories.append(traj)
        
        # Compute summary statistics
        summary = self._compute_summary_statistics(trajectories)
        
        # Identify sensitivity hotspots
        hotspots = self._identify_hotspots(trajectories)
        
        return {
            "run_id": str(run_id),
            "simulation_id": str(uuid4()),
            "trajectories": trajectories,
            "summary_statistics": summary,
            "sensitivity_hotspots": hotspots,
            "parameters": {
                "n_trajectories": self.params.n_trajectories,
                "max_steps": self.params.max_steps,
                "activation_threshold": self.params.activation_threshold,
                "propagation_strength": self.params.propagation_strength,
                "random_seed": self.params.random_seed,
            },
        }

    def _compute_summary_statistics(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """
        Compute aggregate statistics across trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Summary statistics
        """
        completed = [t for t in trajectories if t["status"] == TrajectoryStatus.COMPLETED.value]
        failed = [t for t in trajectories if t["status"] == TrajectoryStatus.FAILED.value]
        
        if not completed:
            return {
                "success_rate": 0.0,
                "failure_rate": 1.0,
                "message": "All trajectories failed",
            }
        
        # Collect activation frequencies
        activation_freq: Dict[str, int] = Counter()
        for traj in completed:
            for step in traj["steps"]:
                activation_freq.update(step["activated_hids"])
        
        # Effect frequencies
        effect_freq: Dict[str, int] = Counter()
        for traj in completed:
            effect_freq.update(traj["final_state"]["top_effects"])
        
        # Trajectory lengths
        lengths = [len(t["steps"]) for t in completed]
        
        return {
            "success_rate": len(completed) / len(trajectories),
            "failure_rate": len(failed) / len(trajectories),
            "total_trajectories": len(trajectories),
            "completed_trajectories": len(completed),
            "avg_trajectory_length": np.mean(lengths) if lengths else 0,
            "std_trajectory_length": np.std(lengths) if lengths else 0,
            "most_frequent_activations": [
                {"hid": hid, "frequency": freq / len(completed)}
                for hid, freq in activation_freq.most_common(10)
            ],
            "most_frequent_effects": [
                {"effect": effect, "frequency": freq / len(completed)}
                for effect, freq in effect_freq.most_common(10)
            ],
        }

    def _identify_hotspots(self, trajectories: List[Dict]) -> List[str]:
        """
        Identify sensitivity hotspots - hypotheses with high variance or influence.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            List of HIDs that are sensitivity hotspots
        """
        completed = [t for t in trajectories if t["status"] == TrajectoryStatus.COMPLETED.value]
        
        if not completed:
            return []
        
        # Track activation variance across trajectories
        activation_matrix = []
        for traj in completed:
            activated_in_traj = set()
            for step in traj["steps"]:
                activated_in_traj.update(step["activated_hids"])
            
            # Binary vector of activations
            row = [1 if hid in activated_in_traj else 0 for hid in self.hypotheses.keys()]
            activation_matrix.append(row)
        
        activation_matrix = np.array(activation_matrix)
        
        # Compute variance for each hypothesis
        variances = np.var(activation_matrix, axis=0)
        
        # Compute mean activation rate
        mean_activation = np.mean(activation_matrix, axis=0)
        
        # Hotspots: high variance and moderate to high activation rate
        hotspot_scores = variances * mean_activation
        
        # Get top hotspots
        hid_list = list(self.hypotheses.keys())
        top_indices = np.argsort(hotspot_scores)[-10:][::-1]
        
        hotspots = [hid_list[i] for i in top_indices if hotspot_scores[i] > 0.01]
        
        return hotspots

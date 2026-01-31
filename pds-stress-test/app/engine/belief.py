"""
Belief Update Engine

Manages probabilistic belief states over hypotheses.
Updates beliefs based on evidence signals using rule-based Bayesian logic.

NO LLMs - this is pure mechanistic reasoning.

Provides both:
1. Functional API: init_priors(), update_beliefs(), validate_belief_state()
2. Class-based API: BeliefUpdateEngine (for backward compatibility)
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import networkx as nx

from app.schemas import BeliefState, Hypothesis, Signal, SignalType, TimeHorizon


# ============================================================================
# FUNCTIONAL API (V1 Primary Interface)
# ============================================================================


def init_priors(
    hypotheses: List[Hypothesis],
    strategy: str = "uniform",
) -> BeliefState:
    """
    Initialize prior belief state over hypotheses.
    
    Args:
        hypotheses: List of hypotheses to track
        strategy: Initialization strategy:
            - "uniform": Equal probability for all (default)
            - "time_horizon": Immediate hypotheses get slightly higher priors
    
    Returns:
        BeliefState with initialized probabilities
    
    Raises:
        ValueError: If strategy is unknown
    """
    if not hypotheses:
        raise ValueError("Cannot initialize priors with empty hypothesis list")
    
    beliefs: Dict[str, float] = {}
    explanation_log: List[str] = []
    
    if strategy == "uniform":
        # Equal probability for all
        prior = 1.0 / len(hypotheses)
        for h in hypotheses:
            beliefs[h.hid] = prior
        explanation_log.append(
            f"Initialized {len(beliefs)} beliefs with uniform priors P={prior:.4f}"
        )
    
    elif strategy == "time_horizon":
        # Weight by time horizon (immediate > short > medium > long)
        weights = {
            TimeHorizon.IMMEDIATE: 1.5,
            TimeHorizon.SHORT_TERM: 1.2,
            TimeHorizon.MEDIUM_TERM: 1.0,
            TimeHorizon.LONG_TERM: 0.8,
        }
        
        # Calculate raw scores
        raw_beliefs = {}
        total_weight = 0.0
        for h in hypotheses:
            weight = weights.get(h.time_horizon, 1.0)
            raw_beliefs[h.hid] = weight
            total_weight += weight
        
        # Normalize to sum to 1
        for hid in raw_beliefs:
            beliefs[hid] = raw_beliefs[hid] / total_weight
        
        explanation_log.append(
            f"Initialized {len(beliefs)} beliefs with time_horizon weighting "
            f"(immediate={weights[TimeHorizon.IMMEDIATE]}, long_term={weights[TimeHorizon.LONG_TERM]})"
        )
    
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy}")
    
    # Validate probabilities sum to approximately 1
    total_prob = sum(beliefs.values())
    if abs(total_prob - 1.0) > 0.001:
        # Renormalize if needed
        for hid in beliefs:
            beliefs[hid] /= total_prob
        explanation_log.append(f"Renormalized beliefs (sum was {total_prob:.6f})")
    
    return BeliefState(
        id=uuid4(),
        run_id=hypotheses[0].run_id if hypotheses else uuid4(),
        beliefs=beliefs,
        timestamp=datetime.now(timezone.utc),
        explanation_log=explanation_log,
    )


def update_beliefs(
    state: BeliefState,
    signals: List[Signal],
    graph: nx.DiGraph,
    *,
    params: Optional[Dict] = None,
) -> BeliefState:
    """
    Update belief state based on evidence signals and graph structure.
    
    Args:
        state: Current belief state
        signals: Evidence signals to process
        graph: Hypothesis relationship graph (nx.DiGraph with 'relation' edge attribute)
        params: Optional parameters:
            - influence_strength: float (default 0.3) - graph propagation strength
            - depends_on_cap: float (default 0.9) - max probability for dependent hypothesis
    
    Returns:
        New BeliefState with updated beliefs and explanation log
    
    Process:
        1. Map signals to affected hypotheses
        2. Apply likelihood multipliers based on signal strength
        3. Propagate effects through graph edges
        4. Renormalize probabilities
        5. Log all actions for auditability
    """
    params = params or {}
    influence_strength = params.get("influence_strength", 0.3)
    depends_on_cap = params.get("depends_on_cap", 0.9)
    
    # Copy current beliefs
    new_beliefs = state.beliefs.copy()
    new_log = state.explanation_log.copy()
    
    if not signals:
        new_log.append("No signals provided - beliefs unchanged")
        return BeliefState(
            id=uuid4(),
            run_id=state.run_id,
            beliefs=new_beliefs,
            timestamp=datetime.now(timezone.utc),
            explanation_log=new_log,
        )
    
    # Step 1: Map signals to hypotheses and apply direct updates
    signal_mapping = _map_signals_to_hypotheses(signals)
    
    for signal in signals:
        affected_hids = signal.affected_hids
        if not affected_hids:
            # Try to infer from signal mapping
            affected_hids = signal_mapping.get(signal.signal_type, [])
        
        if not affected_hids:
            new_log.append(
                f"Signal {signal.signal_type.value}: No affected hypotheses (skipped)"
            )
            continue
        
        # Calculate likelihood multiplier from signal strength
        likelihood = _calculate_likelihood(signal)
        
        for hid in affected_hids:
            if hid not in new_beliefs:
                new_log.append(f"Warning: Signal references unknown HID '{hid}' (skipped)")
                continue
            
            old_belief = new_beliefs[hid]
            
            # Bayesian update: P(H|E) ∝ L(E|H) * P(H)
            # likelihood > 1.0 increases belief, likelihood < 1.0 decreases it
            raw_update = old_belief * likelihood
            new_belief = max(0.01, min(0.99, raw_update))  # Clamp to [0.01, 0.99]
            
            new_beliefs[hid] = new_belief
            direction = "↑" if new_belief > old_belief else "↓" if new_belief < old_belief else "="
            new_log.append(
                f"{hid}: {old_belief:.3f} {direction} {new_belief:.3f} | "
                f"Signal {signal.signal_type.value} (L={likelihood:.2f})"
            )
    
    # Step 2: Propagate through graph edges
    if graph and graph.number_of_edges() > 0:
        propagation_updates = _propagate_through_graph(
            new_beliefs, graph, influence_strength, depends_on_cap
        )
        
        for hid, (old, new, reason) in propagation_updates.items():
            new_beliefs[hid] = new
            direction = "↑" if new > old else "↓"
            new_log.append(f"{hid}: {old:.3f} {direction} {new:.3f} | {reason}")
    
    # Step 3: Renormalize probabilities
    total_prob = sum(new_beliefs.values())
    if abs(total_prob - 1.0) > 0.01:  # More than 1% deviation
        normalization_factor = 1.0 / total_prob
        for hid in new_beliefs:
            new_beliefs[hid] *= normalization_factor
        new_log.append(
            f"Renormalized beliefs (sum was {total_prob:.4f}, factor={normalization_factor:.4f})"
        )
    
    return BeliefState(
        id=uuid4(),
        run_id=state.run_id,
        beliefs=new_beliefs,
        timestamp=datetime.now(timezone.utc),
        explanation_log=new_log,
    )


def validate_belief_state(state: BeliefState) -> List[str]:
    """
    Validate belief state consistency and correctness.
    
    Args:
        state: BeliefState to validate
    
    Returns:
        List of validation error messages (empty if valid)
    
    Checks:
        - All probabilities in [0, 1]
        - Probabilities sum to approximately 1.0
        - No missing or malformed hypothesis IDs
    """
    errors = []
    
    if not state.beliefs:
        errors.append("BeliefState has no beliefs (empty dict)")
        return errors
    
    # Check probability ranges
    for hid, prob in state.beliefs.items():
        if not isinstance(prob, (int, float)):
            errors.append(f"{hid}: probability is not numeric (got {type(prob).__name__})")
            continue
        
        if prob < 0.0 or prob > 1.0:
            errors.append(f"{hid}: probability {prob:.4f} out of range [0, 1]")
        
        if not isinstance(hid, str) or not hid:
            errors.append(f"Invalid hypothesis ID: '{hid}' (must be non-empty string)")
    
    # Check sum
    total_prob = sum(state.beliefs.values())
    if abs(total_prob - 1.0) > 0.05:  # Allow 5% tolerance
        errors.append(
            f"Probabilities sum to {total_prob:.4f} (should be ~1.0, tolerance ±0.05)"
        )
    
    return errors


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _map_signals_to_hypotheses(signals: List[Signal]) -> Dict[SignalType, List[str]]:
    """
    Rule-based mapping of signal categories to hypothesis patterns.
    
    This is a simple lookup table for V1. In production, this would be
    configurable or learned from data.
    """
    # For V1, we rely on signals having explicit affected_hids
    # This function provides defaults if none are specified
    return {
        SignalType.GRIEVANCE: [],  # Must be explicitly mapped
        SignalType.ADMIN_CIRCULAR: [],
        SignalType.MEDIA_REPORT: [],
        SignalType.AUDIT: [],
        SignalType.COURT_OBSERVATION: [],
        SignalType.FIELD_REPORT: [],
    }


def _calculate_likelihood(signal: Signal) -> float:
    """
    Calculate likelihood multiplier from signal properties.
    
    For supporting evidence (positive signals), likelihood should be > 1.0
    to increase belief when multiplied.
    
    Factors:
        - Signal strength (0.0-1.0 maps to 0.5-2.0 multiplier)
        - Signal type credibility weight
        - Recency (could be added in V2)
    
    Returns:
        Multiplier in range [0.5, 2.0] where:
        - > 1.0 increases belief (supporting evidence)
        - < 1.0 decreases belief (contradicting evidence)
        - = 1.0 neutral (no effect)
    """
    # Map signal strength [0, 1] to base multiplier [0.5, 1.5]
    # strength=0.0 -> 0.5x (strong contradiction)
    # strength=0.5 -> 1.0x (neutral)
    # strength=1.0 -> 1.5x (strong support)
    base_multiplier = 0.5 + signal.strength
    
    # Adjust by signal type credibility (adds up to +0.5x boost)
    type_weights = {
        SignalType.AUDIT: 0.5,  # +50% boost to multiplier
        SignalType.COURT_OBSERVATION: 0.4,
        SignalType.ADMIN_CIRCULAR: 0.3,
        SignalType.FIELD_REPORT: 0.2,
        SignalType.MEDIA_REPORT: 0.1,
        SignalType.GRIEVANCE: 0.0,  # No extra boost
    }
    
    type_weight = type_weights.get(signal.signal_type, 0.0)
    multiplier = base_multiplier + type_weight
    
    # Clamp to reasonable range
    return max(0.5, min(2.0, multiplier))


def _propagate_through_graph(
    beliefs: Dict[str, float],
    graph: nx.DiGraph,
    influence_strength: float,
    depends_on_cap: float,
) -> Dict[str, Tuple[float, float, str]]:
    """
    Propagate belief updates through graph structure.
    
    Rules:
        - reinforces: source belief boosts target belief
        - contradicts: source belief suppresses target belief  
        - depends_on: if source belief is low, cap target belief
    
    Returns:
        Dict of {hid: (old_belief, new_belief, reason)}
    """
    updates = {}
    
    for source_hid, target_hid, edge_data in graph.edges(data=True):
        if source_hid not in beliefs or target_hid not in beliefs:
            continue
        
        source_belief = beliefs[source_hid]
        target_belief = beliefs[target_hid]
        relation = edge_data.get("relation", edge_data.get("edge_type"))
        
        if relation == "reinforces":
            # High source belief boosts target
            if source_belief > 0.6:
                adjustment = influence_strength * (source_belief - 0.5)
                new_belief = min(0.99, target_belief + adjustment)
                
                if abs(new_belief - target_belief) > 0.001:
                    updates[target_hid] = (
                        target_belief,
                        new_belief,
                        f"Reinforced by {source_hid} (P={source_belief:.3f})",
                    )
                    beliefs[target_hid] = new_belief  # Update for next iteration
        
        elif relation == "contradicts":
            # High source belief suppresses target
            if source_belief > 0.6:
                adjustment = influence_strength * (source_belief - 0.5)
                new_belief = max(0.01, target_belief - adjustment)
                
                if abs(new_belief - target_belief) > 0.001:
                    updates[target_hid] = (
                        target_belief,
                        new_belief,
                        f"Contradicted by {source_hid} (P={source_belief:.3f})",
                    )
                    beliefs[target_hid] = new_belief
        
        elif relation == "depends_on":
            # Low source belief caps target
            if source_belief < 0.4:
                capped_belief = min(target_belief, source_belief * depends_on_cap)
                
                if capped_belief < target_belief - 0.001:
                    updates[target_hid] = (
                        target_belief,
                        capped_belief,
                        f"Capped by dependency {source_hid} (P={source_belief:.3f})",
                    )
                    beliefs[target_hid] = capped_belief
    
    return updates


# ============================================================================
# CLASS-BASED API (Backward Compatibility)
# ============================================================================


class BeliefUpdateEngine:
    """
    Maintains and updates probabilistic beliefs over hypotheses.
    
    Uses simple Bayesian update rules:
    - P(H|E) ∝ P(E|H) * P(H)
    - Evidence signals adjust beliefs based on relevance and strength
    """

    def __init__(
        self,
        hypotheses: List[Hypothesis],
        prior_belief: float = 0.5,
    ):
        """
        Initialize belief state.
        
        Args:
            hypotheses: List of hypotheses to track
            prior_belief: Default prior probability for all hypotheses
        """
        self.hypotheses = {h.hid: h for h in hypotheses}
        self.beliefs: Dict[str, float] = {
            hid: prior_belief for hid in self.hypotheses.keys()
        }
        self.explanation_log: List[str] = [
            f"Initialized {len(self.beliefs)} beliefs with uniform prior P={prior_belief:.3f}"
        ]

    def get_belief(self, hid: str) -> float:
        """
        Get current belief for a hypothesis.
        
        Args:
            hid: Hypothesis ID
            
        Returns:
            Current probability P(H_i)
            
        Raises:
            KeyError: If HID not found
        """
        return self.beliefs[hid]

    def get_all_beliefs(self) -> Dict[str, float]:
        """Get all current beliefs as a dictionary."""
        return self.beliefs.copy()

    def set_belief(self, hid: str, probability: float, explanation: str) -> None:
        """
        Manually set belief for a hypothesis.
        
        Args:
            hid: Hypothesis ID
            probability: New probability value [0, 1]
            explanation: Reason for this update
            
        Raises:
            KeyError: If HID not found
            ValueError: If probability out of range
        """
        if hid not in self.beliefs:
            raise KeyError(f"Hypothesis {hid} not found")
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"Probability must be in [0, 1], got {probability}")
        
        old_belief = self.beliefs[hid]
        self.beliefs[hid] = probability
        self.explanation_log.append(
            f"{hid}: {old_belief:.3f} → {probability:.3f} | {explanation}"
        )

    def update_from_signal(
        self,
        signal: Signal,
        likelihood_fn: Optional[callable] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Update beliefs based on an evidence signal.
        
        Uses Bayesian update:
        P(H|E) = P(E|H) * P(H) / P(E)
        
        Args:
            signal: Evidence signal
            likelihood_fn: Optional function to compute P(E|H)
                         If None, uses default based on signal strength
            
        Returns:
            Dict of {hid: (old_belief, new_belief)} for affected hypotheses
        """
        updates: Dict[str, Tuple[float, float]] = {}
        
        if not signal.affected_hids:
            self.explanation_log.append(
                f"Signal {signal.id} ({signal.signal_type}) affected no hypotheses"
            )
            return updates
        
        # Default likelihood function based on signal strength
        if likelihood_fn is None:
            def default_likelihood(h: Hypothesis, s: Signal) -> float:
                """
                Compute P(E|H) - probability of observing signal if H is true.
                
                Uses signal strength as base, modulated by time horizon match.
                """
                base_likelihood = 0.5 + (s.strength - 0.5) * 0.8  # Scale to [0.1, 0.9]
                
                # Adjust based on time horizon - immediate effects are more observable
                if h.time_horizon.value == "immediate":
                    return min(0.95, base_likelihood * 1.2)
                elif h.time_horizon.value == "short_term":
                    return base_likelihood
                elif h.time_horizon.value == "medium_term":
                    return base_likelihood * 0.8
                else:  # long_term
                    return base_likelihood * 0.6
            
            likelihood_fn = default_likelihood
        
        # Apply Bayesian update to each affected hypothesis
        for hid in signal.affected_hids:
            if hid not in self.hypotheses:
                self.explanation_log.append(f"Warning: Signal references unknown HID {hid}")
                continue
            
            h = self.hypotheses[hid]
            prior = self.beliefs[hid]
            
            # Compute likelihood P(E|H)
            likelihood_true = likelihood_fn(h, signal)
            likelihood_false = 1.0 - likelihood_true
            
            # Bayesian update
            # P(H|E) = P(E|H) * P(H) / P(E)
            # P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
            evidence_prob = (
                likelihood_true * prior + 
                likelihood_false * (1.0 - prior)
            )
            
            if evidence_prob > 0:
                posterior = (likelihood_true * prior) / evidence_prob
                posterior = max(0.01, min(0.99, posterior))  # Clamp to avoid extremes
            else:
                posterior = prior  # No update if evidence probability is zero
            
            old_belief = self.beliefs[hid]
            self.beliefs[hid] = posterior
            updates[hid] = (old_belief, posterior)
            
            direction = "↑" if posterior > old_belief else ("↓" if posterior < old_belief else "→")
            self.explanation_log.append(
                f"{hid}: {old_belief:.3f} {direction} {posterior:.3f} | "
                f"Signal: {signal.signal_type.value} (strength={signal.strength:.2f}, "
                f"likelihood={likelihood_true:.2f})"
            )
        
        return updates

    def update_from_signals(
        self,
        signals: List[Signal],
        likelihood_fn: Optional[callable] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Update beliefs from multiple signals sequentially.
        
        Args:
            signals: List of evidence signals
            likelihood_fn: Optional custom likelihood function
            
        Returns:
            Dict of {hid: [(old, new), ...]} tracking all updates per hypothesis
        """
        all_updates: Dict[str, List[Tuple[float, float]]] = {}
        
        for signal in signals:
            updates = self.update_from_signal(signal, likelihood_fn)
            for hid, (old, new) in updates.items():
                if hid not in all_updates:
                    all_updates[hid] = []
                all_updates[hid].append((old, new))
        
        return all_updates

    def apply_graph_constraints(
        self,
        graph_edges: List[Dict],
        influence_strength: float = 0.1,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Apply graph-based constraints to beliefs.
        
        Hypotheses that reinforce each other should have correlated beliefs.
        Contradictory hypotheses should have anti-correlated beliefs.
        
        Args:
            graph_edges: List of edge dictionaries from hypothesis graph
            influence_strength: How much neighbors influence belief (0-1)
            
        Returns:
            Dict of {hid: (old_belief, new_belief)} for adjusted beliefs
        """
        updates: Dict[str, Tuple[float, float]] = {}
        adjustments: Dict[str, float] = {hid: 0.0 for hid in self.beliefs.keys()}
        
        # Compute adjustments based on graph structure
        for edge in graph_edges:
            source_hid = edge["source_hid"]
            target_hid = edge["target_hid"]
            edge_type = edge["edge_type"]
            # V1 uses uniform edge weight of 1.0
            edge_weight = edge.get("weight", 1.0)
            
            if source_hid not in self.beliefs or target_hid not in self.beliefs:
                continue
            
            source_belief = self.beliefs[source_hid]
            
            if edge_type == "reinforces":
                # If source is believed, increase target belief
                adjustment = influence_strength * edge_weight * (source_belief - 0.5)
                adjustments[target_hid] += adjustment
                
            elif edge_type == "contradicts":
                # If source is believed, decrease target belief
                adjustment = -influence_strength * edge_weight * (source_belief - 0.5)
                adjustments[target_hid] += adjustment
                
            elif edge_type == "depends_on":
                # If target is not believed, decrease source belief
                if source_belief > 0.5 and self.beliefs[target_hid] < 0.3:
                    adjustment = -influence_strength * edge_weight * 0.5
                    adjustments[source_hid] += adjustment
        
        # Apply adjustments
        for hid, adjustment in adjustments.items():
            if abs(adjustment) > 0.001:  # Only update if meaningful change
                old_belief = self.beliefs[hid]
                new_belief = max(0.01, min(0.99, old_belief + adjustment))
                
                if abs(new_belief - old_belief) > 0.001:
                    self.beliefs[hid] = new_belief
                    updates[hid] = (old_belief, new_belief)
                    
                    direction = "↑" if new_belief > old_belief else "↓"
                    self.explanation_log.append(
                        f"{hid}: {old_belief:.3f} {direction} {new_belief:.3f} | "
                        f"Graph constraint adjustment ({adjustment:+.3f})"
                    )
        
        return updates

    def get_high_confidence_hypotheses(self, threshold: float = 0.7) -> List[str]:
        """
        Get hypotheses with belief above threshold.
        
        Args:
            threshold: Minimum belief threshold
            
        Returns:
            List of HIDs with P(H) >= threshold
        """
        return [hid for hid, belief in self.beliefs.items() if belief >= threshold]

    def get_low_confidence_hypotheses(self, threshold: float = 0.3) -> List[str]:
        """
        Get hypotheses with belief below threshold.
        
        Args:
            threshold: Maximum belief threshold
            
        Returns:
            List of HIDs with P(H) <= threshold
        """
        return [hid for hid, belief in self.beliefs.items() if belief <= threshold]

    def get_uncertain_hypotheses(self, epsilon: float = 0.1) -> List[str]:
        """
        Get hypotheses with beliefs close to 0.5 (maximum uncertainty).
        
        Args:
            epsilon: Distance from 0.5 to consider uncertain
            
        Returns:
            List of HIDs with |P(H) - 0.5| <= epsilon
        """
        return [
            hid for hid, belief in self.beliefs.items()
            if abs(belief - 0.5) <= epsilon
        ]

    def compute_entropy(self) -> float:
        """
        Compute total entropy of the belief state.
        
        Higher entropy = more uncertainty across hypotheses.
        
        Returns:
            Total entropy in nats
        """
        total_entropy = 0.0
        for belief in self.beliefs.values():
            if belief > 0 and belief < 1:
                p = belief
                entropy = -p * math.log(p) - (1 - p) * math.log(1 - p)
                total_entropy += entropy
        return total_entropy

    def get_explanation_log(self) -> List[str]:
        """Get the full explanation log of all updates."""
        return self.explanation_log.copy()

    def to_dict(self) -> Dict:
        """
        Export belief state for serialization.
        
        Returns:
            Dictionary with beliefs and explanation log
        """
        return {
            "beliefs": self.beliefs.copy(),
            "explanation_log": self.explanation_log.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": {
                "entropy": self.compute_entropy(),
                "high_confidence_count": len(self.get_high_confidence_hypotheses()),
                "low_confidence_count": len(self.get_low_confidence_hypotheses()),
                "uncertain_count": len(self.get_uncertain_hypotheses()),
            },
        }

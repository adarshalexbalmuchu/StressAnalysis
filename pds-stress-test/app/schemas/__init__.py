"""
Pydantic v2 Schemas for PDS Stress-Testing Engine

These schemas define the contract layer between API, engine, and storage.
All data flows through these typed schemas to ensure consistency and validation.

Architecture Note:
- Schemas are the single source of truth for data structures
- No business logic should be implemented here
- All validation rules are declarative
- Schemas support both API and storage layer conversions
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# ENUMS - Domain-specific type constraints
# ============================================================================


class EdgeType(str, Enum):
    """Types of directed relationships between hypotheses in the hypothesis graph."""

    REINFORCES = "reinforces"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"


class TimeHorizon(str, Enum):
    """Expected timeframe for hypothesis effects to manifest."""

    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class SignalType(str, Enum):
    """Categories of evidence signals that can update belief states."""

    GRIEVANCE = "grievance"
    ADMIN_CIRCULAR = "admin_circular"
    MEDIA_REPORT = "media_report"
    AUDIT = "audit"
    COURT_OBSERVATION = "court_observation"
    FIELD_REPORT = "field_report"


class RunStatus(str, Enum):
    """Lifecycle status of a policy stress-test run."""

    CREATED = "created"
    HYPOTHESES_LOADED = "hypotheses_loaded"
    GRAPH_BUILT = "graph_built"
    BELIEFS_INITIALIZED = "beliefs_initialized"
    SIMULATING = "simulating"
    COMPLETED = "completed"
    FAILED = "failed"


class TrajectoryStatus(str, Enum):
    """Status of an individual simulated trajectory."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# HYPOTHESIS SCHEMAS
# ============================================================================


class HypothesisBase(BaseModel):
    """Core hypothesis schema - represents a mechanistic adaptation claim."""

    hid: str = Field(..., description="Unique hypothesis identifier (e.g., 'H001')")
    stakeholders: List[str] = Field(..., description="Affected stakeholder groups")
    triggers: List[str] = Field(..., description="Policy changes or conditions that activate this")
    mechanism: str = Field(..., description="Causal mechanism of adaptation")
    primary_effects: List[str] = Field(..., description="Direct consequences")
    secondary_effects: List[str] = Field(
        default_factory=list, description="Indirect/cascading consequences"
    )
    time_horizon: TimeHorizon = Field(..., description="Expected timeframe for manifestation")
    confidence_notes: Optional[str] = Field(
        None, description="Contextual notes on uncertainty"
    )

    @field_validator("hid")
    @classmethod
    def validate_hid_format(cls, v: str) -> str:
        """Ensure HID follows naming convention."""
        if not v.startswith("H") or not v[1:].isdigit():
            raise ValueError("HID must be format H### (e.g., H001, H042)")
        return v

    @field_validator("stakeholders", "triggers", "primary_effects")
    @classmethod
    def validate_non_empty_lists(cls, v: List[str]) -> List[str]:
        """Ensure critical lists are not empty."""
        if not v:
            raise ValueError("This field cannot be empty")
        return v


class HypothesisCreate(HypothesisBase):
    """Schema for creating a new hypothesis."""

    pass


class Hypothesis(HypothesisBase):
    """Full hypothesis with system metadata."""

    id: UUID
    run_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# GRAPH SCHEMAS (Edges and Graph Structure)
# ============================================================================


class Edge(BaseModel):
    """
    Directed edge in the hypothesis graph.
    
    Represents a causal or logical relationship between two hypotheses.
    Edges are directional: source -> target.
    
    V1 uses uniform edge weights (1.0) for all relationships.
    """

    source_hid: str = Field(
        ...,
        description="Source hypothesis HID (tail of directed edge)",
        pattern=r"^H\d+$",
    )
    
    target_hid: str = Field(
        ...,
        description="Target hypothesis HID (head of directed edge)",
        pattern=r"^H\d+$",
    )
    
    edge_type: EdgeType = Field(
        ...,
        description="Type of relationship: reinforces, contradicts, or depends_on",
    )
    
    notes: Optional[str] = Field(
        None,
        description="Human-readable explanation of why this relationship exists",
    )

    @field_validator("source_hid", "target_hid")
    @classmethod
    def validate_hid_format(cls, v: str) -> str:
        """Enforce HID format for edge endpoints."""
        if not v.startswith("H") or not v[1:].isdigit():
            raise ValueError(f"HID must be format H###, got: {v}")
        return v


class EdgeCreate(Edge):
    """Schema for creating a new graph edge via API."""

    pass


# Backward compatibility aliases
GraphEdge = Edge
GraphEdgeCreate = EdgeCreate


class HypothesisGraph(BaseModel):
    """Complete hypothesis graph structure."""

    run_id: UUID
    nodes: List[str] = Field(..., description="List of hypothesis HIDs")
    edges: List[Edge] = Field(..., description="Relationships between hypotheses")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Graph-level metadata (cycles, clusters, etc.)"
    )
    created_at: datetime

    @field_validator("nodes")
    @classmethod
    def validate_node_hids(cls, v: List[str]) -> List[str]:
        """Ensure all nodes follow HID format."""
        for hid in v:
            if not hid.startswith("H") or not hid[1:].isdigit():
                raise ValueError(f"Invalid node HID: {hid}")
        return v


# ============================================================================
# BELIEF STATE SCHEMAS
# ============================================================================


class BeliefState(BaseModel):
    """Probabilistic belief state over all hypotheses at a point in time."""

    id: UUID
    run_id: UUID
    beliefs: Dict[str, float] = Field(
        ..., description="Map of HID -> probability P(H)"
    )
    timestamp: datetime
    explanation_log: List[str] = Field(
        default_factory=list, description="Human-readable update explanations"
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("beliefs")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure all probabilities are in [0, 1]."""
        for hid, prob in v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for {hid} must be in [0, 1], got {prob}")
        return v


class BeliefUpdate(BaseModel):
    """Request to update belief state based on evidence."""

    signal_ids: List[UUID] = Field(..., description="Signal IDs providing evidence")
    update_method: str = Field(
        default="bayesian_simple", description="Update algorithm to use"
    )
    notes: Optional[str] = Field(None, description="Context for this update")


# ============================================================================
# SIGNAL SCHEMAS (Evidence)
# ============================================================================


class SignalBase(BaseModel):
    """Base schema for evidence signals."""

    signal_type: SignalType
    content: str = Field(..., description="Description or summary of the signal")
    source: str = Field(..., description="Source of this evidence")
    date_observed: datetime = Field(..., description="When this signal was observed")
    affected_hids: List[str] = Field(
        ..., description="Hypothesis HIDs this signal provides evidence for/against"
    )
    strength: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Strength/reliability of signal"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional structured data"
    )


class SignalCreate(SignalBase):
    """Schema for creating a new signal."""

    pass


class Signal(SignalBase):
    """Full signal with system metadata."""

    id: UUID
    run_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# TRAJECTORY SCHEMAS (Simulation Results)
# ============================================================================


class SimulationParams(BaseModel):
    """Parameters for Monte Carlo trajectory simulation."""

    n_runs: int = Field(default=2000, ge=1, description="Number of Monte Carlo runs")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    activation_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum probability for activation"
    )
    propagate_steps: int = Field(
        default=2, ge=0, description="Number of graph propagation steps"
    )
    reinforce_delta: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Boost from 'reinforces' edges"
    )
    contradict_delta: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Reduction from 'contradicts' edges"
    )
    top_k: int = Field(default=5, ge=1, description="Number of top trajectories to return")


class TrajectoryStep(BaseModel):
    """A single step in a simulated trajectory."""

    step: int
    activated_hids: List[str] = Field(..., description="Hypotheses active at this step")
    cumulative_effects: List[str] = Field(
        ..., description="Accumulated effects up to this point"
    )
    notes: Optional[str] = None


class Trajectory(BaseModel):
    """A single simulated future trajectory."""

    trajectory_id: str = Field(..., description="Unique identifier for this trajectory")
    run_id: UUID
    steps: List[TrajectoryStep] = Field(..., description="Sequence of states")
    final_state: Dict[str, Any] = Field(
        ..., description="Terminal state characteristics"
    )
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of this path")
    status: TrajectoryStatus = Field(default=TrajectoryStatus.COMPLETED)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Results from a Monte Carlo simulation run."""

    run_id: UUID
    simulation_id: UUID
    trajectories: List[Trajectory] = Field(..., description="All simulated trajectories")
    summary_statistics: Dict[str, Any] = Field(
        ..., description="Aggregate metrics across trajectories"
    )
    sensitivity_hotspots: List[str] = Field(
        ..., description="HIDs with high variance or influence"
    )
    timestamp: datetime
    parameters: Dict[str, Any] = Field(
        ..., description="Simulation parameters used (n_iterations, seed, etc.)"
    )


# ============================================================================
# POLICY RUN SCHEMAS (Top-Level Container)
# ============================================================================


class RunBase(BaseModel):
    """Base schema for a stress-test run."""

    policy_rule: str = Field(
        ..., description="The policy rule being stress-tested"
    )
    domain: str = Field(default="PDS", description="Policy domain")
    description: Optional[str] = Field(None, description="Human-readable context")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Run-specific metadata"
    )


class RunCreate(RunBase):
    """Schema for creating a new run."""

    pass


class Run(RunBase):
    """Full run with system metadata."""

    id: UUID
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    hypothesis_count: int = Field(default=0, description="Number of hypotheses in this run")

    model_config = ConfigDict(from_attributes=True)


class RunSummary(BaseModel):
    """High-level summary of a run."""

    id: UUID
    policy_rule: str
    status: RunStatus
    hypothesis_count: int
    created_at: datetime
    latest_belief_state_timestamp: Optional[datetime] = None
    simulation_count: int = Field(default=0, description="Number of simulations executed")


# ============================================================================
# API RESPONSE WRAPPERS (Standard Response Format)
# ============================================================================


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

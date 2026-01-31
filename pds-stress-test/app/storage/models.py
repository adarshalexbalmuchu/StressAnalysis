"""
SQLAlchemy models for persistent storage.

All models use UUID primary keys and include proper timestamps.
JSONB fields are used for complex artifacts (graphs, belief states, simulation results).
"""

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.schemas import EdgeType, RunStatus, SignalType, TimeHorizon, TrajectoryStatus

Base = declarative_base()


class Run(Base):
    """Represents a single stress-test run for a policy rule."""

    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_rule = Column(String(500), nullable=False, index=True)
    domain = Column(String(100), nullable=False, default="PDS", index=True)
    description = Column(Text, nullable=True)
    status = Column(
        Enum(RunStatus),
        nullable=False,
        default=RunStatus.CREATED,
        index=True,
    )
    hypothesis_count = Column(Integer, nullable=False, default=0)
    metadata = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    hypotheses = relationship("Hypothesis", back_populates="run", cascade="all, delete-orphan")
    graphs = relationship("HypothesisGraphModel", back_populates="run", cascade="all, delete-orphan")
    belief_states = relationship("BeliefStateModel", back_populates="run", cascade="all, delete-orphan")
    signals = relationship("SignalModel", back_populates="run", cascade="all, delete-orphan")
    simulations = relationship("SimulationResultModel", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Run(id={self.id}, policy_rule='{self.policy_rule[:50]}', status={self.status})>"


class Hypothesis(Base):
    """Represents a mechanistic hypothesis about stakeholder adaptation."""

    __tablename__ = "hypotheses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Core hypothesis fields
    hid = Column(String(20), nullable=False, index=True)
    stakeholders = Column(JSONB, nullable=False)  # List[str]
    triggers = Column(JSONB, nullable=False)  # List[str]
    mechanism = Column(Text, nullable=False)
    primary_effects = Column(JSONB, nullable=False)  # List[str]
    secondary_effects = Column(JSONB, nullable=False, default=list)  # List[str]
    time_horizon = Column(Enum(TimeHorizon), nullable=False)
    confidence_notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    run = relationship("Run", back_populates="hypotheses")

    def __repr__(self) -> str:
        return f"<Hypothesis(id={self.id}, hid='{self.hid}', run_id={self.run_id})>"


class HypothesisGraphModel(Base):
    """Stores the hypothesis graph structure as a JSONB artifact."""

    __tablename__ = "hypothesis_graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Graph structure stored as JSONB
    nodes = Column(JSONB, nullable=False)  # List[str] of HIDs
    edges = Column(JSONB, nullable=False)  # List[Dict] of edge specifications
    metadata = Column(JSONB, nullable=False, default=dict)  # Graph-level metadata

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    run = relationship("Run", back_populates="graphs")

    def __repr__(self) -> str:
        return f"<HypothesisGraph(id={self.id}, run_id={self.run_id}, nodes={len(self.nodes) if self.nodes else 0})>"


class BeliefStateModel(Base):
    """Stores belief state snapshots over time."""

    __tablename__ = "belief_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Belief state data
    beliefs = Column(JSONB, nullable=False)  # Dict[str, float] - HID -> probability
    explanation_log = Column(JSONB, nullable=False, default=list)  # List[str]
    
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    # Relationships
    run = relationship("Run", back_populates="belief_states")

    def __repr__(self) -> str:
        return f"<BeliefState(id={self.id}, run_id={self.run_id}, timestamp={self.timestamp})>"


class SignalModel(Base):
    """Represents an evidence signal that can update beliefs."""

    __tablename__ = "signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Signal data
    signal_type = Column(Enum(SignalType), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source = Column(String(500), nullable=False)
    date_observed = Column(DateTime(timezone=True), nullable=False, index=True)
    affected_hids = Column(JSONB, nullable=False)  # List[str]
    strength = Column(Float, nullable=False, default=0.5)
    metadata = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    run = relationship("Run", back_populates="signals")

    def __repr__(self) -> str:
        return f"<Signal(id={self.id}, type={self.signal_type}, run_id={self.run_id})>"


class SimulationResultModel(Base):
    """Stores complete simulation results including all trajectories."""

    __tablename__ = "simulation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Simulation results stored as JSONB
    trajectories = Column(JSONB, nullable=False)  # List[Dict] of trajectory data
    summary_statistics = Column(JSONB, nullable=False)
    sensitivity_hotspots = Column(JSONB, nullable=False)  # List[str] of HIDs
    parameters = Column(JSONB, nullable=False)  # Simulation parameters

    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    # Relationships
    run = relationship("Run", back_populates="simulations")

    def __repr__(self) -> str:
        return f"<SimulationResult(id={self.id}, run_id={self.run_id}, trajectories={len(self.trajectories) if self.trajectories else 0})>"

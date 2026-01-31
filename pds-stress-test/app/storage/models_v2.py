"""
SQLAlchemy 2.0 ORM Models for Policy Stress-Testing Engine.

All artifacts stored as JSON for auditability and reproducibility.
Foreign keys cascade delete. Time-indexed for belief evolution tracking.

NO business logic. NO API imports. NO LLM logic.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text, JSON, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, CHAR
import uuid as uuid_lib

from app.storage.database import Base


# UUID type that works with both PostgreSQL and SQLite
class UUID(TypeDecorator):
    """Platform-independent UUID type."""
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if isinstance(value, uuid_lib.UUID):
                return str(value)
            return value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid_lib.UUID):
            return value
        return uuid_lib.UUID(value)


class PolicyRun(Base):
    """
    Policy stress-test run.
    
    Parent entity for all artifacts in a stress-testing session.
    """

    __tablename__ = "policy_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    policy_rule_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships (cascade delete)
    hypothesis_sets = relationship(
        "HypothesisSet",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    hypothesis_graphs = relationship(
        "HypothesisGraph",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    signals = relationship(
        "SignalSet",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    belief_states = relationship(
        "BeliefState",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="BeliefState.created_at",
    )
    simulations = relationship(
        "SimulationOutput",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<PolicyRun(id={self.id}, name='{self.name}')>"


class HypothesisSet(Base):
    """
    Hypothesis set (seeded or generated) for a run.
    
    Stores hypotheses as JSON array for full auditability.
    """

    __tablename__ = "hypothesis_sets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("policy_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    hypotheses_json: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    run = relationship("PolicyRun", back_populates="hypothesis_sets")

    def __repr__(self) -> str:
        return f"<HypothesisSet(id={self.id}, run_id={self.run_id})>"


class HypothesisGraph(Base):
    """
    Hypothesis graph (edges + validation status) for a run.
    
    Stores graph structure as JSON dict (nodes + edges).
    """

    __tablename__ = "hypothesis_graphs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("policy_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    graph_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    run = relationship("PolicyRun", back_populates="hypothesis_graphs")

    def __repr__(self) -> str:
        return f"<HypothesisGraph(id={self.id}, run_id={self.run_id})>"


class SignalSet(Base):
    """
    Signal set (evidence batch) for a run.
    
    Stores signals as JSON array.
    """

    __tablename__ = "signal_sets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("policy_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    signals_json: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    run = relationship("PolicyRun", back_populates="signals")

    def __repr__(self) -> str:
        return f"<SignalSet(id={self.id}, run_id={self.run_id})>"


class BeliefState(Base):
    """
    Belief state snapshot (probabilities + explanation) for a run.
    
    Time-indexed to support belief evolution tracking.
    Stores belief_json as dict (hid -> probability) and explanation_log as list.
    """

    __tablename__ = "belief_states"

    id: Mapped[uuid.UUID] = mapped_column(UUID(), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("policy_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    belief_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    explanation_log: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # Relationships
    run = relationship("PolicyRun", back_populates="belief_states")

    def __repr__(self) -> str:
        return f"<BeliefState(id={self.id}, run_id={self.run_id}, created_at={self.created_at})>"


class SimulationOutput(Base):
    """
    Simulation output (trajectories + stats) for a run.
    
    Stores both params_json and result_json as dicts.
    """

    __tablename__ = "simulation_outputs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("policy_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    params_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    run = relationship("PolicyRun", back_populates="simulations")

    def __repr__(self) -> str:
        return f"<SimulationOutput(id={self.id}, run_id={self.run_id})>"

"""Initial schema for PDS stress-testing engine.

Implements a JSONB-centric design where complex domain artifacts
(hypotheses, graph, signals, beliefs, simulation results) are stored
as JSONB columns for flexibility and query performance.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-01-31 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables for the policy stress-testing engine."""

    # 1. Policy Runs table (top-level container)
    op.create_table(
        "policy_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("policy_rule_text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_policy_runs_name"), "policy_runs", ["name"])

    # 2. Hypothesis Sets table (JSONB array of hypothesis objects)
    op.create_table(
        "hypothesis_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "hypotheses_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["policy_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_hypothesis_sets_run_id"), "hypothesis_sets", ["run_id"]
    )

    # 3. Hypothesis Graphs table (JSONB for NetworkX node_link_data format)
    op.create_table(
        "hypothesis_graphs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "graph_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["policy_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_hypothesis_graphs_run_id"), "hypothesis_graphs", ["run_id"]
    )

    # 4. Signal Sets table (JSONB array of signal objects)
    op.create_table(
        "signal_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "signals_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["policy_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_signal_sets_run_id"), "signal_sets", ["run_id"]
    )

    # 5. Belief States table (time-indexed JSONB for belief snapshots)
    op.create_table(
        "belief_states",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "belief_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "explanation_log",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["policy_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_belief_states_run_id"), "belief_states", ["run_id"]
    )
    op.create_index(
        op.f("ix_belief_states_created_at"), "belief_states", ["created_at"]
    )

    # 6. Simulation Outputs table (JSONB for params and results)
    op.create_table(
        "simulation_outputs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "params_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "result_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["policy_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_simulation_outputs_run_id"), "simulation_outputs", ["run_id"]
    )


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_index(
        op.f("ix_simulation_outputs_run_id"), table_name="simulation_outputs"
    )
    op.drop_table("simulation_outputs")

    op.drop_index(op.f("ix_belief_states_created_at"), table_name="belief_states")
    op.drop_index(op.f("ix_belief_states_run_id"), table_name="belief_states")
    op.drop_table("belief_states")

    op.drop_index(op.f("ix_signal_sets_run_id"), table_name="signal_sets")
    op.drop_table("signal_sets")

    op.drop_index(
        op.f("ix_hypothesis_graphs_run_id"), table_name="hypothesis_graphs"
    )
    op.drop_table("hypothesis_graphs")

    op.drop_index(
        op.f("ix_hypothesis_sets_run_id"), table_name="hypothesis_sets"
    )
    op.drop_table("hypothesis_sets")

    op.drop_index(op.f("ix_policy_runs_name"), table_name="policy_runs")
    op.drop_table("policy_runs")

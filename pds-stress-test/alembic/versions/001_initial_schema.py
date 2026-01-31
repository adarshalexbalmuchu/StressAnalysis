"""Initial schema for PDS stress-test engine.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-01-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial tables for the stress-test engine."""
    
    # Create enum types
    op.execute("""
        CREATE TYPE runstatus AS ENUM (
            'created', 'hypotheses_loaded', 'graph_built', 
            'beliefs_initialized', 'simulating', 'completed', 'failed'
        )
    """)
    
    op.execute("""
        CREATE TYPE timehorizon AS ENUM (
            'immediate', 'short_term', 'medium_term', 'long_term'
        )
    """)
    
    op.execute("""
        CREATE TYPE signaltype AS ENUM (
            'grievance', 'admin_circular', 'media_report', 
            'audit', 'court_observation', 'field_report'
        )
    """)
    
    # Create runs table
    op.create_table(
        'runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('policy_rule', sa.String(500), nullable=False),
        sa.Column('domain', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('created', 'hypotheses_loaded', 'graph_built', 
                                   'beliefs_initialized', 'simulating', 'completed', 
                                   'failed', name='runstatus'), nullable=False),
        sa.Column('hypothesis_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('metadata', JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('NOW()')),
    )
    op.create_index('ix_runs_policy_rule', 'runs', ['policy_rule'])
    op.create_index('ix_runs_domain', 'runs', ['domain'])
    op.create_index('ix_runs_status', 'runs', ['status'])
    
    # Create hypotheses table
    op.create_table(
        'hypotheses',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('run_id', UUID(as_uuid=True), nullable=False),
        sa.Column('hid', sa.String(20), nullable=False),
        sa.Column('stakeholders', JSONB(), nullable=False),
        sa.Column('triggers', JSONB(), nullable=False),
        sa.Column('mechanism', sa.Text(), nullable=False),
        sa.Column('primary_effects', JSONB(), nullable=False),
        sa.Column('secondary_effects', JSONB(), nullable=False, server_default='[]'),
        sa.Column('time_horizon', sa.Enum('immediate', 'short_term', 'medium_term', 
                                         'long_term', name='timehorizon'), nullable=False),
        sa.Column('confidence_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_hypotheses_run_id', 'hypotheses', ['run_id'])
    op.create_index('ix_hypotheses_hid', 'hypotheses', ['hid'])
    
    # Create hypothesis_graphs table
    op.create_table(
        'hypothesis_graphs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('run_id', UUID(as_uuid=True), nullable=False),
        sa.Column('nodes', JSONB(), nullable=False),
        sa.Column('edges', JSONB(), nullable=False),
        sa.Column('metadata', JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_hypothesis_graphs_run_id', 'hypothesis_graphs', ['run_id'])
    
    # Create belief_states table
    op.create_table(
        'belief_states',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('run_id', UUID(as_uuid=True), nullable=False),
        sa.Column('beliefs', JSONB(), nullable=False),
        sa.Column('explanation_log', JSONB(), nullable=False, server_default='[]'),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_belief_states_run_id', 'belief_states', ['run_id'])
    op.create_index('ix_belief_states_timestamp', 'belief_states', ['timestamp'])
    
    # Create signals table
    op.create_table(
        'signals',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('run_id', UUID(as_uuid=True), nullable=False),
        sa.Column('signal_type', sa.Enum('grievance', 'admin_circular', 'media_report', 
                                        'audit', 'court_observation', 'field_report', 
                                        name='signaltype'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('source', sa.String(500), nullable=False),
        sa.Column('date_observed', sa.DateTime(timezone=True), nullable=False),
        sa.Column('affected_hids', JSONB(), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('metadata', JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_signals_run_id', 'signals', ['run_id'])
    op.create_index('ix_signals_signal_type', 'signals', ['signal_type'])
    op.create_index('ix_signals_date_observed', 'signals', ['date_observed'])
    
    # Create simulation_results table
    op.create_table(
        'simulation_results',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('run_id', UUID(as_uuid=True), nullable=False),
        sa.Column('trajectories', JSONB(), nullable=False),
        sa.Column('summary_statistics', JSONB(), nullable=False),
        sa.Column('sensitivity_hotspots', JSONB(), nullable=False),
        sa.Column('parameters', JSONB(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, 
                 server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_simulation_results_run_id', 'simulation_results', ['run_id'])
    op.create_index('ix_simulation_results_timestamp', 'simulation_results', ['timestamp'])


def downgrade() -> None:
    """Drop all tables and enums."""
    op.drop_index('ix_simulation_results_timestamp', 'simulation_results')
    op.drop_index('ix_simulation_results_run_id', 'simulation_results')
    op.drop_table('simulation_results')
    
    op.drop_index('ix_signals_date_observed', 'signals')
    op.drop_index('ix_signals_signal_type', 'signals')
    op.drop_index('ix_signals_run_id', 'signals')
    op.drop_table('signals')
    
    op.drop_index('ix_belief_states_timestamp', 'belief_states')
    op.drop_index('ix_belief_states_run_id', 'belief_states')
    op.drop_table('belief_states')
    
    op.drop_index('ix_hypothesis_graphs_run_id', 'hypothesis_graphs')
    op.drop_table('hypothesis_graphs')
    
    op.drop_index('ix_hypotheses_hid', 'hypotheses')
    op.drop_index('ix_hypotheses_run_id', 'hypotheses')
    op.drop_table('hypotheses')
    
    op.drop_index('ix_runs_status', 'runs')
    op.drop_index('ix_runs_domain', 'runs')
    op.drop_index('ix_runs_policy_rule', 'runs')
    op.drop_table('runs')
    
    op.execute('DROP TYPE signaltype')
    op.execute('DROP TYPE timehorizon')
    op.execute('DROP TYPE runstatus')

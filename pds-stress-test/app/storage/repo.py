"""
Repository layer for CRUD operations on stress-test artifacts.

Thin persistence layer that wraps SQLAlchemy operations.
NO business logic. NO engine logic. Pure data access.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.utils.json_safe import json_safe
from app.storage.models_v2 import (
    BeliefState,
    HypothesisGraph,
    HypothesisSet,
    PolicyRun,
    SignalSet,
    SimulationOutput,
)


# ============================================================================
# RUN CRUD
# ============================================================================


def create_run(db: Session, name: str, policy_rule_text: str) -> PolicyRun:
    """
    Create a new policy stress-test run.
    
    Args:
        db: Database session
        name: Human-readable run name
        policy_rule_text: Policy rule being stress-tested
        
    Returns:
        Created PolicyRun instance
    """
    run = PolicyRun(
        name=name,
        policy_rule_text=policy_rule_text,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: UUID) -> Optional[PolicyRun]:
    """
    Get a policy run by ID.
    
    Args:
        db: Database session
        run_id: Run UUID
        
    Returns:
        PolicyRun if found, None otherwise
    """
    return db.query(PolicyRun).filter(PolicyRun.id == run_id).first()


def list_runs(db: Session, limit: int = 50, offset: int = 0) -> list[PolicyRun]:
    """
    List policy runs, most recent first.
    
    Args:
        db: Database session
        limit: Maximum number of runs to return
        offset: Number of runs to skip
        
    Returns:
        List of PolicyRun instances
    """
    return (
        db.query(PolicyRun)
        .order_by(desc(PolicyRun.created_at))
        .limit(limit)
        .offset(offset)
        .all()
    )


# ============================================================================
# HYPOTHESIS SET CRUD
# ============================================================================


def save_hypothesis_set(
    db: Session,
    run_id: UUID,
    hypotheses_json: list[dict],
) -> HypothesisSet:
    """
    Save a hypothesis set for a run.
    
    Args:
        db: Database session
        run_id: Parent run UUID
        hypotheses_json: List of hypothesis dicts
        
    Returns:
        Created HypothesisSet instance
    """
    hypothesis_set = HypothesisSet(
        run_id=run_id,
        hypotheses_json=json_safe(hypotheses_json),
    )
    db.add(hypothesis_set)
    db.commit()
    db.refresh(hypothesis_set)
    return hypothesis_set


def get_latest_hypothesis_set(db: Session, run_id: UUID) -> Optional[HypothesisSet]:
    """
    Get the most recent hypothesis set for a run.
    
    Args:
        db: Database session
        run_id: Run UUID
        
    Returns:
        Latest HypothesisSet if exists, None otherwise
    """
    return (
        db.query(HypothesisSet)
        .filter(HypothesisSet.run_id == run_id)
        .order_by(desc(HypothesisSet.created_at))
        .first()
    )


# ============================================================================
# GRAPH CRUD
# ============================================================================


def save_graph(
    db: Session,
    run_id: UUID,
    graph_json: dict,
) -> HypothesisGraph:
    """
    Save a hypothesis graph for a run.
    
    Args:
        db: Database session
        run_id: Parent run UUID
        graph_json: Graph structure as dict (nodes + edges)
        
    Returns:
        Created HypothesisGraph instance
    """
    graph = HypothesisGraph(
        run_id=run_id,
        graph_json=json_safe(graph_json),
    )
    db.add(graph)
    db.commit()
    db.refresh(graph)
    return graph


def get_latest_graph(db: Session, run_id: UUID) -> Optional[HypothesisGraph]:
    """
    Get the most recent hypothesis graph for a run.
    
    Args:
        db: Database session
        run_id: Run UUID
        
    Returns:
        Latest HypothesisGraph if exists, None otherwise
    """
    return (
        db.query(HypothesisGraph)
        .filter(HypothesisGraph.run_id == run_id)
        .order_by(desc(HypothesisGraph.created_at))
        .first()
    )


# ============================================================================
# SIGNAL SET CRUD
# ============================================================================


def save_signals(
    db: Session,
    run_id: UUID,
    signals_json: list[dict],
) -> SignalSet:
    """
    Save a signal set for a run.
    
    Args:
        db: Database session
        run_id: Parent run UUID
        signals_json: List of signal dicts
        
    Returns:
        Created SignalSet instance
    """
    signal_set = SignalSet(
        run_id=run_id,
        signals_json=json_safe(signals_json),
    )
    db.add(signal_set)
    db.commit()
    db.refresh(signal_set)
    return signal_set


def get_latest_signals(db: Session, run_id: UUID) -> Optional[SignalSet]:
    """
    Get the most recent signal set for a run.
    
    Args:
        db: Database session
        run_id: Run UUID
        
    Returns:
        Latest SignalSet if exists, None otherwise
    """
    return (
        db.query(SignalSet)
        .filter(SignalSet.run_id == run_id)
        .order_by(desc(SignalSet.created_at))
        .first()
    )


# ============================================================================
# BELIEF STATE CRUD
# ============================================================================


def save_belief_state(
    db: Session,
    run_id: UUID,
    belief_json: dict,
    explanation_log: list[str],
) -> BeliefState:
    """
    Save a belief state snapshot for a run.
    
    Args:
        db: Database session
        run_id: Parent run UUID
        belief_json: Beliefs dict (hid -> probability)
        explanation_log: List of human-readable explanations
        
    Returns:
        Created BeliefState instance
    """
    belief_state = BeliefState(
        run_id=run_id,
        belief_json=json_safe(belief_json),
        explanation_log=json_safe(explanation_log),
    )
    db.add(belief_state)
    db.commit()
    db.refresh(belief_state)
    return belief_state


def get_latest_belief_state(db: Session, run_id: UUID) -> Optional[BeliefState]:
    """
    Get the most recent belief state for a run.
    
    Args:
        db: Database session
        run_id: Run UUID
        
    Returns:
        Latest BeliefState if exists, None otherwise
    """
    return (
        db.query(BeliefState)
        .filter(BeliefState.run_id == run_id)
        .order_by(desc(BeliefState.created_at))
        .first()
    )


def list_belief_states(
    db: Session,
    run_id: UUID,
    limit: int = 50,
) -> list[BeliefState]:
    """
    List belief states for a run, most recent first.
    
    Args:
        db: Database session
        run_id: Run UUID
        limit: Maximum number of states to return
        
    Returns:
        List of BeliefState instances
    """
    return (
        db.query(BeliefState)
        .filter(BeliefState.run_id == run_id)
        .order_by(desc(BeliefState.created_at))
        .limit(limit)
        .all()
    )


# ============================================================================
# SIMULATION OUTPUT CRUD
# ============================================================================


def save_simulation_output(
    db: Session,
    run_id: UUID,
    params_json: dict,
    result_json: dict,
) -> SimulationOutput:
    """
    Save simulation output for a run.
    
    Args:
        db: Database session
        run_id: Parent run UUID
        params_json: Simulation parameters
        result_json: Simulation results
        
    Returns:
        Created SimulationOutput instance
    """
    simulation = SimulationOutput(
        run_id=run_id,
        params_json=json_safe(params_json),
        result_json=json_safe(result_json),
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)
    return simulation


def get_latest_simulation(db: Session, run_id: UUID) -> Optional[SimulationOutput]:
    """
    Get the most recent simulation output for a run.
    
    Args:
        db: Database session
        run_id: Run UUID
        
    Returns:
        Latest SimulationOutput if exists, None otherwise
    """
    return (
        db.query(SimulationOutput)
        .filter(SimulationOutput.run_id == run_id)
        .order_by(desc(SimulationOutput.created_at))
        .first()
    )


def list_simulations(
    db: Session,
    run_id: UUID,
    limit: int = 50,
) -> list[SimulationOutput]:
    """
    List simulation outputs for a run, most recent first.
    
    Args:
        db: Database session
        run_id: Run UUID
        limit: Maximum number of simulations to return
        
    Returns:
        List of SimulationOutput instances
    """
    return (
        db.query(SimulationOutput)
        .filter(SimulationOutput.run_id == run_id)
        .order_by(desc(SimulationOutput.created_at))
        .limit(limit)
        .all()
    )

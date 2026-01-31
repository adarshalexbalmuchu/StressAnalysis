"""
FastAPI routes for stress-test pipeline execution.

Thin routing layer that orchestrates engine calls and persists results.
Executes V1 workflow: hypotheses → graph → signals → beliefs → simulation.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
import logging

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.engine import belief, graph_builder, simulator
from app.schemas import Hypothesis, HypothesisBase, Signal, SignalBase, SimulationParams, Edge
from app.storage import repo
from app.storage.database import get_db


logger = logging.getLogger(__name__)


def norm_uuid(x) -> UUID:
    """Normalize UUID wrapper to standard uuid.UUID."""
    if isinstance(x, UUID):
        return x
    return UUID(str(x))


router = APIRouter(prefix="/runs/{run_id}", tags=["pipeline"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================


class HypothesesRequest(BaseModel):
    """Request to save hypotheses for a run."""

    hypotheses: list[HypothesisBase] = Field(..., description="List of hypotheses")


class HypothesesResponse(BaseModel):
    """Response after saving hypotheses."""

    id: UUID
    run_id: UUID
    hypotheses_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class GraphRequest(BaseModel):
    """Request to validate and save hypothesis graph."""

    edges: list[Edge] = Field(..., description="List of edge objects")


class GraphResponse(BaseModel):
    """Response after validating and saving graph."""

    id: UUID
    run_id: UUID
    nodes_count: int
    edges_count: int
    is_valid: bool
    validation_errors: list[str]
    created_at: datetime

    class Config:
        from_attributes = True


class SignalsRequest(BaseModel):
    """Request to save signals for a run."""

    signals: list[SignalBase] = Field(..., description="List of signals")


class SignalsResponse(BaseModel):
    """Response after saving signals."""

    id: UUID
    run_id: UUID
    signals_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class InitPriorsRequest(BaseModel):
    """Request to initialize prior beliefs."""

    strategy: str = Field(default="uniform", description="Prior initialization strategy")


class InitPriorsResponse(BaseModel):
    """Response after initializing priors."""

    id: UUID
    run_id: UUID
    belief_count: int
    strategy: str
    created_at: datetime

    class Config:
        from_attributes = True


class UpdateBeliefsRequest(BaseModel):
    """Request to update beliefs with new signals."""

    signals: Optional[list[Signal]] = Field(
        None,
        description="New signals to process (optional, uses latest from DB if omitted)",
    )


class UpdateBeliefsResponse(BaseModel):
    """Response after updating beliefs."""

    id: UUID
    run_id: UUID
    belief_count: int
    explanation_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class SimulateRequest(BaseModel):
    """Request to run Monte Carlo simulation."""

    params: SimulationParams = Field(..., description="Simulation parameters")


class SimulateResponse(BaseModel):
    """Response after running simulation."""

    id: UUID
    run_id: UUID
    trajectory_count: int
    created_at: datetime
    result: dict  # Full simulation result

    class Config:
        from_attributes = True


# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================


@router.post("/hypotheses", response_model=HypothesesResponse, status_code=201)
def save_hypotheses(
    run_id: UUID,
    request: HypothesesRequest,
    db: Session = Depends(get_db),
) -> HypothesesResponse:
    """
    Save hypotheses for a run.
    
    Args:
        run_id: Parent run UUID
        request: Hypotheses to save
        db: Database session
        
    Returns:
        Saved hypothesis set details
        
    Raises:
        HTTPException: 404 if run not found
    """
    # Verify run exists
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Convert Pydantic models to dict for JSONB storage
    hypotheses_json = [h.model_dump() for h in request.hypotheses]
    
    # Save to database
    hypothesis_set = repo.save_hypothesis_set(
        db=db,
        run_id=run_id,
        hypotheses_json=hypotheses_json,
    )
    
    return HypothesesResponse(
        id=norm_uuid(hypothesis_set.id),
        run_id=norm_uuid(hypothesis_set.run_id),
        hypotheses_count=len(hypotheses_json),
        created_at=hypothesis_set.created_at,
    )


@router.post("/graph/validate-and-save", response_model=GraphResponse, status_code=201)
def validate_and_save_graph(
    run_id: UUID,
    request: GraphRequest,
    db: Session = Depends(get_db),
) -> GraphResponse:
    """
    Validate hypothesis graph and save if valid.
    
    Calls graph_builder.validate_graph() then persists.
    
    Args:
        run_id: Parent run UUID
        request: Graph edges
        db: Database session
        
    Returns:
        Graph validation results and saved graph details
        
    Raises:
        HTTPException: 404 if run not found
    """
    # Verify run exists
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Load hypotheses for graph building
    hypothesis_set = repo.get_latest_hypothesis_set(db=db, run_id=run_id)
    if not hypothesis_set:
        raise HTTPException(
            status_code=400,
            detail=f"No hypotheses found for run {run_id}. Upload hypotheses first.",
        )
    
    hypotheses_dicts = hypothesis_set.hypotheses_json
    
    # Build and validate graph using engine
    graph = graph_builder.build_graph(hypotheses=hypotheses_dicts, edges=request.edges)
    errors = graph_builder.validate_graph(hypotheses=hypotheses_dicts, edges=request.edges)
    is_valid = len(errors) == 0
    
    # Convert graph to JSON for storage
    graph_json = nx.node_link_data(graph)
    
    # Save to database
    hypothesis_graph = repo.save_graph(
        db=db,
        run_id=run_id,
        graph_json=graph_json,
    )
    
    return GraphResponse(
        id=norm_uuid(hypothesis_graph.id),
        run_id=norm_uuid(hypothesis_graph.run_id),
        nodes_count=graph.number_of_nodes(),
        edges_count=graph.number_of_edges(),
        is_valid=is_valid,
        validation_errors=errors,
        created_at=hypothesis_graph.created_at,
    )


@router.post("/signals", response_model=SignalsResponse, status_code=201)
def save_signals(
    run_id: UUID,
    request: SignalsRequest,
    db: Session = Depends(get_db),
) -> SignalsResponse:
    """
    Save signals for a run.
    
    Args:
        run_id: Parent run UUID
        request: Signals to save
        db: Database session
        
    Returns:
        Saved signal set details
        
    Raises:
        HTTPException: 404 if run not found
    """
    # Verify run exists
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Convert Pydantic models to dict for JSONB storage
    signals_json = [s.model_dump(mode="json") for s in request.signals]
    
    # Save to database
    try:
        signal_set = repo.save_signals(
            db=db,
            run_id=run_id,
            signals_json=signals_json,
        )
    except Exception:
        logger.exception("Failed to save signals")
        raise
    
    return SignalsResponse(
        id=norm_uuid(signal_set.id),
        run_id=norm_uuid(signal_set.run_id),
        signals_count=len(signals_json),
        created_at=signal_set.created_at,
    )


@router.post("/belief/init", response_model=InitPriorsResponse, status_code=201)
def initialize_priors(
    run_id: UUID,
    request: InitPriorsRequest,
    db: Session = Depends(get_db),
) -> InitPriorsResponse:
    """
    Initialize prior beliefs for hypotheses.
    
    Calls belief.init_priors() then persists.
    
    Args:
        run_id: Parent run UUID
        request: Prior initialization parameters
        db: Database session
        
    Returns:
        Initialized belief state details
        
    Raises:
        HTTPException: 404 if run not found, 400 if hypotheses not found
    """
    # Verify run exists
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Load latest hypotheses
    hypothesis_set = repo.get_latest_hypothesis_set(db=db, run_id=run_id)
    if not hypothesis_set:
        raise HTTPException(
            status_code=400,
            detail=f"No hypotheses found for run {run_id}. Save hypotheses first.",
        )
    
    # Convert JSON to Pydantic models for engine
    # Add system fields that engine might reference
    from datetime import datetime, timezone
    from uuid import uuid4
    hypotheses = [
        Hypothesis(
            **h,
            id=uuid4(),
            run_id=run_id,
            created_at=datetime.now(timezone.utc)
        ) 
        for h in hypothesis_set.hypotheses_json
    ]
    
    # Initialize priors using engine
    belief_state = belief.init_priors(
        hypotheses=hypotheses,
        strategy=request.strategy,
    )
    
    # Save to database
    saved_belief_state = repo.save_belief_state(
        db=db,
        run_id=run_id,
        belief_json=belief_state.beliefs,
        explanation_log=belief_state.explanation_log,
    )
    
    return InitPriorsResponse(
        id=norm_uuid(saved_belief_state.id),
        run_id=norm_uuid(saved_belief_state.run_id),
        belief_count=len(belief_state.beliefs),
        strategy=request.strategy,
        created_at=saved_belief_state.created_at,
    )


@router.post("/belief/update", response_model=UpdateBeliefsResponse, status_code=201)
def update_beliefs(
    run_id: UUID,
    request: UpdateBeliefsRequest,
    db: Session = Depends(get_db),
) -> UpdateBeliefsResponse:
    """
    Update beliefs using Bayesian inference.
    
    Loads latest belief state, hypotheses, graph, and signals from DB.
    Calls belief.update_beliefs() then persists new state.
    
    Args:
        run_id: Parent run UUID
        request: Update parameters (optional new signals)
        db: Database session
        
    Returns:
        Updated belief state details
        
    Raises:
        HTTPException: 404 if run not found, 400 if required artifacts missing
    """
    # Verify run exists
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Load required artifacts from database
    current_belief_state_db = repo.get_latest_belief_state(db=db, run_id=run_id)
    if not current_belief_state_db:
        raise HTTPException(
            status_code=400,
            detail=f"No belief state found for run {run_id}. Initialize priors first.",
        )
    
    hypothesis_set = repo.get_latest_hypothesis_set(db=db, run_id=run_id)
    if not hypothesis_set:
        raise HTTPException(
            status_code=400,
            detail=f"No hypotheses found for run {run_id}.",
        )
    
    hypothesis_graph_db = repo.get_latest_graph(db=db, run_id=run_id)
    if not hypothesis_graph_db:
        raise HTTPException(
            status_code=400,
            detail=f"No hypothesis graph found for run {run_id}.",
        )
    
    # Use provided signals or load from database
    if request.signals:
        signals = request.signals
    else:
        signal_set = repo.get_latest_signals(db=db, run_id=run_id)
        if not signal_set:
            raise HTTPException(
                status_code=400,
                detail=f"No signals found for run {run_id}. Provide signals or save them first.",
            )
        signals = [
            Signal(
                **s,
                id=uuid4(),
                run_id=run_id,
                created_at=datetime.now(timezone.utc)
            )
            for s in signal_set.signals_json
        ]
    
    # Convert artifacts to engine format
    hypotheses = [
        Hypothesis(
            **h,
            id=uuid4(),
            run_id=run_id,
            created_at=datetime.now(timezone.utc)
        )
        for h in hypothesis_set.hypotheses_json
    ]
    current_belief_state = belief.BeliefState(
        beliefs=current_belief_state_db.belief_json,
        explanation_log=current_belief_state_db.explanation_log,
        id=norm_uuid(current_belief_state_db.id),
        run_id=norm_uuid(current_belief_state_db.run_id),
        timestamp=current_belief_state_db.created_at,
    )
    hypothesis_graph = nx.node_link_graph(hypothesis_graph_db.graph_json)
    
    # Update beliefs using engine
    updated_belief_state = belief.update_beliefs(
        state=current_belief_state,
        signals=signals,
        graph=hypothesis_graph,
    )
    
    # Save to database
    saved_belief_state = repo.save_belief_state(
        db=db,
        run_id=run_id,
        belief_json=updated_belief_state.beliefs,
        explanation_log=updated_belief_state.explanation_log,
    )
    
    return UpdateBeliefsResponse(
        id=norm_uuid(saved_belief_state.id),
        run_id=norm_uuid(saved_belief_state.run_id),
        belief_count=len(updated_belief_state.beliefs),
        explanation_count=len(updated_belief_state.explanation_log),
        created_at=saved_belief_state.created_at,
    )


@router.post("/simulate", response_model=SimulateResponse, status_code=201)
def simulate_trajectories_endpoint(
    run_id: UUID,
    request: SimulateRequest,
    db: Session = Depends(get_db),
) -> SimulateResponse:
    """
    Run Monte Carlo trajectory simulation.
    
    Loads latest belief state, hypotheses, and graph from DB.
    Calls simulator.simulate_trajectories() then persists results.
    
    Args:
        run_id: Parent run UUID
        request: Simulation parameters
        db: Database session
        
    Returns:
        Simulation results
        
    Raises:
        HTTPException: 404 if run not found, 400 if required artifacts missing
    """
    # Verify run exists
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Load required artifacts from database
    current_belief_state_db = repo.get_latest_belief_state(db=db, run_id=run_id)
    if not current_belief_state_db:
        raise HTTPException(
            status_code=400,
            detail=f"No belief state found for run {run_id}. Initialize or update beliefs first.",
        )
    
    hypothesis_set = repo.get_latest_hypothesis_set(db=db, run_id=run_id)
    if not hypothesis_set:
        raise HTTPException(
            status_code=400,
            detail=f"No hypotheses found for run {run_id}.",
        )
    
    hypothesis_graph_db = repo.get_latest_graph(db=db, run_id=run_id)
    if not hypothesis_graph_db:
        raise HTTPException(
            status_code=400,
            detail=f"No hypothesis graph found for run {run_id}.",
        )
    
    # Convert artifacts to engine format
    hypotheses = [
        Hypothesis(
            **h,
            id=uuid4(),
            run_id=run_id,
            created_at=datetime.now(timezone.utc)
        )
        for h in hypothesis_set.hypotheses_json
    ]
    current_belief_state = belief.BeliefState(
        beliefs=current_belief_state_db.belief_json,
        explanation_log=current_belief_state_db.explanation_log,
        id=norm_uuid(current_belief_state_db.id),
        run_id=norm_uuid(current_belief_state_db.run_id),
        timestamp=current_belief_state_db.created_at,
    )
    hypothesis_graph = nx.node_link_graph(hypothesis_graph_db.graph_json)
    
    # Run simulation using engine
    simulation_result = simulator.simulate_trajectories(
        hypotheses=hypotheses,
        graph=hypothesis_graph,
        belief=current_belief_state,
        n_runs=request.params.n_runs,
        seed=request.params.seed,
        activation_threshold=request.params.activation_threshold,
        propagate_steps=request.params.propagate_steps,
        reinforce_delta=request.params.reinforce_delta,
        contradict_delta=request.params.contradict_delta,
        top_k=request.params.top_k,
    )
    
    # Convert result to dict for storage
    result_json = {
        "trajectories": [
            {
                "name": t.name,
                "probability": t.probability,
                "active_hypotheses": list(t.active_hypotheses) if isinstance(t.active_hypotheses, set) else t.active_hypotheses,
                "frequency": t.frequency,
            }
            for t in simulation_result.trajectories
        ],
        "total_runs": simulation_result.total_runs,
        "seed": simulation_result.seed,
        "hotspots": simulation_result.hotspots,
        "parameters": simulation_result.parameters,
    }
    
    params_json = request.params.model_dump()
    
    # Save to database
    simulation_output = repo.save_simulation_output(
        db=db,
        run_id=run_id,
        params_json=params_json,
        result_json=result_json,
    )
    
    return SimulateResponse(
        id=norm_uuid(simulation_output.id),
        run_id=norm_uuid(simulation_output.run_id),
        trajectory_count=len(simulation_result.trajectories),
        created_at=simulation_output.created_at,
        result=result_json,
    )

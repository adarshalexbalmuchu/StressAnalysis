"""
FastAPI routes for policy run management.

Thin routing layer: validation → repository → response.
NO business logic. NO engine calls.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.storage import repo
from app.storage.database import get_db


def norm_uuid(x) -> UUID:
    """Normalize UUID wrapper to standard uuid.UUID."""
    if isinstance(x, UUID):
        return x
    return UUID(str(x))


router = APIRouter(prefix="/runs", tags=["runs"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================


class CreateRunRequest(BaseModel):
    """Request to create a new stress-test run."""

    name: str = Field(..., description="Human-readable run name")
    policy_rule_text: str = Field(..., description="Policy rule being stress-tested")


class RunResponse(BaseModel):
    """Response with run details."""

    id: UUID
    name: str
    policy_rule_text: str
    created_at: datetime

    class Config:
        from_attributes = True


class RunListResponse(BaseModel):
    """Response with list of runs."""

    runs: list[RunResponse]
    total: int


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/", response_model=RunResponse, status_code=201)
def create_run(
    request: CreateRunRequest,
    db: Session = Depends(get_db),
) -> RunResponse:
    """
    Create a new policy stress-test run.
    
    Args:
        request: Run creation parameters
        db: Database session
        
    Returns:
        Created run details
    """
    run = repo.create_run(
        db=db,
        name=request.name,
        policy_rule_text=request.policy_rule_text,
    )
    return RunResponse(
        id=norm_uuid(run.id),
        name=run.name,
        policy_rule_text=run.policy_rule_text,
        created_at=run.created_at,
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    run_id: UUID,
    db: Session = Depends(get_db),
) -> RunResponse:
    """
    Get a run by ID.
    
    Args:
        run_id: Run UUID
        db: Database session
        
    Returns:
        Run details
        
    Raises:
        HTTPException: 404 if run not found
    """
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return RunResponse(
        id=norm_uuid(run.id),
        name=run.name,
        policy_rule_text=run.policy_rule_text,
        created_at=run.created_at,
    )


@router.get("/", response_model=RunListResponse)
def list_runs(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> RunListResponse:
    """
    List all runs, most recent first.
    
    Args:
        limit: Maximum number of runs to return (default 50)
        offset: Number of runs to skip (default 0)
        db: Database session
        
    Returns:
        List of runs with total count
    """
    runs = repo.list_runs(db=db, limit=limit, offset=offset)
    return RunListResponse(
        runs=[
            RunResponse(
                id=norm_uuid(r.id),
                name=r.name,
                policy_rule_text=r.policy_rule_text,
                created_at=r.created_at,
            )
            for r in runs
        ],
        total=len(runs),
    )


@router.get("/{run_id}/summary", response_model=dict)
def get_run_summary(
    run_id: UUID,
    db: Session = Depends(get_db)
) -> dict:
    """Get comprehensive summary of a run's artifacts.
    
    Returns latest state of all artifacts:
    - Run metadata
    - Hypotheses count + timestamp
    - Graph nodes/edges + timestamp
    - Signals count + timestamp
    - Latest belief state (top 5 hypotheses by probability) + timestamp
    - Latest simulation (top 3 trajectories) + timestamp
    
    Args:
        run_id: Run UUID
        db: Database session
        
    Returns:
        Summary dict with all artifact states
    """
    # Get run
    run = repo.get_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    summary = {
        "run": {
            "id": str(norm_uuid(run.id)),
            "name": run.name,
            "policy_rule_text": run.policy_rule_text,
            "created_at": run.created_at.isoformat(),
        },
        "hypotheses": None,
        "graph": None,
        "signals": None,
        "belief": None,
        "simulation": None,
    }
    
    # Get latest hypotheses
    hypothesis_set = repo.get_latest_hypothesis_set(db=db, run_id=run_id)
    if hypothesis_set:
        summary["hypotheses"] = {
            "count": len(hypothesis_set.hypotheses_json),
            "timestamp": hypothesis_set.created_at.isoformat(),
        }
    
    # Get latest graph
    graph = repo.get_latest_graph(db=db, run_id=run_id)
    if graph:
        graph_json = graph.graph_json
        summary["graph"] = {
            "nodes": len(graph_json.get("nodes", [])),
            "edges": len(graph_json.get("edges", [])),
            "timestamp": graph.created_at.isoformat(),
        }
    
    # Get latest signals
    signal_set = repo.get_latest_signals(db=db, run_id=run_id)
    if signal_set:
        summary["signals"] = {
            "count": len(signal_set.signals_json),
            "timestamp": signal_set.created_at.isoformat(),
        }
    
    # Get latest belief state
    belief_state = repo.get_latest_belief_state(db=db, run_id=run_id)
    if belief_state:
        beliefs = belief_state.belief_json
        # Get top 5 by probability
        sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)[:5]
        summary["belief"] = {
            "top_5": [{"hid": hid, "probability": prob} for hid, prob in sorted_beliefs],
            "timestamp": belief_state.created_at.isoformat(),
        }
    
    # Get latest simulation
    simulation = repo.get_latest_simulation(db=db, run_id=run_id)
    if simulation:
        result_json = simulation.result_json
        trajectories = result_json.get("trajectories", [])
        summary["simulation"] = {
            "top_3": [
                {
                    "name": t["name"],
                    "probability": t["probability"],
                    "active_hypotheses": t["active_hypotheses"],
                }
                for t in trajectories[:3]
            ],
            "timestamp": simulation.created_at.isoformat(),
        }
    
    return summary

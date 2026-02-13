"""
Advanced API Routes — Temporal Evolution, Comparison, Ingestion, History

These routes extend the V1 pipeline with the remaining backend capabilities:

  - POST /{id}/belief/evolve    — multi-step temporal belief evolution
  - GET  /{id}/belief/history   — retrieve all belief state snapshots
  - POST /compare/llm-vs-system — LLM-vs-System comparison experiment
  - POST /ingest/policy         — extract structured data from raw policy text
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.engine.generator import get_generator, _global_rate_limiter
from app.config import settings
from app.schemas import HypothesisBase, SignalBase, SimulationParams
from app.storage import repo
from app.storage.database import get_db


logger = logging.getLogger(__name__)

advanced_router = APIRouter(tags=["advanced"])


def norm_uuid(x) -> UUID:
    if isinstance(x, UUID):
        return x
    return UUID(str(x))


# ============================================================================
# Request / Response schemas
# ============================================================================


# --- Temporal Evolution ---

class SignalBatch(BaseModel):
    """A batch of signals for one time period."""
    label: str = Field("Period", description="Label for this time step (e.g. 'Month 1')")
    signals: list[SignalBase] = Field(..., min_length=1, description="Signals for this period")


class EvolveRequest(BaseModel):
    """Request to run multi-step temporal belief evolution."""
    signal_batches: list[SignalBatch] = Field(
        ..., min_length=1,
        description="Signal batches, one per time period",
    )
    influence_strength: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Graph propagation strength",
    )


class TimeStepResponse(BaseModel):
    step: int
    label: str
    beliefs: dict[str, float]
    entropy: float
    signals_applied: int
    explanation: list[str]


class EvolveResponse(BaseModel):
    """Response from temporal evolution."""
    run_id: UUID
    steps: list[TimeStepResponse]
    total_signals: int
    initial_entropy: float
    final_entropy: float
    convergence_delta: float
    most_changed_hid: str
    most_changed_delta: float


# --- Belief History ---

class BeliefSnapshot(BaseModel):
    id: UUID
    beliefs: dict[str, float]
    created_at: datetime
    explanation_log: list[str]


class BeliefHistoryResponse(BaseModel):
    run_id: UUID
    snapshots: list[BeliefSnapshot]
    count: int


# --- LLM vs System Comparison ---

class CompareRequest(BaseModel):
    """Request to run LLM-vs-System comparison experiment."""
    policy_text: str = Field(..., min_length=20, description="Policy text to analyze")
    domain: str = Field(default="general", description="Policy domain")
    run_id: UUID | None = Field(
        default=None,
        description="Optional existing run ID for system side. If omitted, uses stub data.",
    )
    test_consistency: bool = Field(
        default=True,
        description="Whether to re-analyze with signals to test LLM consistency",
    )


class CompareResponse(BaseModel):
    """Response from LLM-vs-System comparison."""
    llm_failure_modes: list[dict[str, Any]]
    llm_probabilities: dict[str, float]
    llm_narrative: str
    llm_overall_risk: str

    system_hypotheses: list[dict[str, Any]]
    system_beliefs: dict[str, float]
    system_trajectory_count: int
    system_top_trajectory: str
    system_top_probability: float

    consistency_score: float
    coverage_score: float
    structure_score: float
    explanation: list[str]

    consistency_test: dict[str, Any] | None = None


# --- Policy Ingestion ---

class IngestRequest(BaseModel):
    """Request to ingest a raw policy document."""
    policy_text: str = Field(..., min_length=20, description="Raw policy document text")


class IngestResponse(BaseModel):
    """Structured extraction from policy document."""
    title: str
    domain: str
    stakeholders: list[dict[str, Any]]
    rules: list[dict[str, Any]]
    mechanisms: list[dict[str, Any]]
    risk_indicators: list[dict[str, Any]]
    raw_text_length: int
    source: str
    generator_input: dict[str, Any] = Field(
        description="Pre-formatted input for /generate/hypotheses"
    )


# --- Temporal Simulation ---

class TemporalSimRequest(BaseModel):
    """Request to run temporal Monte Carlo simulation."""
    time_steps: int = Field(default=6, ge=2, le=24, description="Number of time steps")
    step_labels: list[str] | None = Field(
        default=None,
        description="Labels for each time step (e.g. ['Month 1', 'Month 2', ...])",
    )
    n_runs: int = Field(default=1000, ge=100, le=10000, description="Monte Carlo runs")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    activation_threshold: float = Field(default=0.5, ge=0.1, le=0.9)
    reinforce_delta: float = Field(default=0.15, ge=0.0, le=0.5)
    contradict_delta: float = Field(default=0.15, ge=0.0, le=0.5)
    decay_rate: float = Field(default=0.02, ge=0.0, le=0.2, description="Per-step drift toward 0.5")
    top_k: int = Field(default=5, ge=1, le=20)


class TemporalEventResponse(BaseModel):
    step: int
    label: str
    activated_hids: list[str]
    deactivated_hids: list[str]
    belief_snapshot: dict[str, float]
    cascading_effects: list[str]


class TemporalTrajectoryResponse(BaseModel):
    name: str
    events: list[TemporalEventResponse]
    final_active: list[str]
    probability: float
    frequency: int


class TemporalSimResponse(BaseModel):
    """Response from temporal trajectory simulation."""
    run_id: UUID
    trajectories: list[TemporalTrajectoryResponse]
    total_runs: int
    time_steps: int
    seed: int
    parameters: dict[str, Any]


# ============================================================================
# Helper: build generator
# ============================================================================


def _get_gen():
    return get_generator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_tokens=settings.gemini_max_tokens,
    )


# ============================================================================
# Routes: Temporal Evolution
# ============================================================================


@advanced_router.post(
    "/runs/{run_id}/belief/evolve",
    response_model=EvolveResponse,
    summary="Run multi-step temporal belief evolution",
)
def evolve_beliefs_endpoint(
    run_id: UUID,
    req: EvolveRequest,
    db: Session = Depends(get_db),
):
    """
    Evolve beliefs over multiple time periods.

    Requires: hypotheses loaded, graph built, beliefs initialized.
    Each signal batch represents a time period (e.g. a month).
    Demonstrates how weak signals accumulate over time.
    """
    from app.engine.temporal import evolve_beliefs

    # Load run artifacts
    run_obj = repo.get_run(db, run_id)
    if not run_obj:
        raise HTTPException(status_code=404, detail="Run not found")

    # Load hypotheses
    hyp_set = repo.get_latest_hypothesis_set(db, run_id)
    if not hyp_set:
        raise HTTPException(status_code=400, detail="No hypotheses loaded for this run")

    # Load graph
    graph_obj = repo.get_latest_graph(db, run_id)
    if not graph_obj:
        raise HTTPException(status_code=400, detail="No graph built for this run")

    # Load current beliefs
    belief_obj = repo.get_latest_belief_state(db, run_id)
    if not belief_obj:
        raise HTTPException(status_code=400, detail="Beliefs not initialized for this run")

    # Build hypothesis objects
    from app.schemas import Hypothesis as HypSchema
    hypotheses_raw = hyp_set.hypotheses_json
    hypotheses = []
    for h in hypotheses_raw:
        hypotheses.append(HypothesisBase(**h))

    # Build NetworkX graph
    from app.engine.graph import build_graph
    graph_data = graph_obj.graph_json
    edges = graph_data.get("edges", [])
    hids = [h.hid for h in hypotheses]
    G = build_graph(hids, edges)

    # Build signal batches
    signal_batches = []
    batch_labels = []
    for batch in req.signal_batches:
        signals = []
        for sig in batch.signals:
            from app.schemas import Signal
            signals.append(Signal(
                id=uuid4(),
                run_id=run_id,
                created_at=datetime.now(timezone.utc),
                **sig.model_dump(),
            ))
        signal_batches.append(signals)
        batch_labels.append(batch.label)

    # Get initial beliefs
    initial_beliefs = belief_obj.beliefs_json
    if isinstance(initial_beliefs, list):
        # Handle if stored as list of dicts
        initial_beliefs = {b["hid"]: b["probability"] for b in initial_beliefs}

    # Run evolution
    evolution = evolve_beliefs(
        hypotheses=hypotheses,
        graph=G,
        initial_beliefs=initial_beliefs,
        signal_batches=signal_batches,
        batch_labels=batch_labels,
        run_id=run_id,
        influence_strength=req.influence_strength,
    )

    # Convert result
    steps = [
        TimeStepResponse(
            step=s.step,
            label=s.label,
            beliefs=s.beliefs,
            entropy=s.entropy,
            signals_applied=s.signals_applied,
            explanation=s.explanation,
        )
        for s in evolution.steps
    ]

    return EvolveResponse(
        run_id=run_id,
        steps=steps,
        total_signals=evolution.total_signals,
        initial_entropy=evolution.initial_entropy,
        final_entropy=evolution.final_entropy,
        convergence_delta=evolution.convergence_delta,
        most_changed_hid=evolution.most_changed_hid,
        most_changed_delta=evolution.most_changed_delta,
    )


# ============================================================================
# Routes: Belief History
# ============================================================================


@advanced_router.get(
    "/runs/{run_id}/belief/history",
    response_model=BeliefHistoryResponse,
    summary="Retrieve belief state history for a run",
)
def belief_history(
    run_id: UUID,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    Retrieve all belief state snapshots for a run, ordered by time.

    Useful for visualizing how beliefs evolved across updates.
    """
    run_obj = repo.get_run(db, run_id)
    if not run_obj:
        raise HTTPException(status_code=404, detail="Run not found")

    states = repo.list_belief_states(db, run_id, limit=limit)

    snapshots = []
    for s in reversed(states):  # Chronological order (oldest first)
        beliefs = s.beliefs_json
        if isinstance(beliefs, list):
            beliefs = {b["hid"]: b["probability"] for b in beliefs}
        snapshots.append(BeliefSnapshot(
            id=norm_uuid(s.id),
            beliefs=beliefs,
            created_at=s.created_at,
            explanation_log=s.explanation_log_json or [],
        ))

    return BeliefHistoryResponse(
        run_id=run_id,
        snapshots=snapshots,
        count=len(snapshots),
    )


# ============================================================================
# Routes: LLM-vs-System Comparison
# ============================================================================


@advanced_router.post(
    "/compare/llm-vs-system",
    response_model=CompareResponse,
    summary="Run LLM-vs-System comparison experiment",
)
def compare_llm_vs_system(req: CompareRequest, db: Session = Depends(get_db)):
    """
    Compare raw LLM policy analysis against the structured system.

    This is the key experiment from the problem statement:
    "Show that standalone LLMs generate plausible explanations,
    but fail to maintain consistent, evolving belief states."

    If run_id is provided, uses actual system data.
    Otherwise, uses stub data for demonstration.
    """
    from app.engine.comparison import (
        run_llm_baseline,
        run_llm_reanalysis,
        evaluate_consistency,
        compare_approaches,
    )

    gen = _get_gen()

    # Step 1: Get raw LLM analysis
    try:
        llm_analysis = run_llm_baseline(gen, req.policy_text, req.domain)
    except Exception as exc:
        logger.exception("LLM baseline analysis failed")
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {exc}") from exc

    # Step 2: Get system data
    system_hypotheses = []
    system_beliefs = {}
    system_trajectories = []

    if req.run_id:
        # Load real system data
        hyp_set = repo.get_latest_hypothesis_set(db, req.run_id)
        if hyp_set:
            system_hypotheses = hyp_set.hypotheses_json

        belief_obj = repo.get_latest_belief_state(db, req.run_id)
        if belief_obj:
            beliefs = belief_obj.beliefs_json
            if isinstance(beliefs, list):
                system_beliefs = {b["hid"]: b["probability"] for b in beliefs}
            else:
                system_beliefs = beliefs

        sim_obj = repo.get_latest_simulation(db, req.run_id)
        if sim_obj and sim_obj.results_json:
            system_trajectories = sim_obj.results_json.get("trajectories", [])
    else:
        # Use stub data for demo
        system_hypotheses = [
            {"hid": "H001", "mechanism": "Biometric authentication exclusion of elderly",
             "stakeholders": ["Elderly"], "time_horizon": "immediate"},
            {"hid": "H002", "mechanism": "Administrative discretion in exemption processing",
             "stakeholders": ["FPS dealers"], "time_horizon": "short_term"},
            {"hid": "H003", "mechanism": "Technology infrastructure failures",
             "stakeholders": ["Rural beneficiaries"], "time_horizon": "immediate"},
        ]
        system_beliefs = {"H001": 0.72, "H002": 0.45, "H003": 0.61}
        system_trajectories = [
            {"name": "Exclusion cascade", "probability": 0.34,
             "active_hypotheses": ["H001", "H003"]},
            {"name": "Administrative workaround", "probability": 0.28,
             "active_hypotheses": ["H002"]},
        ]

    # Step 3: Test consistency (if requested)
    consistency_result = None
    consistency_dict = None
    if req.test_consistency:
        try:
            test_signals = [
                {"type": "field_report", "content": "50% increase in authentication failures in rural areas"},
                {"type": "media_report", "content": "News report on elderly exclusion from ration shops"},
            ]
            reanalysis = run_llm_reanalysis(gen, req.policy_text, llm_analysis, test_signals)
            consistency_result = evaluate_consistency(llm_analysis, reanalysis)
            consistency_dict = {
                "initial_probabilities": consistency_result.initial_probabilities,
                "updated_probabilities": consistency_result.updated_probabilities,
                "probability_drift": consistency_result.probability_drift,
                "ordering_preserved": consistency_result.ordering_preserved,
                "new_modes_appeared": consistency_result.new_modes_appeared,
                "modes_disappeared": consistency_result.modes_disappeared,
                "explanation": consistency_result.explanation,
            }
        except Exception as exc:
            logger.warning("Consistency test failed: %s", exc)
            consistency_dict = {"error": str(exc)}

    # Step 4: Compare approaches
    comparison = compare_approaches(
        llm_analysis=llm_analysis,
        system_hypotheses=system_hypotheses,
        system_beliefs=system_beliefs,
        system_trajectories=system_trajectories,
        llm_consistency=consistency_result,
    )

    return CompareResponse(
        llm_failure_modes=comparison.llm_failure_modes,
        llm_probabilities=comparison.llm_probabilities,
        llm_narrative=comparison.llm_narrative,
        llm_overall_risk=comparison.llm_overall_risk,
        system_hypotheses=comparison.system_hypotheses,
        system_beliefs=comparison.system_beliefs,
        system_trajectory_count=comparison.system_trajectory_count,
        system_top_trajectory=comparison.system_top_trajectory,
        system_top_probability=comparison.system_top_probability,
        consistency_score=comparison.consistency_score,
        coverage_score=comparison.coverage_score,
        structure_score=comparison.structure_score,
        explanation=comparison.explanation,
        consistency_test=consistency_dict,
    )


# ============================================================================
# Routes: Temporal Trajectory Simulation
# ============================================================================


@advanced_router.post(
    "/runs/{run_id}/simulate/temporal",
    response_model=TemporalSimResponse,
    summary="Run temporal Monte Carlo trajectory simulation",
)
def temporal_simulation_endpoint(
    run_id: UUID,
    req: TemporalSimRequest,
    db: Session = Depends(get_db),
):
    """
    Run multi-step temporal Monte Carlo simulation.

    Unlike the basic simulator (single activation snapshot), this produces
    multi-step trajectories where hypotheses activate/deactivate over time,
    graph propagation creates cascading effects, and beliefs evolve with
    natural decay toward priors.

    Requires: hypotheses loaded, graph built, beliefs initialized.
    """
    from app.engine.simulator import simulate_temporal_trajectories

    # Load run artifacts
    run_obj = repo.get_run(db, run_id)
    if not run_obj:
        raise HTTPException(status_code=404, detail="Run not found")

    hyp_set = repo.get_latest_hypothesis_set(db, run_id)
    if not hyp_set:
        raise HTTPException(status_code=400, detail="No hypotheses loaded for this run")

    graph_obj = repo.get_latest_graph(db, run_id)
    if not graph_obj:
        raise HTTPException(status_code=400, detail="No graph built for this run")

    belief_obj = repo.get_latest_belief_state(db, run_id)
    if not belief_obj:
        raise HTTPException(status_code=400, detail="Beliefs not initialized for this run")

    # Build hypothesis objects
    from app.schemas import Hypothesis as HypSchema
    hypotheses_raw = hyp_set.hypotheses_json
    hypotheses = [HypothesisBase(**h) for h in hypotheses_raw]

    # Build NetworkX graph
    from app.engine.graph import build_graph
    graph_data = graph_obj.graph_json
    edges = graph_data.get("edges", [])
    hids = [h.hid for h in hypotheses]
    G = build_graph(hids, edges)

    # Get current beliefs
    beliefs = belief_obj.beliefs_json
    if isinstance(beliefs, list):
        beliefs = {b["hid"]: b["probability"] for b in beliefs}

    # Build BeliefState
    from app.schemas import BeliefState
    belief_state = BeliefState(
        id=uuid4(),
        run_id=run_id,
        beliefs=beliefs,
        created_at=datetime.now(timezone.utc),
    )

    # Build Hypothesis objects for simulator
    from app.schemas import Hypothesis
    hyp_objects = []
    for h in hypotheses_raw:
        hyp_objects.append(Hypothesis(
            id=uuid4(),
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            **{k: v for k, v in h.items() if k not in ("id", "run_id", "created_at")},
        ))

    # Run temporal simulation
    result = simulate_temporal_trajectories(
        hypotheses=hyp_objects,
        graph=G,
        belief=belief_state,
        time_steps=req.time_steps,
        step_labels=req.step_labels,
        n_runs=req.n_runs,
        seed=req.seed,
        activation_threshold=req.activation_threshold,
        reinforce_delta=req.reinforce_delta,
        contradict_delta=req.contradict_delta,
        decay_rate=req.decay_rate,
        top_k=req.top_k,
    )

    # Convert to response
    traj_responses = []
    for t in result.trajectories:
        events = [
            TemporalEventResponse(
                step=e.step,
                label=e.label,
                activated_hids=e.activated_hids,
                deactivated_hids=e.deactivated_hids,
                belief_snapshot=e.belief_snapshot,
                cascading_effects=e.cascading_effects,
            )
            for e in t.events
        ]
        traj_responses.append(TemporalTrajectoryResponse(
            name=t.name,
            events=events,
            final_active=t.final_active,
            probability=t.probability,
            frequency=t.frequency,
        ))

    return TemporalSimResponse(
        run_id=run_id,
        trajectories=traj_responses,
        total_runs=result.total_runs,
        time_steps=result.time_steps,
        seed=result.seed,
        parameters=result.parameters,
    )


# ============================================================================
# Routes: Policy Ingestion
# ============================================================================


@advanced_router.post(
    "/ingest/policy",
    response_model=IngestResponse,
    summary="Extract structured data from raw policy text",
)
def ingest_policy_endpoint(req: IngestRequest):
    """
    Ingest a raw policy document and extract structured information.

    Extracts: stakeholders, rules, mechanisms, risk indicators.
    Output can be fed directly into /generate/hypotheses.
    """
    from app.engine.ingestor import ingest_policy

    gen = _get_gen()

    try:
        result = ingest_policy(gen, req.policy_text)
    except Exception as exc:
        logger.exception("Policy ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        title=result.title,
        domain=result.domain,
        stakeholders=[
            {"name": s.name, "role": s.role, "description": s.description}
            for s in result.stakeholders
        ],
        rules=[
            {"id": r.id, "text": r.text, "affects": r.affects, "mechanism": r.mechanism}
            for r in result.rules
        ],
        mechanisms=[
            {"name": m.name, "description": m.description, "dependencies": m.dependencies}
            for m in result.mechanisms
        ],
        risk_indicators=[
            {"phrase": ri.phrase, "risk_type": ri.risk_type, "severity": ri.severity}
            for ri in result.risk_indicators
        ],
        raw_text_length=result.raw_text_length,
        source=result.source,
        generator_input=result.to_generator_input(),
    )

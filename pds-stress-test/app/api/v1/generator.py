"""
Generator API Routes — Hypothesis & Edge Drafting via GenAI

These routes sit ALONGSIDE the existing pipeline routes.
They produce JSON drafts that users review before submitting to the
existing /runs/{id}/hypotheses and /runs/{id}/graph/validate-and-save
endpoints.  The V1 engine is never touched.

Workflow:
  1. POST /generate/hypotheses  → returns draft hypothesis JSON
  2. POST /generate/edges       → returns draft edge JSON
  3. POST /generate/full        → returns both in one call
  4. User reviews / edits output
  5. User submits to existing pipeline endpoints
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.engine.generator import (
    GeneratedEdges,
    GeneratedHypotheses,
    _global_rate_limiter,
    get_generator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generator"])


# ============================================================================
# Request / Response schemas (API-local, not stored in DB)
# ============================================================================


class GenerateHypothesesRequest(BaseModel):
    """Request to draft hypotheses from policy text."""

    policy_text: str = Field(
        ...,
        min_length=20,
        description="The policy rule / document excerpt to analyse",
    )
    domain: str = Field(
        default="general",
        description="Policy domain (e.g. PDS, education, fintech)",
    )
    num_hypotheses: int = Field(
        default=15,
        ge=3,
        le=30,
        description="Approximate number of hypotheses to generate",
    )
    stakeholder_hint: str | None = Field(
        default=None,
        description="Optional comma-separated list of key stakeholders",
    )


class GenerateEdgesRequest(BaseModel):
    """Request to draft edges from an existing hypothesis list."""

    hypotheses: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description="List of hypothesis dicts (must include 'hid' and 'mechanism')",
    )


class GenerateFullRequest(BaseModel):
    """Request to draft both hypotheses AND edges in one shot."""

    policy_text: str = Field(..., min_length=20)
    domain: str = Field(default="general")
    num_hypotheses: int = Field(default=15, ge=3, le=30)
    stakeholder_hint: str | None = None


class HypothesesResponse(BaseModel):
    """Response containing drafted hypotheses."""

    hypotheses: list[dict[str, Any]]
    count: int
    model_used: str
    token_usage: dict[str, int]
    rate_limit_remaining: int = Field(description="API calls remaining in current 60s window")
    _note: str = (
        "These are DRAFTS. Review them, then POST to "
        "/api/v1/runs/{run_id}/hypotheses to persist."
    )


class EdgesResponse(BaseModel):
    """Response containing drafted edges."""

    edges: list[dict[str, Any]]
    count: int
    model_used: str
    token_usage: dict[str, int]
    rate_limit_remaining: int = Field(description="API calls remaining in current 60s window")
    _note: str = (
        "These are DRAFTS. Review them, then POST to "
        "/api/v1/runs/{run_id}/graph/validate-and-save to persist."
    )


class FullDraftResponse(BaseModel):
    """Response containing both hypotheses and edges."""

    hypotheses: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    hypothesis_count: int
    edge_count: int
    model_used: str
    token_usage: dict[str, int]
    rate_limit_remaining: int = Field(description="API calls remaining in current 60s window")


class RateLimitStatus(BaseModel):
    """Current rate limit status."""

    remaining_requests: int
    max_requests_per_minute: int = 12
    window_seconds: int = 60
    model: str
    provider: str = "Google Gemini (free tier)"


# ============================================================================
# Helper
# ============================================================================


def _get_gen():
    """Instantiate the correct generator from settings."""
    return get_generator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_tokens=settings.gemini_max_tokens,
    )


# ============================================================================
# Routes
# ============================================================================


@router.get("/status", response_model=RateLimitStatus)
def generator_status():
    """Check generator status and remaining rate limit."""
    return RateLimitStatus(
        remaining_requests=_global_rate_limiter.remaining,
        model=settings.gemini_model,
    )


@router.post("/hypotheses", response_model=HypothesesResponse)
def draft_hypotheses(req: GenerateHypothesesRequest):
    """
    Generate hypothesis drafts from a policy text using GenAI.

    Uses Google Gemini (free tier) when GEMINI_API_KEY is set,
    otherwise falls back to an offline stub for development.

    Rate limited to 12 RPM (within Gemini free tier of 15 RPM).
    """
    try:
        gen = _get_gen()
        result: GeneratedHypotheses = gen.generate_hypotheses(
            policy_text=req.policy_text,
            domain=req.domain,
            num_hypotheses=req.num_hypotheses,
            stakeholder_hint=req.stakeholder_hint,
        )
        return HypothesesResponse(
            hypotheses=result.hypotheses,
            count=len(result.hypotheses),
            model_used=result.model_used,
            token_usage=result.token_usage,
            rate_limit_remaining=_global_rate_limiter.remaining,
        )
    except Exception as exc:
        logger.exception("Hypothesis generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/edges", response_model=EdgesResponse)
def draft_edges(req: GenerateEdgesRequest):
    """
    Generate edge (relationship) drafts between existing hypotheses.

    Requires at least 2 hypotheses with 'hid' and 'mechanism' fields.
    Rate limited to 12 RPM (within Gemini free tier of 15 RPM).
    """
    # Validate that input hypotheses have required fields
    for h in req.hypotheses:
        if "hid" not in h or "mechanism" not in h:
            raise HTTPException(
                status_code=422,
                detail="Each hypothesis must have 'hid' and 'mechanism' fields.",
            )

    try:
        gen = _get_gen()
        result: GeneratedEdges = gen.generate_edges(hypotheses=req.hypotheses)
        return EdgesResponse(
            edges=result.edges,
            count=len(result.edges),
            model_used=result.model_used,
            token_usage=result.token_usage,
            rate_limit_remaining=_global_rate_limiter.remaining,
        )
    except Exception as exc:
        logger.exception("Edge generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/full", response_model=FullDraftResponse)
def draft_full(req: GenerateFullRequest):
    """
    Generate both hypotheses AND edges in a single call.

    This is the recommended entrypoint:
      1. Generates hypotheses from policy text.
      2. Feeds them into edge generation.
      3. Returns a complete draft ready for human review.

    Uses 2 API calls. Rate limited to 12 RPM.
    """
    try:
        gen = _get_gen()

        # Step 1 — hypotheses
        hyp_result: GeneratedHypotheses = gen.generate_hypotheses(
            policy_text=req.policy_text,
            domain=req.domain,
            num_hypotheses=req.num_hypotheses,
            stakeholder_hint=req.stakeholder_hint,
        )

        # Step 2 — edges (using just-generated hypotheses)
        edge_result: GeneratedEdges = gen.generate_edges(
            hypotheses=hyp_result.hypotheses,
        )

        # Merge token usage
        total_usage: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[key] = (
                hyp_result.token_usage.get(key, 0)
                + edge_result.token_usage.get(key, 0)
            )

        return FullDraftResponse(
            hypotheses=hyp_result.hypotheses,
            edges=edge_result.edges,
            hypothesis_count=len(hyp_result.hypotheses),
            edge_count=len(edge_result.edges),
            model_used=hyp_result.model_used,
            token_usage=total_usage,
            rate_limit_remaining=_global_rate_limiter.remaining,
        )
    except Exception as exc:
        logger.exception("Full draft generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

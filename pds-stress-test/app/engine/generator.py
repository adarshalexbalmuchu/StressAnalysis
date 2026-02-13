"""
Generative Hypothesis Drafting Engine (Phase 2 — The Brain)

Uses Google Gemini (free tier) to generate structured hypothesis objects
and graph edges from raw policy text. Outputs match existing V1 Pydantic
schemas exactly, so the downstream engine (graph → belief → simulation)
is never modified.

Design Principles:
  1. GenAI is used ONLY to expand the hypothesis space — never to score
     or decide.
  2. All outputs are machine-readable JSON matching HypothesisBase / Edge.
  3. A fallback stub is provided so the rest of the system works even
     without an API key (useful for tests and offline dev).
  4. The module is pure engine logic — NO FastAPI, NO database imports.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# RATE LIMITER — enforces Gemini free-tier limits (15 RPM)
# ============================================================================

class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Gemini free tier: 15 requests per minute.
    We use 12 RPM as our ceiling to leave headroom for retries.
    """

    def __init__(self, max_calls: int = 12, window_seconds: float = 60.0):
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def wait_if_needed(self) -> float:
        """
        Block the caller until a slot is available.

        Returns:
            The number of seconds the caller was delayed (0.0 if none).
        """
        with self._lock:
            now = time.monotonic()
            # Purge timestamps older than the window
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) < self._max_calls:
                self._timestamps.append(now)
                return 0.0

            # Must wait until the oldest call exits the window
            wait_until = self._timestamps[0] + self._window
            sleep_seconds = max(0.0, wait_until - now + 0.1)  # +0.1s buffer

        logger.info("Rate limit reached — sleeping %.1fs", sleep_seconds)
        time.sleep(sleep_seconds)

        with self._lock:
            self._timestamps.append(time.monotonic())
            return sleep_seconds

    @property
    def remaining(self) -> int:
        """Requests remaining in the current window."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window
            active = [t for t in self._timestamps if t > cutoff]
            return max(0, self._max_calls - len(active))


# Singleton rate limiter shared across all GeminiClient instances
_global_rate_limiter = RateLimiter(max_calls=12, window_seconds=60.0)


# ============================================================================
# PROMPTS — kept as constants so they're easy to audit / version-control
# ============================================================================

HYPOTHESIS_SYSTEM_PROMPT = """\
You are a senior policy analyst specialising in implementation failures.
Your job is to brainstorm *mechanistic failure hypotheses* for a proposed
policy rule.  Each hypothesis must describe:
  • WHO is affected (stakeholders)
  • WHAT triggers the effect
  • HOW the stakeholder adapts (mechanism)
  • primary & secondary effects
  • time horizon (immediate / short_term / medium_term / long_term)

You must NOT recommend, judge, or score.  You ONLY generate hypotheses.

Return a JSON array.  Each element must have EXACTLY these keys:
  hid         — string, format "H001", "H002", …
  stakeholders — list of strings
  triggers     — list of strings
  mechanism    — string, one sentence
  primary_effects   — list of strings
  secondary_effects — list of strings (can be empty)
  time_horizon      — one of: "immediate", "short_term", "medium_term", "long_term"
  confidence_notes  — string or null

Return ONLY the JSON array — no markdown fences, no commentary.
"""

EDGE_SYSTEM_PROMPT = """\
You are a systems‐thinking analyst.  Given a list of policy failure
hypotheses (JSON), identify directed causal relationships between them.

Edge types:
  • "reinforces"  — activating the source makes the target MORE likely
  • "contradicts" — activating the source makes the target LESS likely
  • "depends_on"  — the target cannot activate unless the source has activated

Return a JSON array where each element has EXACTLY these keys:
  source_hid — string (e.g. "H001")
  target_hid — string (e.g. "H003")
  edge_type  — one of: "reinforces", "contradicts", "depends_on"
  notes      — string, one-sentence justification

Return ONLY the JSON array — no markdown fences, no commentary.
"""


# ============================================================================
# Data containers returned by the generator
# ============================================================================

@dataclass
class GeneratedHypotheses:
    """Container for the raw output of hypothesis generation."""

    hypotheses: list[dict[str, Any]]
    model_used: str
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class GeneratedEdges:
    """Container for the raw output of edge generation."""

    edges: list[dict[str, Any]]
    model_used: str
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_text: str = ""


# ============================================================================
# JSON extraction helper
# ============================================================================

def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """
    Robustly extract a JSON array from LLM output.

    Handles:
      • Clean JSON
      • JSON wrapped in markdown ```json ... ``` fences
      • Leading/trailing prose around the array
    """
    # Strip markdown fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        raise ValueError("Parsed JSON is not an array")
    except json.JSONDecodeError:
        pass

    # Fallback: find the first [ ... ] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON array from LLM output: {exc}") from exc

    raise ValueError("No JSON array found in LLM output")


# ============================================================================
# Gemini client wrapper (free tier)
# ============================================================================

class GeminiClient:
    """
    Thin wrapper around google-genai SDK.

    Uses Gemini 2.0 Flash by default (free: 15 RPM / 1 M tokens per day).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Core generation with rate limiting & retry
    # ------------------------------------------------------------------ #

    def _call(
        self,
        system: str,
        user: str,
        max_retries: int = 3,
    ) -> tuple[str, dict[str, int]]:
        """
        Send a prompt pair to Gemini and return (text, token_usage).

        Includes:
          • Pre-call rate limiting (12 RPM ceiling for 15 RPM free tier)
          • Exponential backoff on 429 / 503 errors (up to max_retries)
          • Token usage tracking
        """
        from google.genai import types

        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            # --- rate-limit gate ---
            delay = _global_rate_limiter.wait_if_needed()
            if delay > 0:
                logger.debug("Rate limiter delayed call by %.1fs", delay)

            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=self._temperature,
                        max_output_tokens=self._max_tokens,
                    ),
                )

                text = response.text or ""
                usage: dict[str, int] = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(um, "prompt_token_count", 0),
                        "completion_tokens": getattr(um, "candidates_token_count", 0),
                        "total_tokens": getattr(um, "total_token_count", 0),
                    }
                return text, usage

            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                is_rate_limit = "429" in err_str or "resource" in err_str or "quota" in err_str
                is_server_err = "503" in err_str or "500" in err_str

                if (is_rate_limit or is_server_err) and attempt < max_retries:
                    # Try to extract server-suggested retry delay
                    server_delay = 0
                    import re as _re
                    delay_match = _re.search(r"retry\s*(?:in|Delay['\"]:\s*['\"]?)(\d+)", str(exc))
                    if delay_match:
                        server_delay = int(delay_match.group(1))

                    backoff = max(server_delay, min(2 ** attempt * 5, 60))
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt, max_retries, type(exc).__name__, backoff,
                    )
                    time.sleep(backoff)
                else:
                    raise

        # Should not reach here, but just in case
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_hypotheses(
        self,
        policy_text: str,
        domain: str = "general",
        num_hypotheses: int = 15,
        stakeholder_hint: str | None = None,
    ) -> GeneratedHypotheses:
        """
        Generate failure/success hypotheses for a policy rule.

        Args:
            policy_text: The policy rule or document excerpt.
            domain: Domain context (e.g. "PDS", "education").
            num_hypotheses: Desired number of hypotheses (hint, not hard).
            stakeholder_hint: Optional comma-separated stakeholder list.

        Returns:
            GeneratedHypotheses with a list of dicts matching HypothesisBase.
        """
        user_prompt_parts = [
            f"**Policy rule / document:**\n{policy_text}",
            f"**Domain:** {domain}",
            f"**Generate approximately {num_hypotheses} hypotheses.**",
        ]
        if stakeholder_hint:
            user_prompt_parts.append(
                f"**Key stakeholders to consider:** {stakeholder_hint}"
            )
        user_prompt = "\n\n".join(user_prompt_parts)

        raw_text, usage = self._call(HYPOTHESIS_SYSTEM_PROMPT, user_prompt)
        hypotheses = _extract_json_array(raw_text)

        # Normalise HIDs to ensure sequential format
        for i, h in enumerate(hypotheses, start=1):
            h["hid"] = f"H{i:03d}"

        return GeneratedHypotheses(
            hypotheses=hypotheses,
            model_used=self._model_name,
            token_usage=usage,
            raw_text=raw_text,
        )

    def generate_edges(
        self,
        hypotheses: list[dict[str, Any]],
    ) -> GeneratedEdges:
        """
        Generate causal edges between a set of hypotheses.

        Args:
            hypotheses: List of hypothesis dicts (from generate_hypotheses).

        Returns:
            GeneratedEdges with a list of dicts matching Edge schema.
        """
        # Send a compact representation to save tokens
        compact = [
            {
                "hid": h["hid"],
                "mechanism": h.get("mechanism", ""),
                "stakeholders": h.get("stakeholders", []),
                "primary_effects": h.get("primary_effects", []),
            }
            for h in hypotheses
        ]

        user_prompt = (
            "Here are the hypotheses:\n\n"
            + json.dumps(compact, indent=2)
            + "\n\nIdentify all plausible directed edges between them."
        )

        raw_text, usage = self._call(EDGE_SYSTEM_PROMPT, user_prompt)
        edges = _extract_json_array(raw_text)

        # Validate that source/target HIDs exist
        valid_hids = {h["hid"] for h in hypotheses}
        validated_edges = [
            e
            for e in edges
            if e.get("source_hid") in valid_hids
            and e.get("target_hid") in valid_hids
            and e.get("edge_type") in ("reinforces", "contradicts", "depends_on")
        ]

        return GeneratedEdges(
            edges=validated_edges,
            model_used=self._model_name,
            token_usage=usage,
            raw_text=raw_text,
        )


# ============================================================================
# Offline / stub generator (works without API key)
# ============================================================================

class StubGenerator:
    """
    Deterministic fallback used when no Gemini API key is configured.

    Returns a small set of template hypotheses so the pipeline can be
    exercised end-to-end without any external calls.  Useful for tests,
    CI, and offline development.
    """

    MODEL_NAME = "stub-offline"

    def generate_hypotheses(
        self,
        policy_text: str,
        domain: str = "general",
        num_hypotheses: int = 5,
        stakeholder_hint: str | None = None,
    ) -> GeneratedHypotheses:
        """Return a small set of generic template hypotheses."""
        hypotheses = [
            {
                "hid": "H001",
                "stakeholders": ["Primary beneficiaries"],
                "triggers": ["Policy implementation"],
                "mechanism": (
                    "Beneficiaries most dependent on the system face access "
                    "barriers due to new compliance requirements."
                ),
                "primary_effects": [
                    "Increased difficulty accessing entitlements",
                    "Higher time and cost burden on vulnerable groups",
                ],
                "secondary_effects": [
                    "Loss of trust in the system",
                    "Shift to informal alternatives",
                ],
                "time_horizon": "immediate",
                "confidence_notes": "Generated by offline stub — review required",
            },
            {
                "hid": "H002",
                "stakeholders": ["Frontline administrators"],
                "triggers": ["Policy implementation", "Compliance enforcement"],
                "mechanism": (
                    "Administrators exercise discretionary exceptions that "
                    "create regional variance in policy enforcement."
                ),
                "primary_effects": [
                    "Uneven enforcement across regions",
                    "Administrative overload",
                ],
                "secondary_effects": [
                    "Corruption opportunities in exception handling",
                    "Data inconsistencies in official records",
                ],
                "time_horizon": "short_term",
                "confidence_notes": "Generated by offline stub — review required",
            },
            {
                "hid": "H003",
                "stakeholders": ["Technology providers", "System operators"],
                "triggers": ["Technology deployment", "Infrastructure gaps"],
                "mechanism": (
                    "Technology failures in under-served areas compound "
                    "existing inequalities in service delivery."
                ),
                "primary_effects": [
                    "Service disruption in low-infrastructure zones",
                    "Increased support costs",
                ],
                "secondary_effects": [
                    "Political backlash against technology mandates",
                    "Demand for rollback or exceptions",
                ],
                "time_horizon": "short_term",
                "confidence_notes": "Generated by offline stub — review required",
            },
            {
                "hid": "H004",
                "stakeholders": ["Oversight bodies", "Auditors"],
                "triggers": ["Workaround adoption", "Data inconsistencies"],
                "mechanism": (
                    "Informal workarounds erode the reliability of audit "
                    "trails, making oversight less effective."
                ),
                "primary_effects": [
                    "Reduced audit trail integrity",
                    "Difficulty distinguishing compliance from workaround",
                ],
                "secondary_effects": [
                    "Erosion of accountability mechanisms",
                    "Misplaced confidence in reported metrics",
                ],
                "time_horizon": "medium_term",
                "confidence_notes": "Generated by offline stub — review required",
            },
            {
                "hid": "H005",
                "stakeholders": ["Policy designers", "Legislature"],
                "triggers": ["Cumulative failures", "Political pressure"],
                "mechanism": (
                    "Accumulated failures and political mobilisation create "
                    "pressure for policy rollback or significant amendment."
                ),
                "primary_effects": [
                    "Calls for policy reversal",
                    "Legislative amendments or exceptions",
                ],
                "secondary_effects": [
                    "Precedent set against technology-driven mandates",
                    "Reduced political appetite for future reforms",
                ],
                "time_horizon": "long_term",
                "confidence_notes": "Generated by offline stub — review required",
            },
        ]
        return GeneratedHypotheses(
            hypotheses=hypotheses[: num_hypotheses],
            model_used=self.MODEL_NAME,
            raw_text="(offline stub)",
        )

    def generate_edges(
        self,
        hypotheses: list[dict[str, Any]],
    ) -> GeneratedEdges:
        """Return template edges for the stub hypotheses."""

        valid_hids = {h["hid"] for h in hypotheses}
        all_edges = [
            {
                "source_hid": "H001",
                "target_hid": "H002",
                "edge_type": "reinforces",
                "notes": "Beneficiary access failures create pressure on administrators",
            },
            {
                "source_hid": "H003",
                "target_hid": "H001",
                "edge_type": "reinforces",
                "notes": "Technology failures compound beneficiary access barriers",
            },
            {
                "source_hid": "H002",
                "target_hid": "H004",
                "edge_type": "reinforces",
                "notes": "Administrative workarounds erode audit capabilities",
            },
            {
                "source_hid": "H001",
                "target_hid": "H005",
                "edge_type": "reinforces",
                "notes": "Beneficiary exclusion triggers political rollback pressure",
            },
            {
                "source_hid": "H004",
                "target_hid": "H003",
                "edge_type": "contradicts",
                "notes": "Audit oversight pressures investment in technology fixes",
            },
        ]
        valid_edges = [
            e
            for e in all_edges
            if e["source_hid"] in valid_hids and e["target_hid"] in valid_hids
        ]
        return GeneratedEdges(
            edges=valid_edges,
            model_used=self.MODEL_NAME,
            raw_text="(offline stub)",
        )


# ============================================================================
# Factory — pick the right backend based on config
# ============================================================================

def get_generator(
    api_key: str = "",
    model: str = "gemini-2.0-flash",
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> GeminiClient | StubGenerator:
    """
    Return a generator instance.

    • If api_key is provided → GeminiClient (free tier).
    • Otherwise             → StubGenerator (offline deterministic).
    """
    if api_key:
        logger.info("Using Gemini generator (model=%s)", model)
        return GeminiClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    logger.warning(
        "GEMINI_API_KEY not set — falling back to offline StubGenerator. "
        "Set the env var to enable AI-powered hypothesis generation."
    )
    return StubGenerator()

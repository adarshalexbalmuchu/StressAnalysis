"""
Policy Document Ingestor

Extracts structured policy information from raw text input.
Uses Gemini to parse unstructured policy documents into:
  - Stakeholders (who is affected)
  - Policy rules / clauses
  - Mechanisms (how the policy works)
  - Domain classification

The output feeds directly into the hypothesis generator
to close the ingestion gap (Layer 1).

This module is pure engine logic — NO FastAPI, NO database imports.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


INGESTION_PROMPT = """\
You are a policy analysis expert. Extract structured information
from the following policy document.

Identify:
1. **Stakeholders**: All groups affected by this policy (direct and indirect).
   For each stakeholder, note whether they are a primary beneficiary,
   implementer, or affected party.
2. **Policy Rules**: Each distinct requirement, mandate, or regulation
   in the document. Extract the literal rule and which stakeholders it affects.
3. **Mechanisms**: How the policy works — the operational steps or processes.
4. **Domain**: Classify the domain (e.g., "Public Distribution", "Healthcare",
   "Education", "Taxation", "Environmental", "Social Welfare").
5. **Risk Indicators**: Any phrases suggesting potential failure modes —
   exclusion, delays, discretion, technological dependency, etc.

Return a JSON object with this structure:
{
  "title": "Short title for the policy",
  "domain": "domain classification",
  "stakeholders": [
    {
      "name": "group name",
      "role": "beneficiary | implementer | regulator | affected_party",
      "description": "brief description of their stake"
    }
  ],
  "rules": [
    {
      "id": "R1",
      "text": "the actual rule or requirement",
      "affects": ["stakeholder names"],
      "mechanism": "how it's enforced or implemented"
    }
  ],
  "mechanisms": [
    {
      "name": "mechanism name",
      "description": "how it works",
      "dependencies": ["what it depends on"]
    }
  ],
  "risk_indicators": [
    {
      "phrase": "quoted text from the document",
      "risk_type": "exclusion | delay | discretion | technology | equity | other",
      "severity": "high | medium | low"
    }
  ]
}

Return ONLY the JSON — no markdown fences, no commentary.
"""


@dataclass
class Stakeholder:
    name: str
    role: str  # beneficiary, implementer, regulator, affected_party
    description: str = ""


@dataclass
class PolicyRule:
    id: str
    text: str
    affects: list[str] = field(default_factory=list)
    mechanism: str = ""


@dataclass
class Mechanism:
    name: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class RiskIndicator:
    phrase: str
    risk_type: str  # exclusion, delay, discretion, technology, equity, other
    severity: str = "medium"  # high, medium, low


@dataclass
class IngestionResult:
    """Structured output from policy document ingestion."""

    title: str
    domain: str
    stakeholders: list[Stakeholder]
    rules: list[PolicyRule]
    mechanisms: list[Mechanism]
    risk_indicators: list[RiskIndicator]
    raw_text_length: int
    source: str = "manual"  # manual | upload

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "domain": self.domain,
            "stakeholders": [
                {"name": s.name, "role": s.role, "description": s.description}
                for s in self.stakeholders
            ],
            "rules": [
                {"id": r.id, "text": r.text, "affects": r.affects, "mechanism": r.mechanism}
                for r in self.rules
            ],
            "mechanisms": [
                {"name": m.name, "description": m.description, "dependencies": m.dependencies}
                for m in self.mechanisms
            ],
            "risk_indicators": [
                {"phrase": ri.phrase, "risk_type": ri.risk_type, "severity": ri.severity}
                for ri in self.risk_indicators
            ],
            "raw_text_length": self.raw_text_length,
            "source": self.source,
        }

    def to_generator_input(self) -> dict[str, Any]:
        """
        Convert ingestion result into input suitable for the hypothesis
        generator — bridges Layer 1 → Layer 2.
        """
        stakeholder_names = [s.name for s in self.stakeholders]
        policy_text = f"{self.title}. " + " ".join(r.text for r in self.rules)
        return {
            "policy_text": policy_text,
            "domain": self.domain,
            "stakeholders": stakeholder_names,
            "mechanisms": [m.name for m in self.mechanisms],
            "risk_hints": [ri.phrase for ri in self.risk_indicators],
        }


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _build_stub_result(policy_text: str) -> IngestionResult:
    """
    Deterministic stub for offline/test usage.
    Returns a plausible structure for any policy text.
    """
    # Extract simple keywords to make stub somewhat relevant
    text_lower = policy_text.lower()
    domain = "general"
    if any(w in text_lower for w in ["food", "ration", "pds", "distribution"]):
        domain = "Public Distribution"
    elif any(w in text_lower for w in ["health", "medical", "hospital"]):
        domain = "Healthcare"
    elif any(w in text_lower for w in ["education", "school", "university"]):
        domain = "Education"
    elif any(w in text_lower for w in ["tax", "revenue", "fiscal"]):
        domain = "Taxation"

    return IngestionResult(
        title="Policy Document Analysis",
        domain=domain,
        stakeholders=[
            Stakeholder("Primary Beneficiaries", "beneficiary", "Direct recipients of policy benefits"),
            Stakeholder("Implementation Agency", "implementer", "Responsible for policy execution"),
            Stakeholder("Oversight Body", "regulator", "Monitors compliance and outcomes"),
            Stakeholder("Community Groups", "affected_party", "Indirectly affected populations"),
        ],
        rules=[
            PolicyRule("R1", "Primary eligibility criteria for service access", ["Primary Beneficiaries"], "Verification at point of service"),
            PolicyRule("R2", "Implementation timeline and rollout requirements", ["Implementation Agency"], "Phased deployment schedule"),
            PolicyRule("R3", "Monitoring and reporting obligations", ["Oversight Body"], "Quarterly reporting cycle"),
        ],
        mechanisms=[
            Mechanism("Authentication", "Identity verification process", ["Technology infrastructure", "Trained personnel"]),
            Mechanism("Distribution", "Service delivery mechanism", ["Supply chain", "Local facilities"]),
            Mechanism("Grievance Redressal", "Complaint handling process", ["Administrative capacity"]),
        ],
        risk_indicators=[
            RiskIndicator("verification requirements", "exclusion", "high"),
            RiskIndicator("implementation timeline", "delay", "medium"),
            RiskIndicator("administrative discretion", "discretion", "medium"),
        ],
        raw_text_length=len(policy_text),
        source="manual",
    )


def ingest_policy(
    generator,
    policy_text: str,
) -> IngestionResult:
    """
    Ingest a raw policy document and extract structured information.

    Parameters
    ----------
    generator : GeminiClient | StubGenerator
        The LLM generator to use for extraction.
    policy_text : str
        Raw policy document text.

    Returns
    -------
    IngestionResult
        Structured extraction with stakeholders, rules, mechanisms, risks.
    """
    from app.engine.generator import StubGenerator

    if isinstance(generator, StubGenerator):
        return _build_stub_result(policy_text)

    # Real LLM call
    raw_text, _ = generator._call(
        INGESTION_PROMPT,
        f"**Policy Document:**\n\n{policy_text}",
    )
    parsed = _extract_json(raw_text)

    if not parsed:
        logger.warning("LLM returned unparseable output; falling back to stub")
        return _build_stub_result(policy_text)

    # Build structured result from LLM output
    stakeholders = [
        Stakeholder(
            name=s.get("name", "Unknown"),
            role=s.get("role", "affected_party"),
            description=s.get("description", ""),
        )
        for s in parsed.get("stakeholders", [])
    ]

    rules = [
        PolicyRule(
            id=r.get("id", f"R{i+1}"),
            text=r.get("text", ""),
            affects=r.get("affects", []),
            mechanism=r.get("mechanism", ""),
        )
        for i, r in enumerate(parsed.get("rules", []))
    ]

    mechanisms = [
        Mechanism(
            name=m.get("name", "Unknown"),
            description=m.get("description", ""),
            dependencies=m.get("dependencies", []),
        )
        for m in parsed.get("mechanisms", [])
    ]

    risk_indicators = [
        RiskIndicator(
            phrase=ri.get("phrase", ""),
            risk_type=ri.get("risk_type", "other"),
            severity=ri.get("severity", "medium"),
        )
        for ri in parsed.get("risk_indicators", [])
    ]

    return IngestionResult(
        title=parsed.get("title", "Untitled Policy"),
        domain=parsed.get("domain", "general"),
        stakeholders=stakeholders,
        rules=rules,
        mechanisms=mechanisms,
        risk_indicators=risk_indicators,
        raw_text_length=len(policy_text),
        source="manual",
    )

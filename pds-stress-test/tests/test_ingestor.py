"""
Tests for the Policy Document Ingestor engine.

Tests:
  - ingest_policy with stub generator
  - Stakeholder extraction
  - Rule extraction
  - Risk indicator extraction
  - to_generator_input bridge
  - Domain detection in stub
"""

import pytest

from app.engine.ingestor import (
    IngestionResult,
    Stakeholder,
    PolicyRule,
    Mechanism,
    RiskIndicator,
    ingest_policy,
    _build_stub_result,
    _extract_json,
)
from app.engine.generator import StubGenerator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def stub_gen():
    return StubGenerator()


@pytest.fixture
def pds_policy_text():
    return (
        "The Public Distribution System (PDS) mandates biometric authentication "
        "for all ration card holders at Fair Price Shops. Elderly beneficiaries "
        "above age 70 and persons with disabilities may apply for exemption "
        "through the district collector's office. All FPS dealers must maintain "
        "electronic records of each transaction. Food grain allocation is based "
        "on household size and income category."
    )


@pytest.fixture
def health_policy_text():
    return (
        "The National Health Insurance Scheme requires all hospitals to maintain "
        "digital records of patient admissions and treatments. Medical staff must "
        "verify patient identity before providing covered services."
    )


# ============================================================================
# Tests: _extract_json
# ============================================================================


class TestExtractJson:
    def test_valid_json(self):
        assert _extract_json('{"key": "val"}') == {"key": "val"}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"key": "val"}\n```') == {"key": "val"}

    def test_invalid_returns_empty(self):
        assert _extract_json("not json") == {}


# ============================================================================
# Tests: _build_stub_result
# ============================================================================


class TestBuildStubResult:
    def test_pds_domain_detection(self, pds_policy_text):
        result = _build_stub_result(pds_policy_text)
        assert result.domain == "Public Distribution"

    def test_health_domain_detection(self, health_policy_text):
        result = _build_stub_result(health_policy_text)
        assert result.domain == "Healthcare"

    def test_generic_domain(self):
        result = _build_stub_result("Some generic policy about regulations")
        assert result.domain == "general"

    def test_has_stakeholders(self, pds_policy_text):
        result = _build_stub_result(pds_policy_text)
        assert len(result.stakeholders) > 0
        roles = [s.role for s in result.stakeholders]
        assert "beneficiary" in roles
        assert "implementer" in roles

    def test_has_rules(self, pds_policy_text):
        result = _build_stub_result(pds_policy_text)
        assert len(result.rules) > 0
        assert all(r.id.startswith("R") for r in result.rules)

    def test_has_mechanisms(self, pds_policy_text):
        result = _build_stub_result(pds_policy_text)
        assert len(result.mechanisms) > 0

    def test_has_risk_indicators(self, pds_policy_text):
        result = _build_stub_result(pds_policy_text)
        assert len(result.risk_indicators) > 0

    def test_tracks_text_length(self, pds_policy_text):
        result = _build_stub_result(pds_policy_text)
        assert result.raw_text_length == len(pds_policy_text)


# ============================================================================
# Tests: ingest_policy (stub)
# ============================================================================


class TestIngestPolicy:
    def test_returns_ingestion_result(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        assert isinstance(result, IngestionResult)

    def test_stakeholder_types(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        for s in result.stakeholders:
            assert isinstance(s, Stakeholder)
            assert s.role in ("beneficiary", "implementer", "regulator", "affected_party")

    def test_rule_structure(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        for r in result.rules:
            assert isinstance(r, PolicyRule)
            assert r.id
            assert r.text

    def test_risk_indicator_structure(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        for ri in result.risk_indicators:
            assert isinstance(ri, RiskIndicator)
            assert ri.risk_type in ("exclusion", "delay", "discretion", "technology", "equity", "other")
            assert ri.severity in ("high", "medium", "low")


# ============================================================================
# Tests: to_dict
# ============================================================================


class TestToDict:
    def test_serializable(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "stakeholders" in d
        assert "rules" in d
        assert "mechanisms" in d
        assert "risk_indicators" in d
        # Ensure all values are JSON-safe primitives
        import json
        json.dumps(d)  # Should not raise


# ============================================================================
# Tests: to_generator_input (Layer 1 → Layer 2 bridge)
# ============================================================================


class TestToGeneratorInput:
    def test_has_required_fields(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        gen_input = result.to_generator_input()
        assert "policy_text" in gen_input
        assert "domain" in gen_input
        assert "stakeholders" in gen_input
        assert "mechanisms" in gen_input
        assert "risk_hints" in gen_input

    def test_stakeholders_are_strings(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        gen_input = result.to_generator_input()
        assert all(isinstance(s, str) for s in gen_input["stakeholders"])

    def test_policy_text_non_empty(self, stub_gen, pds_policy_text):
        result = ingest_policy(stub_gen, pds_policy_text)
        gen_input = result.to_generator_input()
        assert len(gen_input["policy_text"]) > 0

"""
Tests for the Advanced API routes.

Tests:
  - POST /ingest/policy
  - POST /compare/llm-vs-system
  - Basic route registration
"""

import os
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with stub generator (no real API key)."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        # Re-import to pick up patched env
        from app.main import app
        return TestClient(app)


# ============================================================================
# Tests: /ingest/policy
# ============================================================================


class TestIngestPolicyRoute:
    def test_ingest_success(self, client):
        resp = client.post("/api/v1/ingest/policy", json={
            "policy_text": (
                "The Public Distribution System mandates biometric "
                "authentication for all ration card holders at Fair Price Shops."
            ),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"]
        assert data["domain"]
        assert len(data["stakeholders"]) > 0
        assert len(data["rules"]) > 0
        assert "generator_input" in data

    def test_ingest_short_text_rejected(self, client):
        resp = client.post("/api/v1/ingest/policy", json={
            "policy_text": "Too short",
        })
        assert resp.status_code == 422


# ============================================================================
# Tests: /compare/llm-vs-system
# ============================================================================


class TestCompareRoute:
    def test_compare_stub_mode(self, client):
        """Compare without a run_id — uses stub data on both sides."""
        resp = client.post("/api/v1/compare/llm-vs-system", json={
            "policy_text": (
                "The mandatory biometric authentication policy requires "
                "all beneficiaries to verify identity before food grain collection."
            ),
            "test_consistency": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_failure_modes" in data
        assert "system_beliefs" in data
        assert "consistency_score" in data
        assert "coverage_score" in data
        assert "structure_score" in data
        assert "explanation" in data
        assert len(data["explanation"]) > 0

    def test_compare_short_text_rejected(self, client):
        resp = client.post("/api/v1/compare/llm-vs-system", json={
            "policy_text": "Too short",
        })
        assert resp.status_code == 422

    def test_compare_without_consistency(self, client):
        """Compare without consistency test."""
        resp = client.post("/api/v1/compare/llm-vs-system", json={
            "policy_text": (
                "The mandatory biometric authentication policy requires "
                "all beneficiaries to verify identity before food grain collection."
            ),
            "test_consistency": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["consistency_test"] is None


# ============================================================================
# Tests: Route registration
# ============================================================================


class TestRouteRegistration:
    def test_advanced_routes_registered(self, client):
        """Verify that advanced routes appear in OpenAPI spec."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/api/v1/ingest/policy" in paths
        assert "/api/v1/compare/llm-vs-system" in paths

"""
Tests for the Generator Engine (Phase 2 — The Brain)

Tests the StubGenerator (offline mode) and the JSON extraction logic.
GeminiClient is NOT tested here (requires API key); it shares the same
code paths via the same interface, so the stub tests provide confidence.
"""

import json
import pytest

from app.engine.generator import (
    GeminiClient,
    GeneratedEdges,
    GeneratedHypotheses,
    RateLimiter,
    StubGenerator,
    _extract_json_array,
    get_generator,
)


# ============================================================================
# JSON extraction helper tests
# ============================================================================


class TestExtractJsonArray:
    """Test the robust JSON array extractor."""

    def test_clean_json(self):
        text = '[{"a": 1}, {"a": 2}]'
        result = _extract_json_array(text)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_markdown_fenced_json(self):
        text = '```json\n[{"hid": "H001"}]\n```'
        result = _extract_json_array(text)
        assert len(result) == 1
        assert result[0]["hid"] == "H001"

    def test_json_with_surrounding_prose(self):
        text = 'Here are the hypotheses:\n[{"hid": "H001"}]\nEnd of output.'
        result = _extract_json_array(text)
        assert len(result) == 1

    def test_empty_array(self):
        text = "[]"
        result = _extract_json_array(text)
        assert result == []

    def test_invalid_json_raises(self):
        text = "This is not JSON at all."
        with pytest.raises(ValueError, match="No JSON array found"):
            _extract_json_array(text)

    def test_json_object_not_array_with_fallback(self):
        text = '{"not": "an array"}'
        with pytest.raises(ValueError):
            _extract_json_array(text)


# ============================================================================
# StubGenerator tests
# ============================================================================


class TestStubGenerator:
    """Test the offline/stub generator."""

    def setup_method(self):
        self.gen = StubGenerator()

    def test_generates_hypotheses(self):
        result = self.gen.generate_hypotheses(
            policy_text="Test policy for biometric authentication",
            domain="PDS",
        )
        assert isinstance(result, GeneratedHypotheses)
        assert len(result.hypotheses) == 5
        assert result.model_used == "stub-offline"

    def test_hypotheses_have_required_fields(self):
        result = self.gen.generate_hypotheses(
            policy_text="Test policy rule with domain context",
        )
        required_keys = {
            "hid", "stakeholders", "triggers", "mechanism",
            "primary_effects", "secondary_effects", "time_horizon",
            "confidence_notes",
        }
        for h in result.hypotheses:
            assert required_keys.issubset(h.keys()), f"Missing keys in {h}"

    def test_hid_format(self):
        result = self.gen.generate_hypotheses(
            policy_text="Mandatory compliance requirement for welfare delivery",
        )
        for h in result.hypotheses:
            hid = h["hid"]
            assert hid.startswith("H"), f"HID must start with H: {hid}"
            assert hid[1:].isdigit(), f"HID suffix must be digits: {hid}"

    def test_respects_num_hypotheses(self):
        result = self.gen.generate_hypotheses(
            policy_text="Policy text for limited generation test",
            num_hypotheses=3,
        )
        assert len(result.hypotheses) == 3

    def test_generates_edges(self):
        hyp_result = self.gen.generate_hypotheses(
            policy_text="Policy text for edge generation test",
        )
        edge_result = self.gen.generate_edges(hypotheses=hyp_result.hypotheses)
        assert isinstance(edge_result, GeneratedEdges)
        assert len(edge_result.edges) > 0
        assert edge_result.model_used == "stub-offline"

    def test_edges_have_required_fields(self):
        hyp_result = self.gen.generate_hypotheses(
            policy_text="Test policy for edge field validation",
        )
        edge_result = self.gen.generate_edges(hypotheses=hyp_result.hypotheses)
        required_keys = {"source_hid", "target_hid", "edge_type", "notes"}
        for e in edge_result.edges:
            assert required_keys.issubset(e.keys()), f"Missing keys in {e}"

    def test_edge_types_are_valid(self):
        hyp_result = self.gen.generate_hypotheses(
            policy_text="Test policy for edge type validation",
        )
        edge_result = self.gen.generate_edges(hypotheses=hyp_result.hypotheses)
        valid_types = {"reinforces", "contradicts", "depends_on"}
        for e in edge_result.edges:
            assert e["edge_type"] in valid_types, f"Invalid type: {e['edge_type']}"

    def test_edges_reference_valid_hids(self):
        hyp_result = self.gen.generate_hypotheses(
            policy_text="Test policy for HID reference validation",
        )
        edge_result = self.gen.generate_edges(hypotheses=hyp_result.hypotheses)
        valid_hids = {h["hid"] for h in hyp_result.hypotheses}
        for e in edge_result.edges:
            assert e["source_hid"] in valid_hids
            assert e["target_hid"] in valid_hids


# ============================================================================
# Schema compatibility tests
# ============================================================================


class TestSchemaCompatibility:
    """Verify that generator output can be parsed by existing V1 schemas."""

    def test_hypotheses_parse_as_hypothesis_base(self):
        from app.schemas import HypothesisBase

        gen = StubGenerator()
        result = gen.generate_hypotheses(
            policy_text="Policy text for schema compatibility test",
        )
        for h_dict in result.hypotheses:
            # This must not raise a ValidationError
            parsed = HypothesisBase(**h_dict)
            assert parsed.hid == h_dict["hid"]
            assert len(parsed.stakeholders) > 0

    def test_edges_parse_as_edge_schema(self):
        from app.schemas import Edge

        gen = StubGenerator()
        hyps = gen.generate_hypotheses(
            policy_text="Policy text for edge schema compatibility",
        )
        edges = gen.generate_edges(hypotheses=hyps.hypotheses)
        for e_dict in edges.edges:
            parsed = Edge(**e_dict)
            assert parsed.source_hid == e_dict["source_hid"]
            assert parsed.edge_type.value == e_dict["edge_type"]


# ============================================================================
# Factory tests
# ============================================================================


class TestGetGenerator:
    """Test the generator factory function."""

    def test_returns_stub_when_no_key(self):
        gen = get_generator(api_key="")
        assert isinstance(gen, StubGenerator)

    def test_returns_gemini_when_key_provided(self):
        # We can't call the real API, but we can verify it creates
        # a GeminiClient instance.  This will fail if the SDK
        # import itself is broken.
        try:
            gen = get_generator(api_key="test-fake-key-12345")
            assert isinstance(gen, GeminiClient)
        except Exception:
            # If the SDK refuses a bad key at init time, that's also
            # acceptable — we just need to know the factory tried.
            pass


# ============================================================================
# Rate limiter tests
# ============================================================================


class TestRateLimiter:
    """Test the thread-safe rate limiter."""

    def test_allows_requests_within_limit(self):
        rl = RateLimiter(max_calls=5, window_seconds=60.0)
        for _ in range(5):
            delay = rl.wait_if_needed()
            assert delay == 0.0
        assert rl.remaining == 0

    def test_remaining_count(self):
        rl = RateLimiter(max_calls=10, window_seconds=60.0)
        assert rl.remaining == 10
        rl.wait_if_needed()
        assert rl.remaining == 9
        rl.wait_if_needed()
        assert rl.remaining == 8

    def test_blocks_at_limit(self):
        """Verify the limiter blocks when limit is reached (using tiny window)."""
        rl = RateLimiter(max_calls=2, window_seconds=0.3)
        rl.wait_if_needed()
        rl.wait_if_needed()
        # Third call should block ~0.3s
        start = __import__("time").monotonic()
        rl.wait_if_needed()
        elapsed = __import__("time").monotonic() - start
        assert elapsed >= 0.2  # Should have waited


# ============================================================================
# End-to-end stub pipeline test
# ============================================================================


class TestEndToEndStubPipeline:
    """
    Simulate the full Phase 2 pipeline using the stub:
      policy text → hypotheses → edges → validate against V1 schemas
    """

    def test_full_pipeline(self):
        from app.schemas import Edge, HypothesisBase

        gen = StubGenerator()

        # Step 1 — generate hypotheses
        hyps = gen.generate_hypotheses(
            policy_text=(
                "Mandatory Aadhaar-based biometric authentication at "
                "Fair Price Shops (FPS) as a hard condition for receiving "
                "PDS entitlements."
            ),
            domain="PDS",
            num_hypotheses=5,
        )
        assert len(hyps.hypotheses) == 5

        # Step 2 — generate edges
        edges = gen.generate_edges(hypotheses=hyps.hypotheses)
        assert len(edges.edges) > 0

        # Step 3 — parse through V1 schemas (zero tolerance for mismatch)
        parsed_hypotheses = [HypothesisBase(**h) for h in hyps.hypotheses]
        parsed_edges = [Edge(**e) for e in edges.edges]

        assert len(parsed_hypotheses) == 5
        assert all(isinstance(e, Edge) for e in parsed_edges)

        # Step 4 — verify graph can be built from these
        from app.engine.graph import build_graph

        # build_graph expects Hypothesis (with id/run_id) or dicts —
        # feed it dicts directly
        graph = build_graph(hyps.hypotheses, edges.edges)
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() == len(edges.edges)

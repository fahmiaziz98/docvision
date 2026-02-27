"""
Unit tests for AgenticWorkflow reflect pattern (graph.py).

Critic node now uses Pydantic structured output (CriticOutput).
Mock response.choices[0].message.parsed instead of raw JSON string.
"""

import warnings

import pytest
from langgraph.types import Command

from docvision.workflows.graph import AgenticWorkflow, CriticOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_response(content: str):
    """Mock VLM response for generate/refine nodes (plain text content)."""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].message.parsed = None
    return resp


def _make_critic_response(score: int, issues: list, needs_revision: bool):
    """Mock VLM response for critic node (structured output via .parsed)."""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.choices[0].message.content = ""   # not used for structured output
    resp.choices[0].message.parsed = CriticOutput(
        score=score,
        issues=issues,
        needs_revision=needs_revision,
    )
    return resp


# ---------------------------------------------------------------------------
# Tests: CriticOutput Pydantic model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCriticOutput:

    def test_valid_instantiation(self):
        obj = CriticOutput(score=8, issues=["minor gap"], needs_revision=False)
        assert obj.score == 8
        assert obj.issues == ["minor gap"]
        assert obj.needs_revision is False

    def test_score_out_of_range_raises(self):
        with pytest.raises(Exception):
            CriticOutput(score=11, issues=[], needs_revision=False)

    def test_score_negative_raises(self):
        with pytest.raises(Exception):
            CriticOutput(score=-1, issues=[], needs_revision=False)

    def test_empty_issues_default(self):
        obj = CriticOutput(score=9, issues=[], needs_revision=False)
        assert obj.issues == []

    def test_field_descriptions_exist(self):
        """Ensure all fields have descriptions for LLM guidance."""
        schema = CriticOutput.model_json_schema()
        props = schema.get("properties", {})
        for field_name in ("score", "issues", "needs_revision"):
            assert field_name in props, f"Missing field: {field_name}"
            assert "description" in props[field_name], (
                f"Field '{field_name}' missing description — LLM won't know what to fill"
            )


# ---------------------------------------------------------------------------
# Tests: AgenticWorkflow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgenticWorkflow:

    def test_max_reflect_cycles_warning(self):
        from unittest.mock import MagicMock
        client = MagicMock()
        with pytest.warns(UserWarning, match="max_reflect_cycles"):
            AgenticWorkflow(vlm_client=client, max_reflect_cycles=3)

    def test_no_warning_at_max_2(self):
        from unittest.mock import MagicMock
        client = MagicMock()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AgenticWorkflow(vlm_client=client, max_reflect_cycles=2)

    @pytest.mark.asyncio
    async def test_happy_path_no_revision(self):
        """generate → critic (score=9) → complete. Only 2 API calls."""
        from unittest.mock import AsyncMock, MagicMock

        call_count = 0
        responses = [
            _make_text_response("<transcription>Clean output</transcription>"),
            _make_critic_response(9, [], False),
        ]

        async def mock_invoke(**kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        client = MagicMock()
        client.invoke = mock_invoke

        workflow = AgenticWorkflow(vlm_client=client, max_reflect_cycles=2)
        state = await workflow.run(image_b64="b64", mime_type="image/jpeg")

        assert state["accumulated_text"] == "Clean output"
        assert state["critic_score"] == 9
        assert state["reflect_iteration"] == 0
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_one_reflect_cycle(self):
        """generate → critic (score=6) → refine → critic (score=9) → complete."""
        from unittest.mock import MagicMock

        responses = [
            _make_text_response("<transcription>Rough output</transcription>"),
            _make_critic_response(6, ["table cut off mid-row"], True),
            _make_text_response("<transcription>Fixed output</transcription>"),
            _make_critic_response(9, [], False),
        ]
        call_idx = 0

        async def mock_invoke(**kwargs):
            nonlocal call_idx
            resp = responses[call_idx]
            call_idx += 1
            return resp

        client = MagicMock()
        client.invoke = mock_invoke

        workflow = AgenticWorkflow(vlm_client=client, max_reflect_cycles=2)
        state = await workflow.run(image_b64="b64", mime_type="image/jpeg")

        assert state["accumulated_text"] == "Fixed output"
        assert state["critic_score"] == 9
        assert state["reflect_iteration"] == 1
        assert call_idx == 4

    @pytest.mark.asyncio
    async def test_max_cycles_stops_loop(self):
        """Stops after max_reflect_cycles even if critic still wants revision."""
        from unittest.mock import MagicMock

        # generate, critic(bad), refine, critic(still bad) — max=2, second cycle stops
        responses = [
            _make_text_response("<transcription>Output</transcription>"),
            _make_critic_response(5, ["issue persists"], True),
            _make_text_response("<transcription>Refined output</transcription>"),
            _make_critic_response(5, ["still an issue"], True),  # should stop here
        ]
        call_idx = 0

        async def mock_invoke(**kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        client = MagicMock()
        client.invoke = mock_invoke

        workflow = AgenticWorkflow(vlm_client=client, max_reflect_cycles=1)
        state = await workflow.run(image_b64="b64", mime_type="image/jpeg")

        assert state["reflect_iteration"] <= 1

    @pytest.mark.asyncio
    async def test_critic_structured_output_is_passed_as_schema(self):
        """Verify output_schema=CriticOutput is forwarded to VLMClient.invoke()."""
        from unittest.mock import MagicMock, call

        call_kwargs = []

        async def mock_invoke(**kwargs):
            call_kwargs.append(kwargs)
            # First call = generate (text), second = critic (structured)
            if len(call_kwargs) == 1:
                return _make_text_response("<transcription>text</transcription>")
            return _make_critic_response(10, [], False)

        client = MagicMock()
        client.invoke = mock_invoke

        workflow = AgenticWorkflow(vlm_client=client, max_reflect_cycles=1)
        await workflow.run(image_b64="b64", mime_type="image/jpeg")

        # Second call (critic) must have output_schema=CriticOutput
        critic_call = call_kwargs[1]
        assert critic_call.get("output_schema") is CriticOutput

    @pytest.mark.asyncio
    async def test_critic_uses_low_temperature(self):
        """Critic node must pass temperature_override=0.1 for deterministic output."""
        call_kwargs = []

        async def mock_invoke(**kwargs):
            call_kwargs.append(kwargs)
            if len(call_kwargs) == 1:
                return _make_text_response("<transcription>text</transcription>")
            return _make_critic_response(9, [], False)

        from unittest.mock import MagicMock
        client = MagicMock()
        client.invoke = mock_invoke

        workflow = AgenticWorkflow(vlm_client=client, max_reflect_cycles=1)
        await workflow.run(image_b64="b64", mime_type="image/jpeg")

        critic_call = call_kwargs[1]
        assert critic_call.get("temperature_override") == 0.1

    @pytest.mark.asyncio
    async def test_empty_response_completes_gracefully(self):
        """Empty VLM response should not crash the workflow."""
        from unittest.mock import MagicMock

        responses = [
            _make_text_response(""),
            _make_critic_response(10, [], False),
        ]
        call_idx = 0

        async def mock_invoke(**kwargs):
            nonlocal call_idx
            resp = responses[min(call_idx, len(responses) - 1)]
            call_idx += 1
            return resp

        client = MagicMock()
        client.invoke = mock_invoke

        workflow = AgenticWorkflow(vlm_client=client, max_reflect_cycles=1)
        state = await workflow.run(image_b64="b64", mime_type="image/jpeg")

        assert state["accumulated_text"] == ""
        assert state["reflect_iteration"] == 0
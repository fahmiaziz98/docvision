import warnings
from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from ..core import (
    CRITIC_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    REFINE_PROMPT,
    VLMClient,
)
from ..core.types import AgenticParseState, CriticOutput
from ..utils import extract_response_text, extract_transcription


class AgenticWorkflow:
    """
    Self-correcting document parsing workflow with critic/refiner reflect pattern.

    Flow:
        generate → critic → [refine → critic]* → complete

    Routing is handled inside nodes via Command objects (modern LangGraph pattern).
    The critic uses Pydantic structured output — no JSON parsing needed.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        system_prompt: Optional[str] = None,
        max_reflect_cycles: int = 2,
    ):
        if max_reflect_cycles > 2:
            warnings.warn(
                f"max_reflect_cycles={max_reflect_cycles} exceeds the recommended maximum of 2. "
                "Token cost multiplier is approximately 4-5x per extra cycle.",
                UserWarning,
                stacklevel=2,
            )

        self._client = vlm_client
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_reflect_cycles = max_reflect_cycles
        self._graph = self._build_graph()

    async def run(self, image_b64: str, mime_type: str) -> Dict[str, Any]:
        """
        Run the full reflect workflow for a single page image.

        Returns final state dict with:
        - accumulated_text: final parsed content
        - critic_score: last critic score
        - critic_issues: last issues found
        - reflect_iteration: number of cycles run
        """
        initial_state = self._make_initial_state(image_b64, mime_type)
        return await self._graph.ainvoke(initial_state)

    def _build_graph(self) -> StateGraph:
        """
        Build and compile the LangGraph state graph.

        Routing happens inside nodes via Command — no add_conditional_edges needed.
        START goes directly to generate (no redundant start node).
        """
        graph = StateGraph(AgenticParseState)

        graph.add_node(
            "generate",
            self._node_generate,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0),
        )
        graph.add_node(
            "critic",
            self._node_critic,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0),
        )
        graph.add_node(
            "refine",
            self._node_refine,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0),
        )
        graph.add_node("complete", self._node_complete)

        graph.add_edge(START, "generate")
        graph.add_edge("generate", "critic")
        graph.add_edge("refine", "critic")
        graph.add_edge("complete", END)

        return graph.compile()

    async def _node_generate(self, state: AgenticParseState) -> Dict[str, Any]:
        """
        Generator node — initial VLM parse of the document image.

        Extracts content from <transcription> tags if present.
        """
        response = await self._client.invoke(
            image_b64=state["image_b64"],
            mime_type=state["mime_type"],
            system_prompt=self._system_prompt,
            user_prompt=state["current_prompt"],
        )

        raw = extract_response_text(response)
        content = extract_transcription(raw) or raw

        return {
            "accumulated_text": content,
            "generation_history": [content],
            "iteration_count": state["iteration_count"] + 1,
        }

    async def _node_critic(
        self, state: AgenticParseState
    ) -> Command[Literal["refine", "complete"]]:
        """
        Critic node — evaluate structural completeness, then route via Command.

        Uses Pydantic structured output (CriticOutput) — schema enforced at API level.
        Uses temperature_override=0.1 for deterministic evaluation.

        Returns Command with both the state update AND the routing decision,
        following the modern LangGraph pattern (no separate conditional edge needed).
        """
        critic_user_prompt = (
            f"Here is the parsed document output to evaluate:```\n{state['accumulated_text']}\n```"
        )

        response = await self._client.invoke(
            image_b64=state["image_b64"],
            mime_type=state["mime_type"],
            system_prompt=CRITIC_PROMPT,
            user_prompt=critic_user_prompt,
            output_schema=CriticOutput,
            temperature_override=0.1,
        )

        critic_result: CriticOutput = response.choices[0].message.parsed

        if critic_result is None:
            return Command(
                update={"critic_score": 0, "critic_issues": []},
                goto="complete",
            )

        needs_revision = critic_result.score < 9 and bool(critic_result.issues)
        cycles_remaining = state["reflect_iteration"] < self._max_reflect_cycles

        goto = "refine" if (needs_revision and cycles_remaining) else "complete"

        return Command(
            update={
                "critic_score": critic_result.score,
                "critic_issues": critic_result.issues,
            },
            goto=goto,
        )

    async def _node_refine(self, state: AgenticParseState) -> Dict[str, Any]:
        """
        Refiner node — targeted fix based on critic issues.

        Fixes ONLY the listed structural issues. Does not rewrite
        content that the critic did not flag.
        """
        issues_text = (
            "\n".join(f"- {issue}" for issue in state["critic_issues"])
            or "- General structural improvements needed"
        )

        refine_user_prompt = REFINE_PROMPT.format(
            issues=issues_text,
            current_output=state["accumulated_text"],
        )

        response = await self._client.invoke(
            image_b64=state["image_b64"],
            mime_type=state["mime_type"],
            system_prompt=self._system_prompt,
            user_prompt=refine_user_prompt,
        )

        raw = extract_response_text(response)
        refined = extract_transcription(raw) or raw

        return {
            "accumulated_text": refined,
            "generation_history": [refined],
            "reflect_iteration": state["reflect_iteration"] + 1,
        }

    async def _node_complete(self, state: AgenticParseState) -> Dict[str, Any]:
        """
        Terminal node — required waypoint before END.
        LangGraph conditional edges via Command cannot point directly to END.
        Intentionally empty; extend here for post-processing if needed.
        """
        return {}

    def _make_initial_state(self, image_b64: str, mime_type: str) -> AgenticParseState:
        """Build a clean initial state for the workflow."""
        return AgenticParseState(
            image_b64=image_b64,
            mime_type=mime_type,
            accumulated_text="",
            iteration_count=0,
            current_prompt=DEFAULT_USER_PROMPT,
            generation_history=[],
            critic_score=0,
            critic_issues=[],
            reflect_iteration=0,
        )

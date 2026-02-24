from typing import Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from ..core import DEFAULT_SYSTEM_PROMPT, VLMClient
from ..core.types import AgenticParseState
from ..utils import (
    # check_max_tokens_hit,
    # detect_retention_loop,
    extract_transcription,
    # has_complete_transcription,
)

MAX_ITERATIONS = 3


class AgenticWorkflow:
    """
    Self-correcting agentic workflow for document parsing using LangGraph.

    This workflow manages the interaction with a Vision Language Model (VLM) to robustly
    extract text from documents. It handles common issues such as:
    - **Token limits**: Automatically continues generation if the output is truncated.
    - **Repetition loops**: Detects and corrects repetitive text generation loops.
    - **Incomplete outputs**: Ensures XML tags are properly closed.

    Attributes:
        vlm_client (VLMClient): The client used to communicate with the VLM.
        system_prompt (str): The system instructions for the VLM.
        user_prompt (str): The initial user prompt template.
        graph (CompiledGraph): The compiled LangGraph executable workflow.
    """

    def __init__(
        self,
        vlm_client: "VLMClient",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_iterations: int = 3,
    ):
        """
        Initialize the AgenticWorkflow.

        Args:
            vlm_client: An instance of VLMClient for making API calls.
            system_prompt: Optional override for the system prompt. Defaults to XML extraction rules.
            user_prompt: Optional override for the initial user prompt.
            max_iterations: Maximum number of iterations to run the workflow.
        """
        self.vlm_client = vlm_client
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_iterations = max_iterations

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Construct and compile the state graph.

        Returns:
            A compiled StateGraph ready for execution.
        """
        builder = StateGraph(AgenticParseState)

        builder.add_node("start", self._start_node)
        builder.add_node(
            "generate",
            self._generate_node,
            retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0),
        )
        builder.add_node("complete", self._complete_node)

        builder.add_edge(START, "start")
        builder.add_edge("start", "generate")

        return builder.compile()

    async def _start_node(self, state: AgenticParseState) -> Dict:
        """
        Initialize the parsing session state.

        Args:
            state: The initial input state.

        Returns:
            A dictionary updating the state with initial values.
        """
        return {
            "iteration_count": 0,
            "accumulated_text": "",
            "current_prompt": self.user_prompt,
            "generation_history": [],
        }

    async def _generate_node(
        self,
        state: AgenticParseState,
    ) -> Command[Literal["generate", "complete"]]:
        """
        The core generation node that calls the VLM and decides the next step.

        It implements the routing logic based on the response analysis:
        1. Checks for safety/iteration limits.
        2. Detects repetition loops.
        3. Checks for completion logic (valid XML tags).
        4. Checks for token truncation.

        Args:
            state: The current state of the parsing workflow.

        Returns:
            A Command object directing the graph to either 'generate' again or 'complete'.
        """
        iteration = state["iteration_count"] + 1

        # Safety limit: Prevent infinite loops
        if iteration > self.max_iterations:
            transcription = extract_transcription(state["accumulated_text"])
            return Command(
                update={
                    "accumulated_text": transcription or state["accumulated_text"],
                    "iteration_count": iteration,
                },
                goto="complete",
            )

        response = await self.vlm_client.invoke(
            image_b64=state["image_b64"],
            mime_type=state["mime_type"],
            system_prompt=self.system_prompt,
            user_prompt=state["current_prompt"],
        )

        response_text = response.choices[0].message.content
        accumulated = state["accumulated_text"] + response_text

        # ========== ROUTING LOGIC ==========

        # Check 1: Repetition loop detection
        text_before_loop = extract_transcription(accumulated)
        if text_before_loop is not None:
            restart_from = (
                text_before_loop[-500:] if len(text_before_loop) > 500 else text_before_loop
            )
            return Command(
                update={
                    "accumulated_text": text_before_loop,
                    "iteration_count": iteration,
                    "current_prompt": DEFAULT_SYSTEM_PROMPT.format(restart_from=restart_from),
                    "generation_history": state.get("generation_history", []) + [response_text],
                },
                goto="generate",
            )

        # Check 2: Successful completion (closing tag found)
        if extract_transcription(accumulated):
            return Command(
                update={
                    "accumulated_text": accumulated,
                    "iteration_count": iteration,
                    "generation_history": state.get("generation_history", []) + [response_text],
                },
                goto="complete",
            )

        # Check 3: Max tokens hit (truncation)
        if extract_transcription(response):
            context = accumulated[-300:] if len(accumulated) > 300 else accumulated
            return Command(
                update={
                    "accumulated_text": accumulated,
                    "iteration_count": iteration,
                    "current_prompt": DEFAULT_SYSTEM_PROMPT.format(context=context),
                    "generation_history": state.get("generation_history", []) + [response_text],
                },
                goto="generate",
            )

        # Check 4: Response finished but incomplete (missing closing tag, no explicit error)
        # This acts as a fallback "continue" mechanism
        context = accumulated[-300:] if len(accumulated) > 300 else accumulated
        return Command(
            update={
                "accumulated_text": accumulated,
                "iteration_count": iteration,
                "current_prompt": DEFAULT_SYSTEM_PROMPT.format(context=context),
                "generation_history": state.get("generation_history", []) + [response_text],
            },
            goto="generate",
        )

    async def _complete_node(self, state: AgenticParseState) -> Command[Literal[END]]:
        """
        Final processing node to extract clean content.

        Args:
            state: The final accumulation state.

        Returns:
            Command to END the workflow with the cleaned text.
        """
        final_text = extract_transcription(state["accumulated_text"])

        return Command(
            update={
                "accumulated_text": final_text or state["accumulated_text"],
            },
            goto=END,
        )

    async def run(self, image_b64: str, mime_type: str) -> dict:
        """
        Execute the workflow on an image.

        Args:
            image_b64: Base64 encoded image string.
            mime_type: MIME type of the image.

        Returns:
            The final state dictionary containing the parsed text.
        """
        result = await self.graph.ainvoke(
            {
                "image_b64": image_b64,
                "mime_type": mime_type,
            }
        )

        return result

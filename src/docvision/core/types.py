import operator
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, List, Optional, TypedDict

from pydantic import BaseModel, Field


class ParsingMode(str, Enum):
    """Available document parsing modes."""

    BASIC_OCR = "ocr_engine"  # ocr parsing
    VLM = "parse_with_vlm"  # Standard single-shot VLM parsing
    AGENTIC = "parse_with_agent"  # Iterative/agentic parsing for long or complex documents


class RotationAngle(int, Enum):
    """Available rotation angles."""

    NONE = 0
    CLOCKWISE_90 = 90
    COUNTER_CLOCKWISE_90 = -90
    ROTATE_180 = 180


@dataclass
class RotationResult:
    """Result of an image rotation."""

    angle: RotationAngle
    confidence: float
    original_angle: float
    applied_rotation: bool


@dataclass
class DocumentMetadata:
    """
    Core metadata for a document.
    """

    file_name: Optional[str] = None
    total_pages: int = 0


@dataclass
class ParseResult:
    """Representation of the result from parsing a single document page."""

    id: str
    content: str
    metadata: dict[str, Any]


class AgenticParseState(TypedDict):
    """
    State dictionary for the agentic reflect parsing workflow.

    Attributes:
        image_b64: Base64 encoded image of the page being parsed.
        mime_type: MIME type of the image.
        accumulated_text: Total text extracted so far.
        iteration_count: Current generator iteration number.
        current_prompt: Prompt for the current generator call.
        generation_history: List tracking incremental generation output.

        --- Reflect pattern fields ---
        critic_score: Quality score from critic agent (0-10).
        critic_issues: Specific issues identified by critic.
        reflect_iteration: Number of reflect cycles completed.
    """

    image_b64: str
    mime_type: str
    accumulated_text: str
    iteration_count: int
    current_prompt: str
    generation_history: Annotated[List[str], operator.add]

    # Reflect pattern
    critic_score: int
    critic_issues: List[str]
    reflect_iteration: int


class CriticOutput(BaseModel):
    score: int = Field(
        ge=0, le=10, description="Completeness score (0-10). 8-10: OK, 0-7: Broken/Incomplete."
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific structural issues (e.g., 'table cut off', 'missing pipes', 'duplicates').",
    )
    needs_revision: bool = Field(description="True if score < 8 and fixes are required.")

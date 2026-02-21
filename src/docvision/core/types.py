import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, List, Optional, TypedDict


class ParsingMode(str, Enum):
    """Available document parsing modes."""

    PDF = "pdf_parsing"  # Native parsing
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
class PageBlock:
    """
    A single content block extracted from a page.
    Tracks its vertical position for reading-order reconstruction.
    """

    kind: str  # "text" | "table" | "pseudo_table" | "chart"
    content: str  # Markdown-formatted string
    y_top: float  # Top y-coordinate (for ordering)
    y_bottom: float  # Bottom y-coordinate
    metadata: dict = field(default_factory=dict)


@dataclass
class NativeParseResult:
    """
    Result from NativePDFParser for a single page.
    """

    markdown: str  # Full page as Markdown string
    metadata: dict[str, Any]  # Structured metadata
    page_number: int
    has_tables: bool
    has_pseudo_tables: bool
    has_charts: bool
    block_count: int


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
    State dictionary used to maintain context during an agentic/iterative parsing loop.

    Attributes:
        image_b64: Base64 encoded image of the page being parsed.
        mime_type: MIME type of the image.
        accumulated_text: The total text extracted so far across iterations.
        iteration_count: Current iteration number.
        current_prompt: The prompt being used for the current iteration.
        generation_history: List of strings tracking the incremental growth of the output.
    """

    image_b64: str
    mime_type: str
    accumulated_text: str
    iteration_count: int
    current_prompt: str
    generation_history: Annotated[List[str], operator.add]

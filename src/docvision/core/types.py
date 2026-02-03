import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, TypedDict


class ImageFormat(str, Enum):
    """Supported image formats for export and processing."""

    PNG = "png"
    JPEG = "jpeg"


class ParsingMode(str, Enum):
    """Available document parsing modes."""

    VLM = "parse_with_vlm"  # Standard single-shot VLM parsing
    AGENTIC = "parse_with_agent"  # Iterative/agentic parsing for long or complex documents


@dataclass
class ParseResult:
    """
    Representation of the result from parsing a single document page.

    Attributes:
        content: The extracted text/markdown content.
        page_number: The index of the page in the source document.
        processing_time: Time taken to process the page in seconds.
        metadata: Additional page-level metadata.
    """

    content: str
    page_number: int
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchParseResult:
    """
    Consolidated result from parsing multiple pages or documents.

    Attributes:
        results: List of individual ParseResult objects.
        total_pages: Total number of pages processed.
        total_time: Cumulative processing time in seconds.
        success_count: Number of pages successfully parsed.
        error_count: Number of pages that failed to parse.
        errors: List of error details for failed pages.
    """

    results: List[ParseResult]
    total_pages: int
    total_time: float
    success_count: int
    error_count: int
    errors: List[Dict[str, Any]] = field(default_factory=list)


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

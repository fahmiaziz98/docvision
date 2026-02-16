import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict


class ImageFormat(str, Enum):
    """Supported image formats for export and processing."""

    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    WEBP = "webp"


class RotationAngle(int, Enum):
    """Available rotation angles."""

    NONE = 0
    CLOCKWISE_90 = 90
    COUNTER_CLOCKWISE_90 = -90
    ROTATE_180 = 180


class ParsingMode(str, Enum):
    """Available document parsing modes."""

    VLM = "parse_with_vlm"  # Standard single-shot VLM parsing
    AGENTIC = "parse_with_agent"  # Iterative/agentic parsing for long or complex documents


@dataclass
class ParserConfig:
    """Unified configuration for DocumentParsingAgent."""

    # VLM Client settings
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    timeout: float = 300.0
    temperature: float = 0.1
    max_tokens: int = 2048

    # Prompts
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    # PDF Rendering
    render_zoom: float = 2.0  # DPI equivalent (300 DPI = ~2.0 zoom)
    post_crop_max_size: int = 2048
    image_format: ImageFormat = ImageFormat.JPEG
    jpeg_quality: int = 95

    # Rotation settings
    enable_auto_rotate: bool = True
    aggressive_mode: bool = True
    use_aspect_ratio_fallback: bool = True
    hough_threshold: int = 200
    min_score_diff: float = 0.15
    analysis_max_size: int = 1500

    # Content cropping
    enable_crop: bool = True
    crop_padding: int = 10
    crop_ignore_bottom_percent: float = 12.0
    crop_max_crop_percent: float = 30.0

    # Debug
    debug_save_path: Optional[str] = None
    progress_callback: Optional[Callable[[int, int], None]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError(
                f"Invalid JPEG quality: {self.jpeg_quality}. Must be between 1 and 100."
            )


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
class RotationResult:
    """
    Representation of the result from rotating an image.

    Attributes:
        angle: The angle by which the image was rotated.
        confidence: The confidence level of the rotation detection.
        original_angle: The original angle detected by the rotation detection algorithm.
        applied_rotation: Whether the rotation was applied to the image.
    """

    angle: RotationAngle
    confidence: float
    original_angle: float
    applied_rotation: bool


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

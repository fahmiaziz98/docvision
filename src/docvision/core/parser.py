import asyncio
import time
from pathlib import Path
from typing import Literal, Optional, Type, Union

from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from ..processing import ImageProcessor
from ..workflows import AgenticWorkflow
from ..workflows.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from .client import VLMClient
from .types import BatchParseResult, ImageFormat, ParseResult, ParsingMode


class DocumentParsingAgent:
    """
    Production-ready document parser using Vision Language Models.

    Supports two parsing modes:
    - VLM: Fast single-shot parsing (default)
    - AGENTIC: Self-correcting multi-turn workflow for maximum quality

    Attributes:
        image_format: Format to use for image encoding (png/jpeg).
        jpeg_quality: Quality setting for JPEG encoding.
        system_prompt: System prompt for transcription instructions.
        user_prompt: Initial user prompt for transcription.
        processor: ImageProcessor instance for PDF/Image handling.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        timeout: float = 300.0,
        dpi: int = 300,
        auto_crop: bool = False,
        resize: bool = True,
        auto_rotate: bool = False,
        max_dimension: int = 2048,
        image_format: Literal["png", "jpeg"] = "jpeg",
        jpeg_quality: int = 95,
        crop_padding: int = 10,
        crop_ignore_bottom_percent: float = 12.0,
        crop_footer_gap_threshold: int = 100,
        crop_column_ink_ratio: float = 0.01,
        crop_row_ink_ratio: float = 0.002,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        debug_save_path: Optional[str] = None,
    ):
        """
        Initialize the DocumentParsingAgent.

        Args:
            base_url: Base URL for OpenAI-compatible API.
            api_key: API key for authentication.
            model_name: Name of the VLM model.
            timeout: Request timeout in seconds.
            dpi: DPI for PDF to image conversion.
            auto_crop: Enable content-aware cropping.
            resize: Enable image resizing.
            auto_rotate: Enable automatic orientation correction.
            max_dimension: Max width/height for processed images.
            image_format: Encoding format.
            jpeg_quality: Quality for JPEG encoding.
            crop_padding: Padding for cropper.
            crop_ignore_bottom_percent: Footer ignore height %.
            crop_footer_gap_threshold: Gap threshold for footer detection.
            crop_column_ink_ratio: Column ink sensitivity.
            crop_row_ink_ratio: Row ink sensitivity.
            system_prompt: Custom system prompt.
            user_prompt: Custom initial user prompt.
            temperature: VLM sampling temperature.
            max_tokens: Max tokens for VLM response.
            debug_save_path: Directory to save debug images.
        """
        self.api_key = api_key or "EMPTY"
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_prompt = user_prompt or DEFAULT_USER_PROMPT

        self._vlm_client = VLMClient(
            base_url=base_url,
            api_key=self.api_key,
            model_name=model_name,
            timeout=int(timeout),
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=3,
            retry_delay=2.0,
        )

        self.processor = ImageProcessor(
            dpi=dpi,
            auto_crop=auto_crop,
            resize=resize,
            auto_rotate=auto_rotate,
            max_dimension=max_dimension,
            crop_padding=crop_padding,
            crop_ignore_bottom_percent=crop_ignore_bottom_percent,
            crop_footer_gap_threshold=crop_footer_gap_threshold,
            crop_column_ink_ratio=crop_column_ink_ratio,
            crop_row_ink_ratio=crop_row_ink_ratio,
            debug_save_path=debug_save_path,
        )

        self._agentic_workflow = None

    def _get_agentic_workflow(self) -> AgenticWorkflow:
        """Lazy initialization of agentic workflow."""
        if self._agentic_workflow is None:
            self._agentic_workflow = AgenticWorkflow(
                vlm_client=self._vlm_client,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
        return self._agentic_workflow

    def parse_image(
        self,
        image: Union[str, Path, Image.Image],
        mode: ParsingMode = ParsingMode.VLM,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ParseResult:
        """
        Parse a single image (Synchronous).

        Args:
            image: Path to image file or PIL Image object.
            mode: Parsing mode (VLM or AGENTIC).
            output_schema: Optional Pydantic model for structured output (VLM mode only).

        Returns:
            A ParseResult object containing the output and metadata.
        """
        if mode == ParsingMode.AGENTIC:
            raise NotImplementedError("Agentic mode requires async. Use aparse_image() instead.")

        start_time = time.time()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        processed_img = self.processor.process_image(image, page_num=0, doc_name="image")

        img_b64, mime_type = self.processor.encode_image(
            processed_img, ImageFormat(self.image_format), self.jpeg_quality
        )

        response = self._vlm_client.call(
            img_b64,
            mime_type,
            self.system_prompt,
            self.user_prompt,
            output_schema=output_schema,
        )

        content = response.choices[0].message.content if response and response.choices else ""
        processing_time = time.time() - start_time

        return ParseResult(
            content=content,
            page_number=0,
            processing_time=processing_time,
            metadata={
                "image_size": processed_img.size,
                "mime_type": mime_type,
                "mode": mode.value,
                "structured": bool(output_schema),
            },
        )

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        mode: ParsingMode = ParsingMode.VLM,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> BatchParseResult:
        """
        Parse a PDF document page by page (Synchronous).

        Args:
            pdf_path: Path to the PDF file.
            mode: Parsing mode (VLM or AGENTIC).
            start_page: First page index to parse.
            end_page: Last page index to parse.
            output_schema: Optional Pydantic model for structured output (VLM mode only).

        Returns:
            A BatchParseResult containing results for all pages.
        """
        if mode == ParsingMode.AGENTIC:
            raise NotImplementedError("Agentic mode requires async. Use aparse_pdf() instead.")

        batch_start = time.time()

        images = self.processor.pdf_to_images(pdf_path, start_page, end_page)
        total_pages = len(images)

        results = []
        errors = []
        success_count = 0

        doc_name = Path(pdf_path).stem

        for idx, img in tqdm(enumerate(images), total=total_pages, desc="Processing pages"):
            page_num = (start_page or 0) + idx

            try:
                processed_img = self.processor.process_image(img, page_num, doc_name)
                img_b64, mime_type = self.processor.encode_image(
                    processed_img, ImageFormat(self.image_format), self.jpeg_quality
                )

                page_start = time.time()
                response = self._vlm_client.call(
                    img_b64,
                    mime_type,
                    self.system_prompt,
                    self.user_prompt,
                    output_schema=output_schema,
                )

                content = (
                    response.choices[0].message.content if response and response.choices else ""
                )
                processing_time = time.time() - page_start

                result = ParseResult(
                    content=content,
                    page_number=page_num,
                    processing_time=processing_time,
                    metadata={
                        "image_size": processed_img.size,
                        "mime_type": mime_type,
                        "mode": mode.value,
                        "structured": bool(output_schema),
                    },
                )

                results.append(result)
                success_count += 1

            except Exception as e:
                errors.append({"page": page_num, "error": str(e), "type": type(e).__name__})

        total_time = time.time() - batch_start

        return BatchParseResult(
            results=results,
            total_pages=total_pages,
            total_time=total_time,
            success_count=success_count,
            error_count=len(errors),
            errors=errors,
        )

    async def aparse_image(
        self,
        image: Union[str, Path, Image.Image],
        mode: ParsingMode = ParsingMode.VLM,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ParseResult:
        """
        Parse a single image (Asynchronous).

        Args:
            image: Path to image file or PIL Image object.
            mode: Parsing mode (VLM or AGENTIC).
            output_schema: Optional Pydantic model for structured output (VLM mode only).

        Returns:
            A ParseResult object.
        """
        if mode == ParsingMode.AGENTIC and output_schema is not None:
            raise ValueError("output_schema is only supported in VLM mode")

        start_time = time.time()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        processed_img = self.processor.process_image(image, page_num=0, doc_name="image")

        img_b64, mime_type = self.processor.encode_image(
            processed_img, ImageFormat(self.image_format), self.jpeg_quality
        )

        if mode == ParsingMode.VLM:
            response = await self._vlm_client.acall(
                img_b64,
                mime_type,
                self.system_prompt,
                self.user_prompt,
                output_schema=output_schema,
            )
            content = response.choices[0].message.content if response and response.choices else ""
            iterations = 1
            generation_history = []

        elif mode == ParsingMode.AGENTIC:
            workflow = self._get_agentic_workflow()
            result = await workflow.run(img_b64, mime_type)
            content = result["accumulated_text"]
            iterations = result["iteration_count"]
            generation_history = result["generation_history"]

        else:
            raise ValueError(f"Invalid parsing mode: {mode}")

        processing_time = time.time() - start_time

        return ParseResult(
            content=content,
            page_number=0,
            processing_time=processing_time,
            metadata={
                "image_size": processed_img.size,
                "mime_type": mime_type,
                "mode": mode.value,
                "iterations": iterations,
                "generation_history": generation_history,
                "structured": bool(output_schema),
            },
        )

    async def aparse_pdf(
        self,
        pdf_path: Union[str, Path],
        mode: ParsingMode = ParsingMode.VLM,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        max_concurrent: int = 3,
    ) -> BatchParseResult:
        """
        Parse a PDF document with optional concurrency (Asynchronous).

        Args:
            pdf_path: Path to the PDF file.
            mode: Parsing mode (VLM or AGENTIC).
            start_page: First page index to parse.
            end_page: Last page index to parse.
            output_schema: Optional Pydantic model for structured output (VLM mode only).
            max_concurrent: Maximum number of concurrent page parsing tasks.

        Returns:
            A BatchParseResult object.
        """
        if mode == ParsingMode.AGENTIC and output_schema is not None:
            raise ValueError("output_schema is only supported in VLM mode")

        batch_start = time.time()

        images = self.processor.pdf_to_images(pdf_path, start_page, end_page)
        total_pages = len(images)

        results = []
        errors = []

        semaphore = asyncio.Semaphore(max_concurrent)
        doc_name = Path(pdf_path).stem

        workflow = self._get_agentic_workflow() if mode == ParsingMode.AGENTIC else None

        async def process_page(idx: int, img: Image.Image):
            page_num = (start_page or 0) + idx

            async with semaphore:
                try:
                    processed_img = self.processor.process_image(img, page_num, doc_name)
                    img_b64, mime_type = self.processor.encode_image(
                        processed_img, ImageFormat(self.image_format), self.jpeg_quality
                    )

                    page_start = time.time()

                    if mode == ParsingMode.VLM:
                        response = await self._vlm_client.acall(
                            img_b64,
                            mime_type,
                            self.system_prompt,
                            self.user_prompt,
                            output_schema=output_schema,
                        )
                        content = (
                            response.choices[0].message.content
                            if response and response.choices
                            else ""
                        )
                        iterations = 1
                        generation_history = []

                    elif mode == ParsingMode.AGENTIC:
                        result = await workflow.run(img_b64, mime_type)
                        content = result["accumulated_text"]
                        iterations = result["iteration_count"]
                        generation_history = result["generation_history"]

                    else:
                        raise ValueError(f"Invalid mode: {mode.value}")

                    processing_time = time.time() - page_start

                    return ParseResult(
                        content=content,
                        page_number=page_num,
                        processing_time=processing_time,
                        metadata={
                            "image_size": processed_img.size,
                            "mime_type": mime_type,
                            "mode": mode.value,
                            "iterations": iterations,
                            "generation_history": generation_history,
                            "structured": bool(output_schema),
                        },
                    )

                except Exception as e:
                    return {
                        "error": True,
                        "page": page_num,
                        "message": str(e),
                        "type": type(e).__name__,
                    }

        tasks = [process_page(idx, img) for idx, img in enumerate(images)]
        page_results = []
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing pages"):
            result = await coro
            page_results.append(result)

        for result in page_results:
            if isinstance(result, dict) and result.get("error"):
                errors.append(
                    {
                        "page": result["page"],
                        "error": result["message"],
                        "type": result["type"],
                    }
                )
            else:
                results.append(result)

        total_time = time.time() - batch_start

        return BatchParseResult(
            results=results,
            total_pages=total_pages,
            total_time=total_time,
            success_count=len(results),
            error_count=len(errors),
            errors=errors,
        )

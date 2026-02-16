import asyncio
import time
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from ..processing.image_v2 import ImageProcessor
from ..workflows import AgenticWorkflow
from ..workflows.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from .client import VLMClient
from .types import BatchParseResult, ParserConfig, ParseResult, ParsingMode


class DocumentParsingAgent:
    """
    Production-ready document parser using Vision Language Models.

    Supports two parsing modes:
    - VLM: Fast single-shot parsing (default)
    - AGENTIC: Self-correcting multi-turn workflow for maximum quality

    Attributes:
        config: Unified configuration for parser and image processing.
        processor: ImageProcessor instance for PDF/Image handling.
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize the DocumentParsingAgent.

        Args:
            config: ParserConfig instance with all settings.
                    If None, uses default configuration.
        """
        self.config = config or ParserConfig()

        # Set default prompts if not provided
        if self.config.system_prompt is None:
            self.config.system_prompt = DEFAULT_SYSTEM_PROMPT
        if self.config.user_prompt is None:
            self.config.user_prompt = DEFAULT_USER_PROMPT

        self._vlm_client = VLMClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key or "EMPTY",
            model_name=self.config.model_name,
            timeout=int(self.config.timeout),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            max_retries=3,
            retry_delay=2.0,
        )

        self.processor = ImageProcessor(config=self.config)

        self._agentic_workflow = None

    def _get_agentic_workflow(self) -> AgenticWorkflow:
        """Lazy initialization of agentic workflow."""
        if self._agentic_workflow is None:
            self._agentic_workflow = AgenticWorkflow(
                vlm_client=self._vlm_client,
                system_prompt=self.config.system_prompt,
                user_prompt=self.config.user_prompt,
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

        if output_schema and mode == ParsingMode.AGENTIC:
            raise ValueError("output_schema is only supported in VLM mode")

        start_time = time.time()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        processed_img = self.processor.process_image(image)

        img_b64, mime_type = ImageProcessor.encode_to_base64(
            processed_img, self.config.image_format, self.config.jpeg_quality
        )

        response = self._vlm_client.call(
            img_b64,
            mime_type,
            self.config.system_prompt,
            self.config.user_prompt,
            output_schema=output_schema,
        )

        content = response.choices[0].message.content if response and response.choices else ""
        processing_time = time.time() - start_time

        # Get image size from numpy array
        img_size = (
            (processed_img.shape[1], processed_img.shape[0])
            if isinstance(processed_img, np.ndarray)
            else processed_img.size
        )

        return ParseResult(
            content=content,
            page_number=0,
            processing_time=processing_time,
            metadata={
                "image_size": img_size,
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
            start_page: First page index to parse (1-indexed).
            end_page: Last page index to parse (1-indexed).
            output_schema: Optional Pydantic model for structured output (VLM mode only).

        Returns:
            A BatchParseResult containing results for all pages.
        """
        if mode == ParsingMode.AGENTIC:
            raise NotImplementedError("Agentic mode requires async. Use aparse_pdf() instead.")

        if output_schema and mode == ParsingMode.AGENTIC:
            raise ValueError("output_schema is only supported in VLM mode")

        batch_start = time.time()

        images = self.processor.pdf_to_images(pdf_path, start_page, end_page)
        total_pages = len(images)

        results = []
        errors = []
        success_count = 0

        for idx, img in tqdm(enumerate(images), total=total_pages, desc="Processing pages"):
            page_num = (start_page or 1) + idx

            try:
                processed_img = self.processor.process_image(img, page_num=page_num)
                img_b64, mime_type = ImageProcessor.encode_to_base64(
                    processed_img, self.config.image_format, self.config.jpeg_quality
                )

                page_start = time.time()
                response = self._vlm_client.call(
                    img_b64,
                    mime_type,
                    self.config.system_prompt,
                    self.config.user_prompt,
                    output_schema=output_schema,
                )

                content = (
                    response.choices[0].message.content if response and response.choices else ""
                )
                processing_time = time.time() - page_start

                # Get image size from numpy array
                img_size = (
                    (processed_img.shape[1], processed_img.shape[0])
                    if isinstance(processed_img, np.ndarray)
                    else processed_img.size
                )

                result = ParseResult(
                    content=content,
                    page_number=page_num,
                    processing_time=processing_time,
                    metadata={
                        "image_size": img_size,
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

        processed_img = self.processor.process_image(image)

        img_b64, mime_type = ImageProcessor.encode_to_base64(
            processed_img, self.config.image_format, self.config.jpeg_quality
        )

        if mode == ParsingMode.VLM:
            response = await self._vlm_client.acall(
                img_b64,
                mime_type,
                self.config.system_prompt,
                self.config.user_prompt,
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

        # Get image size from numpy array
        img_size = (
            (processed_img.shape[1], processed_img.shape[0])
            if isinstance(processed_img, np.ndarray)
            else processed_img.size
        )

        return ParseResult(
            content=content,
            page_number=0,
            processing_time=processing_time,
            metadata={
                "image_size": img_size,
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
            start_page: First page index to parse (1-indexed).
            end_page: Last page index to parse (1-indexed).
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

        workflow = self._get_agentic_workflow() if mode == ParsingMode.AGENTIC else None

        async def process_page(idx: int, img: np.ndarray):
            page_num = (start_page or 1) + idx

            async with semaphore:
                try:
                    processed_img = self.processor.process_image(img, page_num=page_num)
                    img_b64, mime_type = ImageProcessor.encode_to_base64(
                        processed_img, self.config.image_format, self.config.jpeg_quality
                    )

                    page_start = time.time()

                    if mode == ParsingMode.VLM:
                        response = await self._vlm_client.acall(
                            img_b64,
                            mime_type,
                            self.config.system_prompt,
                            self.config.user_prompt,
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

                    # Get image size from numpy array
                    img_size = (
                        (processed_img.shape[1], processed_img.shape[0])
                        if isinstance(processed_img, np.ndarray)
                        else processed_img.size
                    )

                    return ParseResult(
                        content=content,
                        page_number=page_num,
                        processing_time=processing_time,
                        metadata={
                            "image_size": img_size,
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

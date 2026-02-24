import asyncio
import copy
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import fitz
import numpy as np
from pydantic import BaseModel

from ..processing import ImageProcessor
from ..utils.helper import extract_transcription

# from ..workflows.graph import AgenticWorkflow
from .client import VLMClient
from .types import DocumentMetadata, ParseResult, ParsingMode


class DocumentParser:
    """
    A high-level parser for documents (PDFs and Images).

    Supports three parsing modes:
    - BASIC_OCR: PaddleOCR ONNX-based parsing.
    - VLM: Single-shot Vision Language Model parsing.
    - AGENTIC: Self-correcting VLM parsing with critic/refine reflect pattern.
    """

    def __init__(
        self,
        # --- VLM config (required for VLM and AGENTIC modes) ---
        vlm_base_url: Optional[str] = None,
        vlm_model: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        max_iterations: int = 3,
        max_reflect_cycles: int = 2,
        ocr_language: str = "english",
        ocr_model_dir: Optional[Union[str, Path]] = None,
        enable_rotate: bool = True,
        rotate_aggressive_mode: bool = False,
        enable_deskew: bool = True,
        render_zoom: float = 2.0,
        post_crop_max_size: int = 1024,
        max_concurrency: int = 5,
        debug_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the DocumentParser.

        Args:
            vlm_base_url: Base URL for the VLM API.
            vlm_model: Model name to use for vision tasks.
            vlm_api_key: API key for the VLM service.
            temperature: Sampling temperature for VLM calls.
            max_tokens: Maximum tokens to generate per call.
            system_prompt: System prompt override for VLM calls.
            max_iterations: Maximum generator iterations in AGENTIC mode.
            max_reflect_cycles: Maximum reflect (critic/refine) cycles.
                                 Values above 2 will emit a warning.
            ocr_language: Language for BASIC_OCR mode. Default is 'english'.
                          Supported: 'english', 'latin', 'chinese', 'korean',
                          'arabic', 'hindi', 'tamil', 'telugu'.
            ocr_model_dir: Custom path to OCR model directory.
                           If None, models are auto-downloaded to
                           ~/.cache/docvision/models/ on first use.
            enable_rotate: Whether to automatically correct image orientation.
            rotate_aggressive_mode: Aggressive mode rotate
            enable_deskew: Whether to correct small skew angles for OCR.
            render_zoom: DPI multiplier for PDF rendering via fitz.
            post_crop_max_size: Max image dimension after preprocessing (VLM mode).
            max_concurrency: Max concurrent pages being processed.
            debug_dir: Directory to save debug images (optional).
        """
        # Validate max_reflect_cycles
        if max_reflect_cycles > 2:
            warnings.warn(
                f"max_reflect_cycles={max_reflect_cycles} exceeds the recommended maximum of 2. "
                "This will increase token usage significantly.",
                UserWarning,
                stacklevel=2,
            )

        self.max_concurrency = max_concurrency
        self.ocr_language = ocr_language
        self.ocr_model_dir = ocr_model_dir
        self.enable_deskew = enable_deskew

        if vlm_base_url and vlm_model and vlm_api_key:
            self._client = VLMClient(
                base_url=vlm_base_url,
                api_key=vlm_api_key,
                model_name=vlm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            self._client = None

        self.system_prompt = system_prompt

        # self._agentic_workflow = (
        #     AgenticWorkflow(
        #         vlm_client=self._client,
        #         system_prompt=system_prompt,
        #         max_iterations=max_iterations,
        #         # max_reflect_cycles=max_reflect_cycles,
        #     )
        #     if self._client
        #     else None
        # )

        self._image_processor = ImageProcessor(
            render_zoom=render_zoom,
            enable_rotate=enable_rotate,
            rotate_aggressive_mode=rotate_aggressive_mode,
            post_crop_max_size=post_crop_max_size,
            debug_dir=debug_dir,
        )

        self._ocr_engine = None

    def _get_ocr_engine(self):
        """
        Lazy-initialize the OCR engine on first use.
        Models are downloaded automatically if not present.
        """
        if self._ocr_engine is None:
            from ..processing.ocr_engine import OCREngine

            self._ocr_engine = OCREngine(
                language=self.ocr_language,
                ocr_model_dir=self.ocr_model_dir,
                enable_deskew=self.enable_deskew,
            )
        return self._ocr_engine

    async def parse_image(
        self,
        image: Union[str, Path, np.ndarray],
        parsing_mode: ParsingMode = ParsingMode.BASIC_OCR,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        save_path: Optional[Union[str, Path]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ParseResult:
        """
        Parse a single image into Markdown or structured JSON.

        Supported modes:
        - ParsingMode.VLM: Vision Language Model.
        - ParsingMode.BASIC_OCR (default): PaddleOCR ONNX, no VLM credentials required.

        Note: ParsingMode.AGENTIC is not supported for single images.
              Use parse_pdf() with ParsingMode.AGENTIC instead.

        Args:
            image: Image source — file path or numpy BGR array.
            parsing_mode: ParsingMode.VLM or ParsingMode.BASIC_OCR.
            metadata: Optional metadata to attach to the result.
            save_path: Optional path (.json or .md) to save the result.
            output_schema: Optional Pydantic model for structured output (VLM only).

        Returns:
            A ParseResult containing the extracted text and metadata.
        """
        if parsing_mode == ParsingMode.AGENTIC:
            raise ValueError(
                "ParsingMode.AGENTIC is not supported for parse_image(). "
                "Use parse_pdf() with ParsingMode.AGENTIC for multi-page agentic workflows."
            )

        if output_schema is not None and parsing_mode != ParsingMode.VLM:
            raise ValueError(
                f"output_schema is only supported with ParsingMode.VLM, "
                f"got ParsingMode.{parsing_mode.name}."
            )

        start_time = time.time()

        img_array = await asyncio.to_thread(self._load_image, image)

        doc_metadata: Dict[str, Any] = {}
        if isinstance(image, (str, Path)):
            doc_metadata["file_name"] = Path(image).name
        if isinstance(metadata, dict):
            doc_metadata.update(metadata)
        elif isinstance(metadata, DocumentMetadata):
            doc_metadata.update(asdict(metadata))

        if parsing_mode == ParsingMode.BASIC_OCR:
            ocr_engine = self._get_ocr_engine()
            img_preprocessed = await asyncio.to_thread(
                self._image_processor.preprocess_for_ocr, img_array
            )
            content = await ocr_engine.recognize(img_preprocessed)
            doc_metadata["parsing_mode"] = ParsingMode.BASIC_OCR.value
            parse_result = ParseResult(
                id=self._generate_id(content),
                content=content,
                metadata=doc_metadata,
            )
        else:
            if self._client is None:
                raise ValueError(
                    "VLM credentials (vlm_base_url, vlm_api_key, vlm_model) are required "
                    "for ParsingMode.VLM. Use ParsingMode.BASIC_OCR instead."
                )
            img_preprocessed = await asyncio.to_thread(
                self._image_processor.preprocess_for_vlm, img_array
            )
            doc_metadata["parsing_mode"] = ParsingMode.VLM.value
            parse_result = await self._call_vlm(
                img_preprocessed, doc_metadata, output_schema=output_schema
            )

        doc_metadata["processing_time"] = time.time() - start_time

        if save_path:
            self._save_results([parse_result], save_path)

        return parse_result

    async def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        parsing_mode: ParsingMode = ParsingMode.BASIC_OCR,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
    ) -> List[ParseResult]:
        """
        Parse a PDF file page-by-page.

        Args:
            pdf_path: Path to the PDF document.
            parsing_mode: ParsingMode to use (BASIC_OCR, VLM, or AGENTIC).
            start_page: First page to parse (1-indexed, inclusive).
            end_page: Last page to parse (1-indexed, inclusive).
            save_path: Where to save results:
                       - .json path  → structured JSON
                       - .md path    → Markdown, pages separated by '---'
                       - directory   → saves as <pdf_stem>.json inside it
            metadata: Optional base metadata to attach to each page result.
            output_schema: Optional Pydantic model for structured JSON output.
                           Only supported with ParsingMode.VLM.

        Returns:
            List of ParseResult objects, sorted by page number.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if output_schema is not None and parsing_mode != ParsingMode.VLM:
            raise ValueError(
                f"output_schema is only supported with ParsingMode.VLM, "
                f"got ParsingMode.{parsing_mode.name}."
            )

        if parsing_mode in (ParsingMode.VLM, ParsingMode.AGENTIC) and self._client is None:
            raise ValueError(
                f"VLM credentials are required for ParsingMode.{parsing_mode.name}. "
                "Provide vlm_base_url, vlm_api_key, and vlm_model, "
                "or use ParsingMode.BASIC_OCR instead."
            )

        base_metadata: Dict[str, Any] = {"file_name": path.name}
        if isinstance(metadata, dict):
            base_metadata.update(metadata)
        elif isinstance(metadata, DocumentMetadata):
            base_metadata.update(asdict(metadata))

        total_pages = await asyncio.to_thread(self._get_pdf_page_count, path)
        base_metadata["total_pages"] = total_pages

        _start = start_page or 1
        _end = min(end_page or total_pages, total_pages)

        if _start > total_pages:
            raise ValueError(
                f"start_page={_start} exceeds total pages ({total_pages}) in '{path.name}'."
            )

        pages = range(_start, _end + 1)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_with_semaphore(page_n: int, meta: Dict[str, Any]):
            async with semaphore:
                if parsing_mode == ParsingMode.BASIC_OCR:
                    return await self._process_page_ocr(path, page_n, meta)
                elif parsing_mode == ParsingMode.VLM:
                    return await self._process_page_vlm(path, page_n, meta, output_schema)
                elif parsing_mode == ParsingMode.AGENTIC:
                    return await self._process_page_agentic(path, page_n, meta)
                else:
                    raise ValueError(f"Unsupported parsing mode: {parsing_mode}")

        from tqdm.asyncio import tqdm

        tasks = [
            _process_with_semaphore(page_num, copy.deepcopy(base_metadata)) for page_num in pages
        ]

        results: List[ParseResult] = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Parsing {path.name}",
            unit="page",
        ):
            try:
                res = await coro
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Error processing a page: {e}")

        results.sort(key=lambda x: x.metadata.get("page_number", 0))

        if save_path and results:
            self._save_results(results, save_path, path.stem)
        elif save_path and not results:
            print(f"Warning: no content extracted from '{path.name}', nothing saved.")

        return results

    async def _process_page_ocr(
        self,
        pdf_path: Path,
        page_num: int,
        metadata: Dict[str, Any],
    ) -> Optional[ParseResult]:
        """
        Process a single PDF page using PaddleOCR ONNX engine.
        """
        start_time = time.time()

        images = await asyncio.to_thread(
            self._image_processor.pdf_to_images,
            str(pdf_path),
            page_num,
            page_num,
        )

        img = images[0]

        ocr_engine = self._get_ocr_engine()
        page_content = await ocr_engine.recognize(img)

        metadata["page_number"] = page_num
        metadata["processing_time"] = time.time() - start_time
        metadata["parsing_mode"] = ParsingMode.BASIC_OCR.value

        if not page_content.strip():
            return None

        return ParseResult(
            id=self._generate_id(page_content),
            content=page_content,
            metadata=metadata,
        )

    async def _process_page_vlm(
        self,
        pdf_path: Path,
        page_num: int,
        metadata: Dict[str, Any],
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Optional[ParseResult]:
        """
        Process a single PDF page using a Vision Language Model.
        """
        start_time = time.time()

        images = await asyncio.to_thread(
            self._image_processor.pdf_to_images,
            str(pdf_path),
            page_num,
            page_num,
        )

        img = await asyncio.to_thread(
            self._image_processor.preprocess_for_vlm, images[0], page_num=page_num
        )

        result = await self._call_vlm(img, metadata, output_schema=output_schema)

        metadata["page_number"] = page_num
        metadata["processing_time"] = time.time() - start_time
        metadata["parsing_mode"] = ParsingMode.VLM.value

        return result if result else None

    async def _process_page_agentic(
        self,
        pdf_path: Path,
        page_num: int,
        metadata: Dict[str, Any],
    ) -> Optional[ParseResult]:
        """
        Process a single PDF page using the agentic reflect workflow.
        """
        if not self._agentic_workflow:
            raise RuntimeError("Agentic workflow is not initialized. Provide VLM credentials.")

        start_time = time.time()

        images = await asyncio.to_thread(
            self._image_processor.pdf_to_images,
            str(pdf_path),
            page_num,
            page_num,
        )

        img = await asyncio.to_thread(
            self._image_processor.preprocess_for_vlm, images[0], page_num=page_num
        )

        img_b64, mime_type = await asyncio.to_thread(self._image_processor.encode_to_base64, img)

        state = await self._agentic_workflow.run(
            image_b64=img_b64,
            mime_type=mime_type,
        )

        content = state.get("accumulated_text", "")

        metadata["page_number"] = page_num
        metadata["processing_time"] = time.time() - start_time
        metadata["parsing_mode"] = ParsingMode.AGENTIC.value
        metadata["reflect_iterations"] = state.get("reflect_iteration", 0)
        metadata["final_critic_score"] = state.get("critic_score", None)

        if not content.strip():
            return None

        return ParseResult(
            id=self._generate_id(content),
            content=content,
            metadata=metadata,
        )

    async def _call_vlm(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any],
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ParseResult:
        """Internal method to handle VLM communication."""

        img_b64, mime_type = await asyncio.to_thread(self._image_processor.encode_to_base64, image)

        response = await self._client.invoke(
            image_b64=img_b64,
            mime_type=mime_type,
            system_prompt=self.system_prompt,
            output_schema=output_schema,
        )

        if output_schema:
            parsed = response.choices[0].message.parsed
            content = (
                parsed.model_dump_json()
                if hasattr(parsed, "model_dump_json")
                else (json.dumps(parsed) if parsed is not None else "")
            )
        else:
            raw = (
                response.choices[0].message.content.strip()
                if (response and response.choices)
                else ""
            )
            content = extract_transcription(raw) or ""

        return ParseResult(
            id=self._generate_id(content),
            content=content,
            metadata=metadata,
        )

    def _save_results(
        self,
        results: List[ParseResult],
        save_path: Union[str, Path],
        default_stem: str = "output",
    ) -> None:
        """
        Save ParseResult list to a JSON or Markdown file.

        Resolution order:
        1. save_path ends with .json → write JSON to that exact path
        2. save_path ends with .md   → write Markdown to that exact path
        3. Anything else             → treat as directory, write <default_stem>.json inside
        """
        save_path_obj = Path(save_path)
        suffix = save_path_obj.suffix.lower()

        if suffix == ".json":
            self._write_json(results, save_path_obj)
        elif suffix == ".md":
            self._write_markdown(results, save_path_obj)
        else:
            # Directory — auto-create and save as JSON
            save_path_obj.mkdir(parents=True, exist_ok=True)
            self._write_json(results, save_path_obj / f"{default_stem}.json")

    def _write_json(self, results: List[ParseResult], target_path: Path) -> None:
        """Serialize results to JSON. Tries to parse content as JSON if it looks like JSON."""
        target_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = []
        for r in results:
            try:
                content_stripped = r.content.strip() if isinstance(r.content, str) else ""
                if content_stripped.startswith(("{", "[")):
                    item = asdict(r)
                    item["content"] = json.loads(r.content)
                    save_data.append(item)
                else:
                    save_data.append(asdict(r))
            except (json.JSONDecodeError, Exception):
                save_data.append(asdict(r))

        with open(target_path, "w", encoding="utf-8") as f:
            # Single result with already-parsed structured content → unwrap for cleaner output
            if len(save_data) == 1 and isinstance(save_data[0].get("content"), (dict, list)):
                json.dump(save_data[0]["content"], f, indent=2, ensure_ascii=False)
            else:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

    def _write_markdown(self, results: List[ParseResult], target_path: Path) -> None:
        """Write results to Markdown file. Pages are separated by horizontal rules."""
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            for i, r in enumerate(results):
                page_num = r.metadata.get("page_number", i + 1)
                f.write(f"<!-- page {page_num} -->\n")
                f.write(r.content)
                if i < len(results) - 1:
                    f.write("\n\n---\n\n")

    @staticmethod
    def _get_pdf_page_count(pdf_path: Path) -> int:
        """Count PDF pages using fitz (no pdfplumber required)."""
        with fitz.open(pdf_path) as doc:
            return len(doc)

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load an image from path or return the array if already loaded."""
        import cv2

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image}")
            return img
        return image

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate a unique SHA3-256 ID based on content text."""
        import hashlib

        if not isinstance(text, str):
            text = str(text or "")
        return hashlib.sha3_256(text.encode()).hexdigest()

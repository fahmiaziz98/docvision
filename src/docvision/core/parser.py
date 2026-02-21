import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pdfplumber
from pydantic import BaseModel

from ..processing import AutoRotate, ImageProcessor, NativePDFParser
from ..utils.helper import extract_transcription
from ..workflows.graph import AgenticWorkflow
from .client import VLMClient
from .types import DocumentMetadata, ParseResult, ParsingMode


class DocumentParser:
    """
    A high-level parser for documents (PDFs and Images) using a combination
    of native extraction and Vision Language Models (VLMs).
    """

    def __init__(
        self,
        vlm_base_url: Optional[str] = None,
        vlm_model: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_iterations: int = 3,
        system_prompt: Optional[str] = None,
        chart_description: bool = False,
        enable_rotate: bool = True,
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
            max_iterations: Maximum number of iterations for agentic parsing.
            system_prompt: System prompt to use for VLM calls.
            chart_description: Whether to use VLM to describe charts/images in PDFs.
            enable_rotate: Whether to automatically correct image orientation.
            max_concurrency: Maximum number of pages to process concurrently.
        """
        self.max_concurrency = max_concurrency
        if vlm_base_url is None or vlm_model is None or vlm_api_key is None:
            self._client = None
        else:
            self._client = VLMClient(
                base_url=vlm_base_url,
                api_key=vlm_api_key,
                model_name=vlm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        self.system_prompt = system_prompt

        self._agentic_workflow = (
            AgenticWorkflow(
                vlm_client=self._client,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
            )
            if self._client
            else None
        )

        self._rotate = AutoRotate(aggressive_mode=True) if enable_rotate else None
        self._image_processor = ImageProcessor(
            render_zoom=render_zoom,
            enable_rotate=enable_rotate,
            post_crop_max_size=post_crop_max_size,
            debug_dir=debug_dir,
        )
        self._pdf_parser = NativePDFParser(
            vlm_client=self._client,
            image_processor=self._image_processor,
            chart_description=chart_description,
        )

    async def parse_image(
        self,
        image: Union[str, Path, np.ndarray],
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        save_path: Optional[Union[str, Path]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ParseResult:
        """
        Parse a single image into Markdown or JSON using a VLM.

        Args:
            image: Image source (path or numpy BGR array).
            metadata: Optional metadata to attach to the result.
            save_path: Optional path to save the ParseResult.
            output_schema: Optional Pydantic model for structured output parsing.

        Returns:
            A ParseResult containing the extracted text and metadata.
        """
        start_time = time.time()

        img_array = await asyncio.to_thread(self._load_image, image)

        doc_metadata: Dict[str, Any] = {}
        if isinstance(image, (str, Path)):
            doc_metadata["file_name"] = Path(image).name

        if isinstance(metadata, dict):
            doc_metadata.update(metadata)
        elif isinstance(metadata, DocumentMetadata):
            doc_metadata.update(asdict(metadata))

        if self._client is None:
            raise ValueError(
                "You must provide VLM_BASE_URL, VLM_API_KEY & VLM_MODEL, when parsing images"
            )

        if self._rotate:
            img_array, _ = self._rotate.auto_rotate(img_array)

        parse_result = await self._call_vlm(img_array, doc_metadata, output_schema=output_schema)

        duration = time.time() - start_time
        doc_metadata["processing_time"] = duration

        if save_path:
            self._save_results([parse_result], save_path)

        return parse_result

    async def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        parsing_mode: ParsingMode = ParsingMode.PDF,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
    ) -> List[ParseResult]:
        """
        Parse a PDF file page-by-page.

        Args:
            pdf_path: Path to the PDF document.
            parsing_mode: ParsingMode to use.
            start_page: First page to parse (1-indexed).
            end_page: Last page to parse (inclusive).
            save_path: Directory to save the results as a JSON or MD file.
            metadata: Optional base metadata to clone for each page.

        Returns:
            List of ParseResult objects.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if parsing_mode != ParsingMode.PDF and self._client is None:
            raise ValueError(
                f"You must provide VLM_BASE_URL, VLM_API_KEY & VLM_MODEL, when parsing using {parsing_mode.name}"
            )

        # Prepare base metadata
        base_metadata: Dict[str, Any] = {"file_name": path.name}
        if isinstance(metadata, dict):
            base_metadata.update(metadata)
        elif isinstance(metadata, DocumentMetadata):
            base_metadata.update(asdict(metadata))

        try:

            def _get_pdf_info(p: Path):
                with pdfplumber.open(p) as pdf:
                    return len(pdf.pages)

            total_pages = await asyncio.to_thread(_get_pdf_info, path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {path.name}: {e}")

        base_metadata["total_pages"] = total_pages

        _start = start_page or 1
        _end = min(end_page or total_pages, total_pages)
        pages = range(_start, _end + 1)

        results: List[ParseResult] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_page_with_semaphore(page_n: int, meta: Dict[str, Any]):
            async with semaphore:
                if parsing_mode == ParsingMode.PDF:
                    return await self._process_page(path, page_n, meta)
                elif parsing_mode == ParsingMode.VLM:
                    return await self._process_page_vlm(path, page_n, meta)
                elif parsing_mode == ParsingMode.AGENTIC:
                    return await self._process_page_agentic(path, page_n, meta)
                else:
                    raise ValueError(f"Parsing mode: {parsing_mode} not supported")

        import copy

        from tqdm.asyncio import tqdm

        tasks = [
            _process_page_with_semaphore(page_num, copy.deepcopy(base_metadata))
            for page_num in pages
        ]

        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc=f"Parsing {path.name}", unit="page"
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

        return results

    def _save_results(
        self,
        results: List[ParseResult],
        save_path: Union[str, Path],
        default_stem: str = "output",
    ) -> None:
        """
        Save results to either a JSON or MD file.
        """
        save_path_obj = Path(save_path)
        suffix = save_path_obj.suffix.lower()

        is_json = suffix == ".json"
        is_md = suffix == ".md"

        if is_json or is_md:
            target_path = save_path_obj
            save_dir = target_path.parent
        else:
            save_dir = save_path_obj
            target_path = save_dir / f"{default_stem}.json"
            is_json = True

        save_dir.mkdir(parents=True, exist_ok=True)

        if is_json:
            save_data = []
            for r in results:
                try:
                    if isinstance(r.content, str) and (
                        r.content.strip().startswith("{") or r.content.strip().startswith("[")
                    ):
                        content_json = json.loads(r.content)
                        item = asdict(r)
                        item["content"] = content_json
                        save_data.append(item)
                    else:
                        save_data.append(asdict(r))
                except Exception:
                    save_data.append(asdict(r))

            with open(target_path, "w", encoding="utf-8") as f:
                if len(save_data) == 1 and isinstance(save_data[0]["content"], (dict, list)):
                    json.dump(save_data[0]["content"], f, indent=2, ensure_ascii=False)
                else:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
        elif is_md:
            with open(target_path, "w", encoding="utf-8") as f:
                content = "\n\n---\n\n".join(r.content for r in results)
                f.write(content)

    async def _process_page_agentic(
        self,
        pdf_path: Path,
        page_num: int,
        metadata: Dict[str, Any],
    ) -> Optional[ParseResult]:
        """
        Process a single PDF page using the Agentic workflow.
        """
        if not self._agentic_workflow:
            raise RuntimeError("Agentic workflow is not initialized.")

        start_time = time.time()

        def _convert_pdf_to_images():
            return self._image_processor.pdf_to_images(
                str(pdf_path),
                start_page=page_num,
                end_page=page_num,
            )

        images = await asyncio.to_thread(_convert_pdf_to_images)
        img = await asyncio.to_thread(
            self._image_processor.process_image, images[0], page_num=page_num
        )

        def _encode_image():
            return self._image_processor.encode_to_base64(img)

        img_b64, mime_type = await asyncio.to_thread(_encode_image)

        state = await self._agentic_workflow.run(
            image_b64=img_b64,
            mime_type=mime_type,
        )

        content = state.get("accumulated_text", "")

        duration = time.time() - start_time
        metadata["page_number"] = page_num
        metadata["processing_time"] = duration
        metadata["agent_iterations"] = state.get("iteration_count", 0)

        if not content.strip():
            return None

        return ParseResult(
            id=self._generate_id(content),
            content=content,
            metadata=metadata,
        )

    async def _process_page(
        self,
        pdf_path: Path,
        page_num: int,
        metadata: Dict[str, Any],
    ) -> Optional[ParseResult]:
        """
        Process a single PDF page using the standard PDF parser.

        Args:
            pdf_path: Path to the PDF file.
            page_num: The page number to process.
            metadata: Metadata to associate with the page.

        Returns:
            The parsed result if content is found, otherwise None.
        """
        start_time = time.time()

        result = await self._pdf_parser.aparse_page(str(pdf_path), page_num)
        page_content = result.markdown

        duration = time.time() - start_time
        metadata["page_number"] = page_num
        metadata["processing_time"] = duration

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
        Process a single PDF page using a Vision Language Model (VLM).

        Args:
            pdf_path: Path to the PDF file.
            page_num: The page number to process.
            metadata: Metadata associated with the document.

        Returns:
            A list of ParseResult objects if successful, None otherwise.
        """
        start_time = time.time()

        def _convert_pdf_to_images():
            return self._image_processor.pdf_to_images(
                str(pdf_path),
                start_page=page_num,
                end_page=page_num,
            )

        images = await asyncio.to_thread(_convert_pdf_to_images)

        img = await asyncio.to_thread(
            self._image_processor.process_image, images[0], page_num=page_num
        )
        result = await self._call_vlm(img, metadata, output_schema=output_schema)

        duration = time.time() - start_time
        metadata["page_number"] = page_num
        metadata["processing_time"] = duration

        if not result:
            return None

        return result

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Load an image from path or return the array if already loaded.

        Args:
            image: path or image numpy array

        Returns:
            image numpy arra
        """
        import cv2

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image}")
            return img
        return image

    async def _call_vlm(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any],
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ParseResult:
        """
        Internal method to handle the VLM communication.

        Args:
            image: image numpy array
            metadata: metadata image

        Returns:
            Parsing results
        """

        def _encode_image():
            return self._image_processor.encode_to_base64(image)

        img_b64, mime_type = await asyncio.to_thread(_encode_image)

        response = await self._client.invoke(
            image_b64=img_b64,
            mime_type=mime_type,
            system_prompt=self.system_prompt,
            output_schema=output_schema,
        )

        if output_schema:
            parsed = response.choices[0].message.parsed
            if hasattr(parsed, "model_dump_json"):
                content = parsed.model_dump_json()
            else:
                content = json.dumps(parsed) if parsed is not None else ""
        else:
            content = (
                response.choices[0].message.content.strip() if response and response.choices else ""
            )
            content = extract_transcription(content) or ""

        return ParseResult(
            id=self._generate_id(content),
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _generate_id(text: str) -> str:
        """
        Generate a unique SHA3-256 ID based on the content text.
        """
        import hashlib

        if not isinstance(text, str):
            text = str(text or "")

        return hashlib.sha3_256(text.encode()).hexdigest()

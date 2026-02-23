import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import pdfplumber

from ..core import VLMClient
from ..core.types import NativeParseResult, PageBlock
from ..utils import extract_transcription

if TYPE_CHECKING:
    from .image import ImageProcessor


@dataclass
class _ExtractedPageData:
    """
    Holds all raw data extracted from a pdfplumber page WITHIN the context manager.
    This is necessary because page objects become invalid after the PDF file is closed.
    """

    width: float
    height: float
    tables: List[dict]  # {"bbox": tuple, "data": list[list]}
    text_bands: List[dict]  # {"bbox": tuple, "text": str}
    images: List[dict]  # pdfplumber image dicts (x0, top, x1, bottom, ...)
    chart_crops: List[
        dict
    ]  # {"bbox": tuple, "pil_image": PIL.Image, "width": float, "height": float}


class NativePDFParser:
    """
    PDF parser.

    Combines deterministic pdfplumber extraction with optional VLM
    hybrid calls for chart/image description.

    Args:
        vlm_client: VLMClient instance (required only if chart_description=True).
        chart_description: Whether to send cropped chart images to VLM.
        pseudo_table_min_rows: Minimum rows to qualify as pseudo-table.
        pseudo_table_min_cols: Minimum columns to qualify as pseudo-table.
        chart_description_prompt: Custom prompt for VLM chart description.
    """

    DEFAULT_CHART_PROMPT = (
        "Describe this chart or image in one sentence. "
        "Focus on what data or information it conveys."
    )

    def __init__(
        self,
        vlm_client: Optional[VLMClient] = None,
        image_processor: Optional["ImageProcessor"] = None,
        chart_description: bool = True,
        pseudo_table_min_rows: int = 2,
        pseudo_table_min_cols: int = 2,
        chart_description_prompt: Optional[str] = None,
    ):
        if chart_description and vlm_client is None:
            raise ValueError(
                "vlm_client is required when chart_description=True, "
                "you must provide endpoint + api key & model. Or "
                "Pass chart_description=False to disable VLM hybrid calls."
            )

        self.vlm_client = vlm_client
        self.image_processor = image_processor
        self.chart_description = chart_description
        self.pseudo_table_min_rows = pseudo_table_min_rows
        self.pseudo_table_min_cols = pseudo_table_min_cols
        self.chart_prompt = chart_description_prompt or self.DEFAULT_CHART_PROMPT

    async def aparse_page(self, pdf_path: str, page_num: int) -> NativeParseResult:
        """
        Parse a single PDF page asynchronously.

        KEY FIX: All pdfplumber operations (including page.crop() for charts)
        are completed INSIDE the context manager on a single thread.
        Only the resulting data (numpy arrays, text, etc.) is returned.

        Args:
            pdf_path: Path to PDF file.
            page_num: Page number (1-indexed).

        Returns:
            NativeParseResult with markdown + metadata.
        """
        try:
            # Step 1: Extract ALL data from pdfplumber in one thread, file stays open
            # the entire time. chart_crops (PIL images) are also created here.
            extracted = await asyncio.to_thread(self._extract_page_data, pdf_path, page_num)

            # Step 2: Now that file is closed, build blocks.
            # For chart VLM calls, we use the PIL images captured in step 1.
            blocks = await self._build_blocks_from_extracted(extracted)

            return self._build_result(blocks, page_num)
        except Exception as e:
            return NativeParseResult(
                markdown=f"[EXTRACTION ERROR: {e}]",
                metadata={"error": str(e)},
                page_number=page_num,
                has_tables=False,
                has_pseudo_tables=False,
                has_charts=False,
                block_count=0,
            )

    async def aparse_pdf(
        self, pdf_path: str, start_page: int = 1, end_page: Optional[int] = None
    ) -> list[NativeParseResult]:
        """
        Parse multiple pages asynchronously.

        Args:
            pdf_path: Path to PDF file.
            start_page: start page
            end_page: end page

        Returns
            NativeParseResult with markdown + metadata.
        """

        def _get_page_count(path: str) -> int:
            with pdfplumber.open(path) as pdf:
                return len(pdf.pages)

        total = await asyncio.to_thread(_get_page_count, pdf_path)

        end_page = min(end_page or total, total)
        results = []
        for p in range(start_page, end_page + 1):
            results.append(await self.aparse_page(pdf_path, p))
        return results

    def _extract_page_data(self, pdf_path: str, page_num: int) -> _ExtractedPageData:
        """
        Extract all content blocks from a page (sync).

        Args:
            pdf_path: Path to PDF file.
            page_num: Page number (1-indexed).

        Returns:
            _ExtractedPageData with all extracted content.
        """
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            width = page.width
            height = page.height

            tables_data = []
            table_bboxes = []
            found_tables = page.find_tables()
            extracted_tables = page.extract_tables()

            if found_tables and extracted_tables:
                for plumber_table, data in zip(found_tables, extracted_tables):
                    if not data:
                        continue
                    bbox = plumber_table.bbox
                    table_bboxes.append(bbox)
                    tables_data.append({"bbox": bbox, "data": data})

            text_bands = self._extract_text_bands(page, table_bboxes, width, height)

            chart_crops = []
            raw_images = list(page.images)

            for img in raw_images:
                bbox = (img["x0"], img["top"], img["x1"], img["bottom"])

                if any(self._bbox_overlaps(bbox, tb) for tb in table_bboxes):
                    continue

                x0 = max(0, img["x0"])
                top = max(0, img["top"])
                x1 = min(width, img["x1"])
                bottom = min(height, img["bottom"])

                img_width = x1 - x0
                img_height = bottom - top
                if img_width < 50 or img_height < 50:
                    continue

                clipped_bbox = (x0, top, x1, bottom)

                if self.chart_description and self.image_processor:
                    try:
                        cropped_page = page.crop(clipped_bbox)
                        pil_image = cropped_page.to_image(resolution=150).original
                        chart_crops.append(
                            {
                                "bbox": clipped_bbox,
                                "pil_image": pil_image,
                                "width": img_width,
                                "height": img_height,
                                "y_top": img["top"],
                                "y_bottom": img["bottom"],
                            }
                        )
                    except Exception:
                        chart_crops.append(
                            {
                                "bbox": clipped_bbox,
                                "pil_image": None,
                                "width": img_width,
                                "height": img_height,
                                "y_top": img["top"],
                                "y_bottom": img["bottom"],
                            }
                        )
                else:
                    chart_crops.append(
                        {
                            "bbox": clipped_bbox,
                            "pil_image": None,
                            "width": img_width,
                            "height": img_height,
                            "y_top": img["top"],
                            "y_bottom": img["bottom"],
                        }
                    )

        return _ExtractedPageData(
            width=width,
            height=height,
            tables=tables_data,
            text_bands=text_bands,
            images=raw_images,
            chart_crops=chart_crops,
        )

    def _extract_text_bands(
        self,
        page,
        table_bboxes: List[Tuple],
        width: float,
        height: float,
    ) -> List[dict]:
        """
        Extract text from page, sliced around table bounding boxes.
        Must be called while the page/file is still open.

        Args:
            page: Page object from pdfplumber.
            table_bboxes: List of table bounding boxes.
            width: Page width.
            height: Page height.

        Returns:
            List of text bands.
        """
        text_bands = []

        if not table_bboxes:
            raw_text = page.extract_text(layout=True) or ""
            if raw_text.strip():
                text_bands.append(
                    {
                        "bbox": (0, 0, width, height),
                        "text": raw_text.strip(),
                        "y_top": 0,
                        "y_bottom": height,
                    }
                )
            return text_bands

        sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1])
        bands = []
        prev_bottom = 0

        for bbox in sorted_bboxes:
            x0, y_top, x1, y_bottom = bbox
            if y_top > prev_bottom + 5:
                bands.append((0, prev_bottom, width, y_top))
            prev_bottom = y_bottom

        if prev_bottom < height - 5:
            bands.append((0, prev_bottom, width, height))

        for band_bbox in bands:
            try:
                cropped = page.crop(band_bbox)
                text = cropped.extract_text(layout=True) or ""
                if text.strip():
                    text_bands.append(
                        {
                            "bbox": band_bbox,
                            "text": text.strip(),
                            "y_top": band_bbox[1],
                            "y_bottom": band_bbox[3],
                        }
                    )
            except Exception:
                continue

        return text_bands

    async def _build_blocks_from_extracted(self, extracted: _ExtractedPageData) -> List[PageBlock]:
        """
        Build PageBlock list from extracted data. File is already closed at this point.
        VLM chart calls happen here (async), using PIL images captured earlier.
        """
        blocks: List[PageBlock] = []

        # --- Table blocks ---
        for table_info in extracted.tables:
            md = self._format_table(table_info["data"])
            bbox = table_info["bbox"]
            blocks.append(
                PageBlock(
                    kind="table",
                    content=md,
                    y_top=bbox[1],
                    y_bottom=bbox[3],
                    metadata={
                        "rows": len(table_info["data"]),
                        "cols": len(table_info["data"][0]) if table_info["data"] else 0,
                    },
                )
            )

        # --- Text blocks (with pseudo-table detection) ---
        for band in extracted.text_bands:
            text_block = PageBlock(
                kind="text",
                content=band["text"],
                y_top=band["y_top"],
                y_bottom=band["y_bottom"],
                metadata={},
            )
            pseudo = self._detect_pseudo_table(text_block)
            blocks.append(pseudo if pseudo else text_block)

        # --- Chart blocks (async VLM calls, file already closed) ---
        for crop_info in extracted.chart_crops:
            if (
                self.chart_description
                and self.image_processor
                and crop_info["pil_image"] is not None
            ):
                content = await self._describe_chart_with_vlm(crop_info["pil_image"])
            else:
                content = "<chart>Chart or image detected</chart>"

            blocks.append(
                PageBlock(
                    kind="chart",
                    content=content,
                    y_top=crop_info["y_top"],
                    y_bottom=crop_info["y_bottom"],
                    metadata={
                        "bbox": crop_info["bbox"],
                        "width": crop_info["width"],
                        "height": crop_info["height"],
                    },
                )
            )

        blocks.sort(key=lambda b: b.y_top)
        return blocks

    async def _describe_chart_with_vlm(self, pil_image) -> str:
        """
        Send a PIL image to VLM for description. File is already closed.

        Args:
            pil_image: PIL image to describe.

        Returns:
            Description of the chart or image.
        """
        try:
            import cv2
            import numpy as np

            img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            img_np = self.image_processor.process_image(img_np)
            img_b64, mime_type = self.image_processor.encode_to_base64(img_np)

            response = await self.vlm_client.invoke(
                image_b64=img_b64,
                mime_type=mime_type,
                system_prompt="Describe this chart or image in one sentence only.",
                user_prompt=self.chart_prompt,
            )

            description = (
                response.choices[0].message.content.strip()
                if response and response.choices
                else "Chart or image"
            )
            description = extract_transcription(description)
            return f"<chart>{description}</chart>"

        except Exception:
            return "<chart>Chart or image detected</chart>"

    def _format_table(self, data: list[list]) -> str:
        """Convert raw pdfplumber table data to markdown pipe table."""
        if not data:
            return ""

        header = [str(cell or "").strip().replace("\n", " ") for cell in data[0]]
        col_count = len(header)

        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * col_count) + " |")

        for row in data[1:]:
            normalized = [str(cell or "").strip().replace("\n", " ") for cell in row]
            # Pad or trim to match header column count
            while len(normalized) < col_count:
                normalized.append("")
            normalized = normalized[:col_count]
            lines.append("| " + " | ".join(normalized) + " |")

        return "\n".join(lines)

    def _detect_pseudo_table(self, block: PageBlock) -> Optional[PageBlock]:
        """
        Detect space-aligned pseudo-tables in plain text blocks.
        Common in financial reports where tables have no visible borders.

        Returns a new PageBlock with kind="pseudo_table" if detected, else None.
        """
        lines = block.content.split("\n")
        table_lines = [line for line in lines if self._is_table_row(line)]

        if len(table_lines) < self.pseudo_table_min_rows:
            return None

        # Check if enough lines are table-like (>40% of block)
        ratio = len(table_lines) / max(len(lines), 1)
        if ratio < 0.4:
            return None

        md = self._format_pseudo_table(lines)
        if not md:
            return None

        return PageBlock(
            kind="pseudo_table",
            content=md,
            y_top=block.y_top,
            y_bottom=block.y_bottom,
            metadata={"original_lines": len(lines), "table_lines": len(table_lines)},
        )

    def _is_table_row(self, line: str) -> bool:
        """Heuristic: line has 2+ number clusters with wide whitespace gaps."""
        numbers = re.findall(r"[\d,\.]+", line)
        has_multiple_numbers = len(numbers) >= self.pseudo_table_min_cols
        has_wide_gaps = bool(re.search(r"\s{3,}", line))
        return has_multiple_numbers and has_wide_gaps

    def _format_pseudo_table(self, lines: list[str]) -> str:
        """Convert space-aligned lines into a markdown table using column position detection."""
        col_positions = self._detect_column_positions(lines)

        if len(col_positions) < self.pseudo_table_min_cols:
            return ""

        rows = [self._split_by_positions(line, col_positions) for line in lines]
        rows = [r for r in rows if any(c.strip() for c in r)]

        if not rows:
            return ""

        col_count = max(len(r) for r in rows)
        md_lines = []

        # First row as header
        header = rows[0] + [""] * (col_count - len(rows[0]))
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * col_count) + " |")

        for row in rows[1:]:
            padded = row + [""] * (col_count - len(row))
            md_lines.append("| " + " | ".join(padded[:col_count]) + " |")

        return "\n".join(md_lines)

    def _detect_column_positions(self, lines: list[str]) -> list[int]:
        """Detect consistent column positions from number alignment across lines."""
        position_votes: dict[int, int] = {}

        for line in lines:
            for match in re.finditer(r"[\d,\.]{3,}", line):
                bucket = (match.start() // 4) * 4
                position_votes[bucket] = position_votes.get(bucket, 0) + 1

        threshold = max(1, len(lines) * 0.3)
        return sorted(pos for pos, count in position_votes.items() if count >= threshold)

    def _split_by_positions(self, line: str, positions: list[int]) -> list[str]:
        """Split a line into columns at detected positions."""
        cols = []
        prev = 0
        for pos in positions:
            chunk = line[prev:pos].strip() if pos <= len(line) else ""
            if chunk:
                cols.append(chunk)
            prev = pos
        last = line[prev:].strip()
        if last:
            cols.append(last)
        return cols or [""]

    def _build_result(self, blocks: list[PageBlock], page_num: int) -> NativeParseResult:
        """
        Assemble all blocks (already sorted by y_top) into final Markdown + metadata.
        """
        markdown_parts = []
        tables_meta = []
        charts_meta = []
        text_meta = []

        for block in blocks:
            if block.kind == "text":
                markdown_parts.append(block.content)
                text_meta.append({"y_top": block.y_top, "y_bottom": block.y_bottom})

            elif block.kind in ("table", "pseudo_table"):
                markdown_parts.append(block.content)
                tables_meta.append(
                    {
                        "kind": block.kind,
                        "y_top": block.y_top,
                        "y_bottom": block.y_bottom,
                        **block.metadata,
                    }
                )

            elif block.kind == "chart":
                markdown_parts.append(block.content)
                charts_meta.append(
                    {
                        "y_top": block.y_top,
                        "y_bottom": block.y_bottom,
                        **block.metadata,
                    }
                )

        markdown = "\n\n".join(p for p in markdown_parts if p.strip())

        return NativeParseResult(
            markdown=markdown,
            metadata={
                "tables": tables_meta,
                "charts": charts_meta,
                "text_blocks": text_meta,
                "total_blocks": len(blocks),
            },
            page_number=page_num,
            has_tables=any(b.kind in ("table", "pseudo_table") for b in blocks),
            has_pseudo_tables=any(b.kind == "pseudo_table" for b in blocks),
            has_charts=any(b.kind == "chart" for b in blocks),
            block_count=len(blocks),
        )

    @staticmethod
    def _bbox_overlaps(a: tuple, b: tuple) -> bool:
        """Check if two bboxes (x0, top, x1, bottom) overlap."""
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

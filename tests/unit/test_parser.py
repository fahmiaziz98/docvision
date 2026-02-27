from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel

from docvision.core.parser import DocumentParser
from docvision.core.types import ParseResult, ParsingMode


def _make_vlm_response(content: str = "Extracted text content"):
    """Build a mock VLM response.

    Note: _call_vlm calls extract_transcription() on the raw content.
    To get a clean string back, wrap content in <transcription> tags.
    """
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = f"<transcription>{content}</transcription>"
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_vlm_client():
    client = MagicMock()
    client.invoke = AsyncMock(return_value=_make_vlm_response())
    return client


@pytest.fixture
def mock_image_processor():
    processor = MagicMock()
    processor.encode_to_base64.return_value = ("base64string", "image/png")
    processor.preprocess_for_ocr.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    processor.preprocess_for_vlm.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    processor.pdf_to_images.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)]
    return processor


@pytest.fixture
def parser(mock_vlm_client, mock_image_processor):
    with (
        patch("docvision.core.parser.VLMClient", return_value=mock_vlm_client),
        patch("docvision.core.parser.ImageProcessor", return_value=mock_image_processor),
    ):
        p = DocumentParser(
            base_url="https://api.test.com",
            model_name="test-model",
            api_key="test-key",
        )
        p._client = mock_vlm_client
        p._image_processor = mock_image_processor
        return p


@pytest.mark.unit
class TestDocumentParser:
    @pytest.mark.asyncio
    async def test_parse_image_vlm(self, parser, mock_vlm_client):
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(parser, "_load_image", return_value=img_array):
            result = await parser.parse_image("test_image.jpg", parsing_mode=ParsingMode.VLM)

        assert isinstance(result, ParseResult)
        assert result.content == "Extracted text content"
        assert result.metadata["file_name"] == "test_image.jpg"
        assert result.metadata["parsing_mode"] == ParsingMode.VLM.value
        mock_vlm_client.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_image_agentic_raises(self, parser):
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(parser, "_load_image", return_value=img_array):
            with pytest.raises(ValueError, match="AGENTIC"):
                await parser.parse_image("test.jpg", parsing_mode=ParsingMode.AGENTIC)

    @pytest.mark.asyncio
    async def test_parse_image_output_schema_non_vlm_raises(self, parser):
        """output_schema with BASIC_OCR should raise."""

        class TestSchema(BaseModel):
            field: str

        img_array = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(parser, "_load_image", return_value=img_array):
            with pytest.raises(ValueError, match="output_schema"):
                await parser.parse_image(
                    "test.jpg",
                    parsing_mode=ParsingMode.BASIC_OCR,
                    output_schema=TestSchema,
                )

    @pytest.mark.asyncio
    async def test_parse_image_structured_output(self, parser, mock_vlm_client):
        """output_schema should be forwarded to VLM client."""

        class TestModel(BaseModel):
            field: str

        # For structured output, .parsed is used — mock it with the schema instance
        mock_vlm_client.invoke.return_value.choices[0].message.parsed = TestModel(field="hello")

        img_array = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(parser, "_load_image", return_value=img_array):
            await parser.parse_image(
                "test.jpg", parsing_mode=ParsingMode.VLM, output_schema=TestModel
            )

        _, kwargs = mock_vlm_client.invoke.call_args
        assert kwargs["output_schema"] == TestModel

    @pytest.mark.asyncio
    async def test_parse_pdf_vlm_mode(self, parser, mock_vlm_client):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(parser, "_get_pdf_page_count", return_value=2),
            patch.object(parser, "_process_page_vlm") as mock_process,
        ):
            mock_process.side_effect = lambda path, page_n, meta, output_schema=None: (
                ParseResult(
                    id=f"id-{page_n}",
                    content=f"Page {page_n} content",
                    metadata={"page_number": page_n},
                )
            )

            results = await parser.parse_pdf("test.pdf", parsing_mode=ParsingMode.VLM)

        assert len(results) == 2
        assert results[0].metadata["page_number"] == 1
        assert results[1].metadata["page_number"] == 2

    @pytest.mark.asyncio
    async def test_parse_pdf_output_schema_non_vlm_raises(self, parser):
        class TestSchema(BaseModel):
            field: str

        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ValueError, match="output_schema"):
                await parser.parse_pdf(
                    "test.pdf",
                    parsing_mode=ParsingMode.BASIC_OCR,
                    output_schema=TestSchema,
                )

    @pytest.mark.asyncio
    async def test_parse_pdf_output_schema_forwarded_to_pages(self, parser):
        """output_schema should be forwarded to each _process_page_vlm call."""

        class TestSchema(BaseModel):
            field: str

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(parser, "_get_pdf_page_count", return_value=1),
            patch.object(parser, "_process_page_vlm") as mock_process,
        ):
            mock_process.return_value = ParseResult(
                id="id-1", content="content", metadata={"page_number": 1}
            )

            await parser.parse_pdf(
                "test.pdf",
                parsing_mode=ParsingMode.VLM,
                output_schema=TestSchema,
            )

        # _process_page_vlm(path, page_n, meta, output_schema) — 4th positional arg
        args, kwargs = mock_process.call_args
        # output_schema passed positionally (from _process_with_semaphore lambda)
        received_schema = args[3] if len(args) > 3 else kwargs.get("output_schema")
        assert received_schema == TestSchema

    @pytest.mark.asyncio
    async def test_parse_pdf_start_page_exceeds_total_raises(self, parser):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch.object(parser, "_get_pdf_page_count", return_value=3),
        ):
            with pytest.raises(ValueError, match="start_page"):
                await parser.parse_pdf("test.pdf", start_page=10)

    @pytest.mark.asyncio
    async def test_parse_pdf_no_vlm_client_raises(self):
        p = DocumentParser()  # no VLM credentials

        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ValueError, match="credentials"):
                await p.parse_pdf("test.pdf", parsing_mode=ParsingMode.VLM)

    @pytest.mark.asyncio
    async def test_parse_pdf_file_not_found(self, parser):
        with pytest.raises(FileNotFoundError):
            await parser.parse_pdf("nonexistent.pdf")

    def test_max_reflect_cycles_warning(self):
        with pytest.warns(UserWarning, match="max_reflect_cycles"):
            DocumentParser(
                base_url="http://test",
                model_name="test",
                api_key="test",
                max_reflect_cycles=5,
            )

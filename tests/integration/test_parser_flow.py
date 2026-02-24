from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from docvision.core.parser import DocumentParser
from docvision.core.types import ParseResult, ParsingMode


@pytest.fixture
def integration_parser():
    """Parser configured for integration testing with mocked VLM."""
    mock_client = MagicMock()
    mock_client.invoke = AsyncMock()

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "<transcription>Integrated result</transcription>"
    mock_message.parsed = None
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.invoke.return_value = mock_response

    with patch("docvision.core.parser.VLMClient", return_value=mock_client):
        p = DocumentParser(
            vlm_base_url="http://localhost:8080",
            vlm_model="test",
            vlm_api_key="test",
            enable_rotate=False,
        )
        p._client = mock_client
        return p


def _make_fitz_mock(page_count: int = 2):
    """Build a reusable fitz.open() context manager mock."""
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = page_count

    mock_page = MagicMock()
    mock_pix = MagicMock()
    mock_pix.samples = b"\x00" * (10 * 10 * 3)
    mock_pix.height = 10
    mock_pix.width = 10
    mock_pix.n = 3
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.load_page.return_value = mock_page

    mock_fitz = MagicMock()
    mock_fitz.__enter__.return_value = mock_doc
    mock_fitz.__exit__.return_value = False
    return mock_fitz


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pdf_parsing_flow(integration_parser):
    """Full VLM parse_pdf flow — fitz only, no pdfplumber."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("docvision.processing.image.fitz.open", return_value=_make_fitz_mock(2)),
        patch.object(
            integration_parser,
            "_get_pdf_page_count",
            return_value=2,
        ),
    ):
        results = await integration_parser.parse_pdf(
            "integration_test.pdf",
            parsing_mode=ParsingMode.VLM,
        )

    assert len(results) == 2
    for i, result in enumerate(results, start=1):
        assert result.content == "Integrated result"
        assert result.metadata["page_number"] == i


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_parsing_flow(integration_parser):
    """Single image VLM parse flow."""
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch.object(integration_parser, "_load_image", return_value=img_array):
        result = await integration_parser.parse_image("test.png", parsing_mode=ParsingMode.VLM)

    assert isinstance(result, ParseResult)
    assert result.content == "Integrated result"
    assert result.metadata["file_name"] == "test.png"
    assert result.metadata["parsing_mode"] == ParsingMode.VLM.value


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_basic_ocr_mode(integration_parser):
    """BASIC_OCR parse_pdf flow — no VLM calls, uses OCR engine."""
    mock_ocr_engine = MagicMock()
    mock_ocr_engine.recognize = AsyncMock(return_value="OCR extracted text")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("docvision.processing.image.fitz.open", return_value=_make_fitz_mock(1)),
        patch.object(integration_parser, "_get_pdf_page_count", return_value=1),
        patch.object(integration_parser, "_get_ocr_engine", return_value=mock_ocr_engine),
    ):
        results = await integration_parser.parse_pdf(
            "test.pdf",
            parsing_mode=ParsingMode.BASIC_OCR,
        )

    assert len(results) == 1
    assert results[0].content == "OCR extracted text"
    assert results[0].metadata["parsing_mode"] == ParsingMode.BASIC_OCR.value

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

    # Mock VLM response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "<transcription>Integrated result</transcription>"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.invoke.return_value = mock_response

    with (patch("docvision.core.parser.VLMClient", return_value=mock_client),):
        p = DocumentParser(
            vlm_base_url="http://localhost:8080",
            vlm_model="test",
            vlm_api_key="test",
            enable_rotate=False,
        )
        p._client = mock_client
        return p


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pdf_parsing_flow(integration_parser):
    """Test the full flow of parsing a PDF (mocked file access)."""
    pdf_path = "integration_test.pdf"

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pdfplumber.open") as mock_pdf_open,
        patch("docvision.processing.image.fitz.open") as mock_fitz_open,
    ):
        # Mock pdfplumber for page count and metadata
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]  # 2 pages
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        # Mock fitz for image rendering
        mock_fitz_doc = MagicMock()
        mock_fitz_doc.__len__.return_value = 2
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.samples = b"\x00" * (10 * 10 * 3)
        mock_pix.height = 10
        mock_pix.width = 10
        mock_pix.n = 3
        mock_page.get_pixmap.return_value = mock_pix
        mock_fitz_doc.load_page.return_value = mock_page
        mock_fitz_open.return_value = mock_fitz_doc

        # Run parsing
        results = await integration_parser.parse_pdf(pdf_path, parsing_mode=ParsingMode.VLM)

        assert len(results) == 2
        assert results[0].content == "Integrated result"
        assert results[0].metadata["page_number"] == 1
        assert results[1].metadata["page_number"] == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_image_parsing_flow(integration_parser):
    """Test parsing a single image."""
    img_path = "test.png"
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch.object(integration_parser, "_load_image", return_value=img_array),
    ):
        result = await integration_parser.parse_image(img_path)

        assert isinstance(result, ParseResult)
        assert result.content == "Integrated result"
        assert result.metadata["file_name"] == "test.png"

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import numpy as np

from docvision.core.parser import DocumentParsingAgent
from docvision.core.types import ParserConfig, ParsingMode, ParseResult, BatchParseResult


@pytest.fixture
def mock_vlm_client():
    client = MagicMock()
    # Mock async call method
    client.acall = AsyncMock()
    
    # Mock response object structure
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Extracted text content"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    client.acall.return_value = mock_response
    return client


@pytest.fixture
def mock_image_processor():
    processor = MagicMock()
    # Mock encoding
    processor.encode_to_base64.return_value = ("base64string", "image/jpeg")
    # Mock processing
    processor.process_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    # Mock PDF conversion
    processor.pdf_to_images.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)]
    return processor


@pytest.fixture
def agent(mock_vlm_client, mock_image_processor):
    config = ParserConfig(api_key="test_key")
    with patch("docvision.core.parser.VLMClient", return_value=mock_vlm_client), \
         patch("docvision.core.parser.ImageProcessor", return_value=mock_image_processor):
        agent = DocumentParsingAgent(config=config)
        # Inject mocks directly to ensure they are used
        agent._vlm_client = mock_vlm_client
        agent.processor = mock_image_processor
        return agent


@pytest.mark.unit
class TestDocumentParsingAgent:

    @pytest.mark.asyncio
    async def test_aparse_image_vlm_mode(self, agent, mock_vlm_client):
        image_path = "test_image.jpg"
        
        # We need to mock Image.open since we pass a string path
        with patch("PIL.Image.open") as mock_open:
            result = await agent.aparse_image(image_path, mode=ParsingMode.VLM)
            
            assert isinstance(result, ParseResult)
            assert result.content == "Extracted text content"
            assert result.page_number == 0
            
            # Verify client was called
            mock_vlm_client.acall.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_aparse_image_agentic_mode(self, agent):
        # Mock the workflow run method
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value={
            "accumulated_text": "Agentic result",
            "iteration_count": 2,
            "generation_history": ["step1", "step2"]
        })
        
        # Mock Path.exists and Image.open
        with patch.object(agent, '_get_agentic_workflow', return_value=mock_workflow), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("PIL.Image.open"):
            result = await agent.aparse_image("test_image.jpg", mode=ParsingMode.AGENTIC)
            
            assert result.content == "Agentic result"
            assert result.metadata["mode"] == ParsingMode.AGENTIC.value
            assert result.metadata["iterations"] == 2

    @pytest.mark.asyncio
    async def test_aparse_pdf_success(self, agent, mock_vlm_client):
        pdf_path = "test.pdf"
        
        with patch("pathlib.Path.exists", return_value=True):
            result = await agent.aparse_pdf(pdf_path, mode=ParsingMode.VLM)
            
            assert isinstance(result, BatchParseResult)
            assert result.total_pages == 1
            assert result.success_count == 1
            assert len(result.results) == 1
            assert result.results[0].content == "Extracted text content"

    def test_sync_parse_image(self, agent, mock_vlm_client):
        # Should call vlm_client.call, NOT asyncio.run
        with patch("PIL.Image.open"):
            agent.parse_image("test.jpg")
            mock_vlm_client.call.assert_called_once()

    def test_sync_parse_pdf(self, agent, mock_vlm_client):
        # Should call vlm_client.call for each page (1 page in mock)
        agent.parse_pdf("test.pdf")
        assert mock_vlm_client.call.call_count == 1

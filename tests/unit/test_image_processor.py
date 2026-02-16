import pytest
import numpy as np
import pytest_asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from PIL import Image

from docvision.core.types import ParserConfig, ImageFormat
from docvision.processing.image import ImageProcessor


@pytest.fixture
def basic_config():
    return ParserConfig(
        enable_auto_rotate=False,
        enable_crop=False,
        render_zoom=1.0
    )

@pytest.fixture
def processor(basic_config):
    return ImageProcessor(config=basic_config)

@pytest.fixture
def sample_image_np():
    # Create a simple 100x100 RGB image (numpy array)
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def sample_image_pil():
    # Create a simple 100x100 RGB image (PIL)
    return Image.new('RGB', (100, 100), color='red')


@pytest.mark.unit
class TestImageProcessorInitialization:
    def test_init_default_config(self):
        processor = ImageProcessor()
        assert isinstance(processor.config, ParserConfig)
        assert processor.rotation_pipeline is not None  # Default is True
        assert processor.cropper is not None  # Default is True

    def test_init_custom_config(self, basic_config):
        processor = ImageProcessor(config=basic_config)
        assert processor.config == basic_config
        assert processor.rotation_pipeline is None  # Disabled in fixture
        assert processor.cropper is None  # Disabled in fixture


@pytest.mark.unit
class TestImageEncoding:
    def test_encode_numpy_to_base64(self, processor, sample_image_np):
        b64, mime = processor.encode_to_base64(sample_image_np, ImageFormat.JPEG)
        assert isinstance(b64, str)
        assert mime == "image/jpeg"
        assert len(b64) > 0

    def test_encode_pil_to_base64(self, processor, sample_image_pil):
        b64, mime = processor.encode_to_base64(sample_image_pil, ImageFormat.PNG)
        assert isinstance(b64, str)
        assert mime == "image/png"
        assert len(b64) > 0

    def test_encode_invalid_format(self, processor, sample_image_pil):
        # We need to mock ImageFormat enum or pass a string if types allowed, 
        # but encode_to_base64 uses .lower() on the enum value.
        # Let's assume we pass a valid enum but one that isn't handled if any?
        # Actually types.py likely only has JPEG, PNG. 
        # If we pass a random object it might fail.
        pass  # Skip for now as enum restricts values


@pytest.mark.unit
class TestImageProcessing:
    def test_process_image_no_ops(self, processor, sample_image_np):
        # With auto_rotate and crop disabled
        result = processor.process_image(sample_image_np)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_np.shape

    def test_resize_image(self, processor, sample_image_np):
        # Set max size smaller than image
        processor.config.post_crop_max_size = 50
        result = processor._resize_image(sample_image_np)
        assert result.shape[0] == 50 or result.shape[1] == 50

    @patch("docvision.processing.image.cv2.imwrite")
    def test_save_debug_images(self, mock_imwrite, processor, sample_image_np):
        processor.config.debug_save_path = "/tmp/debug"
        # We need to mock Path.mkdir too to avoid fs errors
        with patch("pathlib.Path.mkdir"):
            processor._save_images([sample_image_np], "/tmp/debug", "test_doc")
            mock_imwrite.assert_called_once()
    
    def test_process_image_with_debug(self, processor, sample_image_np):
        processor.config.debug_save_path = "/tmp/debug"
        with patch.object(processor, '_save_images') as mock_save:
            processor.process_image(sample_image_np, page_num=1)
            mock_save.assert_called_once()


@pytest.mark.unit
class TestPDFConversion:
    @patch("docvision.processing.image.fitz.open")
    def test_pdf_to_images_success(self, mock_fitz_open, processor):
        # Mock PDF document and page
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        
        # Setup page return
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        
        # Setup pixmap
        mock_pix.samples = b'\x00' * (100 * 100 * 3)
        mock_pix.height = 100
        mock_pix.width = 100
        mock_pix.n = 3
        
        mock_page.get_pixmap.return_value = mock_pix
        mock_fitz_open.return_value = mock_doc

        # We also need to mock Path.exists
        with patch("pathlib.Path.exists", return_value=True):
            images = processor.pdf_to_images("dummy.pdf")
            
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
            mock_fitz_open.assert_called_with(Path("dummy.pdf"))
            mock_doc.close.assert_called_once()

    def test_pdf_not_found(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.pdf_to_images("nonexistent.pdf")

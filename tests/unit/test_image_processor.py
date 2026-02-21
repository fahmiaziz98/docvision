import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image

from docvision.processing.image import ImageProcessor


@pytest.fixture
def processor():
    return ImageProcessor(
        render_zoom=1,
        enable_rotate=False,
        post_crop_max_size=500
    )


@pytest.fixture
def sample_image_np():
    # Create a simple 100x100 RGB image (numpy array, but ImageProcessor expects BGR for some ops)
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.mark.unit
class TestImageProcessor:
    def test_init(self):
        proc = ImageProcessor(render_zoom=3, enable_rotate=True)
        assert proc.render_zoom == 3
        assert proc.enable_rotate is True
        assert proc.rotation_pipeline is not None

    def test_encode_to_base64_numpy(self, processor, sample_image_np):
        b64, mime = processor.encode_to_base64(sample_image_np)
        assert isinstance(b64, str)
        assert mime == "image/jpeg"
        assert len(b64) > 0

    def test_encode_to_base64_pil(self, processor):
        img_pil = Image.new('RGB', (100, 100), color='red')
        b64, mime = processor.encode_to_base64(img_pil)
        assert isinstance(b64, str)
        assert mime == "image/jpeg"

    def test_process_image_no_ops(self, processor, sample_image_np):
        # enable_rotate is False in fixture
        result = processor.process_image(sample_image_np)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_np.shape

    @patch("docvision.processing.image.fitz.open")
    def test_pdf_to_images_mock(self, mock_fitz_open, processor):
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

        with patch("pathlib.Path.exists", return_value=True):
            images = processor.pdf_to_images("dummy.pdf")
            
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
            mock_fitz_open.assert_called_with(Path("dummy.pdf"))
            mock_doc.close.assert_called_once()

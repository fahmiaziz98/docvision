from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from docvision.processing.image import ImageProcessor


@pytest.fixture
def processor():
    """No-op processor: rotate, crop, deskew all disabled."""
    return ImageProcessor(
        dpi=72,
        enable_rotate=False,
        enable_crop=False,
        enable_deskew=False,
        post_crop_max_size=2000,  # large enough not to resize a 100x100 image
    )


@pytest.fixture
def sample_image_np():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _make_fitz_ctx_mock(page_count: int = 1):
    """Context-manager-compatible fitz.open() mock."""
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = page_count

    mock_pix = MagicMock()
    mock_pix.samples = b"\x00" * (100 * 100 * 3)
    mock_pix.height = 100
    mock_pix.width = 100
    mock_pix.n = 3

    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.load_page.return_value = mock_page

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_doc
    mock_ctx.__exit__.return_value = False
    return mock_ctx


@pytest.mark.unit
class TestImageProcessor:
    def test_init(self):
        proc = ImageProcessor(dpi=150, enable_rotate=True)
        assert proc.dpi == 150
        assert proc.enable_rotate is True
        assert proc.rotation_pipeline is not None

    def test_encode_to_base64_default_png(self, processor, sample_image_np):
        """Default format is now PNG."""
        b64, mime = processor.encode_to_base64(sample_image_np)
        assert isinstance(b64, str)
        assert mime == "image/png"
        assert len(b64) > 0

    def test_encode_to_base64_jpeg(self, processor, sample_image_np):
        b64, mime = processor.encode_to_base64(sample_image_np, img_format="JPEG")
        assert mime == "image/jpeg"

    def test_encode_to_base64_pil(self, processor):
        img_pil = Image.new("RGB", (100, 100), color="red")
        b64, mime = processor.encode_to_base64(img_pil)  # default PNG
        assert isinstance(b64, str)
        assert mime == "image/png"

    def test_encode_to_base64_invalid_format(self, processor, sample_image_np):
        with pytest.raises(ValueError, match="Unsupported"):
            processor.encode_to_base64(sample_image_np, img_format="BMP")

    def test_preprocess_for_vlm_no_ops(self, processor, sample_image_np):
        """With all ops disabled and post_crop_max_size > image size, shape is unchanged."""
        result = processor.preprocess_for_vlm(sample_image_np)
        assert isinstance(result, np.ndarray)
        # Shape should be unchanged - no crop, no resize
        assert result.shape == sample_image_np.shape

    def test_preprocess_for_ocr_returns_array(self, processor, sample_image_np):
        result = processor.preprocess_for_ocr(sample_image_np)
        assert isinstance(result, np.ndarray)

    def test_preprocess_for_vlm_resize(self):
        """When image exceeds post_crop_max_size, it should be downscaled."""
        proc = ImageProcessor(
            dpi=72,
            enable_rotate=False,
            enable_crop=False,
            post_crop_max_size=50,
        )
        large_img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = proc.preprocess_for_vlm(large_img)
        assert max(result.shape[:2]) <= 50

    @patch("docvision.processing.image.fitz.open")
    def test_pdf_to_images_mock(self, mock_fitz_open, processor):
        """fitz.open() is used as a context manager."""
        mock_fitz_open.return_value = _make_fitz_ctx_mock(page_count=1)

        with patch("pathlib.Path.exists", return_value=True):
            images = processor.pdf_to_images("dummy.pdf")

        assert len(images) == 1
        assert isinstance(images[0], np.ndarray)
        mock_fitz_open.assert_called_once()

    @patch("docvision.processing.image.fitz.open")
    def test_pdf_to_images_page_range(self, mock_fitz_open, processor):
        mock_fitz_open.return_value = _make_fitz_ctx_mock(page_count=5)

        with patch("pathlib.Path.exists", return_value=True):
            images = processor.pdf_to_images("dummy.pdf", start_page=2, end_page=3)

        assert len(images) == 2

    def test_pdf_to_images_file_not_found(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.pdf_to_images("nonexistent.pdf")

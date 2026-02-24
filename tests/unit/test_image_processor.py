from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from docvision.processing.image import ImageProcessor


@pytest.fixture
def processor():
    return ImageProcessor(
        render_zoom=1,
        enable_rotate=False,
        enable_deskew=False,
        post_crop_max_size=500,
    )


@pytest.fixture
def sample_image_np():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _make_fitz_ctx_mock(page_count: int = 1):
    """Return a context-manager-compatible fitz.open() mock."""
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
        img_pil = Image.new("RGB", (100, 100), color="red")
        b64, mime = processor.encode_to_base64(img_pil)
        assert isinstance(b64, str)
        assert mime == "image/jpeg"

    def test_encode_to_base64_png(self, processor, sample_image_np):
        b64, mime = processor.encode_to_base64(sample_image_np, img_format="PNG")
        assert mime == "image/png"

    def test_encode_to_base64_invalid_format(self, processor, sample_image_np):
        with pytest.raises(ValueError, match="Unsupported"):
            processor.encode_to_base64(sample_image_np, img_format="BMP")

    def test_preprocess_for_vlm_no_ops(self, processor, sample_image_np):
        # enable_rotate=False, no rotation pipeline active
        result = processor.preprocess_for_vlm(sample_image_np)
        assert isinstance(result, np.ndarray)
        # padding adds 32px total (16 each side) to each dimension
        assert result.shape[0] == 100 + 32
        assert result.shape[1] == 100 + 32

    def test_preprocess_for_ocr_no_rotate(self, processor, sample_image_np):
        result = processor.preprocess_for_ocr(sample_image_np)
        assert isinstance(result, np.ndarray)

    @patch("docvision.processing.image.fitz.open")
    def test_pdf_to_images_mock(self, mock_fitz_open, processor):
        # fitz.open() is used as a context manager: 'with fitz.open(...) as doc:'
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

        # Should render pages 2 and 3 only
        assert len(images) == 2

    def test_pdf_to_images_file_not_found(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.pdf_to_images("nonexistent.pdf")

import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..core.types import ImageFormat, ParserConfig
from .crop import ContentCropper
from .rotate import AutoRotate


class ImageProcessor:
    """
    Handles image processing tasks including PDF conversion, auto-rotation,
    cropping, resizing, and encoding.
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize the ImageProcessor with configuration.

        Args:
            config: Configuration object. If None, default ParserConfig is used.
        """
        self.config = config or ParserConfig()

        self.rotation_pipeline = None
        if self.config.enable_auto_rotate:
            self.rotation_pipeline = AutoRotate(
                hough_threshold=self.config.hough_threshold,
                min_score_diff=self.config.min_score_diff,
                analysis_max_size=self.config.analysis_max_size,
                use_aspect_ratio_fallback=self.config.use_aspect_ratio_fallback,
                aggressive_mode=self.config.aggressive_mode,
            )
        self.cropper = None
        if self.config.enable_crop:
            self.cropper = ContentCropper(
                padding=self.config.crop_padding,
                ignore_bottom_percent=self.config.crop_ignore_bottom_percent,
                max_crop_percent=self.config.crop_max_crop_percent,
            )

    def pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Convert a PDF file into a list of images (numpy arrays).

        Args:
            pdf_path: Path to the PDF file.
            start_page: The first page to convert (1-indexed).
            end_page: The last page to convert (1-indexed, inclusive).

        Returns:
            A list of images as numpy arrays (BGR format).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if start_page is None:
            start_page = 1

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            start_page = max(1, start_page)
            end_page = min(total_pages, end_page or total_pages)

            page_iterator = range(start_page, end_page + 1)
            if self.config.progress_callback is None:
                page_iterator = tqdm(page_iterator, desc="Converting PDF to images")

            images = []
            for page_num in page_iterator:
                page = doc.load_page(page_num - 1)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(self.config.render_zoom, self.config.render_zoom),
                    alpha=False,
                )
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    (pix.height, pix.width, pix.n)
                )

                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                images.append(img)

            doc.close()
            return images
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")

    def process_image(
        self, image: Union[np.ndarray, Image.Image], page_num: Optional[int] = None
    ) -> Union[np.ndarray, Image.Image]:
        """
        Apply enabled processing steps (rotation, cropping, resizing) to an image.

        Args:
            image: The input image (numpy array or PIL Image).
            page_num: Optional page number for naming debug images.

        Returns:
            The processed image.
        """
        if isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = np.array(image)

        try:
            if self.config.enable_auto_rotate:
                img_np, rotation_result = self.rotation_pipeline.auto_rotate(img_np)

            if self.config.enable_crop:
                img_np = self.cropper.crop(img_np)

            img_np = self._resize_image(img_np)

            if self.config.debug_save_path:
                doc_name = (
                    f"processed_page_{page_num}" if page_num is not None else "processed_image"
                )
                self._save_images(
                    [img_np],
                    self.config.debug_save_path,
                    doc_name,
                    ImageFormat.JPEG,
                    self.config.jpeg_quality,
                )

            return img_np
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {e}")

    def _resize_image(
        self, image: Union[np.ndarray, Image.Image]
    ) -> Union[np.ndarray, Image.Image]:
        """
        Resize the image if its dimensions exceed the configured maximum size.

        Args:
            image: Input image.

        Returns:
            Resized image.
        """
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.size

        if max(h, w) > self.config.post_crop_max_size:
            scale = self.config.post_crop_max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return image

    def _save_images(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        output_dir: Union[str, Path],
        doc_name: str = "images",
        img_format: ImageFormat = ImageFormat.JPEG,
        quality: int = 95,
    ) -> List[str]:
        """
        Save a list of images to a specified directory.

        Args:
            images: List of images to save.
            output_dir: Target directory.
            doc_name: Prefix for the filename.
            img_format: Output image format.
            quality: JPEG quality (if applicable).

        Returns:
            A list of paths where the images were saved.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, image in enumerate(images):
            if "page" in doc_name:
                filename = f"{doc_name}.{img_format.lower()}"
            else:
                filename = f"{doc_name}_page_{i}.{img_format.lower()}"

            path = output_dir / filename
            if isinstance(image, np.ndarray):
                cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                image.save(path)
            saved_paths.append(str(path))

        return saved_paths

    @staticmethod
    def encode_to_base64(
        image: Union[np.ndarray, Image.Image],
        img_format: ImageFormat = ImageFormat.JPEG,
        quality: int = 95,
    ) -> Tuple[str, str]:
        """
        Encode an image to a base64 string.

        Args:
            image: Image to encode.
            img_format: Target format for encoding.
            quality: Quality for compression.

        Returns:
            A tuple of (base64_string, mime_type).
        """
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image

        buffer = BytesIO()
        fmt = img_format.lower()

        if fmt in ("jpg", "jpeg"):
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
            mime_type = "image/jpeg"
        elif fmt == "png":
            pil_image.save(buffer, format="PNG", optimize=True)
            mime_type = "image/png"
        elif fmt == "webp":
            pil_image.save(buffer, format="WEBP", quality=quality)
            mime_type = "image/webp"
        else:
            raise ValueError(f"Unsupported format: {img_format}")

        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return b64_data, mime_type

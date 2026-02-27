import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np
from PIL import Image

from .crop import ContentCropper
from .rotate import AutoRotate


class ImageProcessor:
    """
    Handles all image processing for document parsing pipelines.

    Two intentional preprocessing methods:
    - preprocess_for_ocr(): high-contrast, DPI-normalized, deskewed (for PaddleOCR)
    - preprocess_for_vlm(): natural color, smart resize, padded (for Vision LLMs)

    All enhancement logic lives here as private methods — no separate enhance.py.
    """

    def __init__(
        self,
        dpi: float = 300,
        enable_crop: bool = True,
        padding_size: int = 10,
        enable_rotate: bool = True,
        rotate_aggressive_mode: bool = False,
        enable_deskew: bool = True,
        post_crop_max_size: int = 1024,
        debug_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the ImageProcessor.

        Args:
            dpi: DPI for PDF rendering. 2.0 ≈ 144 DPI.
            enable_crop: Enable content cropping to remove whitespace.
            padding_size: Padding size for content cropping.
            enable_rotate: Auto-correct large rotations (90°/180°) via Hough transform.
            rotate_aggressive_mode: Aggressive mode rotate
            enable_deskew: Correct small skew angles (1–5°) before OCR.
            post_crop_max_size: Max image dimension for VLM preprocessing.
                                Recommended: 1536 for 8B models, 2048 for 72B+ models,
                                1024 for 3B and below.
            debug_dir: If set, saves preprocessed images here for inspection.
        """
        self.dpi = dpi
        self.enable_crop = enable_crop
        self.enable_rotate = enable_rotate
        self.enable_deskew = enable_deskew
        self.post_crop_max_size = post_crop_max_size
        self.debug_dir = Path(debug_dir) if debug_dir else None

        self.rotation_pipeline = AutoRotate(aggressive_mode=rotate_aggressive_mode)
        self.content_cropper = ContentCropper(padding=padding_size)

    def pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Render PDF pages to BGR numpy arrays using PyMuPDF (fitz).

        Args:
            pdf_path: Path to the PDF file.
            start_page: First page to render (1-indexed). Default: first page.
            end_page: Last page to render (1-indexed, inclusive). Default: last page.

        Returns:
            List of BGR numpy arrays, one per page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                _start = max(1, start_page or 1)
                _end = min(total_pages, end_page or total_pages)

                images = []
                for page_num in range(_start, _end + 1):
                    page = doc.load_page(page_num - 1)
                    zoom = self.dpi / 72
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom),
                        alpha=False,
                    )
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        (pix.height, pix.width, pix.n)
                    )
                    img = cv2.cvtColor(
                        img,
                        cv2.COLOR_BGRA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR,
                    )
                    images.append(img)

            return images
        except Exception as e:
            raise RuntimeError(f"Failed to render PDF '{pdf_path.name}': {e}") from e

    def preprocess_for_ocr(
        self,
        image: np.ndarray,
        page_num: Optional[int] = None,
    ) -> np.ndarray:
        """
        Preprocess image for PaddleOCR ONNX inference.

        Pipeline:
        1. Auto-rotate  — correct large rotations (90°/180°) [if enable_rotate]
        2. Normalize DPI — ensure min resolution for reliable text detection
        3. CLAHE contrast enhancement — improve text/background separation
        4. Deskew — correct small skew angles 1–5° [if enable_deskew]

        Args:
            image: Input BGR numpy array.
            page_num: Optional page number used for debug image filenames.

        Returns:
            Preprocessed BGR numpy array ready for OCR.
        """
        img = image.copy()

        if self.enable_rotate:
            img, _ = self.rotation_pipeline.auto_rotate(img)

        img = self._normalize_dpi(img)
        img = self._enhance_contrast_clahe(img)

        if self.enable_deskew:
            img = self._deskew(img)

        if self.debug_dir:
            self._save_debug([img], prefix=f"ocr_p{page_num or 0}")

        return img

    def preprocess_for_vlm(
        self,
        image: np.ndarray,
        page_num: Optional[int] = None,
    ) -> np.ndarray:
        """
        Preprocess image for Vision Language Model inference.

        Pipeline:
        1. Auto-rotate    — correct large rotations (90°/180°) [if enable_rotate]
        2. White balance  — correct color cast from lighting
        3. Adaptive resize — cap longest dimension at post_crop_max_size
        4. Unsharp mask   — recover text edge sharpness lost during downscale
        5. Add padding    — prevent content from touching image edges

        Intentionally minimal — VLMs are robust to noise and compression.
        Over-processing destroys color/texture context that VLMs rely on.

        Args:
            image: Input BGR numpy array.
            page_num: Optional page number used for debug image filenames.

        Returns:
            Preprocessed BGR numpy array ready for VLM encoding.
        """
        img = image.copy()

        if self.enable_rotate:
            img, _ = self.rotation_pipeline.auto_rotate(img)

        if self.enable_crop:
            img = self.content_cropper.crop(img)

        # img = self._normalize_white_balance(img)
        img = self._adaptive_resize(img, max_size=self.post_crop_max_size)
        # img = self._unsharp_mask(img)
        # img = self._add_padding(img)

        if self.debug_dir:
            self._save_debug([img], prefix=f"vlm_p{page_num or 0}")

        return img

    @staticmethod
    def encode_to_base64(
        image: Union[np.ndarray, Image.Image],
        img_format: str = "PNG",
        quality: int = 95,
    ) -> Tuple[str, str]:
        """
        Encode an image to base64 string for API transmission.

        Args:
            image: BGR numpy array or PIL Image.
            img_format: Output format — 'JPEG', 'PNG', or 'WEBP'. Default 'JPEG'.
            quality: Compression quality for JPEG/WEBP. Default 95.

        Returns:
            Tuple of (base64_string, mime_type).
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.ndim == 3 and image.shape[2] == 3
                else image
            )
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
            raise ValueError(
                f"Unsupported image format: '{img_format}'. Use 'JPEG', 'PNG', or 'WEBP'."
            )

        return base64.b64encode(buffer.getvalue()).decode("utf-8"), mime_type

    def _normalize_dpi(
        self,
        image: np.ndarray,
        min_height: int = 800,
        max_height: int = 4000,
    ) -> np.ndarray:
        """
        Ensure image has enough vertical resolution for reliable OCR.

        PaddleOCR recognition expects ~32px per text line.
        For typical documents (~40 lines/page), minimum ~800px height is needed.

        Upscales if below min_height, downscales if above max_height.
        """
        h, w = image.shape[:2]

        if h < min_height:
            scale = min_height / h
            return cv2.resize(image, (int(w * scale), min_height), interpolation=cv2.INTER_CUBIC)

        if h > max_height:
            scale = max_height / h
            return cv2.resize(image, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)

        return image

    def _enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Improves text visibility on uneven backgrounds — yellowed paper,
        shadows, camera photos. Applied on the L channel of LAB color space
        to avoid color shifting.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_ch)

        return cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    def _deskew(self, image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
        """
        Correct small skew angles (up to max_angle degrees).

        Uses horizontal projection profile variance maximization:
        a correctly aligned text image has higher variance in row-wise pixel sums
        because text lines create peaks, while gaps create valleys.

        Searches in 0.5° steps. Skips correction if detected angle < 0.3°
        to avoid introducing interpolation artifacts unnecessarily.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        best_angle = 0.0
        best_score = -1.0
        angles = [a * 0.5 for a in range(int(-max_angle * 2), int(max_angle * 2) + 1)]

        for angle in angles:
            rotated = self._rotate_for_deskew(binary, angle)
            score = float(np.var(np.sum(rotated, axis=1).astype(np.float64)))
            if score > best_score:
                best_score = score
                best_angle = angle

        if abs(best_angle) < 0.3:
            return image

        return self._rotate_for_deskew(image, best_angle)

    @staticmethod
    def _rotate_for_deskew(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by an arbitrary angle, expanding canvas to avoid cropping."""
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        m[0, 2] += (new_w / 2.0) - center[0]
        m[1, 2] += (new_h / 2.0) - center[1]

        border = 255 if image.ndim == 2 else (255, 255, 255)
        return cv2.warpAffine(
            image,
            m,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    @staticmethod
    def _normalize_white_balance(image: np.ndarray) -> np.ndarray:
        """
        Gray-world white balance correction.

        Corrects color cast from artificial lighting (warm/cool tones).
        Assumes the average color of the scene should be neutral gray.
        """
        result = image.astype(np.float32)
        means = [np.mean(result[:, :, c]) for c in range(3)]
        mean_gray = sum(means) / 3.0

        for c, mean_c in enumerate(means):
            if mean_c > 0:
                result[:, :, c] *= mean_gray / mean_c

        return np.clip(result, 0, 255).astype(np.uint8)

    def _adaptive_resize(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """
        Resize so the longest side does not exceed max_size.
        LANCZOS for downscaling (best quality), CUBIC for upscaling.
        """
        h, w = image.shape[:2]
        longest = max(h, w)

        if longest <= max_size:
            return image

        scale = max_size / longest
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _unsharp_mask(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Apply unsharp mask to recover edge sharpness lost during resize.

        Uses Gaussian blur as the blurred reference, then blends:
            sharpened = original * (1 + strength) - blurred * strength

        strength=0.5 is conservative — recovers text edges without
        introducing halos or over-sharpening document backgrounds.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _add_padding(
        image: np.ndarray,
        padding: int = 16,
        color: tuple = (255, 255, 255),
    ) -> np.ndarray:
        """
        Add white padding around the image.

        Prevents VLM from missing text flush against the image edge,
        which can happen after cropping or rotation.
        """
        return cv2.copyMakeBorder(
            image,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=color,
        )

    def _save_debug(
        self,
        images: List[np.ndarray],
        prefix: str = "debug",
        quality: int = 95,
    ) -> None:
        """Save preprocessed images to debug_dir for visual inspection."""
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            path = self.debug_dir / f"{prefix}_{i}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])

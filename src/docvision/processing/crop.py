from typing import Optional, Tuple

import cv2
import numpy as np


class ContentCropper:
    """
    Cropper to remove whitespace from document images.

    Uses projection profile analysis to detect main content
    boundaries and remove empty areas.
    """

    def __init__(
        self,
        padding: int = 10,
        ignore_bottom_percent: float = 12.0,
        footer_gap_threshold: int = 100,
        column_ink_ratio: float = 0.01,
        row_ink_ratio: float = 0.002,
        max_crop_percent: float = 30.0,
    ):
        """
        Initialize cropper.

        Args:
            padding: Padding around content (pixels)
            ignore_bottom_percent: Footer ignore area (%)
            footer_gap_threshold: Vertical gap threshold (pixels)
            column_ink_ratio: Min ink ratio for vertical projection
            row_ink_ratio: Min ink ratio for horizontal projection
            max_crop_percent: Maximum cropping percentage to prevent over-crop
        """
        self.padding = padding
        self.ignore_bottom_percent = ignore_bottom_percent
        self.footer_gap_threshold = footer_gap_threshold
        self.column_ink_ratio = column_ink_ratio
        self.row_ink_ratio = row_ink_ratio
        self.max_crop_percent = max_crop_percent

    def _find_main_content_block(self, h_proj: np.ndarray) -> Tuple[int, int]:
        """
        Find main content block based on projection.

        Args:
            h_proj: Horizontal projection

        Returns:
            Tuple (top, bottom) bounds
        """
        idx = np.where(h_proj > 0)[0]
        if len(idx) == 0:
            return 0, len(h_proj) - 1

        gaps = np.diff(idx)
        large_gaps = np.where(gaps > self.footer_gap_threshold)[0]

        if len(large_gaps) == 0:
            return idx[0], idx[-1]

        # Find largest block
        blocks = []
        start = 0
        for g in large_gaps:
            block = idx[start : g + 1]
            blocks.append((block[0], block[-1], len(block)))
            start = g + 1

        # Add last block
        block = idx[start:]
        if len(block) > 0:
            blocks.append((block[0], block[-1], len(block)))

        # Return largest block
        top, bottom, _ = max(blocks, key=lambda x: x[2])
        return top, bottom

    def _projection_bounds(self, proj: np.ndarray, min_pixels: int) -> Optional[Tuple[int, int]]:
        """
        Find bounds from projection profile.

        Args:
            proj: Projection array
            min_pixels: Minimum pixels threshold

        Returns:
            Tuple (start, end) atau None
        """
        idx = np.where(proj > min_pixels)[0]
        if len(idx) == 0:
            return None
        return idx[0], idx[-1]

    def _validate_crop(
        self, original_h: int, original_w: int, top: int, bottom: int, left: int, right: int
    ) -> Tuple[int, int, int, int]:
        """
        Validate cropping to prevent over-crop.

        Args:
            original_h, original_w: Original dimensions
            top, bottom, left, right: Crop boundaries

        Returns:
            Validated boundaries
        """
        h_crop_percent = (1 - (bottom - top) / original_h) * 100
        w_crop_percent = (1 - (right - left) / original_w) * 100

        if h_crop_percent > self.max_crop_percent:
            expand = int(original_h * (h_crop_percent - self.max_crop_percent) / 200)
            top = max(0, top - expand)
            bottom = min(original_h, bottom + expand)

        if w_crop_percent > self.max_crop_percent:
            expand = int(original_w * (w_crop_percent - self.max_crop_percent) / 200)
            left = max(0, left - expand)
            right = min(original_w, right + expand)

        return top, bottom, left, right

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to remove whitespace.

        Args:
            image: Input image (BGR)

        Returns:
            Cropped image
        """
        original_h, original_w = image.shape[:2]
        h, w = original_h, original_w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h_proj = np.sum(binary > 0, axis=1)
        v_proj = np.sum(binary > 0, axis=0)

        if self.ignore_bottom_percent > 0:
            cut = int(h * self.ignore_bottom_percent / 100)
            h_proj[-cut:] = 0

        min_row_pixels = int(w * self.row_ink_ratio)
        min_col_pixels = int(h * self.column_ink_ratio)

        tb = self._projection_bounds(h_proj, min_row_pixels)
        lr = self._projection_bounds(v_proj, min_col_pixels)

        if not tb or not lr:
            return image

        top, bottom = tb
        left, right = lr

        top, bottom = self._find_main_content_block(h_proj)

        top, bottom, left, right = self._validate_crop(
            original_h, original_w, top, bottom, left, right
        )

        top = max(0, top - self.padding)
        bottom = min(original_h, bottom + self.padding)
        left = max(0, left - self.padding)
        right = min(original_w, right + self.padding)

        cropped = image[top:bottom, left:right]

        cropped = self._second_pass(cropped)

        return cropped

    def _second_pass(self, image: np.ndarray) -> np.ndarray:
        """
        Second pass cropping untuk fine-tuning.

        Args:
            image: Input image dari first pass

        Returns:
        Fine-tuned cropped image
        """
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        v_proj = np.sum(binary > 0, axis=0)
        h_proj = np.sum(binary > 0, axis=1)

        min_col = int(h * self.column_ink_ratio)
        min_row = int(w * self.row_ink_ratio)

        lr = self._projection_bounds(v_proj, min_col)
        tb = self._projection_bounds(h_proj, min_row)

        if lr and tb:
            left, right = lr
            top, bottom = tb

            # Validate second pass
            top, bottom, left, right = self._validate_crop(h, w, top, bottom, left, right)

            return image[top:bottom, left:right]

        return image

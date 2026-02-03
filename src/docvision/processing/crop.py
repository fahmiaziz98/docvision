from typing import Optional, Tuple

import cv2
import numpy as np


class ContentCropper:
    """
    Smart content-aware document cropper that identifies and crops the main content block,
    optionally ignoring footers and applying padding.
    """

    def __init__(
        self,
        padding: int = 10,
        ignore_bottom_percent: float = 12.0,
        footer_gap_threshold: int = 100,
        column_ink_ratio: float = 0.01,
        row_ink_ratio: float = 0.002,
    ):
        """
        Initialize the ContentCropper.

        Args:
            padding: Pixels of padding to add around the cropped area.
            ignore_bottom_percent: Percentage of the image height to ignore at the bottom (for footers).
            footer_gap_threshold: Minimum vertical gap in pixels to consider a separate block (like a footer).
            column_ink_ratio: Ratio of ink pixels in a column to consider it non-empty.
            row_ink_ratio: Ratio of ink pixels in a row to consider it non-empty.
        """
        self.padding = padding
        self.ignore_bottom_percent = ignore_bottom_percent
        self.footer_gap_threshold = footer_gap_threshold
        self.column_ink_ratio = column_ink_ratio
        self.row_ink_ratio = row_ink_ratio

    def _find_main_content_block(self, h_proj: np.ndarray) -> Tuple[int, int]:
        """
        Identify the largest continuous block of content in the horizontal projection.

        Args:
            h_proj: Horizontal projection array (sum of ink pixels per row).

        Returns:
            A tuple of (top, bottom) row indices.
        """
        idx = np.where(h_proj > 0)[0]
        if len(idx) == 0:
            return 0, len(h_proj) - 1

        gaps = np.diff(idx)
        large_gaps = np.where(gaps > self.footer_gap_threshold)[0]

        if len(large_gaps) == 0:
            return idx[0], idx[-1]

        blocks = []
        start = 0
        for g in large_gaps:
            block = idx[start : g + 1]
            blocks.append((block[0], block[-1], len(block)))
            start = g + 1

        block = idx[start:]
        blocks.append((block[0], block[-1], len(block)))

        top, bottom, _ = max(blocks, key=lambda x: x[2])
        return top, bottom

    def _projection_bounds(self, proj: np.ndarray, min_pixels: int) -> Optional[Tuple[int, int]]:
        """
        Calculate the first and last indices where the projection exceeds a threshold.

        Args:
            proj: Projection array.
            min_pixels: Minimum number of pixels to consider the row/column as containing ink.

        Returns:
            A tuple of (start, end) indices, or None if no content is found.
        """
        idx = np.where(proj > min_pixels)[0]
        if len(idx) == 0:
            return None
        return idx[0], idx[-1]

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Apply smart cropping to the input image.

        Args:
            image: Input image as a numpy array (BGR).

        Returns:
            The cropped image as a numpy array.
        """
        h, w = image.shape[:2]

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

        top = max(0, top - self.padding)
        bottom = min(h, bottom + self.padding)
        left = max(0, left - self.padding)
        right = min(w, right + self.padding)

        cropped = image[top:bottom, left:right]

        gray2 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, bin2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h2, w2 = bin2.shape
        v2 = np.sum(bin2 > 0, axis=0)
        h2p = np.sum(bin2 > 0, axis=1)

        min_col2 = int(h2 * self.column_ink_ratio)
        min_row2 = int(w2 * self.row_ink_ratio)

        lr2 = self._projection_bounds(v2, min_col2)
        tb2 = self._projection_bounds(h2p, min_row2)

        if lr2 and tb2:
            l2, r2 = lr2
            t2, b2 = tb2
            cropped = cropped[t2:b2, l2:r2]

        return cropped

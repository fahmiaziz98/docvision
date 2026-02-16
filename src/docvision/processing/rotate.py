from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.types import RotationAngle, RotationResult


class OrientationScorer:
    """
    Class to score image orientation based on text characteristics.

    Logic: Text usually has:
    1. Higher density at the top (header)
    2. Horizontal projection with high variance (text lines)
    3. Text blocks with wide aspect ratios at the top
    4. Document aspect ratio (portrait vs landscape)
    """

    def __init__(
        self,
        top_heavy_weight: float = 0.35,
        variance_weight: float = 0.25,
        header_weight: float = 0.15,
        aspect_ratio_weight: float = 0.25,
        max_top_heavy: float = 5.0,
        max_variance: float = 2.0,
        max_header: float = 1.0,
    ):
        """
        Initialize the OrientationScorer.

        Args:
            top_heavy_weight: Weight for top-heavy score (default: 0.35).
            variance_weight: Weight for projection variance score (default: 0.25).
            header_weight: Weight for header density score (default: 0.15).
            aspect_ratio_weight: Weight for document aspect ratio score (default: 0.25).
            max_top_heavy: Maximum top-heavy score (default: 5.0).
            max_variance: Maximum projection variance score (default: 2.0).
            max_header: Maximum header density score (default: 1.0).
        """
        self.top_heavy_weight = top_heavy_weight
        self.variance_weight = variance_weight
        self.header_weight = header_weight
        self.aspect_ratio_weight = aspect_ratio_weight
        self.max_top_heavy = max_top_heavy
        self.max_variance = max_variance
        self.max_header = max_header

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for orientation scoring.

        Args:
            image: Input image (numpy array in BGR format).

        Returns:
            Binary image (numpy array in BGR format).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def calculate_top_heavy_score(self, binary: np.ndarray) -> float:
        """
        Score how top-heavy the document is.

        Args:
            binary: Binary image (white text on black background).

        Returns:
            Top-heavy score (higher = more top-heavy).
        """
        h, w = binary.shape
        top_third = binary[: h // 3, :]
        middle_third = binary[h // 3 : 2 * h // 3, :]
        bottom_third = binary[2 * h // 3 :, :]

        top_density = np.sum(top_third) / (top_third.size + 1e-6)
        middle_density = np.sum(middle_third) / (middle_third.size + 1e-6)
        bottom_density = np.sum(bottom_third) / (bottom_third.size + 1e-6)

        score = (top_density + middle_density * 0.5) / (bottom_density + 1e-6)

        return min(score, self.max_top_heavy)

    def calculate_variance_score(self, binary: np.ndarray) -> float:
        """
        Score based on horizontal projection variance.

        Args:
            binary: Binary image (white text on black background).

        Returns:
            Variance score (higher = more text lines).
        """
        projection = np.sum(binary, axis=1)

        from scipy.ndimage import gaussian_filter1d

        projection = gaussian_filter1d(projection, sigma=2)

        variance = np.var(projection)

        return min(variance / 1000, self.max_variance)

    def calculate_header_score(self, binary: np.ndarray) -> float:
        """
        Score based on header density.

        Args:
            binary: Binary image (white text on black background).

        Returns:
            Header density score (higher = more header content).
        """
        h, w = binary.shape
        top_third = binary[: h // 3, :]

        contours, _ = cv2.findContours(top_third, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        header_blocks = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / (ch + 1e-6)

                if aspect_ratio > 2:
                    header_blocks += 1

        return min(header_blocks / 10, self.max_header)

    def calculate_aspect_ratio_score(self, image: np.ndarray) -> float:
        """
        Score based on document aspect ratio.

        Args:
            image: Original image (BGR format).

        Returns:
            Aspect ratio score (higher = more portrait-like).
        """
        h, w = image.shape[:2]
        ratio = h / w

        if ratio > 1.1:
            return 1.0  # Portrait
        elif ratio < 0.9:
            return 0.0  # Landscape
        else:
            return 0.5  # Square

    def score(self, image: np.ndarray) -> float:
        """
        Calculate total score for image

        Args:
            image: Input Image (BGR)

        Returns:
            Total score orientation (higher = good orientation)
        """
        binary = self.preprocess(image)

        top_heavy = self.calculate_top_heavy_score(binary)
        variance = self.calculate_variance_score(binary)
        header = self.calculate_header_score(binary)
        aspect = self.calculate_aspect_ratio_score(image)

        total_score = (
            top_heavy * self.top_heavy_weight
            + variance * self.variance_weight
            + header * self.header_weight
            + aspect * self.aspect_ratio_weight
        )

        return total_score


class RotationDetector:
    """
    Document image rotation detector.

    Detects rotation using:
    1. Hough Transform for line detection
    2. Orientation scoring to validate rotation direction
    3. Aspect ratio analysis
    """

    def __init__(
        self,
        hough_threshold: int = 200,
        max_lines: int = 50,
        angle_tolerance: float = 5.0,
        min_score_diff: float = 0.15,
        analysis_max_size: int = 1500,
        use_aspect_ratio_fallback: bool = True,
        aggressive_mode: bool = True,
    ):
        """
        Initialize the rotation detector.

        Args:
            hough_threshold: Threshold for Hough transform
            max_lines: Maximum number of lines to analyze
            angle_tolerance: Angle tolerance for micro-rotation correction
            min_score_diff: Minimum score difference to decide rotation
            analysis_max_size: Maximum size for analysis (downsampling)
            use_aspect_ratio_fallback: Use aspect ratio as fallback
            aggressive_mode: Aggressive mode - rotate if Hough detects 90° even if scoring is ambiguous
        """
        self.hough_threshold = hough_threshold
        self.max_lines = max_lines
        self.angle_tolerance = angle_tolerance
        self.min_score_diff = min_score_diff
        self.analysis_max_size = analysis_max_size
        self.use_aspect_ratio_fallback = use_aspect_ratio_fallback
        self.aggressive_mode = aggressive_mode
        self.scorer = OrientationScorer()

    def _prepare_analysis_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for analysis (downsample if too large).

        Args:
            image: Input image

        Returns:
            Prepared image
        """
        h, w = image.shape[:2]

        if max(h, w) > self.analysis_max_size:
            scale = self.analysis_max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return image

    def _detect_angle_hough(self, image: np.ndarray) -> Optional[float]:
        """
        Detect angle using Hough transform.

        Args:
            image: Input image

        Returns:
            Detected angle (in degrees) or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_threshold)

        if lines is None:
            return None

        angles = []
        for line in lines[: self.max_lines]:
            theta = line[0][1]
            angle = np.degrees(theta) - 90

            # Normalize angle to range [-90, 90]
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            angles.append(angle)

        return np.median(angles)

    def _check_aspect_ratio(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check aspect ratio to detect possible 90° rotation.

        Args:
            image: Input image

        Returns:
            Tuple (is_likely_rotated, confidence)
        """
        h, w = image.shape[:2]
        ratio = h / w

        # Text documents are usually portrait (ratio > 1)
        # If landscape (ratio < 1), it's likely rotated 90°
        if ratio < 0.85:
            # Landscape - likely rotated
            confidence = min((0.85 - ratio) * 2, 1.0)
            return True, confidence

        return False, 0.0

    def _test_90_degree_rotations(
        self, image: np.ndarray, detected_angle: float
    ) -> Tuple[RotationAngle, float]:
        """
        Test both 90-degree orientations and choose the best one.

        Args:
            image: Input image
            detected_angle: Angle detected by Hough transform

        Returns:
            Tuple (best rotation angle, confidence)
        """
        test_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        test_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        score_cw = self.scorer.score(test_cw)
        score_ccw = self.scorer.score(test_ccw)
        score_original = self.scorer.score(image)

        # Check if rotation is needed
        best_rotated_score = max(score_cw, score_ccw)
        score_diff = abs(score_cw - score_ccw)

        # If one rotation is much better than original
        improvement = best_rotated_score - score_original

        if improvement > 0.05:  # Significant improvement
            if score_cw > score_ccw:
                return RotationAngle.CLOCKWISE_90, min(score_diff + 0.3, 1.0)
            else:
                return RotationAngle.COUNTER_CLOCKWISE_90, min(score_diff + 0.3, 1.0)

        # If improvement is small but Hough is confident in 90° and aggressive mode is active
        if self.aggressive_mode and 85 <= abs(detected_angle) <= 95:
            # Hough is confident in 90°, choose direction based on detected_angle
            if detected_angle < 0:  # -90° means CCW is needed
                if score_ccw >= score_cw:
                    return RotationAngle.COUNTER_CLOCKWISE_90, 0.7
                else:
                    return RotationAngle.CLOCKWISE_90, 0.6
            else:  # +90° means CW is needed
                if score_cw >= score_ccw:
                    return RotationAngle.CLOCKWISE_90, 0.7
                else:
                    return RotationAngle.COUNTER_CLOCKWISE_90, 0.6

        # Check aspect ratio as fallback
        if self.use_aspect_ratio_fallback:
            is_rotated, aspect_confidence = self._check_aspect_ratio(image)
            if is_rotated and aspect_confidence > 0.5:
                if score_cw > score_ccw:
                    return RotationAngle.CLOCKWISE_90, aspect_confidence
                else:
                    return RotationAngle.COUNTER_CLOCKWISE_90, aspect_confidence

        # If score diff is too small, assume ambiguous
        if score_diff < self.min_score_diff:
            return RotationAngle.NONE, 0.5

        # Choose the better one
        if score_cw > score_ccw:
            return RotationAngle.CLOCKWISE_90, min(score_diff, 1.0)
        else:
            return RotationAngle.COUNTER_CLOCKWISE_90, min(score_diff, 1.0)

    def detect(self, image: np.ndarray) -> RotationResult:
        """
        Detect rotation for image.

        Args:
            image: Input image (BGR)

        Returns:
            RotationResult with rotation information
        """
        analysis_img = self._prepare_analysis_image(image)

        detected_angle = self._detect_angle_hough(analysis_img)

        if detected_angle is None:
            if self.use_aspect_ratio_fallback:
                is_rotated, confidence = self._check_aspect_ratio(image)
                if is_rotated:
                    best_angle, conf = self._test_90_degree_rotations(analysis_img, 90)
                    return RotationResult(
                        angle=best_angle,
                        confidence=conf,
                        original_angle=0.0,
                        applied_rotation=(best_angle != RotationAngle.NONE),
                    )

            return RotationResult(
                angle=RotationAngle.NONE, confidence=1.0, original_angle=0.0, applied_rotation=False
            )

        abs_angle = abs(detected_angle)

        # Case 1: Angle is close to 0 degrees
        if abs_angle < self.angle_tolerance:
            return RotationResult(
                angle=RotationAngle.NONE,
                confidence=1.0,
                original_angle=detected_angle,
                applied_rotation=False,
            )

        # Case 2: Angle is close to 180 degrees
        if abs_angle > (180 - self.angle_tolerance):
            return RotationResult(
                angle=RotationAngle.ROTATE_180,
                confidence=1.0,
                original_angle=detected_angle,
                applied_rotation=True,
            )

        # Case 3: Angle is close to 90 degrees (CW or CCW)
        if 85 <= abs_angle <= 95:
            best_angle, confidence = self._test_90_degree_rotations(analysis_img, detected_angle)

            return RotationResult(
                angle=best_angle,
                confidence=confidence,
                original_angle=detected_angle,
                applied_rotation=(best_angle != RotationAngle.NONE),
            )

        # Case 4: Small angle, perform correction
        if abs_angle < 45:
            return RotationResult(
                angle=RotationAngle.NONE,
                confidence=1.0,
                original_angle=detected_angle,
                applied_rotation=False,
            )

        # Default: no rotation needed
        return RotationResult(
            angle=RotationAngle.NONE,
            confidence=1.0,
            original_angle=detected_angle,
            applied_rotation=False,
        )


class ImageRotator:
    """
    Class for performing image rotations.
    """

    @staticmethod
    def rotate_90_clockwise(image: np.ndarray) -> np.ndarray:
        """Rotate image 90 degrees clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def rotate_90_counter_clockwise(image: np.ndarray) -> np.ndarray:
        """Rotate image 90 degrees counter-clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def rotate_180(image: np.ndarray) -> np.ndarray:
        """Rotate image 180 degrees."""
        return cv2.rotate(image, cv2.ROTATE_180)

    @staticmethod
    def rotate_arbitrary(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate with arbitrary angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        m = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        m[0, 2] += (new_w / 2) - center[0]
        m[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            m,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return rotated

    @classmethod
    def apply_rotation(cls, image: np.ndarray, rotation_angle: RotationAngle) -> np.ndarray:
        """
        Apply rotation.

        Args:
            image: Input image
            rotation_angle: Rotation angle to be applied

        Returns:
            Rotated image
        """
        if rotation_angle == RotationAngle.NONE:
            return image.copy()
        elif rotation_angle == RotationAngle.CLOCKWISE_90:
            return cls.rotate_90_clockwise(image)
        elif rotation_angle == RotationAngle.COUNTER_CLOCKWISE_90:
            return cls.rotate_90_counter_clockwise(image)
        elif rotation_angle == RotationAngle.ROTATE_180:
            return cls.rotate_180(image)
        else:
            return image.copy()


class AutoRotate:
    """
    Auto-rotate images based on detected orientation.
    """

    def __init__(
        self,
        hough_threshold: int = 200,
        min_score_diff: float = 0.15,
        analysis_max_size: int = 1500,
        use_aspect_ratio_fallback: bool = True,
        aggressive_mode: bool = True,
    ):
        """
        Initialize AutoRotate.

        Args:
            hough_threshold: Threshold for Hough transform
            min_score_diff: Minimum score difference to consider a rotation valid
            analysis_max_size: Maximum size for analysis image
            use_aspect_ratio_fallback: Whether to use aspect ratio fallback
            aggressive_mode: Whether to use aggressive mode
        """
        self.detector = RotationDetector(
            hough_threshold=hough_threshold,
            min_score_diff=min_score_diff,
            analysis_max_size=analysis_max_size,
            use_aspect_ratio_fallback=use_aspect_ratio_fallback,
            aggressive_mode=aggressive_mode,
        )
        self.rotator = ImageRotator()

    def auto_rotate(self, image: np.ndarray) -> Tuple[np.ndarray, RotationResult]:
        """
        Auto-rotate image based on detected orientation.

        Args:
            image: Input image (BGR)

        Returns:
            Tuple of (rotated_image, rotation_result)
        """
        rotation_result = self.detector.detect(image)
        rotated_image = self.rotator.apply_rotation(image, rotation_result.angle)
        return rotated_image, rotation_result

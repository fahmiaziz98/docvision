import numpy as np
import pytest

from docvision.core.types import RotationAngle, RotationResult
from docvision.processing.rotate import AutoRotate


@pytest.mark.unit
class TestAutoRotate:
    def test_init(self):
        rotator = AutoRotate(aggressive_mode=True)
        # Check if it was passed to detector
        assert rotator.detector.aggressive_mode is True

    def test_no_rotate_needed(self):
        # Create a blank square image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        rotator = AutoRotate()

        rotated_img, result = rotator.auto_rotate(img)

        assert isinstance(rotated_img, np.ndarray)
        assert isinstance(result, RotationResult)
        # For a blank image, it should typically return NONE
        assert result.angle == RotationAngle.NONE
        assert result.applied_rotation is False

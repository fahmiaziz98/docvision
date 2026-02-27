import numpy as np
import pytest

from docvision.core.types import RotationAngle, RotationResult
from docvision.processing.rotate import AutoRotate, ImageRotator


@pytest.mark.unit
class TestAutoRotate:
    def test_init_default(self):
        rotator = AutoRotate()
        assert rotator.detector is not None
        assert rotator.rotator is not None

    def test_init_aggressive_mode(self):
        rotator = AutoRotate(aggressive_mode=True)
        assert rotator.detector.aggressive_mode is True

    def test_init_non_aggressive(self):
        rotator = AutoRotate(aggressive_mode=False)
        assert rotator.detector.aggressive_mode is False

    def test_auto_rotate_blank_image_no_rotation(self):
        """A blank image should not trigger a rotation."""
        img = np.zeros((200, 150, 3), dtype=np.uint8)
        rotator = AutoRotate()

        rotated_img, result = rotator.auto_rotate(img)

        assert isinstance(rotated_img, np.ndarray)
        assert isinstance(result, RotationResult)
        assert result.angle == RotationAngle.NONE
        assert result.applied_rotation is False

    def test_auto_rotate_returns_correct_types(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rotator = AutoRotate()

        rotated_img, result = rotator.auto_rotate(img)

        assert rotated_img.dtype == np.uint8
        assert hasattr(result, "angle")
        assert hasattr(result, "confidence")
        assert hasattr(result, "original_angle")
        assert hasattr(result, "applied_rotation")

    def test_auto_rotate_preserves_channel_count(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        rotator = AutoRotate()

        rotated_img, _ = rotator.auto_rotate(img)

        assert rotated_img.ndim == 3
        assert rotated_img.shape[2] == 3


@pytest.mark.unit
class TestImageRotator:
    def test_rotate_90_clockwise(self):
        # A 100x200 image (h=100, w=200) rotated 90° CW → h=200, w=100
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ImageRotator.rotate_90_clockwise(img)
        assert result.shape == (200, 100, 3)

    def test_rotate_90_counter_clockwise(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ImageRotator.rotate_90_counter_clockwise(img)
        assert result.shape == (200, 100, 3)

    def test_rotate_180(self):
        # 180° rotation preserves dimensions
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ImageRotator.rotate_180(img)
        assert result.shape == (100, 200, 3)

    def test_apply_rotation_none(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        rotator = ImageRotator()
        result = rotator.apply_rotation(img, RotationAngle.NONE)
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)

    def test_apply_rotation_clockwise_90(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        rotator = ImageRotator()
        result = rotator.apply_rotation(img, RotationAngle.CLOCKWISE_90)
        assert result.shape == (200, 100, 3)

    def test_apply_rotation_counter_clockwise_90(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        rotator = ImageRotator()
        result = rotator.apply_rotation(img, RotationAngle.COUNTER_CLOCKWISE_90)
        assert result.shape == (200, 100, 3)

    def test_apply_rotation_180(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        rotator = ImageRotator()
        result = rotator.apply_rotation(img, RotationAngle.ROTATE_180)
        assert result.shape == (100, 200, 3)


@pytest.mark.unit
class TestRotationAngleEnum:
    def test_values(self):
        assert RotationAngle.NONE == 0
        assert RotationAngle.CLOCKWISE_90 == 90
        assert RotationAngle.COUNTER_CLOCKWISE_90 == -90
        assert RotationAngle.ROTATE_180 == 180

    def test_is_int(self):
        assert isinstance(RotationAngle.NONE, int)
        assert isinstance(RotationAngle.CLOCKWISE_90, int)

"""Tests for mediapipe_video.codec (encode_rle, threshold_mask) and the proto contract."""
import numpy as np
import pytest

from mediapipe_video.codec import encode_rle, threshold_mask
from mediapipe_video.contract import pb


class TestThresholdMask:
    def test_2d_input_returns_2d(self):
        result = threshold_mask(np.ones((2, 3)), 0.5)
        assert result.shape == (2, 3)
        assert result.all()

    def test_3d_single_channel_squeezed(self):
        for shape in ((2, 3, 1), (1, 3, 1)):
            result = threshold_mask(np.ones(shape), 0.5)
            assert result.shape == shape[:2]
            assert result.all()

    def test_threshold_applied_correctly(self):
        data = np.array([[0.3, 0.6], [0.5, 0.9]])
        result = threshold_mask(data, 0.5)
        expected = np.array([[False, True], [False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="unexpected confidence mask shape"):
            threshold_mask(np.ones((2, 3, 2)), 0.5)


class TestEncodeRle:
    def test_all_background(self):
        assert encode_rle([[0, 0], [0, 0]]) == [4]

    def test_all_foreground(self):
        assert encode_rle([[1, 1], [1, 1]]) == [0, 4]

    def test_mixed_row_major_order(self):
        assert encode_rle([[1, 0, 0], [0, 1, 0]]) == [0, 1, 3, 1, 1]

    def test_roundtrip(self):
        mask = np.random.default_rng(42).random((31, 19)) > 0.5
        counts = encode_rle(mask)
        pixels = np.repeat(np.arange(len(counts)) % 2, counts)
        np.testing.assert_array_equal(pixels.reshape(mask.shape), mask)

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError, match="empty mask"):
            encode_rle([])

    def test_rle_sum_equals_total_pixels(self):
        mask = np.random.default_rng(0).random((10, 10)) > 0.5
        counts = encode_rle(mask)
        assert sum(counts) == mask.size


class TestProtoContract:
    def test_oneof_payload_config(self):
        req = pb.VideoRequest(config=pb.VideoConfig(effect_type="bg_replace"))
        assert req.WhichOneof("payload") == "config"

    def test_oneof_payload_frame_replaces_config(self):
        req = pb.VideoRequest(config=pb.VideoConfig(effect_type="bg_replace"))
        req.frame.image_data = b"image"
        assert req.WhichOneof("payload") == "frame"
        assert not req.HasField("config")

    def test_video_mask_status_field(self):
        mask = pb.VideoMask(status="ok", frame_id=1)
        assert mask.status == "ok"
        assert mask.frame_id == 1

"""Shared test fixtures and helpers for the mediapipe-video test suite."""
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import mediapipe as mp
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

from mediapipe_video.config import MediaPipeConfig
from mediapipe_video.contract import pb
from mediapipe_video.segmenter import Worker


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class Aborted(Exception):
    """Raised by MockContext.abort() so tests can assert on gRPC error codes."""
    def __init__(self, code, details):
        self.code = code
        super().__init__(details)


class MockContext:
    """Minimal stand-in for grpc.ServicerContext."""
    active = True

    def is_active(self):
        return self.active

    def abort(self, code, details):
        raise Aborted(code, details)


class CallbackSegmenter:
    """Mock ImageSegmenter that runs the result callback on a background thread.

    Lets tests exercise Worker.Segment() without a real GPU or TFLite runtime.
    """
    def __init__(self, options, process):
        self.options = options
        self.process = process
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True)
        self.closed = True

    def segment_async(self, image, timestamp_ms):
        self.executor.submit(self.process, self.options.result_callback, image, timestamp_ms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_jpeg(width=3, height=2):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


def deliver_result(callback, image, timestamp_ms):
    """Deliver an all-foreground result then invalidate the borrowed numpy view."""
    plane = np.ones((2, 3, 1), dtype=np.float32)
    result = SimpleNamespace(confidence_masks=[SimpleNamespace(numpy_view=lambda: plane)])
    callback(result, image, timestamp_ms)
    plane.fill(0)  # prove the worker copied the data before callback returned


def make_requests(session="call-tcore", timestamps=(0, 200), eos="separate"):
    jpeg = make_jpeg()
    yield pb.VideoRequest(
        session_id=session, stream_id=session,
        config=pb.VideoConfig(effect_type="bg_replace", inference_width=3, inference_height=2),
    )
    for number, timestamp in enumerate(timestamps, 1):
        yield pb.VideoRequest(
            session_id=session, stream_id=session,
            is_eos=(eos == "frame" and number == len(timestamps)),
            frame=pb.VideoFrame(
                frame_id=number, rtp_timestamp=4294967000 + number,
                timestamp_ms=timestamp, encoding="jpeg", image_data=jpeg,
            ),
        )
    if eos == "separate":
        yield pb.VideoRequest(session_id=session, stream_id=session, is_eos=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def context():
    return MockContext()


@pytest.fixture
def live_stream():
    """Yields (worker, segmenters, process_ref) with ImageSegmenter mocked.

    ``process_ref`` is a one-element list; replace ``process_ref[0]`` to change
    how the mock segmenter delivers (or withholds) results for a single test.
    """
    segmenters = []
    process_ref = [deliver_result]

    def create(options):
        seg = CallbackSegmenter(options, lambda cb, img, ts: process_ref[0](cb, img, ts))
        segmenters.append(seg)
        return seg

    with patch.object(mp.tasks.vision.ImageSegmenter, "create_from_options", side_effect=create):
        with patch.object(Path, "read_bytes", return_value=b"model"):
            worker = Worker(MediaPipeConfig(model=Path("selfie.tflite"), delegate="gpu", result_timeout_secs=0.2))
        yield worker, segmenters, process_ref

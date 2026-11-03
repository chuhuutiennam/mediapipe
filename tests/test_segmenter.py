"""Tests for Worker.Segment() — end-to-end stream handling with mocked MediaPipe."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import patch

import grpc
import mediapipe as mp
import pytest

from tests.conftest import Aborted, MockContext, deliver_result, make_requests
from mediapipe_video.config import MediaPipeConfig
from mediapipe_video.segmenter import Worker


class TestSegmentStream:
    def test_happy_path_separate_eos(self, live_stream, context):
        worker, segmenters, _ = live_stream
        responses = list(worker.Segment(make_requests(eos="separate"), context))
        self._assert_ok_responses(responses, segmenters)

    def test_happy_path_frame_eos(self, live_stream, context):
        worker, segmenters, _ = live_stream
        responses = list(worker.Segment(make_requests(eos="frame"), context))
        self._assert_ok_responses(responses, segmenters)

    def test_happy_path_half_close(self, live_stream, context):
        worker, segmenters, _ = live_stream
        responses = list(worker.Segment(make_requests(eos="half-close"), context))
        self._assert_ok_responses(responses, segmenters)

    def _assert_ok_responses(self, responses, segmenters):
        assert [r.mask.frame_id for r in responses] == [1, 2]
        for number, response in enumerate(responses, 1):
            assert response.session_id == "call-tcore"
            assert response.stream_id == "call-tcore"
            assert response.mask.rtp_timestamp == 4294967000 + number
            assert response.mask.status == "ok"
            assert (response.mask.mask_height, response.mask.mask_width) == (2, 3)
            assert list(response.mask.rle_counts) == [0, 6]
            assert response.is_final
        for seg in segmenters:
            assert seg.options.running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM
            assert seg.options.base_options.delegate == mp.tasks.BaseOptions.Delegate.GPU
            assert callable(seg.options.result_callback)
            assert seg.closed

    def test_response_is_yielded_before_next_frame_is_sent(self, live_stream, context):
        worker, _, _ = live_stream
        first_answered = Event()
        gate_results = []
        source = make_requests(timestamps=(0, 200))

        def gated_requests():
            # Message order: config, frame 1, frame 2, EOS. Hold frame 2 until frame 1 is answered.
            for index, req in enumerate(source):
                if index == 2:
                    gate_results.append(first_answered.wait(timeout=2))
                yield req

        responses = []
        for response in worker.Segment(gated_requests(), context):
            responses.append(response)
            first_answered.set()

        assert gate_results == [True]
        assert [r.mask.frame_id for r in responses] == [1, 2]

    def test_concurrent_streams_keep_callbacks_separate(self, live_stream):
        worker, _, process_ref = live_stream
        worker.result_timeout = 3
        barrier = Barrier(2, timeout=2)

        def process(callback, image, timestamp):
            barrier.wait()
            deliver_result(callback, image, timestamp)

        process_ref[0] = process
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(lambda name=name: list(worker.Segment(make_requests(session=name), MockContext())))
                for name in ("call-tcore", "call-taccess")
            ]
            for name, future in zip(("call-tcore", "call-taccess"), futures):
                responses = future.result(timeout=5)
                assert [r.session_id for r in responses] == [name, name]
                assert [r.mask.status for r in responses] == ["ok", "ok"]

    def test_callback_error_returns_error_mask(self, live_stream, context):
        worker, _, process_ref = live_stream
        process_ref[0] = lambda cb, img, ts: cb(
            SimpleNamespace(confidence_masks=None), img, ts
        )
        responses = list(worker.Segment(make_requests(timestamps=(0,)), context))
        assert responses[0].mask.status == "error"
        assert "confidence masks" in responses[0].mask.error_message

    def test_repeated_timestamp_yields_error_then_recovers(self, live_stream, context):
        worker, _, _ = live_stream
        responses = list(worker.Segment(make_requests(timestamps=(0, 0, 200)), context))
        assert [r.mask.status for r in responses] == ["ok", "error", "ok"]
        assert "timestamp_ms must increase" in responses[1].mask.error_message

    def test_timeout_aborts_and_closes_segmenter(self, live_stream):
        worker, segmenters, process_ref = live_stream
        worker.result_timeout = 0.01
        process_ref[0] = lambda *args: None
        with pytest.raises(Aborted) as exc_info:
            list(worker.Segment(make_requests(), MockContext()))
        assert exc_info.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
        assert segmenters[-1].closed

    def test_cancel_ends_stream_silently_and_closes_segmenter(self, live_stream):
        worker, segmenters, process_ref = live_stream
        ctx = MockContext()
        process_ref[0] = lambda *args: setattr(ctx, "active", False)
        responses = list(worker.Segment(make_requests(), ctx))
        assert responses == []
        assert segmenters[-1].closed

    def test_wrong_callback_timestamp_aborts(self, live_stream, context):
        worker, _, process_ref = live_stream
        process_ref[0] = lambda cb, img, ts: deliver_result(cb, img, ts + 1)
        with pytest.raises(Aborted) as exc_info:
            list(worker.Segment(make_requests(), context))
        assert exc_info.value.code == grpc.StatusCode.INTERNAL

    def test_missing_first_config_aborts(self):
        worker = _make_worker_no_patch()
        with pytest.raises(Aborted) as exc_info:
            list(worker.Segment(iter([]), MockContext()))
        assert exc_info.value.code == grpc.StatusCode.INVALID_ARGUMENT

    def test_gpu_init_error_is_not_silently_retried(self):
        with patch.object(
            mp.tasks.vision.ImageSegmenter, "create_from_options",
            side_effect=RuntimeError("no GPU"),
        ) as create:
            with patch.object(Path, "read_bytes", return_value=b"model"):
                with pytest.raises(RuntimeError, match="no GPU"):
                    Worker(MediaPipeConfig(model=Path("selfie.tflite"), delegate="gpu"))
            assert create.call_count == 1


def _make_worker_no_patch():
    with patch.object(
        mp.tasks.vision.ImageSegmenter,
        "create_from_options",
        return_value=_NoOpSegmenter(),
    ), patch.object(Path, "read_bytes", return_value=b"model"):
        return Worker(MediaPipeConfig(model=Path("selfie.tflite"), delegate="gpu"))


class _NoOpSegmenter:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def segment_async(self, *args): pass

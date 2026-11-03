# mediapipe-video

Python gRPC microservice thực hiện **video background segmentation** theo thời gian thực bằng [MediaPipe ImageSegmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter). Nhận JPEG frame từ gateway, trả về binary mask dạng RLE để gateway áp dụng hiệu ứng `bg_blur`, `bg_replace`, hoặc `bg_remove`.

## Yêu cầu

- Python 3.10+
- TFLite model file (ví dụ: `checkpoint/selfie_segmenter.tflite`)
- GPU với EGL/OpenGL ES nếu dùng `--delegate gpu`

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# hoặc: source .venv/bin/activate  # Linux/macOS

pip install -e ".[dev]"
```

Hoặc dùng Makefile (yêu cầu GNU make):

```bash
make install
```

## Chạy

```bash
python -m mediapipe_video --model checkpoint/selfie_segmenter.tflite
```

```bash
make run        # GPU (default)
make run-cpu    # CPU, không cần GPU
```

### Tất cả options

```
--addr ADDR              Địa chỉ bind gRPC server (default: 127.0.0.1:50052)
--model PATH             Đường dẫn TFLite model [bắt buộc]
--foreground-index INT   Index foreground plane trong confidence_masks
                         (default: tự detect — 0 nếu 1 plane, 1 nếu 2 planes)
--max-streams INT        Số concurrent stream tối đa (default: 8)
--delegate {gpu,cpu}     Compute backend (default: gpu)
--result-timeout FLOAT   Giây timeout chờ MediaPipe callback mỗi frame (default: 10.0)
```

Khi sẵn sàng, worker in ra stdout:
```json
{"event": "ready", "addr": "127.0.0.1:50052", "model": "checkpoint/selfie_segmenter.tflite", "mediapipe": "1.0.1", "running_mode": "LIVE_STREAM", "delegate": "gpu"}
```

Dừng bằng `Ctrl+C` — graceful shutdown trong 2 giây.

## Test

```bash
pytest           # hoặc: make test
make cov         # với coverage report
```

Test suite không cần GPU và không cần model file thực — toàn bộ MediaPipe được mock.

## Lint & Type check

```bash
make lint        # ruff check + auto-fix
make typecheck   # mypy
```

## Cấu trúc thư mục

```
mediapipe/
├── pyproject.toml              # Build config, dependencies, tool settings
├── Makefile                    # Shortcut commands
├── README.md
│
├── src/
│   ├── proto/
│   │   └── video.proto         # gRPC contract (shared với Gateway)
│   └── mediapipe_video/        # Python package chính
│       ├── __init__.py
│       ├── __main__.py         # python -m mediapipe_video entry point
│       ├── contract.py         # Compile src/proto/video.proto → video_pb2
│       ├── codec.py            # encode_rle(), threshold_mask()
│       ├── segmenter.py        # class Worker + gRPC stream handler
│       └── server.py           # main() + gRPC server bootstrap
│
├── tests/
│   ├── conftest.py             # Shared fixtures và test helpers
│   ├── test_codec.py           # Tests cho codec và proto contract
│   └── test_segmenter.py       # Tests cho Worker.Segment() end-to-end
│
├── checkpoint/
│   └── selfie_segmenter.tflite   # (không commit — xem .gitignore)
│
└── docs/
    ├── architecture.md
    └── technical.md
```

## Giao tiếp với Gateway

Worker implement gRPC service `video.VideoStream` theo contract trong `src/proto/video.proto`:

```protobuf
service VideoStream {
  rpc Segment(stream VideoRequest) returns (stream VideoResponse);
}
```

Mỗi stream bắt đầu bằng một `VideoConfig`, tiếp theo là các `VideoFrame` (JPEG bytes), kết thúc bằng `is_eos=true`. Worker trả về một `VideoResponse` (chứa RLE mask) cho mỗi frame nhận được.

Chi tiết đầy đủ: [`docs/technical.md`](docs/technical.md).

## Tài liệu

| File | Nội dung |
|---|---|
| [`docs/deployment.md`](docs/deployment.md) | Cài đặt, cấu hình, start/stop, nâng cấp, log, xử lý lỗi |
| [`docs/architecture.md`](docs/architecture.md) | Architecture diagram, module diagram, call flow, sequence diagram |
| [`docs/technical.md`](docs/technical.md) | API, class, interface, thread model, memory model, event model |

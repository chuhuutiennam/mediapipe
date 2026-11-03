# mediapipe-video

Python gRPC microservice thực hiện **video background segmentation** theo thời gian thực bằng [MediaPipe ImageSegmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter). Nhận JPEG frame từ gateway, trả về binary mask dạng RLE để gateway áp dụng hiệu ứng `bg_blur`, `bg_replace`, hoặc `bg_remove`.

**Source code:** https://github.com/chuhuutiennam/mediapipe

## Yêu cầu

- Python 3.10+
- TFLite model file (ví dụ: `checkpoint/selfie_segmenter.tflite`)
- GPU với EGL/OpenGL ES nếu dùng `--delegate gpu`

## Cài đặt

```bash
git clone https://github.com/chuhuutiennam/mediapipe.git
cd mediapipe

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
make run        # delegate theo config (mặc định auto: thử GPU, lỗi thì chuyển CPU)
make run-cpu    # ép CPU, không cần GPU
```

### Cấu hình

Nếu thư mục đang chạy có `etc/config.yaml` thì file này được tự load. Thứ tự ưu tiên:
**CLI flag > file config** (`--config` hoặc `etc/config.yaml`) **> giá trị mặc định trong code**.
Ý nghĩa từng key xem chú thích trong [`etc/config.yaml`](etc/config.yaml).

### Tất cả options

```
--config PATH            File YAML config (default: etc/config.yaml nếu có)
--addr ADDR              Địa chỉ bind gRPC server (default: 127.0.0.1:50052)
--model PATH             Đường dẫn TFLite model [bắt buộc — qua flag hoặc mediapipe.model]
--foreground-index INT   Index foreground plane trong confidence_masks
                         (default: tự detect — 0 nếu 1 plane, 1 nếu 2 planes)
--gpu-device-index INT   GPU dùng để chạy, set CUDA_VISIBLE_DEVICES
                         (default: GPU mặc định của hệ thống)
--max-streams INT        Số concurrent stream tối đa (default: 8)
--delegate MODE          gpu | cpu | auto (default: auto — thử GPU, lỗi thì chuyển CPU)
--result-timeout FLOAT   Giây timeout chờ MediaPipe callback mỗi frame (default: 10.0)
--log-level LEVEL        DEBUG | INFO | WARNING | ERROR (default: INFO)
```

Log dạng JSON, in ra stdout và ghi vào `logs/mediapipe-video.log` (đổi bằng `logging.file`,
rotate hằng ngày). Khi sẵn sàng sẽ có dòng:
```json
{"timestamp": "...", "level": "INFO", "logger": "mediapipe_video.server", "source": "server.py:...", "message": "server ready  addr=127.0.0.1:50052  model=checkpoint/selfie_segmenter.tflite  mediapipe=<version>  delegate=gpu"}
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
├── etc/
│   └── config.yaml             # Config runtime: server, mediapipe, segmentation, logging, metrics
│
├── scripts/                    # Script triển khai trên server Linux
│   ├── check.sh                # Pre-flight check trước khi cài / nâng cấp / chạy
│   ├── install.sh              # Cài đặt đầy đủ vào APP_DIR (hỗ trợ --offline)
│   ├── upgrade.sh              # Nâng cấp lên source mới
│   ├── start.sh                # Start service
│   └── stop.sh                 # Stop service (SIGTERM, quá hạn thì SIGKILL)
│
├── src/
│   ├── proto/
│   │   └── video.proto         # gRPC contract (shared với Gateway)
│   └── mediapipe_video/        # Python package chính
│       ├── __init__.py
│       ├── __main__.py         # python -m mediapipe_video entry point
│       ├── contract.py         # Compile video.proto (bản trong package) → video_pb2
│       ├── codec.py            # encode_rle(), threshold_mask()
│       ├── segmenter.py        # class Worker + gRPC stream handler
│       ├── server.py           # main(): CLI, load config, gRPC server bootstrap
│       ├── config.py           # Dataclass config + YAML loader
│       ├── log.py              # JSON log formatter, rotate file theo ngày
│       ├── metrics.py          # Prometheus metrics + log snapshot định kỳ
│       └── video.proto         # Bản copy của src/proto/video.proto, đóng gói vào wheel
│
├── tests/
│   ├── conftest.py             # Shared fixtures và test helpers
│   ├── test_codec.py           # Tests cho codec và proto contract
│   └── test_segmenter.py       # Tests cho Worker.Segment() end-to-end
│
├── checkpoint/
│   └── selfie_segmenter.tflite   # model mặc định (~244KB, có trong repo)
│
├── wheels/                     # Wheel cho cài offline (install.sh --offline)
├── logs/                       # Log runtime (file .log không commit)
│
└── docs/
    ├── deployment.md
    ├── architecture.md
    ├── technical.md
    └── metrics.md
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
| [`docs/metrics.md`](docs/metrics.md) | Prometheus metrics, query mẫu, alert, giám sát |

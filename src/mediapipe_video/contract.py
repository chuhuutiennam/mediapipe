"""Compile video.proto at runtime and expose the generated video_pb2 module."""
import importlib
import sys
import tempfile
from pathlib import Path

from grpc_tools import protoc


def load_contract():
    """Generate Python bindings from video.proto bundled inside the package."""
    proto_file = Path(__file__).parent / "video.proto"
    proto_dir = proto_file.parent
    with tempfile.TemporaryDirectory(prefix="video-proto-") as generated:
        if protoc.main(["protoc", f"-I{proto_dir}", f"--python_out={generated}", str(proto_file)]):
            raise RuntimeError("cannot compile video.proto")
        sys.path.insert(0, generated)
        try:
            return importlib.import_module("video_pb2")
        finally:
            sys.path.remove(generated)


pb = load_contract()

from __future__ import annotations
import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
from .io import sha256_file

def clip_boxes(boxes, shape):
    h, w = shape[:2]
    result = []
    for row in boxes:
        if len(row) != 4 or not all(np.isfinite(v) for v in row):
            raise ValueError("Face box must contain four finite coordinates")
        x1, y1, x2, y2 = [int(round(v)) for v in row]
        box = [max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Empty face box")
        result.append(box)
    return result

class InsightFaceDetector:
    def __init__(self, root=None, name="buffalo_l", confidence=0.5, det_size=640, provider=None):
        import os
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        root = Path(root).expanduser() if root else Path.home() / ".insightface"
        model_dir = root / "models" / name
        if not list(model_dir.glob("*.onnx")):
            raise FileNotFoundError(f"Pretrained models missing at {model_dir}; no automatic downloads are performed")
        # Provider selection: explicit argument, else COPHOTOMARK_PROVIDER env
        # (set to CUDA on a GPU host), else CPU. CPU stays the default so local
        # and cloud runs are not accidentally mixed (the provider is part of
        # the evaluation signature; switching between validation and test is
        # rejected as a detector/protocol mismatch).
        requested = (provider or os.environ.get("COPHOTOMARK_PROVIDER", "cpu")).lower()
        provider_name = "CUDAExecutionProvider" if requested in {"cuda", "gpu", "cudaexecutionprovider"} else "CPUExecutionProvider"
        import onnxruntime
        if provider_name not in onnxruntime.get_available_providers():
            raise RuntimeError(f"{provider_name} requested but unavailable; installed providers: "
                               f"{onnxruntime.get_available_providers()}. Install onnxruntime-gpu for CUDA "
                               f"or omit the provider setting to run on CPU.")
        start = time.perf_counter()
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name=name, root=str(root), allowed_modules=["detection"], providers=[provider_name])
        self.app.prepare(ctx_id=0 if provider_name == "CUDAExecutionProvider" else -1,
                         det_size=(det_size, det_size), det_thresh=confidence)
        # ORT may advertise CUDA yet silently instantiate a CPU-only session
        # when CUDA/cuDNN libraries are missing. Check the loaded model itself.
        session = self.app.models["detection"].session
        session_providers = session.get_providers()
        if provider_name not in session_providers:
            raise RuntimeError(f"Requested {provider_name}, but detection session loaded {session_providers}. "
                               "Check the ONNXRuntime/CUDA/cuDNN versions; refusing silent CPU fallback.")
        self.info = {"kind": "insightface_pretrained", "model_pack": name, "provider": provider_name,
                     "session_providers": session_providers,
                     "confidence": confidence, "det_size": det_size,
                     "weights_sha256": {p.name: sha256_file(p) for p in sorted(model_dir.glob("*det*.onnx"))},
                     "initialization_ms": (time.perf_counter() - start) * 1000}

    def detect(self, image):
        return clip_boxes([face.bbox.tolist() for face in self.app.get(image)], image.shape)

class FixedBoxes:
    """Explicit oracle geometry control for tests; never a deployment benchmark."""
    def __init__(self, boxes):
        self.boxes = boxes
        self.info = {"kind": "oracle_boxes", "warning": "Ground-truth geometry; excludes detection errors"}

    def detect(self, image):
        return clip_boxes(self.boxes, image.shape)

def iou(a, b):
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, right-left) * max(0, bottom-top)
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(1, area)

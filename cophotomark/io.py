from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROTECTED_SOURCE = Path("C:/Users/dell'/PycharmProjects/CoPhotoMark").resolve()

def writable(path):
    path = Path(path).resolve()
    if path == PROTECTED_SOURCE or PROTECTED_SOURCE in path.parents:
        raise ValueError("Original project is read-only; select a different output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")

def no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON field: {key}")
        out[key] = value
    return out

def read_json(path):
    with open(path, encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=no_duplicates,
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def write_json(path, value):
    writable(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as stream:
        return [json.loads(line, object_pairs_hook=no_duplicates) for line in stream if line.strip()]

def write_jsonl(path, rows):
    with writable(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with writable(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image

def write_image(path, image):
    path = writable(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

from __future__ import annotations
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from .engine import crop
from .io import IMAGE_EXTS, read_image, read_jsonl, sha256_file, write_jsonl
from .watermark import block_count

def inventory(root, dataset, detector, config, limit=None, groups=None):
    root = Path(root).resolve()
    group_map = {r["relative_path"]: r["group_id"] for r in (read_jsonl(groups) if groups else [])}
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        row = {"path": str(path), "relative_path": relative, "dataset": dataset,
               "source_id": dataset + ":" + digest[:20], "source_sha256": digest,
               "group_id": group_map.get(relative, dataset + ":" + digest),
               "grouping": "supplied_identity_or_capture_group" if relative in group_map else "content_hash_only",
               "config_sha256": config.fingerprint}
        try:
            image = read_image(path)
            boxes = detector.detect(image)
            qualified = [box for box in boxes if min(crop(image, box).shape[:2]) >= config.min_roi and block_count(crop(image, box), config) >= config.min_blocks]
            row.update(status="ok", shape=list(image.shape), detected_boxes=boxes, detected_faces=len(boxes),
                       feasible_roi_count=len(qualified), multi_face_detected=len(boxes) >= 2,
                       short_edges=[min(b[2]-b[0], b[3]-b[1]) for b in boxes])
        except Exception as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows

def split_sources(rows, seed=20261031, validation=0.2, development=0.2):
    if not 0 < validation < 1 or not 0 <= development < 1 or validation + development >= 1:
        raise ValueError("Invalid split fractions")
    # Dedupe identical encoded files BEFORE split; keep a provenance list.
    unique = {}
    for row in rows:
        digest = row["source_sha256"]
        if digest in unique:
            if unique[digest]["group_id"] != row["group_id"]:
                raise ValueError("Duplicate content has conflicting group labels; fix grouping first")
            unique[digest].setdefault("duplicate_paths", []).append(row["path"])
        else:
            unique[digest] = dict(row)
    # Union declared capture/identity groups with exact decoded-pixel duplicates.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    pixel_groups = {}
    for row in unique.values():
        group = row["group_id"]
        find(group)
        if row.get("status") == "ok":
            image = read_image(row["path"])
            pixel_hash = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
            row["pixel_sha256"] = pixel_hash
            if pixel_hash in pixel_groups:
                union(group, pixel_groups[pixel_hash])
            pixel_groups[pixel_hash] = group
    grouped = {}
    for row in unique.values():
        row["group_id"] = find(row["group_id"])
        grouped.setdefault(row["group_id"], []).append(row)
    # Stratify groups by dataset and maximum detected face-count band.
    strata = {}
    for group, members in grouped.items():
        count = max(r.get("detected_faces", 0) for r in members)
        band = "0" if count == 0 else "1" if count == 1 else "2-3" if count <= 3 else "4-6" if count <= 6 else "7+"
        strata.setdefault((members[0]["dataset"], band), []).append(group)
    rng = np.random.default_rng(seed)
    for groups in strata.values():
        rng.shuffle(groups)
        n_val, n_dev = round(len(groups)*validation), round(len(groups)*development)
        for i, group in enumerate(groups):
            split = "validation" if i < n_val else "development" if i < n_val+n_dev else "test"
            for row in grouped[group]:
                row.update(split=split, split_seed=seed)
    return list(unique.values())

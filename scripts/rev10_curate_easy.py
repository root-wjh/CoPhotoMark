# -*- coding: utf-8 -*-
"""Curate the 'easy' WIDER cohort for the revision re-run.

Selects images whose every detected face is large (short edge >= 128 px) and
qualified (>= 3 blocks), 2-3 faces for the multi-face cohort plus a small
single-face cohort for the N=1 row.  Per-event cap for diversity.

Usage:
  python scripts/rev10_curate_easy.py \
      --inventory runs/rev10_easy_20261116/wider_pool_inventory.jsonl \
      --images-root D:\\WIDER_train\\images \
      --output assets/wider_easy_r10 --multi 32 --single 8
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--multi", type=int, default=32)
    ap.add_argument("--single", type=int, default=8)
    ap.add_argument("--min-short-edge", type=int, default=128)
    ap.add_argument("--per-event-cap", type=int, default=4)
    args = ap.parse_args()

    rows = []
    with open(args.inventory, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print("pool:", len(rows))

    def event_of(r):
        return Path(r["relative_path"]).parts[0]

    multi, single = [], []
    for r in rows:
        if r.get("status") != "ok":
            continue
        n = r["detected_faces"]
        se = r.get("short_edges") or []
        if not se:
            continue
        if r["feasible_roi_count"] != n:
            continue
        if min(se) < args.min_short_edge:
            continue
        if 2 <= n <= 3:
            multi.append((min(se), r))
        elif n == 1:
            single.append((se[0], r))

    multi.sort(key=lambda t: -t[0])
    single.sort(key=lambda t: -t[0])
    multi2 = [(se, r) for se, r in multi if r["detected_faces"] == 2]
    multi3 = [(se, r) for se, r in multi if r["detected_faces"] == 3]
    print(f"candidates: multi2={len(multi2)} multi3={len(multi3)} single={len(single)}")

    def take(pool, k):
        cap = defaultdict(int)
        picked = []
        for se, r in pool:
            ev = event_of(r)
            if cap[ev] >= args.per_event_cap:
                continue
            cap[ev] += 1
            picked.append((se, r))
            if len(picked) >= k:
                break
        return picked

    half = args.multi // 2
    picked = (take(multi2, half) + take(multi3, args.multi - half)
              + take(single, args.single))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for se, r in picked:
        src = Path(r["path"])
        dst = out / f"{event_of(r)}__{src.name}"
        shutil.copy2(src, dst)
        manifest.append({"event": event_of(r), "faces": r["detected_faces"],
                         "min_short_edge": se, "file": dst.name})
        print(f"  {dst.name}  faces={r['detected_faces']} min_se={se}")

    with open(out / "curate_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    evs = len({m["event"] for m in manifest})
    print(f"SELECTED {len(manifest)} images across {evs} events -> {out}")


if __name__ == "__main__":
    main()

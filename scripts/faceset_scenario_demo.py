"""Face-set tampering scenario driver (R1-1 / Table5-Figure5 reviewer request).

Scenarios exercised on REAL detected multi-face images:
  delete / duplicate / reorder / insert_unmarked / insert_marked /
  copy_marked_face / replace / watermark_transplant / boundary_tamper /
  erase_reembed_wrong_key / detector_miss_control.

This is a SMALL-SCALE engineering demo with the conservative uncalibrated
exact-tag-match working point. The paper-strength result must use
scripts/run_pipeline.py on the full multi-face corpus with independent
validation calibration. Input images are only read, never modified.
"""
import argparse
import json
import hashlib
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cophotomark.dataset import inventory
from cophotomark.detector import InsightFaceDetector
from cophotomark.experiments import build_trials, enroll_manifest, evaluate, summarize
from cophotomark.io import write_csv, write_json, write_jsonl
from cophotomark.protocol import Config

# Reviewer scenario -> internal attack name and expected security behaviour.
SCENARIOS = [
    ("delete", "删除一张已登记脸", "必须不通过(拒绝/不可判定)"),
    ("duplicate", "复制同一张脸(产生重复编号)", "必须拒绝"),
    ("reorder_exact", "人脸空间重排(完整ROI无插值移动)", "协议允许重排；实际可用率须测量，不能预先归因所有失败"),
    ("reorder", "重排+缩放重采样(几何压力，非纯重排)", "不保证通过，用于区分纯重排与几何失真"),
    ("insert_unmarked", "新增未加水印的人脸", "必须不通过(人数不一致)"),
    ("insert_marked", "插入带其他水印的脸", "必须不通过"),
    ("copy_marked_face", "跨图整张带水印脸移植", "必须不通过"),
    ("replace", "用未水印供体脸替换", "必须不通过"),
    ("watermark_transplant", "残差移植", "必须不通过"),
    ("boundary_tamper", "脸部边界篡改", "压力测试；保留充分有效水印时可能通过，不能声称检测所有像素改动"),
    ("erase_reembed_wrong_key", "擦除后用随机错密钥重嵌", "必须不通过"),
    ("detector_miss_control", "受控漏检一张脸(非视觉攻击对照)", "必须不通过；与自然漏检区分"),
    ("clean", "无攻击基线", "应通过"),
]
SCENARIO_NAMES = [name for name, _, _ in SCENARIOS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of multi-face photos (read-only)")
    parser.add_argument("--dataset", default="wider_multi_face_demo")
    parser.add_argument("--config", default="configs/full64.json")
    parser.add_argument("--output", default="runs/faceset_demo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--clean-only", action="store_true", help="Development check only: no attacks generated; preserve enrollment failures")
    parser.add_argument("--key-file", help="Reuse a private experimental key for paired development checks; do not publish")
    parser.add_argument("--model-root")
    parser.add_argument("--provider", choices=["cpu", "cuda"],
                        help="InsightFace detection provider; default cpu or $COPHOTOMARK_PROVIDER")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    # Prepare the run root before writing the private demo key beneath it.
    out.mkdir(parents=True, exist_ok=True)
    config = Config(**json.loads((root / args.config).read_text(encoding="utf-8")))
    detector = InsightFaceDetector(root=args.model_root, provider=args.provider)

    print(">>> inventory (real detector, read-only input) ...", flush=True)
    rows = inventory(args.input, args.dataset, detector, config, args.limit)
    write_jsonl(out / "inventory.jsonl", rows)
    multi = [r for r in rows if r.get("status") == "ok" and r.get("detected_faces", 0) >= 2]
    single_or_bad = [r for r in rows if r not in multi]
    print(f"    total={len(rows)} multi_face={len(multi)} excluded={len(single_or_bad)}", flush=True)
    # Uncalibrated demo: every source is a development split so the frozen
    # validation gate is bypassed; threshold stays at conservative exact match.
    for row in multi:
        row["split"] = "development"

    key_path = Path(args.key_file).resolve() if args.key_file else out / "keys" / "key0.json"
    if not args.key_file:
        write_json(key_path, {"version": 1, "master_key_hex": secrets.token_hex(32),
                              "purpose": "face-set scenario demo key"})

    print(">>> enroll ...", flush=True)
    records = enroll_manifest(multi, out / "enrolled", detector, config, [key_path],
                              "CoPhotoMark faceset scenario demo", run_seed=20261102)
    enrolled_all = [r for r in records if r.get("status") == "enrolled"]
    # Face-set scenarios need FULL-COVERAGE multi-face enrollment (every
    # detected face enrolled, N>=2); partial coverage would let a suppressed
    # excluded face masquerade as a miss-detection false accept.
    enrolled = [r for r in enrolled_all
                if r.get("coverage") == "all_detected_faces" and r.get("enrolled_faces", 0) >= 2]
    partial = [r for r in enrolled_all if r not in enrolled]
    print(f"    enrolled={len(enrolled_all)} full_coverage_multi_face={len(enrolled)} "
          f"partial_or_single_kept_out={len(partial)} failed={len(records)-len(enrolled_all)}", flush=True)
    if not enrolled:
        raise SystemExit("No FULL-COVERAGE image with >=2 eligible non-overlapping >=64px faces")

    print(">>> build face-set attack trials ...", flush=True)
    # Minimal plan: clean baseline plus the full face-operation battery.
    plan = [{"name": "clean", "label": "genuine"}]
    if args.clean_only:
        trials = []
        for record in enrolled:
            trials.append({**{k: record[k] for k in ("source_id", "source_sha256", "group_id", "split", "dataset", "claim_path", "key_path", "key_id", "enrolled_faces", "coverage")},
                           "trial_id": hashlib.sha256((record["source_id"]+record["key_id"]+"clean").encode()).hexdigest()[:24],
                           "image_path": record["watermarked_path"], "metadata_path": None,
                           "baseline_detected_faces": record["detected_faces"],
                           "attack": "clean", "attack_spec": {"name": "clean"}, "label": "genuine",
                           "provenance": "development_clean_diagnostic_only"})
    else:
        trials = build_trials(enrolled, out / "attacks", plan=plan, repeats=args.repeats,
                              seed=20261102, face_operations=True)

    # Save exactly the trials that are evaluated. No external-sidecar assisted
    # results are mixed into this deployment-channel demo.
    write_jsonl(out / "attacks" / "trials.jsonl", trials)

    print(">>> verify every trial (blind + auto), conservative uncalibrated threshold ...", flush=True)
    scores = evaluate(trials, out / "scores", detector, config, modes=("blind", "auto"),
                      calibration=None, use_voting=True, warmup=1)

    print(">>> summarize ...", flush=True)
    summary = summarize(scores, records)
    table = []
    for attack, zh, expectation in SCENARIOS:
        for mode in ("blind", "auto"):
            hit = [r for r in summary["by_attack"]
                   if r["attack"] == attack and r["mode"] == mode]
            if not hit:
                table.append({"reviewer_scenario": zh, "attack": attack, "mode": mode,
                              "trials": 0, "note": "not generated (e.g. fewer than 2 enrolled faces)"})
                continue
            r = hit[0]
            table.append({"reviewer_scenario": zh, "attack": attack, "mode": mode,
                          "expectation": expectation, "trials": r["trials"],
                          "source_images": r["source_images"],
                          "accepted": r["accepted"], "rejected": r["rejected"],
                          "undecidable": r["undecidable"],
                          "accept_rate": round(r["accept_rate"], 4),
                          "reject_rate": round(r["explicit_reject_rate"], 4),
                          "undecidable_rate": round(r["undecidable_rate"], 4),
                          "cluster_ci95_accept": r["accept_rate_cluster_ci95"]})
    write_csv(out / "faceset_scenarios.csv", table)
    write_json(out / "summary.json", summary)
    from cophotomark.plots import plot_examples
    plot_examples(scores, out / "examples")
    write_json(out / "demo_disclaimer.json", {
        "purpose": "Reviewer R1-1 face-set scenarios on multi-face images",
        "clean_only": args.clean_only,
        "scale": "SMALL demo on whatever multi-face photos were supplied; NOT the paper-scale test set",
        "threshold": "uncalibrated exact tag match (conservative); use run_pipeline.py + validation calibration for paper results",
        "input_inventory_total": len(rows), "multi_face_images": len(multi),
        "full_coverage_multi_face_enrolled": len(enrolled),
        "partial_coverage_or_single_face_excluded_from_scenario_table": len(partial),
        "enrollment_failures": len(records) - len(enrolled_all),
        "scenario_mapping": [{"attack": a, "scenario": zh, "expectation": e} for a, zh, e in SCENARIOS]})

    print("\n=== Face-set scenario results ===")
    for row in table:
        if row["trials"]:
            print(f"{row['attack']:<24} {row['mode']:<6} n={row['trials']:<3} "
                  f"A={row['accepted']} R={row['rejected']} U={row['undecidable']}  {row['reviewer_scenario']}")
    print(f"\nTable written: {out / 'faceset_scenarios.csv'}")


if __name__ == "__main__":
    main()

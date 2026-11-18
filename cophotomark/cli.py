from __future__ import annotations
import argparse
import secrets
from pathlib import Path
from .dataset import inventory, split_sources
from .detector import FixedBoxes, InsightFaceDetector
from .engine import enroll, verify
from .experiments import build_trials, enroll_manifest, evaluate, load_key, summarize, far_curve, evaluation_signature
from .io import read_image, read_json, read_jsonl, writable, write_csv, write_image, write_json, write_jsonl
from .protocol import Config
from .statistics import calibrate

def config_from(path):
    return Config(**read_json(path)) if path else Config()

def detector_from(args):
    if getattr(args, "oracle_boxes", None):
        return FixedBoxes(read_json(args.oracle_boxes))
    return InsightFaceDetector(root=getattr(args, "model_root", None),
                               provider=getattr(args, "provider", None))

def common(parser, detector=False):
    parser.add_argument("--config", help="Public JSON protocol configuration")
    if detector:
        parser.add_argument("--model-root", help="Parent containing models/buffalo_l; default ~/.insightface")
        parser.add_argument("--provider", choices=["cpu", "cuda"],
                            help="Detection provider; default cpu or $COPHOTOMARK_PROVIDER. Keep the SAME provider for validation and test")
        parser.add_argument("--oracle-boxes", help="JSON box list, TEST/ORACLE CONTROL ONLY; never use for primary reported results")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent CoPhotoMark revision research toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("keygen")
    p.add_argument("--output", required=True)
    p = commands.add_parser("inventory")
    common(p, True)
    p.add_argument("--input", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True); p.add_argument("--groups")
    p.add_argument("--limit", type=int)
    p = commands.add_parser("split")
    p.add_argument("--input", nargs="+", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--validation", type=float, default=0.2); p.add_argument("--development", type=float, default=0.2)
    p = commands.add_parser("enroll")
    common(p, True)
    p.add_argument("--input", required=True); p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--output", required=True); p.add_argument("--message", default="CoPhotoMark research")
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--debug-metadata", action="store_true", help="Also write optional .cpm.json copies; default stores metadata only in PNG")
    p = commands.add_parser("embed-one")
    common(p, True)
    p.add_argument("--image", required=True); p.add_argument("--key", required=True)
    p.add_argument("--output", required=True); p.add_argument("--message", default="CoPhotoMark research")
    p.add_argument("--debug-metadata", action="store_true")
    p = commands.add_parser("verify-one")
    common(p, True)
    p.add_argument("--image", required=True); p.add_argument("--key", required=True)
    p.add_argument("--claim"); p.add_argument("--metadata",
               help="Optional external .cpm.json sidecar (debug/preserved-channel control); "
                    "by default metadata is read from the channel embedded inside the image")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["blind", "metadata", "auto"], default="auto")
    p.add_argument("--calibration", help="Frozen validation thresholds; omission uses conservative exact tag match, labeled uncalibrated")
    p = commands.add_parser("attacks")
    p.add_argument("--enrollments", required=True); p.add_argument("--output", required=True)
    p.add_argument("--plan"); p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261031); p.add_argument("--no-face-operations", action="store_true")
    p = commands.add_parser("evaluate")
    common(p, True)
    p.add_argument("--trials", required=True); p.add_argument("--output", required=True)
    p.add_argument("--split", required=True, choices=["development", "validation", "test"])
    p.add_argument("--calibration"); p.add_argument("--modes", nargs="+", choices=["blind", "metadata", "auto"], default=["blind", "auto"])
    p.add_argument("--no-voting", action="store_true"); p.add_argument("--warmup", type=int, default=1)
    p = commands.add_parser("calibrate")
    common(p)
    p.add_argument("--scores", required=True); p.add_argument("--output", required=True)
    p.add_argument("--target-far", type=float, default=0.001)
    p = commands.add_parser("report")
    p.add_argument("--scores", required=True); p.add_argument("--enrollments"); p.add_argument("--output", required=True)
    p = commands.add_parser("plot")
    p.add_argument("--scores", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "keygen":
        if Path(args.output).exists():
            raise FileExistsError("Will not overwrite an existing key")
        write_json(args.output, {"version": 1, "master_key_hex": secrets.token_hex(32), "purpose": "Keep private; exclude from public repository"})
    elif args.command == "inventory":
        rows = inventory(args.input, args.dataset, detector_from(args), config_from(args.config), args.limit, args.groups)
        write_jsonl(args.output, rows)
        print(f"Inventoried {len(rows)} files; this is not a count of completed evaluations")
    elif args.command == "split":
        rows = [r for path in args.input for r in read_jsonl(path)]
        rows = split_sources(rows, args.seed, args.validation, args.development)
        write_jsonl(args.output, rows)
        print({split: sum(r["split"] == split for r in rows) for split in ("development", "validation", "test")})
    elif args.command == "enroll":
        enroll_manifest(read_jsonl(args.input), args.output, detector_from(args), config_from(args.config), args.keys, args.message, args.seed, args.debug_metadata)
    elif args.command == "embed-one":
        result = enroll(read_image(args.image), detector_from(args), config_from(args.config), load_key(args.key), args.message)
        out = Path(args.output)
        if out.suffix.lower() != ".png":
            raise ValueError("Enrollment outputs lossless PNG; lossy formats are evaluated as attacks")
        write_image(out, result.pop("image"))  # self-contained: metadata embedded in Y LSBs
        metadata = result.pop("metadata")
        if args.debug_metadata:
            write_json(out.with_suffix(".cpm.json"), metadata)
        write_json(out.with_suffix(".claim.json"), result.pop("claim"))
        write_json(out.with_suffix(".enrollment.json"), result)
    elif args.command == "verify-one":
        config = config_from(args.config)
        detector = detector_from(args)
        threshold = 1.0
        if args.calibration:
            calibration = read_json(args.calibration)
            if calibration["config_sha256"] != config.fingerprint:
                raise ValueError("Calibration/config mismatch")
            if calibration.get("evaluation_signature") != evaluation_signature(detector, config):
                raise ValueError("Detector/voting protocol differs from validation calibration")
            threshold = calibration["modes"][args.mode]["threshold"]
            if threshold is None:
                raise ValueError("Validation could not meet requested FAR")
        result = verify(read_image(args.image), detector, config, load_key(args.key),
                        read_json(args.claim) if args.claim else None,
                        read_json(args.metadata) if args.metadata and args.mode != "blind" else None,
                        args.mode, threshold)
        result["threshold_status"] = "validation_calibrated" if args.calibration else "uncalibrated_exact_match_control"
        write_json(args.output, result)
        print(result["decision"], result["reason"])
    elif args.command == "attacks":
        build_trials(read_jsonl(args.enrollments), args.output, read_json(args.plan) if args.plan else None,
                     args.repeats, args.seed, not args.no_face_operations)
    elif args.command == "evaluate":
        rows = [r for r in read_jsonl(args.trials) if r["split"] == args.split]
        if not rows:
            raise ValueError("No trials in requested split")
        evaluate(rows, args.output, detector_from(args), config_from(args.config), args.modes,
                 read_json(args.calibration) if args.calibration else None, not args.no_voting, args.warmup)
    elif args.command == "calibrate":
        if not 0 <= args.target_far < 1:
            raise ValueError("target-far must be in [0,1)")
        result = calibrate(read_jsonl(args.scores), config_from(args.config), args.target_far)
        write_json(args.output, result)
        write_csv(Path(args.output).with_suffix(".roc.csv"), [r for v in result["modes"].values() for r in v["roc"]])
    elif args.command == "report":
        from .experiments import summarize_r9
        enrollments_data = read_jsonl(args.enrollments) if args.enrollments else None
        result = summarize(read_jsonl(args.scores), enrollments_data)
        write_json(Path(args.output)/"summary.json", result)
        write_csv(Path(args.output)/"by_attack.csv", result["by_attack"])
        main = [r for r in read_jsonl(args.scores) if r.get("coverage") == "all_detected_faces" and r.get("enrolled_faces", 0) >= 2]
        write_csv(Path(args.output)/"by_attack_full_coverage_multiface.csv", summarize(main)["by_attack"])
        clean_ok = {(r["source_id"], r["key_id"], r["mode"]) for r in main
                    if r["attack"] == "clean" and r["decision"] == "ACCEPTED"}
        paired = [r for r in main if (r["source_id"], r["key_id"], r["mode"]) in clean_ok]
        write_csv(Path(args.output)/"by_attack_clean_accepted_control.csv", summarize(paired)["by_attack"])
        write_csv(Path(args.output)/"far_theory_vs_empirical.csv", far_curve(read_jsonl(args.scores)))
        # R9 supplementary curves (undecidable / exclusion rate vs. N, face-size bands)
        r9 = summarize_r9(read_jsonl(args.scores), enrollments_data)
        write_json(Path(args.output)/"r9_metrics.json", r9)
        write_csv(Path(args.output)/"r9_by_N.csv", r9["by_N"])
        write_csv(Path(args.output)/"r9_by_attack_N.csv", r9["by_attack_N"])
        write_csv(Path(args.output)/"r9_face_size.csv", r9["per_face_size"])
    elif args.command == "plot":
        from .experiments import summarize_r9
        from .plots import plot_scores, plot_r9
        scores = read_jsonl(args.scores)
        plot_scores(scores, args.output)
        # Also generate R9 plots if enrollments are available (optional arg)
        # For CLI, we run R9 when the report has already been written.
        # plot_r9 needs a second call after report writes r9_metrics.json
        # Here we try to auto-detect the report directory:
        report_dir = Path(args.output).parent
        enrollments = report_dir.parent / "enrolled" / "enrollment.jsonl"
        if enrollments.exists():
            r9 = summarize_r9(scores, read_jsonl(str(enrollments)))
            plot_r9(r9, Path(args.output) / "r9_plots")

if __name__ == "__main__":
    main()

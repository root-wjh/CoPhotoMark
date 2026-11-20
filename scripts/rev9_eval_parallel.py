import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cophotomark.io import read_jsonl, write_jsonl, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True)
    parser.add_argument("--split", required=True, choices=["development", "validation", "test"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--provider", default="cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(args.trials) if r["split"] == args.split]
    if not rows:
        raise SystemExit("No trials in requested split")
    chunks = [rows[i::args.workers] for i in range(args.workers)]
    part_paths = []
    for i, chunk in enumerate(chunks):
        part = out / f"_trials_part_{i}.jsonl"
        write_jsonl(part, chunk)
        part_paths.append(part)
    print(f"{len(rows)} trials -> {args.workers} parts", flush=True)

    procs = []
    started = time.perf_counter()
    for i, part in enumerate(part_paths):
        cmd = [sys.executable, "-m", "cophotomark", "evaluate", "--trials", str(part),
               "--split", args.split, "--config", args.config, "--output", str(out / f"_part_{i}"),
               "--provider", args.provider]
        if args.calibration:
            cmd += ["--calibration", args.calibration]
        procs.append(subprocess.Popen(cmd, cwd=str(root), env={**__import__("os").environ, "PYTHONPATH": str(root)}))
    failed = False
    for p in procs:
        if p.wait() != 0:
            failed = True
    if failed:
        raise SystemExit("A parallel evaluate worker failed")
    results = []
    for i in range(args.workers):
        results.extend(read_jsonl(out / f"_part_{i}" / "scores.jsonl"))
    write_jsonl(out / "scores.jsonl", results)
    env = read_json(out / "_part_0" / "environment.json")
    env["parallel_workers"] = args.workers
    env["wall_seconds_total"] = round(time.perf_counter() - started, 1)
    from cophotomark.io import write_json
    write_json(out / "environment.json", env)
    for i in range(args.workers):
        for stale in (out / f"_part_{i}", out / f"_trials_part_{i}.jsonl"):
            if stale.is_dir():
                for f in stale.iterdir():
                    f.unlink()
                stale.rmdir()
            elif stale.exists():
                stale.unlink()
    print(f"wrote {len(results)} rows to {out / 'scores.jsonl'} in {time.perf_counter()-started:.0f} s", flush=True)


if __name__ == "__main__":
    main()

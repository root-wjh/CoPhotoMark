"""Re-render Tables 5/8 and the ROC report from frozen validation/test scores.

This never re-evaluates images and never reads test rows during calibration:
the calibration file must have been produced from the validation split only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cophotomark.io import read_json, read_jsonl
from cophotomark.protocol import Config
from cophotomark.threshold_revision import report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing run directory with validation/test scores")
    parser.add_argument("--calibration", required=True, help="Frozen calibration.json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    run = Path(args.run)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(**read_json(args.config))
    calibration = read_json(args.calibration)
    if calibration["config_sha256"] != config.fingerprint:
        raise ValueError("Calibration/config mismatch")
    validation = read_jsonl(run / "validation" / "scores.jsonl")
    test = read_jsonl(run / "test" / "scores.jsonl")
    records = read_jsonl(run / "enrolled" / "enrollment.jsonl")
    report(validation, test, calibration, config, output, records)
    print("Wrote", output)


if __name__ == "__main__":
    main()

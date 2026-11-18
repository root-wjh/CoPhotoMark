"""Offline ROC/FAR/FRR and frozen-threshold report from validation scores.

Usage:
  python scripts/calibrate_scores.py \
      --scores  runs/<run>/validation/scores.jsonl \
      --config  configs/full64_step18.json \
      --sources runs/<run>/sources.jsonl \
      --output  runs/<run>/calibration_report

Reads only the held-out validation split.  No test row is read, so the report
cannot re-tune a threshold on test outcomes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

from cophotomark.calibration import calibrate_units
from cophotomark.io import read_json, read_jsonl
from cophotomark.protocol import Config


def accepted(row, threshold):
    if not row.get("structural_ok") or row.get("status") == "error":
        return False
    score = float(row.get("raw_score", row.get("score", 0.0)))
    return score >= threshold


def image_level(rows, modes, calibration):
    out = []
    for mode in modes:
        threshold = calibration["modes"][mode]["threshold"]
        subset = [r for r in rows if r.get("mode") == mode]
        clean = [r for r in subset if r.get("attack") == "clean" and r.get("label") == "genuine"]
        genuine = [r for r in subset if r.get("label") == "genuine"]
        negative = [r for r in subset if r.get("label") == "negative"]
        out.append({
            "mode": mode,
            "threshold": threshold,
            "tau": calibration["modes"][mode]["tau"],
            "clean_images": len(clean),
            "clean_acceptance": sum(accepted(r, threshold) for r in clean) / len(clean) if clean else None,
            "genuine_trials": len(genuine),
            "genuine_acceptance": sum(accepted(r, threshold) for r in genuine) / len(genuine) if genuine else None,
            "negative_trials": len(negative),
            "image_far": sum(accepted(r, threshold) for r in negative) / len(negative) if negative else None,
        })
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sources")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(**read_json(args.config))
    rows = read_jsonl(args.scores)
    calibration = calibrate_units(rows, config)
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    roc_rows, far_rows = [], []
    for name, unit in calibration["modes"].items():
        if "roc" not in unit:
            continue
        for point in unit["roc"]:
            roc_rows.append({"unit": name, **point})
        for point in unit["roc"]:
            far_rows.append({
                "unit": name, "tau": point["tau"], "threshold": point["threshold"],
                "bound_single_face": float(binom.cdf(point["tau"], 48, 0.5)),
                "empirical_far": point["far"], "empirical_frr": point["frr"],
                "false_accepts": point["false_accepts"], "true_accepts": point["true_accepts"],
            })
    for filename, table in (("roc_data", roc_rows), ("far_frr_by_tau", far_rows)):
        if table:
            with (output / f"{filename}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)

    image = image_level(rows, ("auto", "blind"), calibration)
    with (output / "image_level.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image[0]))
        writer.writeheader()
        writer.writerows(image)

    if args.sources:
        sources = read_jsonl(args.sources)
        with (output / "validation_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["source_id", "source_sha256", "group_id", "split", "detected_faces", "enrolled_faces", "coverage"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sources)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    colors = {"auto": "#176b91", "blind": "#d07824", "set": "#3f7d3f"}
    for name, unit in calibration["modes"].items():
        if "roc" not in unit:
            continue
        roc = unit["roc"]
        axes[0].plot([p["far"] for p in roc], [1 - p["frr"] for p in roc],
                     "-", color=colors.get(name, "gray"), label=name)
        axes[1].plot([p["threshold"] for p in roc], [p["frr"] for p in roc],
                     "-", color=colors.get(name, "gray"), label=name)
        axes[0].plot(unit["validation"]["far"], 1 - unit["validation"]["frr"], "o",
                     color=colors.get(name, "gray"))
    axes[0].set(xlabel="False acceptance rate (FAR)", ylabel="True acceptance rate (1-FRR)",
                xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Tag threshold gamma", ylabel="False rejection rate (FRR)",
                xlim=(0, 1.03), ylim=(-0.02, 1.02))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(output / "roc_frr.png", dpi=200)
    fig.savefig(output / "roc_frr.pdf")
    plt.close(fig)

    lines = ["# Held-out validation calibration", "",
             f"Config `{config.fingerprint[:16]}`; tag length B = {config.tag_bits} bits.", "",
             "| Unit | budget | N_faces | tau | gamma | bound | measured FAR | measured FRR | samples (pos/neg) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, unit in calibration["modes"].items():
        if "roc" not in unit:
            lines.append(f"| {name} | {unit['budget']} | - | - | infeasible | - | - | - | - |")
            continue
        val = unit["validation"]
        lines.append(
            f"| {name} | {unit['budget']:.0e} | {unit['max_faces']} | {unit['tau']} | "
            f"{unit['threshold']:.4f} | {unit['bound_at_threshold']:.2e} | "
            f"{val['far']:.4f} ({val['false_accepts']}/{val['negative_samples']}) | "
            f"{val['frr']:.4f} | {val['positive_samples']}/{val['negative_samples']} |")
    lines += ["", "Image-level decisions at the frozen thresholds:", "",
              "| Mode | gamma | clean acceptance | genuine acceptance | image FAR |",
              "|---|---|---|---|---|"]
    for row in image:
        lines.append(
            f"| {row['mode']} | {row['threshold']:.4f} | {row['clean_acceptance']:.4f} | "
            f"{row['genuine_acceptance']:.4f} | {row['image_far']:.4f} |")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

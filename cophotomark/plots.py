from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .io import writable, write_csv, read_image
from .statistics import cluster_bootstrap

def plot_scores(rows, output):
    output = Path(output)
    plot_rows = []
    for dimension in ("enrolled_faces", "source_short_edge"):
        groups = defaultdict(list)
        for row in rows:
            value = row.get(dimension)
            if value is None or row.get("label") != "genuine":
                continue
            if dimension == "enrolled_faces":
                band = "1" if value == 1 else "2-3" if value <= 3 else "4-6" if value <= 6 else "7+"
            else:
                band = "<64" if value < 64 else "64-95" if value < 96 else "96-127" if value < 128 else "128+"
            groups[(row["mode"], band)].append(row)
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for mode in sorted({k[0] for k in groups}):
            order = ["1", "2-3", "4-6", "7+"] if dimension == "enrolled_faces" else ["<64", "64-95", "96-127", "128+"]
            for band in order:
                data = groups.get((mode, band), [])
                if not data:
                    continue
                def metric(sample):
                    return sum(r["decision"] == "ACCEPTED" for r in sample)/len(sample)
                mean, ci = metric(data), cluster_bootstrap(data, metric)
                x = order.index(band) + (-0.06 if mode == "blind" else 0.06)
                ax.plot(x, mean, "o", label=mode if band == next(b for b in order if (mode,b) in groups) else None)
                if ci[0] is not None:
                    ax.vlines(x, ci[0], ci[1])
                n_sources = len({r["source_id"] for r in data})
                ax.annotate(f"n={n_sources}", (x, mean), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
                plot_rows.append({"dimension": dimension, "band": band, "mode": mode, "sources": n_sources,
                                  "trials": len(data), "genuine_accept_rate": mean, "ci_low": ci[0], "ci_high": ci[1]})
            ax.set_xticks(range(len(order)), order)
        ax.set(xlabel=dimension, ylabel="Genuine acceptance (source-cluster 95% CI)", ylim=(-0.03,1.12))
        if groups:
            ax.legend()
        fig.savefig(writable(output/f"{dimension}.png"), dpi=180)
        plt.close(fig)
    write_csv(output/"curve_data.csv", plot_rows)
    plot_examples(rows, output / "examples")

def plot_r9(r9_dict, output):
    """R9 supplementary plots: undecidable rate vs. N, face-size distribution."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # 1. Undecidable / accept rate vs. enrollment face count N
    by_n = r9_dict.get("by_N", [])
    if by_n:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained", sharex=True)
        for mode in sorted({r["mode"] for r in by_n}):
            subset = sorted([r for r in by_n if r["mode"] == mode], key=lambda r: r["enrolled_faces_N"])
            xs = [r["enrolled_faces_N"] for r in subset]
            accepts = [r["accept_rate"] for r in subset]
            undecs = [r["undecidable_rate"] for r in subset]
            axes[0].plot(xs, accepts, "o-", label=mode)
            axes[1].plot(xs, undecs, "s-", label=mode)
        axes[0].set(xlabel="Enrolled faces N", ylabel="Accept rate", title="Accept vs. N", ylim=(-0.02, 1.05))
        axes[1].set(xlabel="Enrolled faces N", ylabel="Undecidable rate", title="Undecidable vs. N", ylim=(-0.02, 1.05))
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.savefig(writable(output/"r9_by_N.png"), dpi=180)
        plt.close(fig)
    write_csv(output/"r9_by_N.csv", by_n)

    # 2. Undecidable by attack x N (grouped bar)
    by_attack_n = r9_dict.get("by_attack_N", [])
    if by_attack_n:
        # Focus on a few key attacks for readability
        key_attacks = ["clean", "delete", "duplicate", "reorder", "insert_unmarked", "insert_marked",
                       "replace", "copy_marked_face", "watermark_transplant", "boundary_tamper",
                       "jpeg", "crop", "cropout", "blur"]
        subset = [r for r in by_attack_n if r["attack"] in key_attacks and r["enrolled_faces_N"] <= 6]
        if subset:
            fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
            n_vals = sorted({r["enrolled_faces_N"] for r in subset})
            modes = sorted({r["mode"] for r in subset})
            width = 0.8 / len(modes) / len(key_attacks)
            for mi, mode in enumerate(modes):
                for ai, atk in enumerate(key_attacks):
                    xs = []; ys = []
                    for r in subset:
                        if r["mode"] == mode and r["attack"] == atk:
                            xs.append(r["enrolled_faces_N"] + (ai - len(key_attacks)/2 + 0.5) * width * 2 + (mi - 0.5) * width)
                            ys.append(r["undecidable_rate"])
                    if xs:
                        ax.scatter(xs, ys, s=30, label=f"{mode}/{atk}" if mi==0 else None, alpha=0.7)
            ax.set(xlabel="Enrolled faces N", ylabel="Undecidable rate")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=2)
            fig.savefig(writable(output/"r9_undecidable_by_attack_N.png"), dpi=180)
            plt.close(fig)
        write_csv(output/"r9_by_attack_N.csv", by_attack_n)

    # 3. Per-face size distribution
    face_size = r9_dict.get("per_face_size", [])
    if face_size:
        order = ["<40", "40-63", "64-95", "96-127", "128+"]
        splits = sorted({r["split"] for r in face_size})
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        width = 0.8 / max(len(splits), 1)
        for si, split in enumerate(splits):
            split_data = {r["band"]: r for r in face_size if r["split"] == split}
            xs = [order.index(b) for b in order if b in split_data]
            ys = [split_data[b]["faces"] for b in order if b in split_data]
            xs_shift = [x + (si - 0.5) * width for x in xs]
            ax.bar(xs_shift, ys, width, label=split)
        ax.set(xticks=range(len(order)), xticklabels=order, xlabel="Face short-edge band (px)", ylabel="Face count")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
        fig.savefig(writable(output/"r9_face_size_distribution.png"), dpi=180)
        plt.close(fig)
    write_csv(output/"r9_face_size.csv", face_size)

def plot_examples(rows, output, limit=3):
    """Deterministic source selection independent of verification success.

    Each contact sheet includes all available modes and labels the three-state
    decision. Missing/generated-but-skipped scenarios remain visible in the
    coverage CSV; these figures are examples, never aggregate evidence.
    """
    output = Path(output)
    attacks = ["clean", "delete", "duplicate", "reorder_exact", "reorder",
               "insert_unmarked", "insert_marked", "replace", "copy_marked_face",
               "watermark_transplant", "boundary_tamper", "detector_miss_control"]
    sources = sorted({r["source_id"] for r in rows})[:limit]
    manifest = []
    for source in sources:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), layout="constrained")
        fig.suptitle(f"Source: {source}\nFirst repetition/key by ID; examples only (not selected by outcome)", fontsize=11)
        for ax, attack in zip(axes.flat, attacks):
            subset = [r for r in rows if r["source_id"] == source and r["attack"] == attack]
            ax.axis("off")
            if not subset:
                ax.set_title(f"{attack}\nnot generated / skipped", fontsize=9)
                continue
            trial = min(r["trial_id"] for r in subset)
            selected = [r for r in subset if r["trial_id"] == trial]
            row = selected[0]
            ax.imshow(read_image(row["image_path"])[:, :, ::-1])
            decisions = "; ".join(f"{r['mode']}: {r['decision']}" for r in selected)
            ax.set_title(f"{attack}\n{decisions}", fontsize=8)
            manifest.append({"source_id": source, "attack": attack, "trial_id": trial,
                             "image_path": row["image_path"], "decisions": decisions})
        import hashlib
        name = hashlib.sha256(source.encode()).hexdigest()[:16]
        fig.savefig(writable(output / f"source_{name}.png"), dpi=140)
        plt.close(fig)
    write_csv(output / "examples_manifest.csv", manifest)

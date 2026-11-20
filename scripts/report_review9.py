from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cophotomark.io import read_json, read_jsonl, write_csv, write_json
from cophotomark.statistics import cluster_bootstrap
from cophotomark.calibration import DEFAULT_UNITS, _face_vectors, _image_vectors, tag_bound

# (attack-name, comment, scenario, expectation).  Specs sharing a name are
# split further by their parameter string in the table.
SCENARIOS = [
    ("delete", 1, "删除一张已登记人脸", "必须不通过(拒绝/不可判定)"),
    ("duplicate", 1, "复制同一张人脸(产生重复编号)", "必须拒绝"),
    ("replace", 1, "用未加水印供体脸替换", "必须不通过"),
    ("insert_unmarked", 1, "新增未加水印人脸", "必须不通过(人数不一致)"),
    ("insert_marked", 1, "插入带其他水印的人脸", "必须不通过"),
    ("detector_miss_control", 1, "受控漏检一张脸(对照)", "必须不通过"),
    ("reorder_exact", 1, "人脸空间重排(整ROI无插值平移)", "协议允许重排,度量可用率"),
    ("reorder", 1, "重排+重采样(几何压力)", "几何失真可破坏证据,不保证通过"),
    ("copy_marked_face", 2, "跨图整张带水印脸复制移植", "必须不通过"),
    ("watermark_transplant", 2, "水印残差移植", "必须不通过"),
    ("collusion", 2, "共谋(同宿主多密钥版本平均)", "必须不通过"),
    ("erase_reembed_wrong_key", 2, "擦除后用错误密钥重嵌入", "必须不通过"),
    ("boundary_tamper", 2, "ROI边界篡改", "压力测试,不声称检测所有像素改动"),
    ("rotate", 2, "旋转", "良性几何攻击,度量鲁棒性"),
    ("resize", 2, "缩放", "良性几何攻击,度量鲁棒性"),
    ("stretch", 2, "拉伸", "良性几何攻击,度量鲁棒性"),
    ("screenshot_simulation", 2, "截图(数字模拟)", "良性攻击,度量鲁棒性"),
    ("jpeg", 8, "JPEG重压缩(平台重压缩)", "端到端性能场景"),
    ("platform_simulation", 8, "反复上传下载处理(数字模拟)", "端到端性能场景"),
    ("metadata_strip", 8, "元数据剥离", "端到端性能场景(转盲回退)"),
    ("format_conversion", 8, "格式转换(WebP)", "端到端性能场景"),
    ("wrong_key", 3, "错误密钥(完整候选搜索)", "必须不通过"),
    ("wrong_claim", 3, "错误消息", "必须不通过"),
]

TIMING_GROUPS = {
    "clean": ["clean"],
    "lossy_compression": ["jpeg", "platform_simulation"],
    "geometric": ["resize", "stretch", "rotate", "screenshot_simulation"],
    "metadata_and_format": ["metadata_strip", "format_conversion"],
    "face_set_tampering": ["delete", "duplicate", "replace", "copy_marked_face",
                           "watermark_transplant", "insert_unmarked", "insert_marked",
                           "boundary_tamper", "reorder", "reorder_exact", "collusion",
                           "erase_reembed_wrong_key", "detector_miss_control"],
    "wrong_key_controls": ["wrong_key", "wrong_claim"],
}


def spec_label(spec):
    if not isinstance(spec, dict):
        return ""
    skip = {"name", "label", "note"}
    parts = [f"{k}={v}" for k, v in sorted(spec.items()) if k not in skip]
    return ";".join(parts)


def rates(rows):
    n = len(rows)
    if n == 0:
        return None
    ac = sum(r.get("decision") == "ACCEPTED" for r in rows)
    rj = sum(r.get("decision") == "REJECTED" for r in rows)
    un = sum(r.get("decision") == "UNDECIDABLE" for r in rows)

    def metric(sample):
        return sum(r.get("decision") == "ACCEPTED" for r in sample) / len(sample)

    ci = cluster_bootstrap(rows, metric)
    return {"trials": n, "source_images": len({r["source_id"] for r in rows}),
            "keys": len({r["key_id"] for r in rows}),
            "accepted": ac, "rejected": rj, "undecidable": un,
            "accept_rate": round(ac / n, 4), "reject_rate": round(rj / n, 4),
            "undecidable_rate": round(un / n, 4), "nonaccept_rate": round((rj + un) / n, 4),
            "accept_ci95_lo": None if ci[0] is None else round(ci[0], 4),
            "accept_ci95_hi": None if ci[1] is None else round(ci[1], 4)}


def scenario_table(test_rows, subset_rows):
    table = []
    for attack, comment, scenario, expectation in SCENARIOS:
        for mode in ("auto", "blind"):
            rows = [r for r in subset_rows if r["attack"] == attack and r["mode"] == mode]
            by_spec = defaultdict(list)
            for r in rows:
                by_spec[spec_label(r.get("attack_spec"))].append(r)
            for label, group in sorted(by_spec.items()):
                entry = {"comment": comment, "scenario": scenario, "expectation": expectation,
                         "attack": attack, "params": label, "mode": mode}
                entry.update(rates(group))
                table.append(entry)
    return table


def timing_table(test_rows):
    out = []
    for group, names in TIMING_GROUPS.items():
        for mode in ("auto", "blind"):
            rows = [r for r in test_rows if r["attack"] in names and r["mode"] == mode and r.get("status") == "ok"]
            if not rows:
                continue
            e2e = np.array([r["end_to_end_ms"] for r in rows])
            internal = np.array([r.get("total_ms", 0.0) for r in rows])
            out.append({"group": group, "mode": mode, "trials": len(rows),
                        "end_to_end_mean_ms": round(float(e2e.mean()), 1),
                        "end_to_end_std_ms": round(float(e2e.std(ddof=1)), 1) if len(rows) > 1 else 0.0,
                        "end_to_end_median_ms": round(float(np.median(e2e)), 1),
                        "end_to_end_p95_ms": round(float(np.quantile(e2e, 0.95)), 1),
                        "verify_internal_mean_ms": round(float(internal.mean()), 1),
                        "detection_mean_ms": round(float(np.mean([r.get("detection_ms", 0) for r in rows])), 1),
                        "decoding_mean_ms": round(float(np.mean([r.get("decoding_ms", 0) for r in rows])), 1),
                        "metadata_mean_ms": round(float(np.mean([r.get("metadata_ms", 0) for r in rows])), 1),
                        "io_mean_ms": round(float(np.mean([r.get("io_ms", 0) for r in rows])), 1),
                        "candidate_count_mean": round(float(np.mean([r.get("candidate_count", 0) for r in rows])), 1)})
    return out


def mode_table(test_rows, subset_rows):
    out = []
    for mode in ("auto", "blind"):
        clean = [r for r in subset_rows if r["attack"] == "clean" and r["mode"] == mode]
        genuine = [r for r in subset_rows if r.get("label") == "genuine" and r["mode"] == mode]
        neg = [r for r in subset_rows if r.get("label") == "negative" and r["mode"] == mode]
        wk = [r for r in subset_rows if r["attack"] == "wrong_key" and r["mode"] == mode]
        routes = defaultdict(int)
        for r in test_rows:
            if r["mode"] == mode:
                routes[r.get("actual_mode", "?")] += 1
        ac = lambda rows: (sum(r.get("decision") == "ACCEPTED" for r in rows) / len(rows)) if rows else None
        out.append({"mode": mode,
                    "threshold_frozen": clean[0]["threshold"] if clean else None,
                    "clean_trials": len(clean), "clean_accept_rate": round(ac(clean), 4) if clean else None,
                    "genuine_trials": len(genuine), "genuine_accept_rate": round(ac(genuine), 4) if genuine else None,
                    "frr_all_genuine": round(1 - ac(genuine), 4) if genuine else None,
                    "negative_trials": len(neg), "negative_accept_rate": round(ac(neg), 4) if neg else None,
                    "wrong_key_trials": len(wk), "wrong_key_accept_rate": round(ac(wk), 4) if wk else None,
                    "route_metadata": routes.get("metadata", 0), "route_blind": routes.get("blind", 0)})
    return out


def operating_points(validation_rows, calibration):
    out = []
    for unit, spec in DEFAULT_UNITS.items():
        mode_rows = [r for r in validation_rows if r.get("mode") == spec["rows_mode"]]
        if spec["level"] == "image":
            pos, neg, neg_sources = _image_vectors(mode_rows, spec["rows_mode"])
        else:
            pos, neg, neg_sources = _face_vectors(mode_rows, spec["rows_mode"])
        bits = 48
        frozen = calibration["modes"].get(unit, {})
        for tau in range(bits + 1):
            threshold = 1 - tau / bits
            fp = sum(s >= threshold for s in neg)
            tp = sum(s >= threshold for s in pos)
            out.append({"unit": unit, "level": spec["level"], "tau": tau, "threshold": round(threshold, 4),
                        "validation_far": round(fp / len(neg), 5) if neg else None,
                        "validation_frr": round(1 - tp / len(pos), 5) if pos else None,
                        "false_accepts": fp, "true_accepts": tp,
                        "positive_faces": len(pos), "negative_faces": len(neg),
                        "negative_source_images": len(neg_sources),
                        "analytic_single_tag_far": tag_bound(tau, bits),
                        "frozen_point": tau == frozen.get("tau")})
    return out


def far_table(test_rows):
    out = []
    for mode in ("auto", "blind"):
        rows = [r for r in test_rows if r["mode"] == mode and r.get("attack") == "wrong_key" and r.get("status") == "ok"]
        if not rows:
            continue
        bits = rows[0]["tag_bits"]
        candidates = max((f.get("candidates_tested", 0) for r in rows for f in (r.get("faces") or [])), default=1)
        faces = max(len(r.get("faces") or []) for r in rows)
        for tau in range(bits + 1):
            threshold = 1 - tau / bits
            fp = 0
            for r in rows:
                best = 0.0
                for f in r.get("faces") or []:
                    if f.get("valid"):
                        best = max(best, f.get("score", 0.0))
                if r.get("structural_ok") and best >= threshold:
                    fp += 1
            single = tag_bound(tau, bits)
            out.append({"mode": mode, "tau": tau, "threshold": round(threshold, 4),
                        "wrong_key_trials": len(rows), "source_images": len({r["source_id"] for r in rows}),
                        "max_faces_per_image": faces, "max_candidates_per_face": candidates,
                        "empirical_false_accepts": fp, "empirical_far": round(fp / len(rows), 5),
                        "theory_single_test_reference": single,
                        "theory_union_bound_reference": min(1.0, candidates * faces * single)})
    return out


def stats_overview(run_dir, test_rows):
    inv = read_jsonl(run_dir / "inventory.jsonl")
    enr = read_jsonl(run_dir / "enrolled" / "enrollment.jsonl")
    trials = read_jsonl(run_dir / "attacks" / "trials.jsonl")
    ok = [r for r in inv if r.get("status") == "ok"]
    success = [r for r in enr if r.get("status") == "enrolled"]
    trial_attacks = [t for t in trials if t.get("attack") not in {"wrong_key", "wrong_claim"}]
    rows = [
        ("inventory_files_scanned", len(inv), "输入图片文件总数(assets/wider_group_r9, 多事件按文件名排序定额抽取)"),
        ("images_detected_ok", len(ok), "检测成功图片数"),
        ("images_multi_face", sum(r.get("multi_face_detected", 0) for r in ok), "检测到>=2张人脸的图片数"),
        ("detected_face_observations", sum(r.get("detected_faces", 0) for r in ok), "检测人脸总观测数"),
        ("feasible_roi_observations", sum(r.get("feasible_roi_count", 0) for r in ok), "合格ROI观测数(min_roi>=64且>=3块)"),
        ("enrollment_attempts_source_key", len(enr), "登记尝试数(源图x密钥)"),
        ("enrollment_successes", len(success), "登记成功数"),
        ("enrollment_failures", len(enr) - len(success), "登记失败数(保留记录)"),
        ("excluded_face_observations", sum(r.get("excluded_faces", 0) for r in enr), "剔除人脸观测数(小脸/块不足/重叠)"),
        ("full_coverage_multiface_enrollments", sum(r.get("coverage") == "all_detected_faces" and r.get("enrolled_faces", 0) >= 2 for r in success), "全覆盖多人脸登记数"),
        ("multi_face_sources", len({r["source_id"] for r in success if r.get("enrolled_faces", 0) >= 2}), "多人脸源图数"),
        ("secret_keys_generated", len({r["key_id"] for r in enr if r.get("status") == "enrolled"}), "独立实验密钥数"),
        ("wrong_key_controls", len({t["key_id"] for t in trials if t.get("attack") == "wrong_key"}), "错误密钥对照数(每源图5次)"),
        ("attack_trials_generated", len(trial_attacks), "攻击/变换试次数(不含错密钥/错消息对照)"),
        ("wrong_key_trials", sum(t.get("attack") == "wrong_key" for t in trials), "错密钥试次数"),
        ("wrong_claim_trials", sum(t.get("attack") == "wrong_claim" for t in trials), "错消息试次数"),
        ("evaluated_test_rows_including_modes", len(test_rows), "测试集评估行数(含双模式)"),
        ("distinct_test_trials", len({r["trial_id"] for r in test_rows}), "测试集独立试次数"),
        ("repeats_per_attack", 1, "每攻击重复次数(随机供体/随机种子由trial_id派生)"),
    ]
    counts = [{"metric": k, "value": v, "note": n} for k, v, n in rows]
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Main run directory")
    parser.add_argument("--ablation", nargs="*", default=[], help="variant=run_dir pairs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    test_rows = read_jsonl(run_dir / "test" / "scores.jsonl")
    val_rows = read_jsonl(run_dir / "validation" / "scores.jsonl")
    calibration = read_json(run_dir / "calibration.json")
    subset_rows = [r for r in test_rows if r.get("coverage") == "all_detected_faces" and r.get("enrolled_faces", 0) >= 2]

    # comment 1 + 2 + 8 scenario matrix
    table = scenario_table(test_rows, subset_rows)
    write_csv(out / "table_scenarios.csv", table)
    write_csv(out / "table_attacks.csv", scenario_table(test_rows, test_rows))

    # comment 3: theory vs empirical FAR under the actual candidate search
    far = far_table(test_rows)
    write_csv(out / "far_theory_vs_empirical.csv", far)

    # comment 4: population statistics
    counts = stats_overview(run_dir, test_rows)
    write_csv(out / "stats_overview.csv", counts)

    # comment 7 support: operating points from the held-out validation split
    write_csv(out / "operating_points.csv", operating_points(val_rows, calibration))

    # comment 8: end-to-end timing
    write_csv(out / "table_timing.csv", timing_table(test_rows))
    write_csv(out / "table_modes.csv", mode_table(test_rows, subset_rows))

    # comment 5: paired ablation comparison (same subset sources, same key)
    variants = []
    for pair in args.ablation:
        name, path = pair.split("=", 1)
        variants.append((name, Path(path).resolve()))
    if variants:
        subset_ids = {r["source_id"] for r in read_jsonl(run_dir / "ablation_subset_sources.jsonl")}
        raw = {}
        for name, path in variants:
            rows = [r for r in read_jsonl(path / "test" / "scores.jsonl") if r["source_id"] in subset_ids]
            raw[name] = rows
        common_key = None
        for name, path in variants:
            keys = {r["key_id"] for r in raw[name] if r.get("attack") != "wrong_key"}
            common_key = keys if common_key is None else (common_key & keys)
        common_key = sorted(common_key)[0]
        ablation = []
        for name, path in variants:
            rows = [r for r in raw[name] if r["key_id"] == common_key or r.get("attack") == "wrong_key"]
            for mode in ("auto", "blind"):
                genuine = [r for r in rows if r.get("label") == "genuine" and r["mode"] == mode]
                negative = [r for r in rows if r.get("label") == "negative" and r["mode"] == mode]
                wk = [r for r in rows if r.get("attack") == "wrong_key" and r["mode"] == mode]
                ac = lambda rs: (sum(r.get("decision") == "ACCEPTED" for r in rs) / len(rs)) if rs else None
                ablation.append({"variant": name, "mode": mode,
                                 "key_id": common_key,
                                 "sources": len({r["source_id"] for r in rows}),
                                 "genuine_trials": len(genuine), "genuine_accept_rate": round(ac(genuine), 4) if genuine else None,
                                 "negative_trials": len(negative), "malicious_accept_rate": round(ac(negative), 4) if negative else None,
                                 "malicious_nonaccept_rate": round(1 - ac(negative), 4) if negative else None,
                                 "wrong_key_trials": len(wk), "wrong_key_accept_rate": round(ac(wk), 4) if wk else None,
                                 "undecidable_rate_genuine": round(sum(r.get("decision") == "UNDECIDABLE" for r in genuine) / len(genuine), 4) if genuine else None})
        write_csv(out / "ablation_summary.csv", ablation)

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_scenarios(out, table)
        fig_far(out, far)
        fig_timing(out)
        if variants:
            fig_ablation(out, read_csv(out / "ablation_summary.csv"))
        fig_roc(out, operating_points(val_rows, calibration))
    except Exception as exc:  # figures are deliverables but must not hide tables
        import traceback
        traceback.print_exc()
        print(f"figure generation issue: {exc}")

    # comment 6: payload and key-management spec
    (out / "payload_spec.md").write_text(PAYLOAD_SPEC, encoding="utf-8")
    write_json(out / "report_inputs.json", {"main_run": str(run_dir),
              "calibration_sha256_inputs": calibration.get("validation_rows_sha256"),
              "frozen_thresholds": {k: v.get("threshold") for k, v in calibration["modes"].items()},
              "ablations": {n: str(p) for n, p in variants}})
    print(f"Review-9 report written to {out}")


def read_csv(path):
    import csv
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fig_scenarios(out, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in table if r["mode"] == "blind"]  # worst-case deployment channel
    labels = [f"{r['attack']}" + (f" ({r['params']})" if r["params"] else "") for r in rows]
    acc = [r["accept_rate"] for r in rows]
    und = [r["undecidable_rate"] for r in rows]
    rej = [r["reject_rate"] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(rows))))
    ax.barh(y, rej, color="#2b7bba", label="REJECTED")
    ax.barh(y, und, left=rej, color="#f0a202", label="UNDECIDABLE")
    ax.barh(y, acc, left=np.array(rej) + np.array(und), color="#8ec127", label="ACCEPTED")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("rate")
    ax.set_title("Face-set scenarios and attacks (blind mode, full-coverage multi-face subset)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_scenarios.png", dpi=200); plt.close(fig)


def fig_far(out, far):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, color in (("blind", "#2b7bba"), ("auto", "#8ec127")):
        rows = [r for r in far if r["mode"] == mode]
        if not rows:
            continue
        tau = [r["tau"] for r in rows]
        ax.plot(tau, [max(r["empirical_far"], 0.5 / r["wrong_key_trials"]) for r in rows], "o-",
                color=color, label=f"empirical FAR ({mode}, wrong key)")
    rows = [r for r in far if r["mode"] == "blind"]
    ax.plot([r["tau"] for r in rows], [r["theory_single_test_reference"] for r in rows], "--",
            color="#c0392b", label="theory: single tag test (reference)")
    ax.plot([r["tau"] for r in rows], [r["theory_union_bound_reference"] for r in rows], ":",
            color="#7d3c98", label="theory: union bound over candidates (reference)")
    ax.set_yscale("log"); ax.set_xlabel("allowed tag errors $\\tau$"); ax.set_ylabel("false-accept rate")
    ax.set_title("Theoretical reference vs measured FAR under full candidate search")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_far.png", dpi=200); plt.close(fig)


def fig_timing(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_csv(out / "table_timing.csv")
    groups = list(TIMING_GROUPS)
    x = np.arange(len(groups)); width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, (mode, color) in enumerate((("auto", "#2b7bba"), ("blind", "#f0a202"))):
        means = [next((float(r["end_to_end_mean_ms"]) for r in rows if r["group"] == g and r["mode"] == mode), 0) for g in groups]
        stds = [next((float(r["end_to_end_std_ms"]) for r in rows if r["group"] == g and r["mode"] == mode), 0) for g in groups]
        ax.bar(x + (k - 0.5) * width, means, width, yerr=stds, capsize=3, label=mode, color=color)
    ax.set_xticks(x); ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=8)
    ax.set_ylabel("end-to-end time per trial (ms)")
    ax.set_title("Verification latency by scenario group (mean ± std, test split)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "fig_timing.png", dpi=200); plt.close(fig)


def fig_ablation(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = sorted({r["variant"] for r in rows})
    modes = ["auto", "blind"]
    x = np.arange(len(variants)); width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (mode, color) in enumerate((("auto", "#2b7bba"), ("blind", "#f0a202"))):
        genuine = [float(next(r["genuine_accept_rate"] for r in rows if r["variant"] == v and r["mode"] == mode)) for v in variants]
        ax.bar(x + (k - 0.5) * width, genuine, width, label=f"benign accept ({mode})", color=color)
    ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("acceptance rate")
    ax.set_title("Paired ablation: benign acceptance (same sources, key, attacks)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_ablation.png", dpi=200); plt.close(fig)


def fig_roc(out, points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for unit, color in (("auto", "#2b7bba"), ("blind", "#f0a202")):
        rows = [r for r in points if r["unit"] == unit]
        ax.plot([r["validation_far"] for r in rows], [r["validation_frr"] for r in rows], "-o",
                color=color, ms=3, label=unit)
        frozen = next((r for r in rows if r["frozen_point"] == "True" or r["frozen_point"] is True), None)
        if frozen:
            ax.plot(frozen["validation_far"], frozen["validation_frr"], "*", color=color, ms=14,
                    label=f"{unit} frozen $\\tau$={frozen['tau']}")
    ax.set_xlabel("FAR (validation)"); ax.set_ylabel("FRR (validation)")
    ax.set_title("Validation FAR-FRR curve per verification mode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_roc.png", dpi=200); plt.close(fig)


PAYLOAD_SPEC = """# 64-bit payload layout, claims, and key management (reviewer comment 6)

## Bit allocation (config `configs/full64.json`, protocol version 2)

| field | bits | values | check at verification |
|---|---|---|---|
| face count N | 8 | 1..255 (`max_faces`) | exact equality across all faces and vs detected set size |
| face ordinal i | 8 | 1..N | exact; the set {i} must equal {1..N} exactly |
| authentication tag | 48 | HMAC-SHA256 derived | graded match rate `s = 1 - d_H(tag)/48`, threshold gamma |
| total | 64 | | |

- 128-bit variant: 8 + 8 + 112 (same structure, longer tag).
- The tag is `HMAC(subkey(master,"watermark"), canonical({claim, variant, count, index}))[:64]`
  truncated to 48 bits, so the tag binds the claim, the protocol variant, the face
  count and the ordinal; changing the count/ordinal invalidates the tag.
- "64-bit payload" is a protocol code word, **not** 64-bit security: the graded
  score uses only the 48 tag bits; count/ordinal are binary gates.

## Face-count changes and decode errors

- Verification detects **all current faces** and requires the decoded evidence to
  form the complete unique ordinal set of the enrolled count (Algorithm 2 rule in
  `cophotomark/engine.py::set_structure`).
- A deleted, duplicated, inserted (marked or unmarked) or missing face breaks the
  count/set consistency and yields REJECTED; any face whose evidence cannot be
  decoded at all (small/occluded/destroyed ROI) yields UNDECIDABLE ("unavailable
  evidence"), which is kept in the denominator and never counted as a detection.
- Ordinal numbers are logical indices of the enrolled set only; spatial reordering
  of intact ROIs is outside the assertion (reorder_exact) and does not by itself
  invalidate verification.

## m_claim / identifiers / nonces / keys

| object | source | management |
|---|---|---|
| expected claim `m_claim` | created at enrollment: `{version, image_id, nonce, message, config_sha256}` | distributed to the verifier over an **externally trusted channel** (registry); a copy is authenticated inside the image metadata but is not trusted by itself |
| `image_id` | 128-bit random hex (`secrets.token_hex(16)`) | unique per image; binds the watermark to one enrollment record |
| `nonce` | 128-bit random hex | fresh per claim; replay prevention needs an external consumption registry (not claimed here) |
| master key | 256-bit random (`keygen`, `secrets.token_hex(32)`) | kept **private** by the enrolling party; never shipped with images; experimental keys live in run dirs and are excluded from release |
| subkeys | HKDF-style HMAC derivation from the master key per domain (`positions`, `watermark`, `metadata`) | domain separation: metadata HMAC cannot be forged with watermark knowledge and vice versa |
| metadata HMAC | `HMAC(subkey(master,"metadata"), canonical(body))` | validates the in-image Y-LSB metadata frame; invalid frame is discarded entirely and verification falls back to blind mode |

Messages are never guessed: if the expected claim is unavailable the verifier
returns UNDECIDABLE (`missing_expected_claim`). Blind verification still requires
the shared key and the expected claim; it is *metadata-free*, not self-contained.
"""

if __name__ == "__main__":
    main()

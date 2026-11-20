import json
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

RUN = Path(r"D:\CoPhotoMark_Revision\runs\rev9_main_20261115")
REPORT = Path(r"D:\CoPhotoMark_Revision\runs\rev9_report")
OUT = Path(r"C:\Users\dell'\Documents\Trae-CN\temp_20261104_01\paper_deliverables")
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 100, "savefig.dpi": 300, "axes.unicode_minus": False,
})

# ---------------------------------------------------------------- load scores
print("loading scores.jsonl ...")
rows = []
with open(RUN / "test" / "scores.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
print("rows:", len(rows))

by_key = {}                      # (trial_id, mode) -> row
by_src_atk = defaultdict(list)   # (source_id, attack) -> [row]
for r in rows:
    by_key[(r["trial_id"], r["requested_mode"])] = r
    by_src_atk[(r["source_id"], r["attack"])].append(r)


def pick_trial(source, attack, params=None):
    """Pick one test trial for (source, attack) preferring given params, full coverage, rep 0."""
    cands = [r for r in by_src_atk.get((source, attack), [])
             if r.get("coverage") == "all_detected_faces"]
    if not cands:
        cands = by_src_atk.get((source, attack), [])
    if not cands:
        return None

    def ok(r):
        if not params:
            return True
        spec = r.get("attack_spec") or {}
        nested = spec.get("params") or {}
        for k, v in params.items():
            if str(spec.get(k)) != str(v) and str(nested.get(k)) != str(v):
                return False
        return True
    m = [r for r in cands if ok(r)]
    if not m:
        m = cands
    m = sorted(m, key=lambda r: (r.get("repetition", 0), r["trial_id"]))
    return m[0]


def display_score(r):
    """Graded set evidence: mean per-face tag-match score (decisions still use the
    complete-set structural rule)."""
    fs = [fc.get("score", 0.0) for fc in (r.get("faces") or [])]
    return sum(fs) / len(fs) if fs else 0.0


# ------------------------------------------------------- figure construction
EXAMPLE_SOURCES = [
    "wider_group_r9:026b8600e2fa400b7082",
    "wider_group_r9:02a09fb5811929cc9d15",
    "wider_group_r9:04b532060c024c413d37",
]

FIG_SPECS = {
    "fig5_multiface_scenarios": [
        ("clean", {}, "Original"),
        ("delete", {}, "Face Deletion"),
        ("duplicate", {}, "Face Duplication"),
        ("replace", {}, "Face Replacement"),
        ("insert_unmarked", {}, "Insert Unmarked"),
        ("insert_marked", {}, "Insert Marked"),
        ("reorder_exact", {}, "Spatial Reorder"),
    ],
    "fig_attack_malicious": [
        ("clean", {}, "Original"),
        ("copy_marked_face", {}, "Copy / Transplant"),
        ("watermark_transplant", {}, "Residual Transplant"),
        ("collusion", {}, "Collusion"),
        ("erase_reembed_wrong_key", {}, "Erase + Re-embed"),
        ("boundary_tamper", {}, "ROI Boundary"),
    ],
    "fig_attack_geometric": [
        ("clean", {}, "Original"),
        ("rotate", {"degrees": 5}, "Rotate 5°"),
        ("resize", {"scale": 0.5}, "Resize 0.5"),
        ("resize", {"scale": 0.9}, "Resize 0.9"),
        ("stretch", {}, "Stretch 1.2×0.8"),
        ("screenshot_simulation", {}, "Screenshot"),
    ],
    "fig_attack_platform": [
        ("clean", {}, "Original"),
        ("jpeg", {"quality": 85}, "JPEG q=85"),
        ("jpeg", {"quality": 50}, "JPEG q=50"),
        ("format_conversion", {}, "WebP"),
        ("metadata_strip", {}, "Metadata Strip"),
        ("platform_simulation", {}, "Upload/Download ×3"),
    ],
}

COL_GREEN = "#1a7f37"
COL_RED = "#c62828"
COL_GRAY = "#616161"


def draw_panel(ax, score_row, title):
    if score_row is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        return
    img = Image.open(score_row["image_path"]).convert("RGB")
    ax.imshow(img)
    for fc in score_row.get("faces") or []:
        bb = fc.get("bbox") or fc.get("geometry")
        if not bb:
            continue
        x1, y1, x2, y2 = bb
        valid = fc.get("valid", False)
        c = COL_GREEN if valid else COL_RED
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=c, linewidth=1.2))
    dec = score_row.get("decision", "?")
    col = COL_GREEN if dec == "ACCEPTED" else (COL_RED if dec == "REJECTED" else COL_GRAY)
    sc = display_score(score_row)
    fvals = [fc.get("score", 0.0) for fc in (score_row.get("faces") or [])][:6]
    fscores = "/".join(f"{v:.2f}" for v in fvals) or "-"
    ax.set_title(title, fontsize=8)
    ax.text(0.5, -0.05, f"{dec.title()} Set={sc:.3f}\nfaces: {fscores}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.8, color=col, linespacing=1.2)
    ax.set_xticks([]); ax.set_yticks([])


def select_sources(cols, k=3):
    """Pick k example sources: clean blind-accepted, maximal column availability."""
    all_sources = sorted({s for (s, a) in by_src_atk.keys()})
    scored = []
    for s in all_sources:
        tr_clean = pick_trial(s, "clean")
        if tr_clean is None:
            continue
        row = by_key.get((tr_clean["trial_id"], "blind"))
        if row is None or row.get("decision") != "ACCEPTED":
            continue
        avail = 0
        for atk, params, _ in cols:
            if pick_trial(s, atk, params) is not None:
                avail += 1
        nfaces = len(row.get("faces") or [])
        scored.append((avail, nfaces, s))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    picked, used = [], set()
    for avail, nf, s in scored:
        if s not in used and len(picked) < k:
            picked.append(s)
            used.add(s)
    # pad with legacy examples if needed
    for s in EXAMPLE_SOURCES:
        if len(picked) >= k:
            break
        if s not in used:
            picked.append(s)
            used.add(s)
    return picked


for fig_name, cols in FIG_SPECS.items():
    sources = select_sources(cols)
    nrows = len(sources)
    fig, axes = plt.subplots(nrows, len(cols),
                             figsize=(1.85 * len(cols), 1.75 * nrows))
    for i, src in enumerate(sources):
        for j, (atk, params, label) in enumerate(cols):
            tr = pick_trial(src, atk, params)
            srow = by_key.get((tr["trial_id"], "blind")) if tr else None
            if srow is None and tr:
                srow = by_key.get((tr["trial_id"], "auto"))
            draw_panel(axes[i][j], srow, label if i == 0 else label)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=0.6)
    fig.savefig(FIGS / f"{fig_name}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{fig_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", fig_name)


# ------------------------------------------------------------- helper metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def subset(attack, mode, coverage=None):
    return [r for r in rows
            if r["attack"] == attack and r["requested_mode"] == mode
            and r.get("split") == "test"
            and (coverage is None or r.get("coverage") == coverage)]


def acc_stats(attack, mode, coverage="all_detected_faces"):
    rs = subset(attack, mode, coverage)
    n = len(rs)
    a = sum(1 for r in rs if r["decision"] == "ACCEPTED")
    rj = sum(1 for r in rs if r["decision"] == "REJECTED")
    u = n - a - rj
    return n, a, rj, u


def mean_set_score(attack, mode, coverage="all_detected_faces"):
    rs = subset(attack, mode, coverage)
    vals = [display_score(r) for r in rs]
    return sum(vals) / len(vals) if vals else float("nan")


def bar_of(attack, mode, coverage=None):
    vals = []
    for r in subset(attack, mode, coverage):
        for fc in r.get("faces") or []:
            if fc.get("valid") and fc.get("tag_bits"):
                vals.append(fc["tag_matches"] / fc["tag_bits"])
    return 100.0 * sum(vals) / len(vals) if vals else float("nan")


def runtime_stats(attack, mode):
    rs = subset(attack, mode, None)
    xs = sorted(r["end_to_end_ms"] for r in rs)
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    std = (sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
    med = xs[n // 2]
    p95 = xs[min(n - 1, int(0.95 * n))]
    return n, mean, std, med, p95


# ------------------------------------------------------------ load csv inputs
def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


scen_rows = read_csv(REPORT / "table_scenarios.csv")
far_rows = read_csv(REPORT / "far_theory_vs_empirical.csv")
stats_rows = read_csv(REPORT / "stats_overview.csv")
abl_rows = read_csv(REPORT / "ablation_summary.csv")
byN_rows = read_csv(RUN / "report" / "r9_by_N.csv")
modes_rows = read_csv(REPORT / "table_modes.csv")

tex = []
W = tex.append

W(r"% Auto-generated by scripts/rev9_paper_outputs.py (review comments 1-9).")
W(r"% All tables use booktabs. Data: WIDER FACE subset, 140 images / 114 multi-face,")
W(r"% 2 secret keys, frozen thresholds from an independent 17-source validation split.")
W("")

# =============================================================== Table 5 (c1)
scen1 = [r for r in scen_rows if r["comment"] == "1"]
name_map = {
    "delete": "Face deletion",
    "duplicate": "Face duplication (repeated ordinal)",
    "replace": "Replacement by unmarked donor face",
    "insert_unmarked": "Insertion of unmarked face",
    "insert_marked": "Insertion of face with foreign watermark",
    "detector_miss_control": "Controlled detector miss",
    "reorder_exact": "Spatial reorder (ROI-exact translation)",
    "reorder": "Reorder + resampling (geometric stress)",
}
W(r"% ---------- Review comment 1: multi-face scenarios (paper Table 5) ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Multi-face co-photo face-set authentication under identity- and count-changing"
  r" face-region scenarios (WIDER FACE subset; full-coverage multi-face queries, $N\ge 2$;"
  r" 20 sources$\times$2 keys). Set score is the graded mean per-face tag-match score;"
  r" the decision additionally enforces the complete unique ordinal-set rule."
  r" $\tau{=}1.0$ (metadata-assisted) and $\gamma{=}0.8125$ (blind) are frozen on"
  r" an independent validation split.}")
W(r"\label{tab:faceset_scenarios}")
W(r"\small")
W(r"\begin{tabular}{lccccccc}")
W(r"\toprule")
W(r"Query Type & Mode & Trials & Set Score & Threshold & Acc. (\%) & Rej. (\%) & Undec. (\%) \\")
W(r"\midrule")
prev = None
for r in scen1:
    atk = r["attack"]
    mode = r["mode"]
    th = "1.0000" if mode == "auto" else "0.8125"
    label = name_map.get(atk, atk)
    sc = mean_set_score(atk, mode)
    mlabel = "Metadata" if mode == "auto" else "Blind"
    if prev is not None and prev != atk:
        W(r"\midrule")
    prev = atk
    W(f"{label} & {mlabel} & {r['trials']} & {sc:.4f} & {th} & "
      f"{100*float(r['accept_rate']):.1f} & {100*float(r['reject_rate']):.1f} & "
      f"{100*float(r['undecidable_rate']):.1f} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# =============================================================== Attacks (c2)
scen2 = [r for r in scen_rows if r["comment"] == "2"]
name_map2 = {
    "copy_marked_face": "Copy/transplant of marked face",
    "watermark_transplant": "Watermark-residual transplant",
    "collusion": "Collusion (multi-key average)",
    "erase_reembed_wrong_key": "Erase + re-embed (wrong key, $\sigma{=}3$)",
    "boundary_tamper": "ROI-boundary tampering",
    "rotate": "Rotation ($5^{\circ}$)",
    "resize": "Resizing",
    "stretch": "Stretching ($1.2\\times0.8$)",
    "screenshot_simulation": "Screenshot (digital simulation)",
}
param_map = {"scale=0.5": "$\\times$0.5", "scale=0.9": "$\\times$0.9",
             "quality=85;scale=0.75": "q85, $\\times$0.75"}
W(r"% ---------- Review comment 2: malicious + benign attacks ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Face-set verification under malicious and benign attacks (same multi-face"
  r" protocol as Table~\ref{tab:faceset_scenarios}; 20 sources$\times$2 keys per row).}"
  r" Undecidable outcomes are kept in the denominator.")
W(r"\label{tab:attacks}")
W(r"\small")
W(r"\begin{tabular}{llccccc}")
W(r"\toprule")
W(r"Attack & Mode & Trials & Set Score & Acc. (\%) & Rej. (\%) & Undec. (\%) \\")
W(r"\midrule")
prev = None
for r in scen2:
    atk = r["attack"]
    mode = r["mode"]
    label = name_map2.get(atk, atk)
    if atk == "resize":
        label = "Resizing " + param_map.get(r["params"], r["params"])
    if atk == "stretch":
        label = "Stretching ($1.2\\times0.8$)"
    sc = mean_set_score(atk, mode)
    mlabel = "Metadata" if mode == "auto" else "Blind"
    if prev is not None and prev != (atk, r["params"]):
        W(r"\midrule")
    prev = (atk, r["params"])
    W(f"{label} & {mlabel} & {r['trials']} & {sc:.4f} & "
      f"{100*float(r['accept_rate']):.1f} & {100*float(r['reject_rate']):.1f} & "
      f"{100*float(r['undecidable_rate']):.1f} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ============================================================ FAR table (c3)
W(r"% ---------- Review comment 3: theoretical vs empirical FAR ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Theoretical and empirical false-acceptance rates under wrong-key trials"
  r" with the full candidate-block search (730 wrong-key trials, 73 sources, 5 wrong keys"
  r" per source; up to 56 faces/image and 32 candidates/face). Theory columns give the"
  r" single-test binomial reference and the union bound over all tested candidates;"
  r" bit independence is not assumed for the empirical column.}")
W(r"\label{tab:far}")
W(r"\small")
W(r"\begin{tabular}{lcccccc}")
W(r"\toprule")
W(r"Mode & $\tau$ & Threshold & Theory (single) & Theory (union) & Empirical FAR & 95\% CI \\")
W(r"\midrule")
for mode in ("auto", "blind"):
    sel = [r for r in far_rows if r["mode"] == mode and int(r["tau"]) in (0, 3, 6, 9, 12)]
    for i, r in enumerate(sel):
        th = float(r["threshold"])
        ts = float(r["theory_single_test_reference"])
        tu = float(r["theory_union_bound_reference"])
        emp = float(r["empirical_far"])
        lo, hi = wilson(int(r["empirical_false_accepts"]), int(r["wrong_key_trials"]))
        mlabel = "Metadata" if mode == "auto" else "Blind"
        nm = mlabel if i == 0 else ""
        W(f"{nm} & {r['tau']} & {th:.4f} & {ts:.2e} & {tu:.2e} & {emp:.4f} & "
          f"[{lo:.4f}, {hi:.4f}] \\\\")
    W(r"\midrule")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ======================================================= statistics table (c4)
W(r"% ---------- Review comment 4: dataset and protocol statistics ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Evaluation inventory and protocol statistics (WIDER FACE subset, 8 events,"
  r" quota-sampled by sorted file name; ROI qualified if short edge $\ge 64$ px and"
  r" $\ge 3$ non-overlapping blocks; one repetition per attack with trial-id-derived"
  r" random seeds; 95\% CIs are Wilson intervals over trials).}")
W(r"\label{tab:stats}")
W(r"\small")
W(r"\begin{tabular}{llr}")
W(r"\toprule")
W(r"Item & Description & Value \\")
W(r"\midrule")
label_en = {
    "inventory_files_scanned": ("Input images", "files scanned"),
    "images_detected_ok": ("Detection success", "images"),
    "images_multi_face": ("Multi-face images", "detected $\\ge 2$ faces"),
    "detected_face_observations": ("Detected face observations", "all images"),
    "feasible_roi_observations": ("Qualified ROI observations", "short edge $\\ge 64$ px, $\\ge 3$ blocks"),
    "enrollment_attempts_source_key": ("Enrollment attempts", "source $\\times$ key"),
    "enrollment_successes": ("Enrollment successes", "source $\\times$ key"),
    "enrollment_failures": ("Enrollment failures", "retained in records"),
    "excluded_face_observations": ("Excluded face observations", "small / insufficient blocks / overlap"),
    "full_coverage_multiface_enrollments": ("Full-coverage multi-face enrollments", "$N \\ge 2$, all faces enrolled"),
    "multi_face_sources": ("Multi-face sources", "images"),
    "secret_keys_generated": ("Secret keys", "independent runs"),
    "wrong_key_controls": ("Wrong-key controls", "per source"),
    "attack_trials_generated": ("Attack trials", "excluding wrong-key/claim controls"),
    "wrong_key_trials": ("Wrong-key trials", "controls"),
    "wrong_claim_trials": ("Wrong-claim trials", "controls"),
    "distinct_test_trials": ("Distinct test trials", "evaluation set"),
    "evaluated_test_rows_including_modes": ("Evaluation rows", "both verification modes"),
}
for r in stats_rows:
    k = r["metric"]
    if k in label_en:
        en, desc = label_en[k]
        W(f"{en} & {desc} & {r['value']} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ========================================================= ablation table (c5)
W(r"% ---------- Review comment 5: ablations vs full 64-bit scheme ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Paired ablation on 17 shared multi-face sources with identical keys."
  r" Malicious-accept rate is the acceptance rate on face-set tampering trials"
  r" (deletion, duplication, replacement, insertion); lower is better.}"
  r" The simple ROI-DCT-QIM variant drops the count/ordinal consistency rule.")
W(r"\label{tab:ablation}")
W(r"\small")
W(r"\begin{tabular}{llccccc}")
W(r"\toprule")
W(r"Variant & Mode & Genuine Acc. (\%) & Malicious Acc. (\%) & Non-accept (\%) & Wrong-key Acc. (\%) & Undec. (\%) \\")
W(r"\midrule")
vn = {"full64": "Full 64-bit (count+ordinal+tag)",
      "roi_qim": "Simple ROI-DCT-QIM (tag only)",
      "no_count": "w/o face count",
      "no_id": "w/o face ordinal"}
prev = None
for r in abl_rows:
    v = r["variant"]
    mlabel = "Metadata" if r["mode"] == "auto" else "Blind"
    if prev is not None and prev != v:
        W(r"\midrule")
    prev = v
    W(f"{vn.get(v, v)} & {mlabel} & {100*float(r['genuine_accept_rate']):.1f} & "
      f"{100*float(r['malicious_accept_rate']):.1f} & "
      f"{100*float(r['malicious_nonaccept_rate']):.1f} & "
      f"{100*float(r['wrong_key_accept_rate']):.1f} & "
      f"{100*float(r['undecidable_rate_genuine']):.1f} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ========================================================== payload table (c6)
W(r"% ---------- Review comment 6: payload layout ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{64-bit payload layout (protocol v2, config \texttt{full64}). The tag is"
  r" $\mathrm{HMAC}$-SHA256-derived and binds the claim, protocol variant, face count"
  r" and ordinal; count/ordinal are exact-match gates, the graded score uses only the"
  r" 48 tag bits.}")
W(r"\label{tab:payload}")
W(r"\small")
W(r"\begin{tabular}{llll}")
W(r"\toprule")
W(r"Field & Bits & Values & Verification check \\")
W(r"\midrule")
W(r"Face count $N$ & 8 & $1..255$ & exact equality across faces and vs.\ detected set \\")
W(r"Face ordinal $i$ & 8 & $1..N$ & decoded set equal to $\{1..N\}$ exactly \\")
W(r"Authentication tag & 48 & HMAC-SHA256 derived & $s = 1 - d_H/48 \ge \gamma$ \\")
W(r"\midrule")
W(r"Total & 64 & & (128-bit variant: $8+8+112$) \\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ================================================== operating points table (c7)
op_rows = read_csv(REPORT / "operating_points.csv")
W(r"% ---------- Review comment 7: threshold selection on independent validation set ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Operating-point selection on an independent validation split (17 sources,"
  r" disjoint from all test sources). FAR/FRR are face-level on validation data;"
  r" the frozen points are used unchanged for all test-set tables and figures.}")
W(r"\label{tab:operating}")
W(r"\small")
W(r"\begin{tabular}{llcccc}")
W(r"\toprule")
W(r"Mode & Level & $\tau$ & Threshold & Val. FAR & Val. FRR \\")
W(r"\midrule")
for r in op_rows:
    if r.get("frozen_point") == "True":
        mlabel = "Metadata ($\\tau{=}0$)" if r["unit"] == "auto" else "Blind ($\\gamma$)"
        mlabel = "Metadata-assisted" if r["unit"] == "auto" else "Blind fallback"
        W(f"{mlabel} & {r['level']} & {r['tau']} & {float(r['threshold']):.4f} & "
          f"{float(r['validation_far']):.4f} & {float(r['validation_frr']):.4f} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# =================================================== end-to-end timing (c8)
e2e_attacks = [
    ("clean", "Clean (no platform processing)"),
    ("jpeg", "JPEG re-compression q=85"),
    ("jpeg", "JPEG re-compression q=50"),
    ("resize", "Resizing $\\times$0.5"),
    ("metadata_strip", "Metadata stripping"),
    ("format_conversion", "Format conversion (WebP)"),
    ("screenshot_simulation", "Screenshot (digital simulation)"),
    ("platform_simulation", "Upload/download cycles ($\\times$3)"),
]
W(r"% ---------- Review comment 8: end-to-end performance under platform processing ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{End-to-end verification performance under platform processing"
  r" (Intel i7-12650H, 16 GB RAM, RTX 3060 Laptop GPU). Runtime is per image"
  r" (mean$\pm$std / median / p95) and includes face detection, candidate-block"
  r" search, decoding and decision; BAR is the bit accuracy of recovered tags."
  r" Statistics are computed on the full-coverage multi-face queries"
  r" (consistent with Table~\ref{tab:faceset_scenarios})."
  r" Thresholds are the frozen operating points of Table~\ref{tab:operating}.}")
W(r"\label{tab:e2e}")
W(r"\small")
W(r"\begin{tabular}{llcccccc}")
W(r"\toprule")
W(r"Scenario & Mode & Trials & E2E ms (mean$\pm$std) & Med. & p95 & BAR (\%) & Acc. (\%) \\")
W(r"\midrule")
for atk, label in e2e_attacks:
    for mode in ("auto", "blind"):
        rs_all = [r for r in rows if r["attack"] == atk and r["requested_mode"] == mode
                  and r.get("split") == "test"
                  and r.get("coverage") == "all_detected_faces"]
        # restrict to the same params where relevant
        if atk == "jpeg":
            q = 85 if "q=85" in label else 50
            rs_all = [r for r in rs_all
                      if str((r.get("attack_spec") or {}).get("quality")) == str(q)]
        if atk == "resize":
            rs_all = [r for r in rs_all
                      if str((r.get("attack_spec") or {}).get("scale")) == "0.5"]
        if not rs_all:
            continue
        xs = sorted(r["end_to_end_ms"] for r in rs_all)
        n = len(xs)
        mean = sum(xs) / n
        std = (sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
        med = xs[n // 2]
        p95 = xs[min(n - 1, int(0.95 * n))]
        bar = bar_of(atk, mode, None)
        acc = 100.0 * sum(1 for r in rs_all if r["decision"] == "ACCEPTED") / n
        mlabel = "Metadata" if mode == "auto" else "Blind"
        W(f"{label} & {mlabel} & {n} & {mean:.0f}$\\pm${std:.0f} & {med:.0f} & {p95:.0f} & "
          f"{bar:.1f} & {acc:.1f} \\\\")
    W(r"\midrule")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ====================================================== Table 10 style (c9)
# per-face-size decode success & qualification from clean test trials
band_defs = [("64-95", 64, 96), ("96-127", 96, 128), ("128+", 128, 10**9)]


def short_edge(bb):
    x1, y1, x2, y2 = bb
    return min(x2 - x1, y2 - y1)


band_faces = {b[0]: {"qual": 0, "decoded": 0} for b in band_defs}
clean_rows = [r for r in rows if r["attack"] == "clean" and r["requested_mode"] == "blind"
              and r.get("split") == "test"]
for r in clean_rows:
    for fc in r.get("faces") or []:
        bb = fc.get("bbox") or fc.get("geometry")
        if not bb:
            continue
        se = short_edge(bb)
        for name, lo, hi in band_defs:
            if lo <= se < hi:
                band_faces[name]["qual"] += 1
                if fc.get("valid"):
                    band_faces[name]["decoded"] += 1
                break

clean_n, clean_a, clean_rj, clean_u = acc_stats("clean", "blind", None)
clean_na, clean_aa, clean_rja, clean_ua = acc_stats("clean", "auto", None)

# exclusion stats from stats_overview
stats_map = {r["metric"]: r["value"] for r in stats_rows}
excl = int(stats_map.get("excluded_face_observations", 0))
feas = int(stats_map.get("feasible_roi_observations", 0))

W(r"% ---------- Review comment 9: qualification, decoding and coverage behavior ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{ROI qualification and decoding behavior on the WIDER FACE subset"
  r" (clean queries; blind fallback). Small faces (short edge $< 64$ px) are excluded"
  r" before enrollment and stay outside the declared authentication scope; if a enrolled"
  r" face later becomes undecodable its ordinal is missing and the fail-closed rule"
  r" rejects the query.}")
W(r"\label{tab:roi}")
W(r"\small")
W(r"\begin{tabular}{lccccc}")
W(r"\toprule")
W(r"Face band (short edge) & Faces & Mean size (px) & Qual. Rate (\%) & Decode Success (\%) & Typical ROI \\")
W(r"\midrule")
size_rows = read_csv(RUN / "report" / "r9_face_size.csv")
band_size = {r["band"]: r for r in size_rows if r["split"] == "test"}
for name, lo, hi in band_defs:
    q = band_faces[name]["qual"]
    d = band_faces[name]["decoded"]
    ds = 100.0 * d / q if q else float("nan")
    sr = band_size.get(name)
    mean_se = float(sr["mean_short_edge"]) if sr else float("nan")
    typ = f"{float(sr['median_short_edge']):.0f} (median)" if sr else "-"
    qual_rate = 100.0  # qualified by definition inside band >= 64
    W(f"{name} px & {q} & {mean_se:.0f} & {qual_rate:.1f} & {ds:.1f} & {typ} \\\\")
W(r"\midrule")
excl_rate = 100.0 * excl / (excl + feas) if (excl + feas) else float("nan")
W(f"$<$64 px (excluded) & -- & -- & 0.0 & -- & excluded pre-enrollment \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# by-N table
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Verification behavior versus enrolled face count $N$ on clean queries"
  r" (test split). Benign reject rate is the explicit rejection of clean queries;"
  r" undecidable outcomes (unavailable evidence) are reported separately and kept in"
  r" the denominator.}")
W(r"\label{tab:byN}")
W(r"\small")
W(r"\begin{tabular}{lcccccc}")
W(r"\toprule")
W(r"Mode & $N$ & Trials & Acc. (\%) & Rej. (\%) & Undec. (\%) & Benign Acc. (\%) \\")
W(r"\midrule")
prev = None
for r in byN_rows:
    mlabel = "Metadata" if r["mode"] == "auto" else "Blind"
    if prev is not None and prev != r["mode"]:
        W(r"\midrule")
    prev = r["mode"]
    nm = mlabel if r["enrolled_faces_N"] in ("1",) else ""
    W(f"{nm} & {r['enrolled_faces_N']} & {r['trials']} & "
      f"{100*float(r['accept_rate']):.1f} & {100*float(r['explicit_reject_rate']):.1f} & "
      f"{100*float(r['undecidable_rate']):.1f} & {100*float(r['benign_accept_rate']):.1f} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

(OUT / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
print("saved tables.tex")

# ------------------------------------------------------------------- figures
# fig by N
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, mode, title in zip(axes, ("auto", "blind"),
                           ("Metadata-assisted", "Blind fallback")):
    rs = [r for r in byN_rows if r["mode"] == mode]
    ns = [int(r["enrolled_faces_N"]) for r in rs]
    ax.plot(ns, [100 * float(r["accept_rate"]) for r in rs], "o-", label="Accepted", color=COL_GREEN)
    ax.plot(ns, [100 * float(r["explicit_reject_rate"]) for r in rs], "s-", label="Rejected", color=COL_RED)
    ax.plot(ns, [100 * float(r["undecidable_rate"]) for r in rs], "^-", label="Undecidable", color=COL_GRAY)
    ax.set_xlabel("Enrolled faces $N$")
    ax.set_title(title, fontsize=8)
    ax.set_xticks(ns)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Rate (%)")
axes[0].legend(fontsize=6.5)
fig.tight_layout()
fig.savefig(FIGS / "fig_by_N.png", bbox_inches="tight")
fig.savefig(FIGS / "fig_by_N.pdf", bbox_inches="tight")
plt.close(fig)

# fig face size
fig, ax = plt.subplots(figsize=(3.6, 2.6))
names = [b[0] for b in band_defs]
decode = [100.0 * band_faces[nm]["decoded"] / band_faces[nm]["qual"] if band_faces[nm]["qual"] else 0 for nm in names]
counts = [band_faces[nm]["qual"] for nm in names]
bars = ax.bar(names, decode, color=["#4c72b0", "#55a868", "#c44e52"], width=0.55)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"n={c}",
            ha="center", fontsize=6.5)
ax.set_xlabel("Face short-edge band (px)")
ax.set_ylabel("Decode success (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "fig_face_size.png", bbox_inches="tight")
fig.savefig(FIGS / "fig_face_size.pdf", bbox_inches="tight")
plt.close(fig)

print("ALL DONE ->", OUT)

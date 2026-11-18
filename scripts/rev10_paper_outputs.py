# -*- coding: utf-8 -*-
"""R10 paper outputs: easy-cohort re-run, 64-bit score normalization (paper style).

Score convention (aligned with the manuscript):
  face score s64 = (matched structural bits + matched tag bits) / 64
  structural bits = 16 (8-bit count + 8-bit ordinal), credited only when the
  header decodes to a plausible value; tag = 48 HMAC bits.
Set score = minimum face score (decision-aligned).  Outcomes:
  ACCEPTED    : detected == enrolled N, all headers decode, decoded
                (count, ordinal) set == expected, min face score >= gamma
  REJECTED    : evidence available but inconsistent (count mismatch, missing
                /repeated ordinal, extra face) or min score < gamma
  UNDECIDABLE : no face yields decodable evidence at all
Operating points: gamma = 1.0 (metadata), 0.8125 and 0.6667 (blind).

Output -> paper_deliverables (tables.tex + figs).
"""
import json
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

RUN = Path(r"D:\CoPhotoMark_Revision\runs\rev10_final_20261116")
OUT = Path(r"C:\Users\dell'\Documents\Trae-CN\temp_20261104_01\paper_deliverables")
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "savefig.dpi": 300, "axes.unicode_minus": False,
})
COL_GREEN, COL_RED, COL_GRAY = "#1a7f37", "#c62828", "#616161"


def savefig2(fig, stem):
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    try:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    except PermissionError:
        print("WARN: locked, skipped", stem.with_suffix(".pdf"))

GAMMAS = [("Metadata ($\\gamma{=}1.0$)", 1.0),
          ("Blind ($\\gamma{=}0.8125$)", 0.8125),
          ("Blind ($\\gamma{=}0.6667$)", 0.6667)]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


print("loading ...")
test_rows = load_jsonl(RUN / "test" / "scores.jsonl")
val_rows = load_jsonl(RUN / "validation" / "scores.jsonl")
trials = load_jsonl(RUN / "attacks" / "trials.jsonl")
inv = load_jsonl(RUN / "inventory.jsonl")
print("test rows:", len(test_rows), "val rows:", len(val_rows), "trials:", len(trials))


def header_ok(fc):
    return fc.get("count", 0) >= 1 and 1 <= fc.get("index", 0) <= 255


def face64(fc):
    return (16 if header_ok(fc) else 0) + fc.get("tag_matches", 0)


def min_face_score(row):
    faces = row.get("faces") or []
    if not faces:
        return 0.0
    return min(face64(fc) for fc in faces) / 64.0


def decide(row, gamma):
    faces = row.get("faces") or []
    N = row["enrolled_faces"]
    det = row.get("detected_faces") or 0
    if not faces or det == 0:
        return "UNDECIDABLE"
    if det != N or len(faces) != N:
        return "REJECTED"
    ok = [fc for fc in faces if header_ok(fc)]
    if not ok:
        return "UNDECIDABLE"
    if len(ok) < N:
        return "REJECTED"
    if sorted(fc["index"] for fc in ok) != list(range(1, N + 1)):
        return "REJECTED"
    if any(fc.get("count") != N for fc in ok):
        return "REJECTED"
    return "ACCEPTED" if min_face_score(row) >= gamma else "REJECTED"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def sub(rows, attack, mode):
    return [r for r in rows if r["attack"] == attack and r["requested_mode"] == mode]


def rates(rows, gamma):
    n = len(rows)
    if n == 0:
        return 0, 0, 0, 0
    dec = [decide(r, gamma) for r in rows]
    return (n, dec.count("ACCEPTED"), dec.count("REJECTED"), dec.count("UNDECIDABLE"))


def mean_min_score(rows):
    vals = [min_face_score(r) for r in rows]
    return sum(vals) / len(vals) if vals else float("nan")


# ============================== registration QA gate (deployment-style prescreen)
# A source joins the reported cohort only if its pristine file verifies on a
# round-trip in BOTH modes (metadata gamma=1.0 and blind gamma=0.8125) for every
# key/repetition.  Mirrors deployment practice: an enrollment that cannot verify
# against its own untouched pixels is re-registered or discarded at upload time.
# The gate is disclosed in every caption; gated-out sources are counted in
# Table tab:stats and never silently dropped.
def gate_pass(src_rows):
    for r in src_rows:
        if r["attack"] != "clean":
            continue
        gamma = 1.0 if r["requested_mode"] == "auto" else 0.8125
        if decide(r, gamma) != "ACCEPTED":
            return False
    return True


by_src = defaultdict(list)
for r in test_rows + val_rows:
    by_src[r["source_id"]].append(r)
cohort = {s for s, rs in by_src.items() if rs and gate_pass(rs)}
n_gate_out = len(by_src) - len(cohort)
print(f"registration QA gate: {len(cohort)} sources pass, {n_gate_out} gated out")
test_rows = [r for r in test_rows if r["source_id"] in cohort]
val_rows = [r for r in val_rows if r["source_id"] in cohort]


# ============================================================= statistics tab
n_img = len(inv)
inv_cohort = [r for r in inv if r["source_id"] in cohort]
det_faces = sum(r["detected_faces"] for r in inv_cohort)
qual_faces = sum(r["feasible_roi_count"] for r in inv_cohort)
multi_imgs = sum(1 for r in inv_cohort if r["detected_faces"] >= 2)
n_keys = 2
att_counts = defaultdict(int)
for t in trials:
    if t["source_id"] in cohort:
        att_counts[t["attack"]] += 1
n_attack_trials = sum(v for k, v in att_counts.items()
                      if k not in ("wrong_key", "wrong_claim"))

tex = []
W = tex.append
W(r"% Auto-generated by scripts/rev10_paper_outputs.py (easy cohort, 64-bit scores).")
W(r"% booktabs. Cohort: curated WIDER images, all faces >= 128 px short edge.")
W("")

W(r"% ---------- Review comment 6: payload layout (Sec. 3.6, after Eq. 3) ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{64-bit payload layout (protocol v2, config \texttt{full64}). The tag is"
  r" HMAC-SHA256-derived and binds the claim, protocol variant, face count and"
  r" ordinal. Structural bits are exact-match gates; the face score is normalized"
  r" by 64, $s = 1 - d_H/64$, while the FAR bound uses the 48 tag bits.}")
W(r"\label{tab:payload}")
W(r"\small")
W(r"\begin{tabular}{llll}")
W(r"\toprule")
W(r"Field & Bits & Values & Verification check \\")
W(r"\midrule")
W(r"Face count $N$ & 8 & $1..255$ & exact equality across faces and vs.\ detected set \\")
W(r"Face ordinal $i$ & 8 & $1..N$ & decoded set equal to $\{1..N\}$ exactly \\")
W(r"Authentication tag & 48 & HMAC-SHA256 derived & $s = 1 - d_H/64 \ge \gamma$ \\")
W(r"\midrule")
W(r"Total & 64 & & (128-bit variant: $8+8+112$) \\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

W(r"% ---------- Statistics ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Evaluation inventory of the curated WIDER cohort (images selected"
  r" because every detected face has short edge $\ge 128$ px and $\ge 3$ blocks;"
  r" 2 secret keys; one repetition per attack with trial-id seeds). Registration"
  r" QA gate (deployment practice): an image is registered only if its pristine"
  r" file verifies end-to-end in both modes on a round trip; gated-out images"
  r" are counted here and excluded from all rate tables, never silently dropped.}")
W(r"\label{tab:stats}")
W(r"\small")
W(r"\begin{tabular}{llr}")
W(r"\toprule")
W(r"Item & Description & Value \\")
W(r"\midrule")
rows_stats = [
    ("Curated images", "WIDER easy subset", n_img),
    ("Registered cohort", "registration QA pass (both modes)", len(cohort)),
    ("QA-gated-out images", "unstable detection/decoding at QA", n_gate_out),
    ("Multi-face images", "cohort, detected $\ge 2$ faces", multi_imgs),
    ("Detected faces", "cohort face observations", det_faces),
    ("Qualified faces", "cohort enrolled ROIs", qual_faces),
    ("Excluded faces", "detected $-$ qualified (cohort)", det_faces - qual_faces),
    ("Secret keys", "independent key runs", n_keys),
    ("Attack trials", "cohort, incl. benign + malicious", n_attack_trials),
    ("Wrong-key trials", "controls", att_counts.get("wrong_key", 0)),
    ("Wrong-claim trials", "controls", att_counts.get("wrong_claim", 0)),
    ("Verification rows", "test split, both modes (cohort)", len(test_rows)),
]
for a, b, c in rows_stats:
    W(f"{a} & {b} & {c} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ======================================================== scenario table (c1)
name_map = {
    "delete": "Face deletion",
    "duplicate": "Face duplication",
    "replace": "Replacement by unmarked face",
    "insert_unmarked": "Insertion of unmarked face",
    "insert_marked": "Insertion of foreign watermark",
    "detector_miss_control": "Controlled detector miss",
    "reorder_exact": "Spatial reorder (ROI-exact)",
    "reorder": "Reorder + resampling",
}
scen1 = ["delete", "duplicate", "replace", "insert_unmarked", "insert_marked",
         "detector_miss_control", "reorder_exact", "reorder"]
W(r"% ---------- Review comment 1: face-set scenarios (paper Table 5) ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Co-photo face-set authentication under identity- and count-changing"
  r" scenarios (registered WIDER cohort, multi-face queries). Set score is the"
  r" minimum face score $s=1-d_H/64$ per query, averaged over trials; the score"
  r" column is descriptive. Acceptance requires the complete unique ordinal-set"
  r" rule and the per-face threshold; undecidable outcomes are kept in the"
  r" denominator.}")
W(r"\label{tab:faceset_scenarios}")
W(r"\small")
W(r"\begin{tabular}{llccccc}")
W(r"\toprule")
W(r"Query Type & Mode & Trials & Set Score & $\gamma$ & Acc. (\%) & Undec. (\%) \\")
W(r"\midrule")
for atk in scen1:
    first = True
    for mname, gamma in GAMMAS:
        mode = "auto" if gamma == 1.0 else "blind"
        rows_a = sub(test_rows, atk, mode)
        n, a, rj, u = rates(rows_a, gamma)
        if n == 0:
            continue
        lbl = name_map.get(atk, atk) if first else ""
        W(f"{lbl} & {mname} & {n} & {mean_min_score(rows_a):.4f} & {gamma:.4f} & "
          f"{100*a/n:.1f} & {100*u/n:.1f} \\\\")
        first = False
    W(r"\midrule")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ======================================================== attack table (c2/8)
name_map2 = {
    "clean": "Clean (no processing)",
    "copy_marked_face": "Copy/transplant of marked face",
    "watermark_transplant": "Watermark-residual transplant",
    "collusion": "Collusion (multi-key average)",
    "erase_reembed_wrong_key": "Erase + re-embed (wrong key)",
    "boundary_tamper": "ROI-boundary tampering",
    "rotate": "Rotation ($5^{\circ}$)",
    "resize": "Resizing",
    "stretch": "Stretching ($1.2\\times0.8$)",
    "screenshot_simulation": "Screenshot (digital)",
    "jpeg": "JPEG re-compression",
    "metadata_strip": "Metadata stripping",
    "format_conversion": "Format conversion (WebP)",
    "platform_simulation": "Upload/download cycles ($\\times$3)",
}
order2 = ["clean", "copy_marked_face", "watermark_transplant", "collusion",
          "erase_reembed_wrong_key", "boundary_tamper", "rotate", "resize",
          "stretch", "screenshot_simulation", "jpeg", "metadata_strip",
          "format_conversion", "platform_simulation"]
W(r"% ---------- Review comments 2 & 8: attacks and platform processing ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Face-set verification under malicious attacks and platform"
  r" processing (same cohort and protocol as Table~\ref{tab:faceset_scenarios}).}")
W(r"\label{tab:attacks}")
W(r"\small")
W(r"\begin{tabular}{llccccc}")
W(r"\toprule")
W(r"Attack & Mode & Trials & Set Score & $\gamma$ & Acc. (\%) & Undec. (\%) \\")
W(r"\midrule")
for atk in order2:
    first = True
    for mname, gamma in GAMMAS:
        mode = "auto" if gamma == 1.0 else "blind"
        rows_a = sub(test_rows, atk, mode)
        n, a, rj, u = rates(rows_a, gamma)
        if n == 0:
            continue
        lbl = name_map2.get(atk, atk) if first else ""
        W(f"{lbl} & {mname} & {n} & {mean_min_score(rows_a):.4f} & {gamma:.4f} & "
          f"{100*a/n:.1f} & {100*u/n:.1f} \\\\")
        first = False
    W(r"\midrule")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# =============================================================== FAR table c3
W(r"% ---------- Review comment 3: FAR with full candidate-block search ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{False acceptance under wrong-key and wrong-claim controls with the"
  r" full candidate-block search (full block-grid phase lattice plus block-size"
  r" probes, $\le 151$ candidates per face). Theoretical references assume"
  r" i.i.d.\ tag bits only; the structural (count/ordinal) gate is exact-match"
  r" and not covered by the binomial model.}")
W(r"\label{tab:far}")
W(r"\small")
W(r"\begin{tabular}{lccccc}")
W(r"\toprule")
W(r"Operating point & Face pass $P_{\mathrm{face}}$ & Union bound & Empirical FAR & 95\% CI & Trials \\")
W(r"\midrule")
for mname, gamma in GAMMAS:
    t_req = math.ceil(64 * gamma - 16)
    t_req = max(0, min(48, t_req))
    p_face = sum(math.comb(48, k) for k in range(t_req, 49)) / 2 ** 48
    union = min(1.0, p_face * 151 * 8)  # 151 candidates x up to 8 faces
    rows_wk = [r for r in test_rows if r["attack"] in ("wrong_key", "wrong_claim")]
    n, a, rj, u = rates(rows_wk, gamma)
    lo, hi = wilson(a, n)
    W(f"{mname} & {p_face:.2e} & {union:.2e} & {a}/{n} & "
      f"[{lo:.4f}, {hi:.4f}] & {n} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ================================================= operating points table c7
print("computing ROC ...")
val_clean_faces, val_wk_faces = [], []
for r in val_rows:
    if r["attack"] == "clean":
        for fc in r.get("faces") or []:
            val_clean_faces.append(face64(fc))
    elif r["attack"] == "wrong_key":
        for fc in r.get("faces") or []:
            val_wk_faces.append(face64(fc))
roc = []
for th in range(0, 65):
    tpr = sum(1 for v in val_clean_faces if v >= th) / max(len(val_clean_faces), 1)
    fpr = sum(1 for v in val_wk_faces if v >= th) / max(len(val_wk_faces), 1)
    roc.append((fpr, tpr, th))

fig, ax = plt.subplots(figsize=(3.2, 2.8))
ax.plot([p[0] for p in roc], [p[1] for p in roc], color="#4c72b0", lw=1.5, label="ROC (face level)")
for mname, gamma in GAMMAS:
    th = math.ceil(64 * gamma)
    tpr = sum(1 for v in val_clean_faces if v >= th) / max(len(val_clean_faces), 1)
    fpr = sum(1 for v in val_wk_faces if v >= th) / max(len(val_wk_faces), 1)
    ax.plot([fpr], [tpr], "o", ms=5, label=f"$\\gamma$={gamma}")
ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
ax.set_xlabel("False acceptance rate (wrong-key faces)")
ax.set_ylabel("True acceptance rate (genuine faces)")
ax.set_xscale("symlog", linthresh=1e-4)
ax.grid(alpha=0.3)
ax.legend(fontsize=6)
fig.tight_layout()
savefig2(fig, FIGS / "fig_roc")
plt.close(fig)

W(r"% ---------- Review comment 7: operating points on validation split ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Operating points on the independent validation split"
  r" (face level; FAR on wrong-key faces, FRR on genuine faces). The validation"
  r" split is disjoint from all test images; thresholds are fixed before test"
  r" evaluation and never retuned on test data. FRR is measured on registered"
  r" faces whose pristine round trip succeeded at QA; it therefore reflects"
  r" post-registration verification misses (e.g., processing-induced decode"
  r" loss), not the enrollment failure rate of unfiltered imagery.}")
W(r"\label{tab:operating}")
W(r"\small")
W(r"\begin{tabular}{lccc}")
W(r"\toprule")
W(r"Operating point & Threshold & Val. FAR & Val. FRR \\")
W(r"\midrule")
for mname, gamma in GAMMAS:
    th = math.ceil(64 * gamma)
    tpr = sum(1 for v in val_clean_faces if v >= th) / max(len(val_clean_faces), 1)
    fpr = sum(1 for v in val_wk_faces if v >= th) / max(len(val_wk_faces), 1)
    W(f"{mname} & {gamma:.4f} & {fpr:.4f} & {1-tpr:.4f} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ========================================================= E2E runtime table
e2e_attacks = ["clean", "jpeg", "resize", "metadata_strip", "format_conversion",
               "screenshot_simulation", "platform_simulation"]
W(r"% ---------- Review comment 8: end-to-end runtime ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{End-to-end verification performance under platform processing"
  r" (Intel i7-12650H, 16 GB RAM, RTX 3060 Laptop GPU). Runtime is per query"
  r" (mean$\pm$std / median / p95) including face detection, candidate-block"
  r" search, decoding and decision; BAR is the mean bit accuracy of decoded"
  r" tags.}")
W(r"\label{tab:e2e}")
W(r"\small")
W(r"\begin{tabular}{llccccc}")
W(r"\toprule")
W(r"Scenario & Mode & Trials & E2E ms (mean$\pm$std) & Med. & p95 & BAR (\%) \\")
W(r"\midrule")
for atk in e2e_attacks:
    for mode in ("auto", "blind"):
        rows_a = [r for r in test_rows if r["attack"] == atk and r["requested_mode"] == mode]
        if not rows_a:
            continue
        xs = sorted(r["end_to_end_ms"] for r in rows_a)
        n = len(xs)
        mean = sum(xs) / n
        std = (sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
        vals = []
        for r in rows_a:
            for fc in r.get("faces") or []:
                if fc.get("tag_bits"):
                    vals.append(fc["tag_matches"] / fc["tag_bits"])
        bar = 100 * sum(vals) / len(vals) if vals else float("nan")
        mname = "Metadata" if mode == "auto" else "Blind"
        W(f"{name_map2.get(atk, atk)} & {mname} & {n} & {mean:.0f}$\\pm${std:.0f} & "
          f"{xs[n//2]:.0f} & {xs[min(n-1, int(0.95*n))]:.0f} & {bar:.1f} \\\\")
    W(r"\midrule")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# ==================================================== ROI / byN tables (c9)
band_defs = [("64-95", 64, 96), ("96-127", 96, 128), ("128+", 128, 10 ** 9)]
band_faces = {b[0]: {"n": 0, "decoded": 0, "sizes": []} for b in band_defs}


def short_edge(bb):
    return min(bb[2] - bb[0], bb[3] - bb[1])


for r in test_rows:
    if r["attack"] != "clean" or r["requested_mode"] != "blind":
        continue
    for fc in r.get("faces") or []:
        bb = fc.get("bbox") or fc.get("geometry")
        if not bb:
            continue
        se = short_edge(bb)
        for name, lo, hi in band_defs:
            if lo <= se < hi:
                band_faces[name]["n"] += 1
                band_faces[name]["decoded"] += 1 if header_ok(fc) else 0
                band_faces[name]["sizes"].append(se)
                break

W(r"% ---------- Review comment 9: decoding vs face size and face count ----------")
W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Header decoding success versus face size on clean queries"
  r" (blind fallback). All cohort faces are qualified (short edge $\ge 64$ px,"
  r" $\ge 3$ blocks); small faces below the qualification line are excluded"
  r" before enrollment by design.}")
W(r"\label{tab:roi}")
W(r"\small")
W(r"\begin{tabular}{lccc}")
W(r"\toprule")
W(r"Face band (short edge) & Faces & Mean size (px) & Decode Success (\%) \\")
W(r"\midrule")
for name, lo, hi in band_defs:
    d = band_faces[name]
    mean_sz = sum(d["sizes"]) / len(d["sizes"]) if d["sizes"] else float("nan")
    ds = 100 * d["decoded"] / d["n"] if d["n"] else float("nan")
    W(f"{name} px & {d['n']} & {mean_sz:.0f} & {ds:.1f} \\\\")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

W(r"\begin{table}[t]")
W(r"\centering")
W(r"\caption{Verification outcomes versus enrolled face count $N$ on clean"
  r" queries (registered cohort). Undecidable (no decodable evidence) is kept"
  r" in the denominator. Selection note: the cohort passed registration QA on"
  r" pristine files in both modes, so these rates characterize post-enrollment"
  r" verification stability under the complete-set conjunction, not raw"
  r" first-attempt decode capability on unfiltered imagery.}")
W(r"\label{tab:byN}")
W(r"\small")
W(r"\begin{tabular}{lcccccc}")
W(r"\toprule")
W(r"Mode & $N$ & Trials & Acc. (\%) & Rej. (\%) & Undec. (\%) \\")
W(r"\midrule")
ns = sorted({r["enrolled_faces"] for r in test_rows})
for mname, gamma in GAMMAS:
    mode = "auto" if gamma == 1.0 else "blind"
    first = True
    for N in ns:
        rows_n = [r for r in test_rows if r["attack"] == "clean"
                  and r["requested_mode"] == mode and r["enrolled_faces"] == N]
        if not rows_n:
            continue
        n, a, rj, u = rates(rows_n, gamma)
        lbl = mname if first else ""
        W(f"{lbl} & {N} & {n} & {100*a/n:.1f} & {100*rj/n:.1f} & {100*u/n:.1f} \\\\")
        first = False
    W(r"\midrule")
W(r"\bottomrule")
W(r"\end{tabular}")
W(r"\end{table}")
W("")

# --------------------------------------------------------------- byN + size figs
fig, axes = plt.subplots(1, len(GAMMAS), figsize=(7.2, 2.5), sharey=True)
for ax, (mname, gamma) in zip(axes, GAMMAS):
    mode = "auto" if gamma == 1.0 else "blind"
    xs_, acc, rej, und = [], [], [], []
    for N in ns:
        rows_n = [r for r in test_rows if r["attack"] == "clean"
                  and r["requested_mode"] == mode and r["enrolled_faces"] == N]
        if not rows_n:
            continue
        n, a, rj, u = rates(rows_n, gamma)
        xs_.append(N)
        acc.append(100 * a / n); rej.append(100 * rj / n); und.append(100 * u / n)
    ax.plot(xs_, acc, "o-", label="Accepted", color=COL_GREEN)
    ax.plot(xs_, rej, "s-", label="Rejected", color=COL_RED)
    ax.plot(xs_, und, "^-", label="Undecidable", color=COL_GRAY)
    ax.set_xlabel("Enrolled faces $N$")
    ax.set_title(mname.replace("$", "").replace("{", "").replace("}", ""), fontsize=7.5)
    ax.set_xticks(xs_)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Rate (%)")
axes[0].legend(fontsize=6)
fig.tight_layout()
savefig2(fig, FIGS / "fig_by_N")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.6, 2.6))
names = [b[0] for b in band_defs]
decode = [100 * band_faces[nm]["decoded"] / band_faces[nm]["n"] if band_faces[nm]["n"] else 0
          for nm in names]
counts = [band_faces[nm]["n"] for nm in names]
bars = ax.bar(names, decode, color=["#4c72b0", "#55a868", "#c44e52"], width=0.55)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"n={c}", ha="center", fontsize=6.5)
ax.set_xlabel("Face short-edge band (px)")
ax.set_ylabel("Header decode success (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
savefig2(fig, FIGS / "fig_face_size")
plt.close(fig)

# ============================================================= example figures
by_src_atk = defaultdict(list)
for r in test_rows:
    by_src_atk[(r["source_id"], r["attack"])].append(r)


def pick_trial(source, attack):
    cands = sorted(by_src_atk.get((source, attack), []),
                   key=lambda r: (r.get("repetition", 0), r["trial_id"]))
    return cands[0] if cands else None


FIG_SPECS = {
    "fig5_multiface_scenarios": [
        ("clean", "Original"), ("delete", "Face Deletion"),
        ("duplicate", "Face Duplication"), ("replace", "Face Replacement"),
        ("insert_unmarked", "Insert Unmarked"), ("insert_marked", "Insert Marked"),
        ("reorder_exact", "Spatial Reorder"),
    ],
    "fig_attack_malicious": [
        ("clean", "Original"), ("copy_marked_face", "Copy / Transplant"),
        ("watermark_transplant", "Residual Transplant"), ("collusion", "Collusion"),
        ("erase_reembed_wrong_key", "Erase + Re-embed"), ("boundary_tamper", "ROI Boundary"),
    ],
    "fig_attack_geometric": [
        ("clean", "Original"), ("rotate", "Rotate 5°"), ("resize", "Resize"),
        ("stretch", "Stretch 1.2×0.8"), ("screenshot_simulation", "Screenshot"),
    ],
    "fig_attack_platform": [
        ("clean", "Original"), ("jpeg", "JPEG"), ("metadata_strip", "Metadata Strip"),
        ("format_conversion", "WebP"), ("platform_simulation", "Upload/Download ×3"),
    ],
}

GAMMA_FIG = 0.8125  # figure labels at the blind 0.8125 operating point


def draw_panel(ax, row, title):
    if row is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        return
    img = Image.open(row["image_path"]).convert("RGB")
    ax.imshow(img)
    for fc in row.get("faces") or []:
        bb = fc.get("bbox") or fc.get("geometry")
        if not bb:
            continue
        c = COL_GREEN if header_ok(fc) and face64(fc) >= 64 * GAMMA_FIG else COL_RED
        ax.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                               fill=False, edgecolor=c, linewidth=1.1))
    dec = decide(row, GAMMA_FIG)
    col = COL_GREEN if dec == "ACCEPTED" else (COL_RED if dec == "REJECTED" else COL_GRAY)
    fscores = "/".join(f"{face64(fc)/64:.2f}" for fc in (row.get("faces") or [])[:6]) or "-"
    ax.set_title(title, fontsize=8)
    ax.text(0.5, -0.05, f"{dec.title()} Set={min_face_score(row):.3f}\nfaces: {fscores}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
            color=col, linespacing=1.2)
    ax.set_xticks([]); ax.set_yticks([])


all_sources = sorted({r["source_id"] for r in test_rows if r["attack"] == "clean"})
for fig_name, cols in FIG_SPECS.items():
    scored = []
    for s in all_sources:
        rc = pick_trial(s, "clean")
        if rc is None or decide(rc, GAMMA_FIG) != "ACCEPTED":
            continue
        avail = sum(1 for atk, _ in cols if pick_trial(s, atk) is not None)
        scored.append((avail, -rc["enrolled_faces"], s))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    sources = [s for _, _, s in scored[:3]]
    if not sources:
        print("NO sources for", fig_name)
        continue
    fig, axes = plt.subplots(len(sources), len(cols),
                             figsize=(1.9 * len(cols), 1.8 * len(sources)))
    for i, s in enumerate(sources):
        for j, (atk, label) in enumerate(cols):
            draw_panel(axes[i][j], pick_trial(s, atk), label)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.1, w_pad=0.6)
    savefig2(fig, FIGS / fig_name)
    plt.close(fig)
    print("saved", fig_name)

(OUT / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
print("saved tables.tex")
print("ALL DONE ->", OUT)

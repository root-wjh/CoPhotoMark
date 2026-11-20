from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import beta, binom, rankdata

from .io import canonical

# Decision units.  `rows_mode` selects which evaluated pipeline supplies the
# scores; `level` selects the statistical unit.
DEFAULT_UNITS = {
    # Exact localization and authenticated parameters: the strictest point is
    # affordable, so the budget is negligible.
    "auto": {"budget": 1e-14, "max_faces": 1, "rows_mode": "auto", "level": "face"},
    # Blind fallback: localization/parameter search adds uncertainty, so a
    # finite per-ROI budget is allowed.  1e-5 -> tau=9 (P=7.61e-6).
    "blind": {"budget": 1e-5, "max_faces": 1, "rows_mode": "blind", "level": "face"},
    # Set-level co-photo claim: the per-face budget is divided over the
    # supported group size (N<=20), giving the same tau=9 and a set-level
    # union bound of 1.5e-4.
    "set": {"budget": 2e-4, "max_faces": 20, "rows_mode": "auto", "level": "image"},
}


def tag_bound(tau: int, bits: int) -> float:
    """P_fa for one independent-uniform wrong-key tag test."""
    if not 0 <= tau <= bits:
        raise ValueError("tau outside tag length")
    return float(binom.cdf(tau, bits, 0.5))


def select_tau(bits: int, budget: float, max_faces: int = 1) -> int | None:
    """Largest admissible tag-error tolerance under the union-bound budget."""
    if bits < 1 or max_faces < 1 or not 0 < budget <= 1:
        raise ValueError("Invalid calibration budget")
    best = None
    for tau in range(bits + 1):
        if max_faces * tag_bound(tau, bits) <= budget:
            best = tau
        else:
            break
    return best


def _face_vectors(rows, mode):
    """Per-ROI samples: genuine faces vs wrong-key faces.

    A face that fails structural/header decoding contributes score 0, so
    undecidable faces stay in the FRR denominator.  Wrong-key rows make every
    face a genuine negative sample; identity-attack rows are *not* mixed in
    because their untouched faces are still valid evidence.
    """
    pos, neg = [], []
    neg_sources = set()
    for row in rows:
        if row.get("mode") != mode or row.get("status") == "error":
            continue
        faces = row.get("faces") or []
        synthetic = False
        if not faces and row.get("score") is not None:
            faces = [{"valid": row.get("structural_ok", True), "score": row.get("score", 0.0)}]
            synthetic = True
        if row.get("label") == "genuine":
            for face in faces:
                pos.append(float(face.get("score", 0.0)) if face.get("valid") else 0.0)
        elif row.get("label") == "negative" and (row.get("attack") == "wrong_key" or synthetic):
            for face in faces:
                neg.append(float(face.get("score", 0.0)) if face.get("valid") else 0.0)
            neg_sources.add(row.get("source_id"))
    return pos, neg, neg_sources


def _image_vectors(rows, mode):
    """Whole-image samples for the set-level decision (min over protected faces)."""
    pos, neg = [], []
    neg_sources = set()
    for row in rows:
        if row.get("mode") != mode or row.get("status") == "error":
            continue
        score = float(row.get("raw_score", row.get("score", 0.0))) if row.get("structural_ok") else 0.0
        if row.get("label") == "genuine":
            pos.append(score)
        elif row.get("label") == "negative":
            neg.append(score)
            neg_sources.add(row.get("source_id"))
    return pos, neg, neg_sources


def _auc(pos, neg):
    if not pos or not neg:
        return None
    scores = np.asarray(pos + neg, dtype=float)
    ranks = rankdata(scores)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _cp(successes, total, alpha=0.05):
    if total == 0:
        return [None, None]
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [low, high]


def _zero_error_upper(total, alpha=0.05):
    return None if total == 0 else float(1 - alpha ** (1 / total))


def _curve(pos, neg, bits):
    points = []
    for tau in range(bits + 1):
        threshold = 1 - tau / bits
        fp = sum(score >= threshold for score in neg)
        tp = sum(score >= threshold for score in pos)
        points.append({
            "tau": tau,
            "threshold": threshold,
            "far": fp / len(neg) if neg else None,
            "frr": 1 - tp / len(pos) if pos else None,
            "false_accepts": fp,
            "true_accepts": tp,
            "negative_samples": len(neg),
            "positive_samples": len(pos),
        })
    return points


def calibrate_units(rows, config, units=None, target_far=None):
    """Freeze one operating point per decision unit and report validation ROC.

    `target_far` is accepted for backward compatibility only; the unit budgets
    in `units` are the selector.  It is recorded, never fitted.
    """
    if not rows or any(r.get("split") != "validation" for r in rows):
        raise ValueError("Calibration accepts only independently held-out validation rows")
    if any(r.get("status") == "error" for r in rows):
        raise ValueError("Fix runtime errors before calibration; unavailable evidence is logged separately")
    signatures = {r.get("evaluation_signature", "unspecified_control") for r in rows}
    if len(signatures) != 1:
        raise ValueError("Do not mix detector/voting protocols during calibration")
    if not any(r.get("label") == "genuine" for r in rows) or not any(r.get("label") == "negative" for r in rows):
        raise ValueError("Both genuine and negative validation trials required")

    bits = int(config.tag_bits)
    units = units or DEFAULT_UNITS
    modes = {}
    for name, spec in units.items():
        mode_rows = [r for r in rows if r.get("mode") == spec["rows_mode"]]
        if not mode_rows:
            continue
        if spec["level"] == "face":
            pos, neg, neg_sources = _face_vectors(mode_rows, spec["rows_mode"])
        else:
            pos, neg, neg_sources = _image_vectors(mode_rows, spec["rows_mode"])
        if not pos or not neg:
            raise ValueError(f"Unit {name} lacks positive or negative validation samples")

        tau = select_tau(bits, spec["budget"], spec["max_faces"])
        if tau is None:
            modes[name] = {"status": "budget_infeasible", "threshold": None, "budget": spec["budget"]}
            continue
        threshold = 1 - tau / bits
        curve = _curve(pos, neg, bits)
        chosen = next(p for p in curve if p["tau"] == tau)
        far, frr = chosen["far"], chosen["frr"]
        fp = chosen["false_accepts"]
        modes[name] = {
            "status": "calibrated_by_budget",
            "selection_rule": "max tau with max_faces*P_fa(tau) <= budget",
            "budget": spec["budget"],
            "max_faces": spec["max_faces"],
            "tag_bits": bits,
            "tau": tau,
            "threshold": threshold,
            "bound_at_threshold": tag_bound(tau, bits),
            "union_bound_at_threshold": spec["max_faces"] * tag_bound(tau, bits),
            "level": spec["level"],
            "rows_mode": spec["rows_mode"],
            "validation": {
                "far": far,
                "frr": frr,
                "false_accepts": fp,
                "true_accepts": chosen["true_accepts"],
                "positive_samples": len(pos),
                "negative_samples": len(neg),
                "negative_source_images": len(neg_sources),
                "far_ci95_trialwise": _cp(fp, len(neg)),
                "zero_false_accept_upper95_sources": _zero_error_upper(len(neg_sources)) if fp == 0 else None,
            },
            "auc": _auc(pos, neg),
            "approximate_eer": (far + frr) / 2,
            "roc": curve,
        }

    return {
        "version": 2,
        "config_sha256": config.fingerprint,
        "tag_bits": bits,
        "target_far_recorded_not_fitted": target_far,
        "evaluation_signature": next(iter(signatures)),
        "validation_source_ids": sorted({r["source_id"] for r in rows}),
        "validation_groups": sorted({r.get("group_id", r["source_id"]) for r in rows}),
        "validation_hashes": sorted({r.get("source_sha256", "") for r in rows} - {""}),
        "validation_rows_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
        "modes": modes,
    }

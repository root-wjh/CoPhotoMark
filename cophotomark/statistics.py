from __future__ import annotations
import math
import numpy as np
from scipy.stats import beta, binom
from .io import canonical
import hashlib

def clopper_pearson(successes, total, alpha=0.05):
    if total == 0:
        return [None, None]
    return [0.0 if successes == 0 else float(beta.ppf(alpha/2, successes, total-successes+1)),
            1.0 if successes == total else float(beta.ppf(1-alpha/2, successes+1, total-successes))]

def zero_error_upper(total, alpha=0.05):
    return None if total == 0 else float(1-alpha**(1/total))

def binomial_reference(tag_bits, allowed_errors, candidates):
    single = float(binom.cdf(allowed_errors, tag_bits, 0.5))
    return {"independent_bit_single_candidate_reference": single,
            "union_bound_reference": min(1.0, candidates*single),
            "candidates": candidates,
            "qualification": "Reference under independent unbiased wrong-key tag bits; NOT an adaptive-tampering guarantee"}

def accepted(row, threshold):
    return bool(row.get("structural_ok")) and row.get("decision") != "UNDECIDABLE" and row.get("score", 0) >= threshold

def calibrate(rows, config, target_far=0.001, units=None):
    """Budget-based calibration on the 48-bit tag (R2-5).

    The old empirical-FAR selector is removed: because the count/ordinal check
    rejects almost every wrong-key candidate, empirical FAR is zero at every
    threshold and a min-FRR rule collapses to the lowest threshold.  The
    selector is now the analytic tag bound with an explicit per-unit budget;
    the validation split reports ROC/FAR/FRR and verifies no leakage.
    """
    from .calibration import calibrate_units
    return calibrate_units(rows, config, units=units, target_far=target_far)

def cluster_bootstrap(rows, metric, seed=20261031, repeats=1000):
    groups = {}
    for row in rows:
        groups.setdefault(row["source_id"], []).append(row)
    names = list(groups)
    if len(names) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = [r for i in rng.integers(0, len(names), len(names)) for r in groups[names[i]]]
        value = metric(sample)
        if value is not None:
            values.append(value)
    return [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else [None, None]

"""Statistical apparatus for benchmark comparison.

Implements the apparatus specified in ben.md:
  - paired bootstrap CI on metric(method_a) - metric(method_b)
  - Wilcoxon signed-rank p-value on paired deltas
  - Holm-Bonferroni correction across a family of tests
  - Cohen's d on paired deltas
  - rank-biserial r as Wilcoxon effect size
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedResult:
    delta_mean: float
    ci_lo: float
    ci_hi: float
    wilcoxon_p: float
    cohens_d: float
    rank_biserial: float
    n: int


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
    ci: float = 0.95, seed: int = 0,
) -> tuple[float, float, float]:
    """95% paired-bootstrap CI on mean(a) - mean(b).

    Resamples the (a_i, b_i) pairs with replacement.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if n != len(b) or n == 0:
        return float("nan"), float("nan"), float("nan")
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    alpha = 1.0 - ci
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1.0 - alpha / 2))
    mean = float((a - b).mean())
    return mean, float(lo), float(hi)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test. Returns (W, two-sided p-value, r effect size).

    Returns (p_value, rank_biserial).
    """
    try:
        from scipy.stats import wilcoxon
        d = np.asarray(a) - np.asarray(b)
        d = d[d != 0]
        if len(d) < 2:
            return float("nan"), float("nan")
        stat, p = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        # rank-biserial: 1 - (2W) / (n(n+1)/2) is one convention.
        ranks = np.argsort(np.argsort(np.abs(d))) + 1
        W_pos = ranks[d > 0].sum()
        W_neg = ranks[d < 0].sum()
        total = W_pos + W_neg
        r = (W_pos - W_neg) / total if total > 0 else float("nan")
        return float(p), float(r)
    except Exception:
        return float("nan"), float("nan")


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    if len(d) == 0:
        return float("nan")
    s = d.std(ddof=1) if len(d) > 1 else 0.0
    if s == 0:
        return float("nan")
    return float(d.mean() / s)


def compare_paired(a: np.ndarray, b: np.ndarray,
                   n_boot: int = 10_000) -> PairedResult:
    mean, lo, hi = paired_bootstrap_ci(a, b, n_boot=n_boot)
    p, r = wilcoxon_signed_rank(a, b)
    d = cohens_d_paired(a, b)
    return PairedResult(
        delta_mean=mean,
        ci_lo=lo, ci_hi=hi,
        wilcoxon_p=p,
        cohens_d=d,
        rank_biserial=r,
        n=len(a),
    )


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjustment for a family of p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        raw = p_values[idx]
        adj = min(1.0, raw * (m - i))
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted.tolist()

"""Bootstrap confidence intervals for the low-FPR points of a U-RMIA report.

Reads the per-target ``attack_result_*.npz`` archives that ``report_urmia_attack``
already writes (they carry the raw per-sample ``scores`` and ``memberships``), so
this re-analyses a finished run **without touching models, signals, or training**.

Two problems with the summary's ``TPR@0.1%FPR`` column motivate it:

1. With |U| = 500 unseen samples the available FPR grid is 0, 0.002, 0.004, ...
   so "TPR at 0.1% FPR" can only resolve to *zero false positives* -- which is
   why ``one_tenth_fpr`` equals ``zero_fpr`` in every row of every run. That
   point is decided by where the single highest-scoring negative lands, so one
   unlucky negative drives it to 0.
2. A point estimate of 16/500 vs 0/500 carries no error bar, so "this scorer
   finds individuals the other cannot" is unfalsifiable as reported.

The fix here is to report **TPR at a fixed false-positive count** (stable under
the resolution limit) with a stratified bootstrap CI over the audit samples.

Usage::

    python analyze_tpr.py demo_urmia_online/report/exp
    python analyze_tpr.py demo_urmia_neggrad_plus/report/exp --fp 0 1 5 10
"""

import argparse
import glob
import os
import re

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve

# Order the summary tables use, so output lines up with urmia_online_summary.json.
_ROLE_ORDER = ["original", "unlearned", "retrained"]
_ATTACK_ORDER = [
    # Old labels kept alongside the new ones so runs made before the 2026-08-10
    # rename still sort into the same columns.
    "urmia_offline",
    "offline",
    "urmia_online",
    "rmia_online",
    "ulira",
    "conf_global",
    "mentr_global",
    "ulira_mentr",
]


def tpr_at_fp(pos: np.ndarray, neg: np.ndarray, n_fp: int) -> float:
    """TPR when exactly ``n_fp`` negatives are allowed above the threshold.

    Uses the (n_fp+1)-th largest negative as the cut, so the operating point is
    defined by a *count* rather than a rate -- unaffected by |U| being too small
    to resolve 0.1% FPR.

    Args:
        pos (np.ndarray): Scores of the forget (member) samples.
        neg (np.ndarray): Scores of the unseen (non-member) samples.
        n_fp (int): Number of false positives permitted.

    Returns:
        float: Fraction of members scoring strictly above the cut.
    """
    if neg.size <= n_fp:
        return float("nan")
    # Descending sort; index n_fp is the (n_fp+1)-th largest.
    threshold = np.partition(neg, -(n_fp + 1))[-(n_fp + 1)]
    return float(np.mean(pos > threshold))


def bootstrap_ci(
    pos: np.ndarray,
    neg: np.ndarray,
    n_fp: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple:
    """Stratified percentile bootstrap CI for ``tpr_at_fp``.

    Members and non-members are resampled separately so both group sizes stay
    fixed -- the quantity of interest is conditional on |F| and |U|.

    Returns:
        tuple: ``(lo, hi)`` 95% percentile bounds.
    """
    stats = np.empty(n_boot)
    for b in range(n_boot):
        p = rng.choice(pos, size=pos.size, replace=True)
        n = rng.choice(neg, size=neg.size, replace=True)
        stats[b] = tpr_at_fp(p, n, n_fp)
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def _sort_key(name: str) -> tuple:
    """Order results by role then attack, unknown names last but stable."""
    for r_i, role in enumerate(_ROLE_ORDER):
        if name == role:  # level-2 archives are named by role alone
            return (r_i, -1)
        if name.startswith(role + "_"):
            attack = name[len(role) + 1 :]
            a_i = _ATTACK_ORDER.index(attack) if attack in _ATTACK_ORDER else len(_ATTACK_ORDER)
            return (r_i, a_i)
    return (len(_ROLE_ORDER), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", help="A run's report/exp directory")
    parser.add_argument(
        "--fp",
        type=int,
        nargs="+",
        default=[0, 1, 5],
        help="False-positive counts to report (default: 0 1 5)",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    paths = glob.glob(os.path.join(args.report_dir, "attack_result_*.npz"))
    if not paths:
        raise SystemExit(f"No attack_result_*.npz under {args.report_dir}")

    named = []
    for path in paths:
        name = re.sub(r"^attack_result_|\.npz$", "", os.path.basename(path))
        named.append((name, path))
    named.sort(key=lambda item: _sort_key(item[0]))

    rng = np.random.default_rng(args.seed)
    header = f"{'target':<28}{'AUC':>8}{'|F|':>6}{'|U|':>6}"
    for k in args.fp:
        header += f"{'TPR@' + str(k) + 'FP':>22}"
    print(header)
    print("-" * len(header))

    for name, path in named:
        with np.load(path) as data:
            if "scores" not in data or "memberships" not in data:
                print(f"{name:<28}  (archive predates raw-score saving, skipped)")
                continue
            scores = data["scores"].ravel().astype(float)
            members = data["memberships"].ravel().astype(int)

        pos, neg = scores[members == 1], scores[members == 0]
        fpr, tpr, _ = roc_curve(members, scores)
        line = f"{name:<28}{sk_auc(fpr, tpr):>8.4f}{pos.size:>6}{neg.size:>6}"
        for k in args.fp:
            point = tpr_at_fp(pos, neg, k)
            lo, hi = bootstrap_ci(pos, neg, k, args.n_boot, rng)
            # Counts are what the claim is really about; rates hide n.
            cell = f"{point:.3f} [{lo:.3f},{hi:.3f}]"
            line += f"{cell:>22}"
        print(line)

    print(
        "\nTPR@kFP = fraction of forget points scoring above the (k+1)-th highest "
        "unseen score.\n95% stratified percentile bootstrap over the audit samples "
        f"({args.n_boot} resamples).\nOverlapping intervals between two scorers mean "
        "this run cannot separate them at that\noperating point."
    )


if __name__ == "__main__":
    main()

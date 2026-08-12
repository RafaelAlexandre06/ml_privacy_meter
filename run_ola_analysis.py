"""Re-analyse a finished OLA run. No models, no signals, no GPU.

``run_ola.py`` answers "can an attacker tell a member from a non-member" three
times over -- once restricted to the outer layer, once to the inner layer, once
to all members. What it never asks is whether an attacker can tell an
*attenuated member* from an *ordinary member*. Every subset in
``run_ola.py:232`` uses non-members as the negative class, so that contrast is
absent from the report. The OLA paper names exactly this gap in its future-work
section ("a similar three-way hypothesis test for OLA is an interesting
direction"), and it matters because the two questions can have opposite answers:
an attenuated point can be perfectly indistinguishable from a non-member while
being trivially separable from the members around it.

That contrast is recoverable from what is already on disk.
``ola_utils.report_ola_attack`` saves the raw ``scores`` and ``memberships`` into
every archive, and the ``all_members`` subset mask is ``np.ones(dataset_size)``
-- so ``attack_result_<role>_all_members_<scorer>.npz`` holds **every score in
dataset order**. Combined with ``ola_outer_layer.npz`` and ``ola_splits.npz``,
each of the following costs a few seconds of numpy:

1. **A vs B** -- outer layer against inner-layer members, the missing leg.
2. **Two-sided AUC and excess over the onion floor** for the cells the summary
   already reports, so a run finished before those fields existed gets them
   without re-running the audit stage.
3. **ASR tie census** -- how much of the outer layer was actually chosen by
   attackability rather than by the tie-break.
4. **Bootstrap CIs on the Privacy Onion claim**, via ``analyze_tpr``.

Usage::

    python run_ola_analysis.py --cf configs/ola/cifar10.yaml
    python run_ola_analysis.py --cf configs/ola/cifar10.yaml --n-boot 2000
"""

import argparse
import json
import os
import time

import numpy as np
import yaml

from analyze_tpr import bootstrap_ci, tpr_at_fp
from audit import compute_attack_results
from ola_utils import (
    AUDIT_SUBSETS,
    OLA_SPLITS_FILE,
    OUTER_LAYER_FILE,
    TARGET_ROLES,
    mann_whitney_null_se,
    two_sided_auc,
)
from util import setup_log

# False-positive *counts*, not rates. With ~25,000 non-members these are the 0%,
# 0.1% and 1% operating points, but a count stays meaningful if the audit set
# changes size, and it is what analyze_tpr bootstraps.
DEFAULT_FP_COUNTS = [0, 25, 250]


def load_dataset_ordered_scores(report_exp: str, logger) -> tuple:
    """Load every ``all_members`` archive as a full-length, dataset-ordered vector.

    The ``all_members`` subset mask is ``np.ones(dataset_size, dtype=bool)``, so
    ``scores[rows]`` in ``run_ola.py`` is the identity and the saved vector is in
    dataset order. That is the property everything downstream relies on. Note the
    guard lives in ``main`` (which cross-checks the loaded membership vector
    against ``ola_splits.npz``), not in this loader -- a change to ``subset_rows``
    would otherwise silently reindex every contrast below while still producing
    plausible AUCs.

    Args:
        report_exp (str): ``<log_dir>/report/exp``.
        logger (logging.Logger): Logger object.

    Returns:
        tuple: ``(scores, is_member)`` where ``scores`` is
            ``{role: {scorer: np.ndarray}}`` and ``is_member`` is the shared
            (dataset_size,) 0/1 membership vector.
    """
    scores, is_member = {}, None
    for role in TARGET_ROLES:
        prefix = f"attack_result_{role}_all_members_"
        found = {}
        for fname in sorted(os.listdir(report_exp)):
            if not (fname.startswith(prefix) and fname.endswith(".npz")):
                continue
            scorer = fname[len(prefix) : -len(".npz")]
            with np.load(os.path.join(report_exp, fname)) as data:
                if "scores" not in data:
                    raise SystemExit(
                        f"{fname} predates raw-score saving; re-run the audit stage."
                    )
                found[scorer] = data["scores"].ravel().astype(float)
                members = data["memberships"].ravel().astype(int)
            if is_member is None:
                is_member = members
            elif not np.array_equal(is_member, members):
                raise SystemExit(
                    f"{fname} disagrees with earlier archives on which samples are "
                    "members. The report directory mixes runs -- use a clean one."
                )
        if not found:
            raise SystemExit(
                f"No '{prefix}*.npz' under {report_exp}. This script needs the "
                "all_members subset, which is the only one saved in dataset order."
            )
        scores[role] = found

    scorers = sorted(scores[TARGET_ROLES[0]])
    for role in TARGET_ROLES:
        if sorted(scores[role]) != scorers:
            raise SystemExit(
                f"Role '{role}' has scorers {sorted(scores[role])}, expected "
                f"{scorers}. The report directory is incomplete."
            )
    logger.info(
        "Loaded %d dataset-ordered score vectors (%d roles x %d scorers), "
        "%d samples, %d members.",
        len(TARGET_ROLES) * len(scorers),
        len(TARGET_ROLES),
        len(scorers),
        is_member.size,
        int(is_member.sum()),
    )
    return scores, is_member


def asr_tie_census(
    asr: np.ndarray, target_train: np.ndarray, outer_indices: np.ndarray, logger
) -> dict:
    """Measure how much of the outer layer the tie-break chose, not the attack.

    ASR is a mean of at most one binary outcome per selection model, so a
    32-model selection half gives it 33 possible values. When more members share
    the cut value than there is room for, ``select_outer_layer``'s stable argsort
    admits the lowest-indexed ones. On CIFAR-10 the dataset index is class-
    ordered, so a large tie skews the outer layer toward the early classes and
    every number in the report inherits that.

    Args:
        asr (np.ndarray): Per-sample ASR over the whole dataset.
        target_train (np.ndarray): The target's member indices.
        outer_indices (np.ndarray): The selected outer layer.
        logger (logging.Logger): Logger object.

    Returns:
        dict: Census fields, including ``arbitrary_fraction``.
    """
    member_asr = np.nan_to_num(asr[target_train], nan=-1.0)
    cut = float(np.min(np.nan_to_num(asr[outer_indices], nan=-1.0)))
    tied = int(np.sum(member_asr == cut))
    tied_admitted = int(
        np.sum(np.nan_to_num(asr[outer_indices], nan=-1.0) == cut)
    )
    above = int(np.sum(member_asr > cut))
    distinct = int(np.unique(member_asr).size)
    arbitrary = tied_admitted / len(outer_indices) if len(outer_indices) else 0.0

    logger.info("")
    logger.info("=== ASR TIE CENSUS ===")
    logger.info("Distinct ASR values among members : %d", distinct)
    logger.info("Cut ASR                           : %.4f", cut)
    logger.info("Members strictly above the cut    : %d", above)
    logger.info("Members tied at the cut           : %d", tied)
    logger.info("Tied members admitted             : %d", tied_admitted)
    logger.info(
        "Fraction of the outer layer decided by the tie-break rather than by "
        "ASR: %.1f%%",
        100.0 * arbitrary,
    )
    if tied > 2 * tied_admitted and tied_admitted > 0:
        logger.warning(
            "Ties dominate the cut (%d contenders for %d places). The outer "
            "layer is largely a low-index slice of an equivalence class, not "
            "the %d most attackable points. More shadow models would give ASR "
            "the resolution to rank inside the tie.",
            tied,
            tied_admitted,
            len(outer_indices),
        )
    return {
        "distinct_member_asr_values": distinct,
        "cut": cut,
        "members_above_cut": above,
        "members_tied_at_cut": tied,
        "tied_admitted": tied_admitted,
        "arbitrary_fraction": float(arbitrary),
    }


def outer_vs_inner(scores: dict, outer_mask: np.ndarray, member_mask: np.ndarray,
                   logger) -> dict:
    """The A-vs-B contrast: attenuated members against ordinary members.

    Non-members are excluded entirely. Both classes were in the training set, so
    a separation here cannot be explained by the difficulty confound that makes
    the outer-vs-non-member numbers so hard to read -- the ``onion`` model scores
    the outer layer almost exactly as ``ola`` does precisely because those points
    are intrinsically hard. Against *members*, that excuse is gone.

    Args:
        scores (dict): ``{role: {scorer: dataset-ordered scores}}``.
        outer_mask (np.ndarray): Boolean, True for outer-layer points.
        member_mask (np.ndarray): Boolean, True for the target's members.
        logger (logging.Logger): Logger object.

    Returns:
        dict: ``{role: {scorer: result}}``.
    """
    rows = member_mask
    labels = outer_mask[rows].astype(int)  # 1 = attenuated, 0 = ordinary member

    logger.info("")
    logger.info("=== A vs B: OUTER LAYER vs INNER-LAYER MEMBERS ===")
    logger.info(
        "%d attenuated vs %d ordinary members. Non-members excluded, so the "
        "difficulty floor that dominates the outer-vs-non-member cells does not "
        "apply here.",
        int(labels.sum()),
        int((1 - labels).sum()),
    )
    header = f"{'role':<10}"
    for scorer in sorted(scores[TARGET_ROLES[0]]):
        header += f" {scorer + '_auc':>16} {scorer + '_2sided':>16}"
    logger.info(header)

    out = {}
    for role in TARGET_ROLES:
        line = f"{role:<10}"
        out[role] = {}
        for scorer in sorted(scores[role]):
            res = compute_attack_results(scores[role][scorer][rows], labels)
            out[role][scorer] = {
                "auc": float(res["auc"]),
                "two_sided_auc": two_sided_auc(res["auc"]),
                "one_fpr": float(res["one_fpr"]),
                "one_tenth_fpr": float(res["one_tenth_fpr"]),
                "zero_fpr": float(res["zero_fpr"]),
            }
            line += (
                f" {res['auc']:>16.4f} {two_sided_auc(res['auc']):>16.4f}"
            )
        logger.info(line)
    logger.info(
        "'vanilla' is the control: it applies no attenuation, so whatever it "
        "reads here is how separable these points were to begin with. Read "
        "'ola' against 'vanilla', not against 0.5."
    )
    return out


def recompute_subsets(scores: dict, is_member: np.ndarray, outer_mask: np.ndarray,
                      n_nonmembers: int, logger) -> dict:
    """Rebuild the summary's cells with two-sided AUC and the onion contrast.

    Reproduces ``run_ola.py``'s ``subset_rows`` exactly so the ``auc`` values can
    be cross-checked against ``ola_summary.json``, then adds the two fields the
    summary was missing when this run was produced.

    Args:
        scores (dict): ``{role: {scorer: dataset-ordered scores}}``.
        is_member (np.ndarray): 0/1 membership over the dataset.
        outer_mask (np.ndarray): Boolean, True for outer-layer points.
        n_nonmembers (int): Size of the negative class.
        logger (logging.Logger): Logger object.

    Returns:
        dict: ``{subset: {role: {scorer: result}}}`` plus an ``excess_over_onion``
            block keyed the same way ``log_ola_summary`` keys it.
    """
    non_members = is_member == 0
    subset_rows = {
        "outer": outer_mask | non_members,
        "inner": ((is_member == 1) & ~outer_mask) | non_members,
        "all_members": np.ones(is_member.size, dtype=bool),
    }
    # One SE per subset: the outer block has ~1,000 positives against the same
    # non-members the inner block has ~24,000 against.
    subset_se = {
        subset: mann_whitney_null_se(
            int((rows & (is_member == 1)).sum()), n_nonmembers
        )
        for subset, rows in subset_rows.items()
    }

    out = {}
    for subset in AUDIT_SUBSETS:
        rows = subset_rows[subset]
        out[subset] = {}
        logger.info("")
        logger.info("=== %s (two-sided) ===", subset.upper())
        header = f"{'role':<10}"
        for scorer in sorted(scores[TARGET_ROLES[0]]):
            header += f" {scorer:>16}"
        logger.info(header)
        for role in TARGET_ROLES:
            line = f"{role:<10}"
            out[subset][role] = {}
            for scorer in sorted(scores[role]):
                res = compute_attack_results(scores[role][scorer][rows],
                                             is_member[rows])
                out[subset][role][scorer] = {
                    "auc": float(res["auc"]),
                    "two_sided_auc": two_sided_auc(res["auc"]),
                    "one_fpr": float(res["one_fpr"]),
                    "one_tenth_fpr": float(res["one_tenth_fpr"]),
                    "zero_fpr": float(res["zero_fpr"]),
                }
                line += f" {two_sided_auc(res['auc']):>16.4f}"
            logger.info(line)

    excess = {}
    logger.info("")
    logger.info("=== EXCESS OVER THE ONION FLOOR (two-sided) ===")
    logger.info("%-12s%9s%s", "subset", "null_se",
                "".join(f"{s:>18}" for s in sorted(scores[TARGET_ROLES[0]])))
    for subset in AUDIT_SUBSETS:
        excess[subset] = {}
        cells = ""
        for scorer in sorted(scores[TARGET_ROLES[0]]):
            defended = out[subset]["ola"][scorer]["two_sided_auc"]
            floor = out[subset]["onion"][scorer]["two_sided_auc"]
            excess[subset][scorer] = {
                "ola": defended,
                "onion": floor,
                "excess": defended - floor,
                "excess_null_se": float((defended - floor) / subset_se[subset]),
                "null_se": float(subset_se[subset]),
            }
            cells += (
                f"{defended - floor:>+11.4f}"
                f"/{(defended - floor) / subset_se[subset]:>+5.1f}"
            )
        logger.info("%-12s%9.5f%s", subset, subset_se[subset], cells)
    logger.info(
        "The 'onion' model was retrained without the outer layer, so its column "
        "is the ground-truth floor: separation attributable to those points "
        "being atypical rather than to their membership. Excess near zero means "
        "attenuation leaks no more than deletion. The SE is the null SE of one "
        "AUC, used here only as a scale for a difference of two correlated ones."
    )

    out["excess_over_onion"] = excess
    out["null_se"] = {s: float(v) for s, v in subset_se.items()}
    return out


def poe_bootstrap(scores: dict, is_member: np.ndarray, outer_mask: np.ndarray,
                  fp_counts: list, n_boot: int, seed: int, logger) -> dict:
    """Bootstrap CIs for the Privacy Onion claim on the inner layer.

    The headline result -- deleting the outer layer raises inner-layer
    TPR@0.1%FPR from 0.0426 to 0.0646 -- is a pair of point estimates with no
    error bar. This resamples members and non-members separately (the quantity is
    conditional on both group sizes) using ``analyze_tpr``'s existing helpers.

    Args:
        scores (dict): ``{role: {scorer: dataset-ordered scores}}``.
        is_member (np.ndarray): 0/1 membership over the dataset.
        outer_mask (np.ndarray): Boolean, True for outer-layer points.
        fp_counts (list): False-positive counts to report at.
        n_boot (int): Bootstrap resamples.
        seed (int): RNG seed.
        logger (logging.Logger): Logger object.

    Returns:
        dict: ``{role: {scorer: {"fp<k>": {...}}}}``.
    """
    inner = (is_member == 1) & ~outer_mask
    non_members = is_member == 0
    rng = np.random.default_rng(seed)

    logger.info("")
    logger.info("=== PRIVACY ONION EFFECT: inner-layer TPR at fixed FP counts ===")
    logger.info(
        "%d inner members vs %d non-members, %d bootstrap resamples.",
        int(inner.sum()),
        int(non_members.sum()),
        n_boot,
    )
    header = f"{'role':<10}{'scorer':<14}"
    for k in fp_counts:
        header += f"{'TPR@' + str(k) + 'FP':>24}"
    logger.info(header)

    out = {}
    for role in TARGET_ROLES:
        out[role] = {}
        for scorer in sorted(scores[role]):
            vec = scores[role][scorer]
            pos, neg = vec[inner], vec[non_members]
            out[role][scorer] = {}
            line = f"{role:<10}{scorer:<14}"
            for k in fp_counts:
                point = tpr_at_fp(pos, neg, k)
                lo, hi = bootstrap_ci(pos, neg, k, n_boot, rng)
                out[role][scorer][f"fp{k}"] = {
                    "tpr": point, "ci_lo": lo, "ci_hi": hi
                }
                line += f"{f'{point:.4f} [{lo:.4f},{hi:.4f}]':>24}"
            logger.info(line)
    logger.info(
        "The onion effect is real for a scorer only if 'onion' sits above "
        "'vanilla' with non-overlapping intervals. Overlap means this run "
        "cannot resolve the promotion at that operating point."
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Re-analyse a finished OLA run.")
    parser.add_argument("--cf", type=str, default="configs/ola/cifar10.yaml")
    parser.add_argument(
        "--fp",
        type=int,
        nargs="+",
        default=DEFAULT_FP_COUNTS,
        help="False-positive counts for the bootstrap (default: 0 25 250)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Bootstrap resamples. 500 keeps the whole script under a minute; "
        "raise it before quoting an interval in a write-up.",
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    log_dir = configs["run"]["log_dir"]
    report_dir = f"{log_dir}/report"
    report_exp = f"{report_dir}/exp"
    models_dir = f"{log_dir}/models"
    for path in (report_exp, models_dir):
        if not os.path.isdir(path):
            raise SystemExit(f"{path} not found -- run run_ola.py first.")

    logger = setup_log(report_dir, "ola_analysis", configs["run"]["time_log"])
    start_time = time.time()

    with np.load(f"{models_dir}/{OUTER_LAYER_FILE}") as data:
        outer_indices = data["outer_indices"]
        asr = data["asr"]
    with np.load(f"{models_dir}/{OLA_SPLITS_FILE}") as data:
        target_train = data["target_train"]

    scores, is_member = load_dataset_ordered_scores(report_exp, logger)

    # The archives and the persisted splits are written by different stages; if
    # they disagree, every contrast below is computed against the wrong labels.
    if not np.array_equal(np.flatnonzero(is_member == 1), np.sort(target_train)):
        raise SystemExit(
            "The saved membership vector does not match ola_splits.npz. The "
            "report and the splits come from different runs."
        )
    outer_mask = np.zeros(is_member.size, dtype=bool)
    outer_mask[outer_indices] = True
    if not np.all(is_member[outer_indices] == 1):
        raise SystemExit("Outer-layer indices include non-members.")

    analysis = {
        "log_dir": log_dir,
        "n_outer": int(len(outer_indices)),
        "n_members": int(is_member.sum()),
        "n_nonmembers": int((is_member == 0).sum()),
    }
    analysis["asr_ties"] = asr_tie_census(asr, target_train, outer_indices, logger)
    analysis["subsets"] = recompute_subsets(
        scores, is_member, outer_mask, analysis["n_nonmembers"], logger
    )
    analysis["outer_vs_inner"] = outer_vs_inner(
        scores, outer_mask, is_member == 1, logger
    )
    if not args.skip_bootstrap:
        analysis["onion_effect"] = poe_bootstrap(
            scores, is_member, outer_mask, args.fp, args.n_boot, args.seed, logger
        )

    with open(f"{report_dir}/ola_analysis.json", "w") as f:
        json.dump(analysis, f, indent=4)
    logger.info("")
    logger.info(
        "Wrote %s/ola_analysis.json in %0.1f s.", report_dir, time.time() - start_time
    )


if __name__ == "__main__":
    main()

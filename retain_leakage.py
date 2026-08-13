"""Retain-set leakage: does unlearning F make the *kept* points more exposed?

Hayes et al., *Inexact Unlearning Needs More Careful Evaluations* (Google
DeepMind, 2024), make a second privacy claim beyond forget-vs-unseen: after
unlearning the forget set F, points in the retain set R -- data no one asked to
delete -- can become **more** vulnerable to a standard membership inference
attack. On their two SOTA unlearners, over half of retain points had *increased*
leakage after unlearning. This is a population-level harm the forget-vs-unseen
audit does not see.

The measurement is nearly free in this pipeline. The reference models a level-2
or level-3 run already trained are, *for retain points*, a valid **standard**-MIA
shadow set: each reference trains on a random half of R (``urmia_utils`` at
level 2, ``urmia_online_utils`` at level 3) and never unlearns retain points, so
a given retain point is trained-on in ~half the references and never-trained in
the other ~half. That per-reference membership is already persisted
(``ref_memberships`` at level 2, ``ref_retain_membership`` at level 3). The only
thing missing is a signal pass over a sample of retain points; the run scripts
add that under ``signal_tag="_retain"`` and hand the arrays here.

Unlike forget/unseen points -- which are OUT of every reference, so only the
one-sided offline scorer is defined -- retain points have a genuine IN/OUT
reference structure, so **every** scorer applies: ``run_ulira`` (LiRA, the
paper's method), ``run_urmia_online``, ``run_rmia_offline_masked`` and the
reference-free ``run_global_threshold``.

Two products per scorer:

**Paired leakage increase (the paper's headline).** For each retain point r,
score it under the target ``original`` and the target ``unlearned`` using r's own
trained/never reference distributions, and take ``delta = unlearned - original``.
``frac_increased = mean(delta > 0)`` is the fraction of retain points that got
*more* exposed -- the ">half" number. ``retrained`` is a control: unlearning
never happened to it, so its delta versus ``original`` should sit near zero.

**Aggregate retain-vs-unseen AUC.** Treating the retain sample as members and the
unseen set U as non-members gives a standard-MIA AUC/TPR per target, so the rise
in TPR@1%FPR after unlearning is visible next to the forget table. Only defined
for scorers that behave with an all-OUT non-member group (``urmia_offline``,
``urmia_online``, ``conf_global``); U has no IN references, so ``ulira``'s
per-example Gaussian is undefined there and ``ulira`` is reported paired-only,
which is the method-faithful reading anyway.

This module is pure numpy/scipy and reuses the scorers in
``modules/mia/attacks/urmia.py`` unchanged. Run ``python retain_leakage.py`` for
the self-test.
"""

import numpy as np

from audit import compute_attack_results
from modules.mia.attacks.urmia import (
    run_rmia_offline_masked,
    run_urmia_online,
    run_ulira,
    run_global_threshold,
)

# Signals are ordered [original, unlearned, retrained, ref_0 .. ref_{K-1}], the
# same layout ``get_urmia_signals`` produces.
TARGET_ROLES = ["original", "unlearned", "retrained"]
REF_COL_START = len(TARGET_ROLES)

# All scorers apply to retain points. Only these three give a clean retain-vs-U
# AUC (U is OUT of every reference); ulira is paired-only -- see module docstring.
SCORERS = ["urmia_offline", "urmia_online", "ulira", "conf_global"]
AUC_SCORERS = ["urmia_offline", "urmia_online", "conf_global"]

# Offset the retain-audit sampler from the split RNG so the retain sample is not
# correlated with any reference's retain half.
_RETAIN_SALT = 0x5EED


def select_retain_audit(retain_indices: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Deterministically sample retain-audit points from R.

    No change is made to the persisted splits: the sample is derived from the run
    seed on demand, so existing ``*_splits.npz`` files and their strict validation
    are untouched.

    Args:
        retain_indices (np.ndarray): All retain indices from the run's splits.
        size (int): Number of retain points to audit. Clamped to ``|R|``.
        seed (int): The run's ``random_seed``.

    Returns:
        np.ndarray: Sampled retain indices (a subset of ``retain_indices``).
    """
    retain_indices = np.asarray(retain_indices)
    rng = np.random.default_rng(int(seed) + _RETAIN_SALT)
    if size >= len(retain_indices):
        return retain_indices.copy()
    return rng.choice(retain_indices, size=int(size), replace=False)


def retain_ref_in(splits: dict, retain_audit_idx: np.ndarray) -> np.ndarray:
    """Per-reference trained/never mask for the retain-audit points.

    Auto-detects the level: level 3 persists ``ref_retain_membership``, level 2
    persists ``ref_memberships`` (both ``(K, dataset_size)`` bool, True where the
    reference trained on that point). Returned transposed to the
    ``(num_samples, K)`` orientation the scorers expect, where True means the
    point was **trained on** (a member) in that reference.

    Args:
        splits (dict): The run's splits dict.
        retain_audit_idx (np.ndarray): Indices returned by
            :func:`select_retain_audit`.

    Returns:
        np.ndarray: Bool mask, shape ``(len(retain_audit_idx), K)``.

    Raises:
        KeyError: If neither membership key is present.
    """
    if "ref_retain_membership" in splits:
        ref = splits["ref_retain_membership"]  # level 3
    elif "ref_memberships" in splits:
        ref = splits["ref_memberships"]  # level 2
    else:
        raise KeyError(
            "splits carries neither 'ref_retain_membership' (level 3) nor "
            "'ref_memberships' (level 2); cannot build retain reference masks."
        )
    return np.asarray(ref)[:, retain_audit_idx].T.astype(bool)


def _score(
    name: str,
    target: np.ndarray,
    ref_signals: np.ndarray,
    ref_in: np.ndarray,
    z_target: np.ndarray,
    z_ref: np.ndarray,
    offline_a: float,
) -> np.ndarray:
    """Dispatch to one of the reused scorers (higher = more member-like)."""
    if name == "urmia_offline":
        return run_rmia_offline_masked(
            target, ref_signals, ref_in, z_target, z_ref, offline_a
        )
    if name == "urmia_online":
        return run_urmia_online(
            target, ref_signals, ref_in, z_target, z_ref, offline_a
        )
    if name == "ulira":
        return run_ulira(target, ref_signals, ref_in)
    if name == "conf_global":
        return run_global_threshold(target, higher_is_member=True)
    raise ValueError(f"Unknown scorer: {name!r}")


def _paired(
    name: str,
    retain_signals: np.ndarray,
    ref_in: np.ndarray,
    z_signals: np.ndarray,
    offline_a: float,
) -> dict:
    """Per-example leakage increase for one scorer over the retain sample."""
    ref = retain_signals[:, REF_COL_START:]
    z_ref = z_signals[:, REF_COL_START:]

    def role_score(col):
        return _score(
            name,
            retain_signals[:, col],
            ref,
            ref_in,
            z_signals[:, col],
            z_ref,
            offline_a,
        )

    s_orig = role_score(0)
    s_unl = role_score(1)
    s_ret = role_score(2)
    delta = s_unl - s_orig
    delta_ctrl = s_ret - s_orig
    return {
        "frac_increased": float((delta > 0).mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_retrained_vs_original_mean": float(delta_ctrl.mean()),
    }


def _aggregate(
    name: str,
    retain_signals: np.ndarray,
    ref_in: np.ndarray,
    unseen_block: np.ndarray,
    z_signals: np.ndarray,
    offline_a: float,
) -> dict:
    """Retain-vs-unseen standard-MIA AUC/TPR per target for one scorer.

    Members are the retain sample (trained in every target); non-members are the
    unseen set U (never trained, OUT of every reference). Scores each target's
    stacked retain+U rows in a single call, so the AUC is over one consistent
    ranking.
    """
    n_ret = retain_signals.shape[0]
    n_uns = unseen_block.shape[0]
    memberships = np.concatenate([np.ones(n_ret, bool), np.zeros(n_uns, bool)])

    ref = np.concatenate(
        [retain_signals[:, REF_COL_START:], unseen_block[:, REF_COL_START:]], axis=0
    )
    # U is OUT of every reference; retain keeps its trained/never mask.
    rin = np.concatenate(
        [ref_in, np.zeros((n_uns, ref_in.shape[1]), bool)], axis=0
    )
    z_ref = z_signals[:, REF_COL_START:]

    out = {}
    for role, col in zip(TARGET_ROLES, range(len(TARGET_ROLES))):
        target = np.concatenate([retain_signals[:, col], unseen_block[:, col]])
        z_target = z_signals[:, col]
        scores = _score(name, target, ref, rin, z_target, z_ref, offline_a)
        res = compute_attack_results(scores, memberships)
        out[role] = {
            "auc": float(res["auc"]),
            "one_fpr": float(res["one_fpr"]),
            "one_tenth_fpr": float(res["one_tenth_fpr"]),
            "zero_fpr": float(res["zero_fpr"]),
        }
    return out


def compute_retain_leakage(
    retain_signals: np.ndarray,
    ref_in: np.ndarray,
    unseen_block: np.ndarray,
    z_signals: np.ndarray,
    offline_a: float = 0.3,
    scorers: list = None,
) -> dict:
    """Full retain-leakage analysis over all scorers.

    Args:
        retain_signals (np.ndarray): ``(RA, 3 + K)`` true-class softmax signals for
            the retain-audit points, column layout ``[original, unlearned,
            retrained, ref_0..]``.
        ref_in (np.ndarray): ``(RA, K)`` bool, True where the retain point was
            trained on (a member) in that reference. From :func:`retain_ref_in`.
        unseen_block (np.ndarray): ``(U, 3 + K)`` signals for the unseen set U,
            sliced from the run's audit-signal cache. Non-members for the
            aggregate AUC.
        z_signals (np.ndarray): ``(P, 3 + K)`` population signals.
        offline_a (float): RMIA offline coefficient.
        scorers (list): Subset of :data:`SCORERS`; defaults to all.

    Returns:
        dict: ``retain_audit_size``, ``unseen_size``, and per-scorer ``paired``
            (all scorers) and ``aggregate`` (AUC scorers only).
    """
    if scorers is None:
        scorers = SCORERS

    result = {
        "retain_audit_size": int(retain_signals.shape[0]),
        "unseen_size": int(unseen_block.shape[0]),
        "scorers": {},
    }
    for name in scorers:
        entry = {"paired": _paired(name, retain_signals, ref_in, z_signals, offline_a)}
        if name in AUC_SCORERS:
            entry["aggregate"] = _aggregate(
                name, retain_signals, ref_in, unseen_block, z_signals, offline_a
            )
        result["scorers"][name] = entry
    return result


def format_retain_rows(analysis: dict) -> list:
    """Render the analysis as log lines, matching the U-RMIA summary style.

    Args:
        analysis (dict): Output of :func:`compute_retain_leakage`.

    Returns:
        list[str]: Lines to log.
    """
    lines = [
        "Retain-set leakage (members = %d-point R sample, U = %d non-members)"
        % (analysis["retain_audit_size"], analysis["unseen_size"]),
        "  delta = score(unlearned) - score(original); frac_inc = fraction of "
        "retain points whose leakage rose.",
        f"{'scorer':<15} {'frac_inc':>8} {'d_mean':>8} {'d_med':>8} "
        f"{'ctrl_d':>8} {'auc_orig':>9} {'auc_unl':>8} {'tpr1_orig':>10} {'tpr1_unl':>9}",
    ]
    for name, entry in analysis["scorers"].items():
        p = entry["paired"]
        agg = entry.get("aggregate")
        if agg:
            auc_orig = f"{agg['original']['auc']:.4f}"
            auc_unl = f"{agg['unlearned']['auc']:.4f}"
            tpr1_orig = f"{agg['original']['one_fpr']:.4f}"
            tpr1_unl = f"{agg['unlearned']['one_fpr']:.4f}"
        else:
            auc_orig = auc_unl = tpr1_orig = tpr1_unl = "n/a"
        lines.append(
            f"{name:<15} {p['frac_increased']:>8.3f} {p['delta_mean']:>8.4f} "
            f"{p['delta_median']:>8.4f} {p['delta_retrained_vs_original_mean']:>8.4f} "
            f"{auc_orig:>9} {auc_unl:>8} {tpr1_orig:>10} {tpr1_unl:>9}"
        )
    lines.append(
        "  Reading it: frac_inc > 0.5 means unlearning exposed a majority of "
        "kept points more than before (Hayes et al. 2024)."
    )
    lines.append(
        "  ctrl_d (retrained - original) should sit near 0: unlearning never "
        "touched retrained, so a large value flags a confound from F's removal."
    )
    return lines


def self_test(seed: int = 0) -> None:
    """Synthesise a retain sample with a planted leakage increase and recover it.

    Retain points are members (high confidence, matching their IN references);
    the ``unlearned`` target is made more member-like than ``original`` (delta > 0)
    while ``retrained`` matches ``original`` (control near 0). U points are OUT of
    every reference and low-confidence (non-members). Runs without any cluster
    output.

    Args:
        seed (int): RNG seed.

    Raises:
        AssertionError: If the planted increase is not recovered.
    """
    rng = np.random.default_rng(seed)
    n_ret, n_uns, num_ref, n_pop = 500, 500, 16, 2000
    sigma = 0.6

    def sig(logit_mean, shape, difficulty=0.0):
        z = logit_mean + difficulty + rng.normal(0.0, sigma, size=shape)
        return 1.0 / (1.0 + np.exp(-z))  # sigmoid -> (0, 1) confidence

    # Per-point difficulty shared across all models, the thing calibration removes.
    diff = rng.normal(0.0, 1.0, size=n_ret)

    # Reference IN/OUT assignment for retain points: each IN in ~half.
    ranks = rng.random((num_ref, n_ret)).argsort(0).argsort(0)
    ref_in = (ranks < num_ref // 2).T  # (n_ret, K)

    ref_ret = np.where(
        ref_in,
        sig(2.0, (n_ret, num_ref), diff[:, None]),   # trained -> confident
        sig(-2.0, (n_ret, num_ref), diff[:, None]),  # never -> not
    )
    # original at +1 (member but below the IN peak), unlearned at +2 (on the IN
    # peak -> more member-like: leakage rose), retrained back at +1 (control).
    ret_orig = sig(1.0, n_ret, diff)
    ret_unl = sig(2.0, n_ret, diff)
    ret_ret = sig(1.0, n_ret, diff)
    retain_signals = np.column_stack([ret_orig, ret_unl, ret_ret, ref_ret])

    # Unseen U: OUT of every reference, low confidence in every target.
    diff_u = rng.normal(0.0, 1.0, size=n_uns)
    uns_ref = sig(-2.0, (n_uns, num_ref), diff_u[:, None])
    uns_tgt = sig(-2.0, n_uns, diff_u)
    unseen_block = np.column_stack([uns_tgt, uns_tgt, uns_tgt, uns_ref])

    # Population z: never trained, low confidence.
    diff_z = rng.normal(0.0, 1.0, size=n_pop)
    z_ref = sig(-2.0, (n_pop, num_ref), diff_z[:, None])
    z_tgt = sig(-2.0, n_pop, diff_z)
    z_signals = np.column_stack([z_tgt, z_tgt, z_tgt, z_ref])

    analysis = compute_retain_leakage(
        retain_signals, ref_in, unseen_block, z_signals
    )
    print("\n".join(format_retain_rows(analysis)))

    for name, entry in analysis["scorers"].items():
        p = entry["paired"]
        assert p["frac_increased"] > 0.5, f"{name}: frac_increased {p['frac_increased']}"
        assert p["delta_mean"] > 0, f"{name}: delta_mean {p['delta_mean']}"
        assert abs(p["delta_retrained_vs_original_mean"]) < 0.1, (
            f"{name}: control delta {p['delta_retrained_vs_original_mean']} not ~0"
        )
        if "aggregate" in entry:
            # A loose above-chance floor: separable synthetic members/non-members
            # must be caught. Not tighter, because urmia_online's empirical p(x)
            # is mildly degraded when U has no IN references (see AUC_SCORERS note).
            assert entry["aggregate"]["original"]["auc"] > 0.6, (
                f"{name}: retain-vs-U AUC {entry['aggregate']['original']['auc']} "
                "below the above-chance floor"
            )
    print("\nself-test passed: planted leakage increase recovered on all scorers.\n")


if __name__ == "__main__":
    self_test()

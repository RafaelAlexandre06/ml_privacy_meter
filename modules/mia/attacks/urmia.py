"""Strong, unlearning-aware attack scorers for level-3 U-RMIA.

These scorers operate on cached true-class softmax signals over a *single*
train-then-unlearn reference set. For each audited sample x and reference k,
``ref_in_membership[x, k]`` is True iff x was **trained on then unlearned** in
reference k (IN-unlearned), and False iff x was **never trained** in reference k
(OUT). Population (z) samples come from a held-out set disjoint from the training
pool, so they are OUT of every reference.

Three scorers, all pure numpy/scipy on the shared signals:

- ``run_rmia_offline_masked`` — the *naive* arm. Offline RMIA (the level-2 math)
  but selecting each sample's OUT reference columns per-row via ``~ref_in``. One-
  sided ("member = higher confidence"), so it is fooled by over-unlearning. Kept
  for the naive-vs-strong contrast.
- ``run_urmia_online`` — RMIA-family strong arm. Estimates p(x) *empirically* from
  the IN and OUT reference means (instead of the ``offline_a`` guess), keeping
  RMIA's population-ratio structure. Two-sided via the calibration.
- ``run_ulira`` — LiRA-style strong arm (the paper's actual method). Logit-scales
  the signals, fits per-sample IN/OUT Gaussians, and scores with the log-
  likelihood ratio. No population z. Naturally two-sided; needs enough reference
  models per sample for stable Gaussians (small-K fallback included).

This module does not modify ``modules/mia/attacks/rmia.py``.
"""

from typing import Optional

import numpy as np
from scipy.stats import norm

_EPS = 1e-6


def _row_masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of ``values`` over the True entries of ``mask``, per row.

    Rows with no True entry fall back to the plain row mean so the result is
    always finite.

    Args:
        values (np.ndarray): Shape (num_samples, num_models).
        mask (np.ndarray): Bool, same shape; True entries are averaged.

    Returns:
        np.ndarray: Shape (num_samples,).
    """
    counts = mask.sum(axis=1)
    summed = np.where(mask, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = summed / counts
    # Fallback for degenerate rows (no masked entry).
    degenerate = counts == 0
    if np.any(degenerate):
        means[degenerate] = values[degenerate].mean(axis=1)
    return means


def run_rmia_offline_masked(
    target_signals: np.ndarray,
    ref_signals: np.ndarray,
    ref_in_membership: np.ndarray,
    z_target_signals: np.ndarray,
    z_ref_signals: np.ndarray,
    offline_a: float,
) -> np.ndarray:
    """Naive offline RMIA using each sample's OUT reference columns only.

    Same math as the level-2 offline attack, but because this run's references
    are train-then-unlearn (each audited sample is IN for ~half of them), the OUT
    baseline for a sample is taken over the references where it was OUT
    (``~ref_in_membership``). This reproduces the one-sided attack that gets
    fooled by over-unlearning, for the naive-vs-strong contrast.

    Args:
        target_signals (np.ndarray): Target confidences, shape (num_samples,).
        ref_signals (np.ndarray): Reference confidences, shape (num_samples, K).
        ref_in_membership (np.ndarray): Bool IN mask, shape (num_samples, K).
        z_target_signals (np.ndarray): Target confidences on population, (num_pop,).
        z_ref_signals (np.ndarray): Reference confidences on population, (num_pop, K).
        offline_a (float): Offline coefficient for the p() approximation.

    Returns:
        np.ndarray: MIA score per sample (higher = more likely member).
    """
    out_mask = ~ref_in_membership
    mean_out_x = _row_masked_mean(ref_signals, out_mask)
    mean_x = (1 + offline_a) / 2 * mean_out_x + (1 - offline_a) / 2
    prob_ratio_x = target_signals.ravel() / mean_x

    # Population points are OUT of all references.
    mean_out_z = z_ref_signals.mean(axis=1)
    mean_z = (1 + offline_a) / 2 * mean_out_z + (1 - offline_a) / 2
    prob_ratio_z = z_target_signals.ravel() / mean_z

    ratios = prob_ratio_x[:, np.newaxis] / prob_ratio_z
    return np.average(ratios > 1.0, axis=1)


def run_urmia_online(
    target_signals: np.ndarray,
    ref_signals: np.ndarray,
    ref_in_membership: np.ndarray,
    z_target_signals: np.ndarray,
    z_ref_signals: np.ndarray,
    offline_a: float,
) -> np.ndarray:
    """RMIA-family strong scorer with an empirically calibrated p(x).

    Uses the train-then-unlearn (IN) and never-trained (OUT) reference means to
    estimate p(x) = (mean_in_x + mean_out_x) / 2 directly, instead of the offline
    linear approximation. Population points have no IN observations, so p(z) keeps
    the offline form. The RMIA population-ratio structure is unchanged.

    Args:
        target_signals (np.ndarray): Target confidences, shape (num_samples,).
        ref_signals (np.ndarray): Reference confidences, shape (num_samples, K).
        ref_in_membership (np.ndarray): Bool IN mask, shape (num_samples, K).
        z_target_signals (np.ndarray): Target confidences on population, (num_pop,).
        z_ref_signals (np.ndarray): Reference confidences on population, (num_pop, K).
        offline_a (float): Offline coefficient for the population p(z) term.

    Returns:
        np.ndarray: MIA score per sample (higher = more likely member).
    """
    in_mask = ref_in_membership
    out_mask = ~ref_in_membership
    mean_in_x = _row_masked_mean(ref_signals, in_mask)
    mean_out_x = _row_masked_mean(ref_signals, out_mask)
    # Empirical member/non-member average confidence for x.
    mean_x = 0.5 * (mean_in_x + mean_out_x)
    prob_ratio_x = target_signals.ravel() / np.clip(mean_x, _EPS, None)

    mean_out_z = z_ref_signals.mean(axis=1)
    mean_z = (1 + offline_a) / 2 * mean_out_z + (1 - offline_a) / 2
    prob_ratio_z = z_target_signals.ravel() / mean_z

    ratios = prob_ratio_x[:, np.newaxis] / prob_ratio_z
    return np.average(ratios > 1.0, axis=1)


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit-scale confidences, clipped to keep values finite."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def run_ulira(
    target_signals: np.ndarray,
    ref_signals: np.ndarray,
    ref_in_membership: np.ndarray,
    min_std: float = 1e-3,
) -> np.ndarray:
    """LiRA-style two-sided scorer (the paper's U-LiRA, adapted).

    Logit-scales all confidences, then for each sample fits a Gaussian to the
    IN-unlearned reference values and another to the OUT reference values, and
    scores with the log-likelihood ratio of the target under IN vs OUT. Naturally
    two-sided: whether the target's value is abnormally high or abnormally low, it
    is scored as a member if it matches the IN distribution better than OUT.

    No population z is used (LiRA does not use it). When a sample has fewer than
    two IN or two OUT references (unstable variance at small K), that group's std
    falls back to the pooled std across all samples.

    Args:
        target_signals (np.ndarray): Target confidences, shape (num_samples,).
        ref_signals (np.ndarray): Reference confidences, shape (num_samples, K).
        ref_in_membership (np.ndarray): Bool IN mask, shape (num_samples, K).
        min_std (float): Lower bound on any fitted standard deviation.

    Returns:
        np.ndarray: Log-likelihood-ratio score per sample (higher = more likely
            IN-unlearned, i.e. member).
    """
    phi_target = _logit(target_signals.ravel())
    phi_ref = _logit(ref_signals)

    in_mask = ref_in_membership
    out_mask = ~ref_in_membership

    in_counts = in_mask.sum(axis=1)
    out_counts = out_mask.sum(axis=1)

    mean_in = _row_masked_mean(phi_ref, in_mask)
    mean_out = _row_masked_mean(phi_ref, out_mask)

    def _masked_std(values, mask, counts, means):
        # Population (biased) std over masked entries; unstable rows handled below.
        sq = np.where(mask, (values - means[:, None]) ** 2, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = sq / counts
        std = np.sqrt(np.where(counts > 0, var, 0.0))
        return std

    std_in = _masked_std(phi_ref, in_mask, in_counts, mean_in)
    std_out = _masked_std(phi_ref, out_mask, out_counts, mean_out)

    # Small-K fallback: rows with <2 observations get the pooled std of that group.
    pooled_in = np.median(std_in[in_counts >= 2]) if np.any(in_counts >= 2) else min_std
    pooled_out = np.median(std_out[out_counts >= 2]) if np.any(out_counts >= 2) else min_std
    std_in = np.where(in_counts >= 2, std_in, pooled_in)
    std_out = np.where(out_counts >= 2, std_out, pooled_out)

    std_in = np.clip(std_in, min_std, None)
    std_out = np.clip(std_out, min_std, None)

    log_in = norm.logpdf(phi_target, loc=mean_in, scale=std_in)
    log_out = norm.logpdf(phi_target, loc=mean_out, scale=std_out)
    return log_in - log_out

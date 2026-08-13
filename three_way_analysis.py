"""Three-state hypothesis test: was this point trained, unlearned, or never seen?

The AUC columns in a level-3 summary answer a *two*-state question -- forget vs
unseen -- and collapse it further into a single ranking statistic. That hides the
distinction this script exists to expose:

    H_trained    x was trained on and never unlearned   (leak: nothing happened)
    H_unlearned  x was trained then unlearned           (leak: deletion advertised)
    H_never      x was never trained                    (the privacy goal)

An unlearner can fail in two opposite directions. It can leave a point looking
*trained* (under-unlearned, residual memorization), or it can push a point past
the never-trained baseline into a region no untrained model would occupy
(over-unlearned). The second is still a leak: the model is not hiding x, it is
advertising that x was deleted, which reveals membership in the original
training set. The RMIA-family scorers are monotone in confidence, so they score
an advertised point *low* and file it as a non-member -- the two failure modes
partially cancel and the reported AUC drifts toward the floor. See
``documentation/urmia_findings.md`` section 8.

This script reads a finished **level-3** run and never touches models or
training. Everything comes from the cached signal matrix and the persisted
IN/OUT reference assignment:

    <log_dir>/signals/urmia_signals.npy       (2*forget_size, 3 + K)
    <log_dir>/models/urmia_online_splits.npz  ref_unlearn_membership (K, 2*forget_size)

Two products, in increasing order of assumption:

**Part A -- direction, from references only.** For each audit point, the K//2
train-then-unlearn references give an IN mean and the other K//2 give an OUT
mean. Their difference on the logit scale is a direct measurement of where this
unlearner leaves a point relative to never having trained on it: positive means
residual memorization, negative means overshoot. No target model is involved, so
this characterises the *unlearning algorithm* rather than one model. These
quantities are already computed inside ``run_urmia_online``
(``modules/mia/attacks/urmia.py``) and discarded; here they are the output.

**Part B -- the three-way posterior.** Per point, fit a Gaussian to the IN
references (H_unlearned) and another to the OUT references (H_never), then score
each target model's observed signal under each hypothesis and report the argmax.
H_trained needs a third reference distribution, which the level-3 run does not
currently save -- ``prepare_online_reference_models`` pickles only the
post-unlearn state dict -- so it is estimated instead (see ``--trained-source``).

**On what this is.** A diagnostic, not an attack. The ``shift`` estimator for
H_trained uses the ground-truth F/U labels, which an attacker does not have. It
answers "what state does this model's behaviour on the forget set resemble?",
which is the question an auditor asks, not the question an adversary answers.
The ``base`` source removes that dependence and is the honest upgrade path.

Usage::

    python three_way_analysis.py sweep_l3_cifar10_neggrad_plus
    python three_way_analysis.py <log_dir> --json out.json
    python three_way_analysis.py --self-test
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import norm

# Signals are ordered [original, unlearned, retrained, ref_0 .. ref_{K-1}].
TARGET_ROLES = ["original", "unlearned", "retrained"]
REF_COL_START = len(TARGET_ROLES)

HYPOTHESES = ["trained", "unlearned", "never"]

_EPS = 1e-6
# Below this many references per side the per-point Gaussians are too unstable to
# fit; the pooled std is substituted (same policy as run_ulira).
_MIN_PER_SIDE = 2


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit-scale confidences, clipped to keep values finite.

    Matches ``modules/mia/attacks/urmia.py`` so the numbers here sit on the same
    scale as the U-LiRA scores in the run's report.

    Args:
        p (np.ndarray): Probabilities in [0, 1].

    Returns:
        np.ndarray: Logit-transformed values.
    """
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _masked_moments(values: np.ndarray, mask: np.ndarray, min_std: float = 1e-3):
    """Per-row mean and std over the True entries of ``mask``.

    Rows with fewer than ``_MIN_PER_SIDE`` observations get the median std of the
    rows that do have enough, which is the small-K fallback ``run_ulira`` uses.

    Args:
        values (np.ndarray): Shape (num_samples, num_models).
        mask (np.ndarray): Bool, same shape.
        min_std (float): Lower bound on the returned std.

    Returns:
        tuple[np.ndarray, np.ndarray]: Means and stds, each (num_samples,).
    """
    counts = mask.sum(axis=1)
    summed = np.where(mask, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = summed / counts
    degenerate = counts == 0
    if np.any(degenerate):
        means[degenerate] = values[degenerate].mean(axis=1)

    sq = np.where(mask, (values - means[:, None]) ** 2, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = sq / counts
    stds = np.sqrt(np.where(counts > 0, var, 0.0))

    enough = counts >= _MIN_PER_SIDE
    pooled = np.median(stds[enough]) if np.any(enough) else min_std
    stds = np.where(enough, stds, pooled)
    return means, np.clip(stds, min_std, None)


def load_run(log_dir: str):
    """Load the cached signals and the level-3 IN/OUT reference assignment.

    Args:
        log_dir (str): A finished level-3 run directory.

    Returns:
        dict: ``phi`` (2F, 3+K) logit signals, ``ref_in`` (2F, K) bool IN mask,
            ``forget_size`` int, ``num_ref_models`` int.

    Raises:
        FileNotFoundError: If the signal or split file is missing.
        ValueError: If the run is level 2 (no ``ref_unlearn_membership``), or the
            signal and split shapes disagree.
    """
    signal_path = os.path.join(log_dir, "signals", "urmia_signals.npy")
    splits_path = os.path.join(log_dir, "models", "urmia_online_splits.npz")

    if not os.path.exists(signal_path):
        raise FileNotFoundError(f"No cached signals at {signal_path}")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"No level-3 splits at {splits_path}. A level-2 run (urmia_splits.npz) "
            "has no train-then-unlearn references, so the unlearned hypothesis "
            "cannot be estimated and this analysis does not apply."
        )

    signals = np.load(signal_path)
    splits = np.load(splits_path)
    if "ref_unlearn_membership" not in splits:
        raise ValueError(
            f"{splits_path} has no 'ref_unlearn_membership'. This is a level-2 "
            "split file; three-way analysis needs level-3 references."
        )

    # (K, 2F) -> (2F, K), the orientation every scorer works in.
    ref_in = np.asarray(splits["ref_unlearn_membership"]).T.astype(bool)
    num_ref_models = ref_in.shape[1]
    forget_size = len(splits["forget_indices"])

    if signals.shape != (ref_in.shape[0], REF_COL_START + num_ref_models):
        raise ValueError(
            f"Signal matrix {signals.shape} disagrees with the splits: expected "
            f"({ref_in.shape[0]}, {REF_COL_START + num_ref_models}). The signal "
            "cache is stale -- delete it and re-run the audit stage."
        )

    return {
        "phi": _logit(signals),
        "ref_in": ref_in,
        "forget_size": forget_size,
        "num_ref_models": num_ref_models,
    }


def reference_states(phi: np.ndarray, ref_in: np.ndarray) -> dict:
    """Per-point unlearned and never-trained reference distributions (Part A).

    For audit point x, ``ref_in[x, k]`` is True iff x was trained-then-unlearned
    in reference k. The IN columns therefore sample the *unlearned* state for x
    and the OUT columns sample the *never-trained* state, both on the same
    architecture and recipe as the target.

    ``delta = mean_in - mean_out`` is the quantity of interest: how far this
    unlearner moves a point away from never having been trained, and in which
    direction. It is measured per point, from ``K//2`` models per side, with no
    target model involved.

    Args:
        phi (np.ndarray): Logit signals, shape (num_audit, 3 + K).
        ref_in (np.ndarray): Bool IN mask, shape (num_audit, K).

    Returns:
        dict: ``mean_in``, ``std_in``, ``mean_out``, ``std_out``, ``delta``.
    """
    ref_phi = phi[:, REF_COL_START:]
    mean_in, std_in = _masked_moments(ref_phi, ref_in)
    mean_out, std_out = _masked_moments(ref_phi, ~ref_in)
    return {
        "mean_in": mean_in,
        "std_in": std_in,
        "mean_out": mean_out,
        "std_out": std_out,
        "delta": mean_in - mean_out,
    }


def estimate_trained_state(
    phi: np.ndarray,
    states: dict,
    forget_size: int,
    source: str = "shift",
    base_signals: np.ndarray = None,
) -> dict:
    """Estimate the trained-but-not-unlearned distribution per point.

    Level-3 references are pickled only *after* unlearning
    (``urmia_online_utils.prepare_online_reference_models``), so the trained state
    has no reference set of its own. Two ways to supply one:

    ``base``
        Signals from pre-unlearn reference checkpoints, if a run was made that
        persisted them. Per-point mean and std from the same K//2 models that
        were later unlearned -- the clean version, no label leakage.

    ``shift`` (default)
        Estimate a single global logit shift from the ``original`` model, whose
        forget points *are* in the trained state: ``delta_hat`` is the mean of
        ``phi(original, x) - mean_out(x)`` over forget points. The per-point
        location is then ``mean_out(x) + delta_hat``.

        This uses the ground-truth F/U labels, which makes it an auditor's
        diagnostic rather than an attack. It carries its own control: the same
        statistic over the *unseen* points must come out near zero, because
        ``original`` never trained on those either. ``control_shift`` reports it;
        a value far from zero means the estimate is contaminated and the trained
        column of Part B should not be trusted.

    Args:
        phi (np.ndarray): Logit signals, shape (num_audit, 3 + K).
        states (dict): Output of :func:`reference_states`.
        forget_size (int): Number of forget points (rows 0..forget_size-1).
        source (str): ``"shift"`` or ``"base"``.
        base_signals (np.ndarray): Pre-unlearn reference signals (num_audit, K),
            already logit-scaled. Required when ``source="base"``.

    Returns:
        dict: ``mean``, ``std`` (both (num_audit,)), plus ``delta_hat`` and
            ``control_shift`` diagnostics when ``source="shift"``.

    Raises:
        ValueError: On an unknown source, or ``base`` without ``base_signals``.
    """
    if source == "base":
        if base_signals is None:
            raise ValueError("source='base' requires base_signals")
        mean, std = _masked_moments(base_signals, np.ones_like(base_signals, bool))
        return {"mean": mean, "std": std}

    if source != "shift":
        raise ValueError(f"Unknown trained-state source: {source!r}")

    residual = phi[:, 0] - states["mean_out"]  # column 0 is `original`
    delta_hat = float(residual[:forget_size].mean())
    control_shift = float(residual[forget_size:].mean())
    # Spread of the residual on forget points bounds the trained-state std from
    # above: it mixes genuine per-model variation with per-point variation the
    # single `original` model cannot separate.
    std_hat = float(residual[:forget_size].std())

    return {
        "mean": states["mean_out"] + delta_hat,
        "std": np.full(phi.shape[0], max(std_hat, 1e-3)),
        "delta_hat": delta_hat,
        "control_shift": control_shift,
    }


def three_way_posterior(
    target_phi: np.ndarray, states: dict, trained: dict
) -> np.ndarray:
    """Posterior over {trained, unlearned, never} for every audit point.

    Flat prior over the three states, so the posterior is the normalised
    likelihood. A flat prior is the right choice here because the question is
    "which state does this behaviour resemble", not "how often does each state
    occur" -- the base rates are fixed by the experiment, not by nature.

    Args:
        target_phi (np.ndarray): Target logit signals, shape (num_audit,).
        states (dict): Output of :func:`reference_states`.
        trained (dict): Output of :func:`estimate_trained_state`.

    Returns:
        np.ndarray: Shape (num_audit, 3), columns ordered as :data:`HYPOTHESES`.
    """
    log_lik = np.stack(
        [
            norm.logpdf(target_phi, loc=trained["mean"], scale=trained["std"]),
            norm.logpdf(target_phi, loc=states["mean_in"], scale=states["std_in"]),
            norm.logpdf(target_phi, loc=states["mean_out"], scale=states["std_out"]),
        ],
        axis=1,
    )
    log_lik -= log_lik.max(axis=1, keepdims=True)
    lik = np.exp(log_lik)
    return lik / lik.sum(axis=1, keepdims=True)


def advertised_mask(
    target_phi: np.ndarray, states: dict, n_sigma: float = 2.0
) -> np.ndarray:
    """Points pushed below the never-trained baseline by more than ``n_sigma``.

    The over-unlearning signature, stated as an absolute rather than a
    comparative claim: the target's confidence on x is lower than any plausible
    never-trained model's would be. Such a point is not hidden -- its abnormality
    is itself evidence that x was in the original training set.

    Args:
        target_phi (np.ndarray): Target logit signals, shape (num_audit,).
        states (dict): Output of :func:`reference_states`.
        n_sigma (float): How many never-trained stds below the mean to require.

    Returns:
        np.ndarray: Bool mask, shape (num_audit,).
    """
    return target_phi < states["mean_out"] - n_sigma * states["std_out"]


def analyse(run: dict, trained_source: str = "shift", n_sigma: float = 2.0) -> dict:
    """Run Part A and Part B over all three target models.

    Args:
        run (dict): Output of :func:`load_run`.
        trained_source (str): Passed to :func:`estimate_trained_state`.
        n_sigma (float): Passed to :func:`advertised_mask`.

    Returns:
        dict: ``direction`` (Part A) and ``targets`` (Part B, keyed by role).
    """
    phi = run["phi"]
    forget_size = run["forget_size"]
    states = reference_states(phi, run["ref_in"])
    trained = estimate_trained_state(phi, states, forget_size, trained_source)

    delta = states["delta"]
    d_forget, d_unseen = delta[:forget_size], delta[forget_size:]
    direction = {
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "frac_overshoot": float((delta < 0).mean()),
        # F and U are drawn from the same pool and are assigned IN/OUT without
        # regard to which is which, so these two must agree. A gap means the
        # reference assignment leaked the F/U label.
        "delta_mean_forget": float(d_forget.mean()),
        "delta_mean_unseen": float(d_unseen.mean()),
    }
    if trained_source == "shift":
        direction["trained_delta_hat"] = trained["delta_hat"]
        direction["trained_control_shift"] = trained["control_shift"]

    results = {}
    for col, role in enumerate(TARGET_ROLES):
        target_phi = phi[:, col]
        post = three_way_posterior(target_phi, states, trained)
        label = post.argmax(axis=1)
        adv = advertised_mask(target_phi, states, n_sigma)

        def _fractions(sl):
            return {h: float((label[sl] == i).mean()) for i, h in enumerate(HYPOTHESES)}

        f_sl, u_sl = slice(0, forget_size), slice(forget_size, None)
        results[role] = {
            "forget": _fractions(f_sl),
            "unseen": _fractions(u_sl),
            "mean_posterior_forget": {
                h: float(post[f_sl, i].mean()) for i, h in enumerate(HYPOTHESES)
            },
            "advertised_forget": float(adv[f_sl].mean()),
            "advertised_unseen": float(adv[u_sl].mean()),
        }
    return {"direction": direction, "targets": results}


def format_report(analysis: dict, trained_source: str) -> str:
    """Render the analysis as the text block printed to stdout.

    Args:
        analysis (dict): Output of :func:`analyse`.
        trained_source (str): Which trained-state estimator was used.

    Returns:
        str: Multi-line report.
    """
    d = analysis["direction"]
    lines = [
        "",
        "Part A - direction of unlearning (references only, no target model)",
        "-" * 70,
        f"  mean logit shift IN vs OUT     {d['delta_mean']:+.4f}",
        f"  median                         {d['delta_median']:+.4f}",
        f"  fraction pushed below baseline {d['frac_overshoot']:.3f}",
        "",
        "  Split by group -- these must agree; a gap means the reference IN/OUT",
        "  assignment leaked the forget/unseen label.",
        f"    forget {d['delta_mean_forget']:+.4f}   unseen {d['delta_mean_unseen']:+.4f}",
    ]
    if "trained_control_shift" in d:
        lines += [
            "",
            f"  trained-state shift estimate   {d['trained_delta_hat']:+.4f}",
            f"  its control (must be ~0)       {d['trained_control_shift']:+.4f}",
        ]

    lines += [
        "",
        f"Part B - three-way classification (trained source: {trained_source})",
        "-" * 70,
        f"{'target':<12} {'group':<8} {'trained':>9} {'unlearned':>10} {'never':>9} "
        f"{'advert.':>9}",
    ]
    for role, res in analysis["targets"].items():
        for group in ("forget", "unseen"):
            frac = res[group]
            lines.append(
                f"{role:<12} {group:<8} {frac['trained']:>9.3f} "
                f"{frac['unlearned']:>10.3f} {frac['never']:>9.3f} "
                f"{res['advertised_' + group]:>9.3f}"
            )

    lines += [
        "",
        "Reading it:",
        "  original / forget  should read 'trained'  -- if not, the whole analysis",
        "                     is miscalibrated and nothing below it means anything.",
        "  original / unseen  should read 'never'    -- same check, other direction.",
        "  retrained / both   should read 'never'    -- this is the floor.",
        "  unlearned / forget is the result: 'trained' = under-unlearned,",
        "                     'never' = success, 'unlearned' = a distinguishable",
        "                     third state, i.e. deletion is detectable.",
        "  advert.            fraction pushed below the never-trained baseline;",
        "                     over-unlearning that RMIA scores as non-membership.",
        "",
    ]
    return "\n".join(lines)


def self_test(seed: int = 0) -> None:
    """Synthesise a run with known ground truth and check the classifier recovers it.

    Builds signals from three separated Gaussians on the logit scale, with the
    same layout a real run produces, then asserts that each target model's forget
    points land in the state they were generated from. Runs without any cluster
    output, which is the only way to exercise this file on a machine that has no
    level-3 run on disk.

    Args:
        seed (int): RNG seed.

    Raises:
        AssertionError: If recovery fails on any of the three targets.
    """
    rng = np.random.default_rng(seed)
    forget_size, num_ref = 500, 16
    num_audit = 2 * forget_size

    mu_never, mu_unlearned, mu_trained, sigma = -2.0, -0.5, 3.0, 0.6
    # Per-point difficulty, shared across models: this is what per-sample
    # calibration exists to remove, so the synthetic data must have it.
    difficulty = rng.normal(0.0, 1.5, size=num_audit)

    in_per_sample = num_ref // 2
    ranks = rng.random((num_ref, num_audit)).argsort(axis=0).argsort(axis=0)
    ref_in = (ranks < in_per_sample).T

    def draw(mu, shape):
        return mu + difficulty[:, None] + rng.normal(0.0, sigma, size=shape)

    ref_phi = np.where(
        ref_in,
        draw(mu_unlearned, (num_audit, num_ref)),
        draw(mu_never, (num_audit, num_ref)),
    )

    # original: forget points trained, unseen points never trained.
    original = np.where(
        np.arange(num_audit) < forget_size,
        draw(mu_trained, (num_audit, 1))[:, 0],
        draw(mu_never, (num_audit, 1))[:, 0],
    )
    # unlearned: forget points in the unlearned state, unseen never trained.
    unlearned = np.where(
        np.arange(num_audit) < forget_size,
        draw(mu_unlearned, (num_audit, 1))[:, 0],
        draw(mu_never, (num_audit, 1))[:, 0],
    )
    retrained = draw(mu_never, (num_audit, 1))[:, 0]

    phi = np.column_stack([original, unlearned, retrained, ref_phi])
    run = {
        "phi": phi,
        "ref_in": ref_in,
        "forget_size": forget_size,
        "num_ref_models": num_ref,
    }
    analysis = analyse(run)
    print(format_report(analysis, "shift"))

    checks = [
        ("original", "forget", "trained"),
        ("original", "unseen", "never"),
        ("unlearned", "forget", "unlearned"),
        ("unlearned", "unseen", "never"),
        ("retrained", "forget", "never"),
        ("retrained", "unseen", "never"),
    ]
    for role, group, expected in checks:
        frac = analysis["targets"][role][group]
        winner = max(frac, key=frac.get)
        assert winner == expected, (
            f"{role}/{group}: expected majority '{expected}', got '{winner}' {frac}"
        )
        assert frac[expected] > 0.5, f"{role}/{group}: {expected} only {frac[expected]}"

    d = analysis["direction"]
    # mu_unlearned > mu_never by construction, so delta must be positive, and the
    # forget/unseen split must agree since both were generated identically.
    assert d["delta_mean"] > 0, d
    assert abs(d["delta_mean_forget"] - d["delta_mean_unseen"]) < 0.2, d
    assert abs(d["trained_control_shift"]) < 0.2, d
    print("self-test passed: all six target/group cells recovered.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir", nargs="?", help="A finished level-3 run directory"
    )
    parser.add_argument(
        "--trained-source",
        choices=["shift", "base"],
        default="shift",
        help="How to estimate the trained-but-not-unlearned state (default: shift)",
    )
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=2.0,
        help="Never-trained stds below the mean to count as advertised (default: 2)",
    )
    parser.add_argument("--json", help="Also write the analysis to this JSON path")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run on synthetic data with known ground truth and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.log_dir:
        parser.error("log_dir is required unless --self-test is given")

    run = load_run(args.log_dir)
    analysis = analyse(run, args.trained_source, args.n_sigma)
    print(f"\nRun: {args.log_dir}")
    print(
        f"  {run['forget_size']} forget / {run['forget_size']} unseen points, "
        f"K = {run['num_ref_models']} references"
    )
    print(format_report(analysis, args.trained_source))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(analysis, f, indent=4)
        print(f"Written to {args.json}\n")


if __name__ == "__main__":
    main()

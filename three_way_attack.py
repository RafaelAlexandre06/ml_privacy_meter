"""Attack-grade three-way membership test for a finished level-3 U-RMIA run.

After unlearning, an audit point x sits in one of three worlds:

    trained    x was trained on and never unlearned      (Hayes et al.: retain)
    unlearned  x was trained on, then unlearned           (forget)
    never      x was never trained on                     (test)

``three_way_analysis.py`` already classifies against these worlds but has no
reference distribution for ``trained``: level-3 references are pickled only
after unlearning, so that world is estimated from the ground-truth F/U labels
and the result is a diagnostic, not an attack. This module is the honest
version. It needs the run to have been made with ``audit.three_way: true``,
which persists every reference's pre-unlearn checkpoint
(``online_reference_k_base.pkl``) and adds one signal pass over them
(``signals/urmia_signals_base.npy``).

Per audit point the K references split into K//2 that trained-then-unlearned
it (IN) and K//2 that never saw it (OUT). All three worlds come from references
alone, no labels:

    trained    base signals of the IN refs     (same models, before unlearning)
    unlearned  post-unlearn signals of the IN refs
    never      post-unlearn signals of the OUT refs

This is the three-way hypothesis test of Hayes et al. 2024, Appendix D.2. The
two-way scorers in the main summary are monotone in confidence, so a point
pushed *below* the never-trained baseline (over-unlearned) scores as a
non-member and cancels against under-unlearned points. Here a point is read by
whichever world it resembles, so an advertised deletion is a detection rather
than a cancellation.

One gap is measured rather than assumed. The retain-world models of Hayes et
al. had other points unlearned from them; our ``trained`` world is the
untouched pre-unlearn checkpoint. ``collateral`` -- how much unlearning *other*
points moved the OUT references on x -- is exactly that difference. It must sit
near zero and agree between the forget and unseen halves; otherwise unlearning
is moving non-members and the never world is not what it claims to be.

Usage::

    python three_way_attack.py sweep_l3_cifar10_neggrad_plus
    python three_way_attack.py <log_dir> --json out.json
    python three_way_attack.py --self-test
"""

import argparse
import json
import os

import numpy as np

from three_way_analysis import (
    HYPOTHESES,
    REF_COL_START,
    TARGET_ROLES,
    _logit,
    _masked_moments,
    advertised_mask,
    three_way_posterior,
)

BASE_SIGNAL_FILE = "urmia_signals_base.npy"
GROUPS = ["forget", "unseen"]


def load_run(log_dir: str) -> dict:
    """Load the audit signals, the base signals and the reference assignment.

    Args:
        log_dir (str): A finished level-3 run made with ``audit.three_way: true``.

    Returns:
        dict: ``signals`` (2F, 3+K) and ``base_signals`` (2F, K) true-class
            confidences, ``ref_in`` (2F, K) bool, ``forget_size``,
            ``num_ref_models``.

    Raises:
        FileNotFoundError: If a signal or split file is missing.
        ValueError: If the run is level 2, or the shapes disagree.
    """
    signal_path = os.path.join(log_dir, "signals", "urmia_signals.npy")
    base_path = os.path.join(log_dir, "signals", BASE_SIGNAL_FILE)
    splits_path = os.path.join(log_dir, "models", "urmia_online_splits.npz")

    if not os.path.exists(signal_path):
        raise FileNotFoundError(f"No cached signals at {signal_path}")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"No level-3 splits at {splits_path}. A level-2 run has no "
            "train-then-unlearn references, so the three worlds cannot be fit."
        )
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"No base signals at {base_path}. Re-run run_urmia_online.py with "
            "audit.three_way: true. A run made without it cannot be upgraded: "
            "the pre-unlearn reference checkpoints were never saved."
        )

    signals = np.load(signal_path)
    base_signals = np.load(base_path)
    splits = np.load(splits_path)
    if "ref_unlearn_membership" not in splits:
        raise ValueError(f"{splits_path} is a level-2 split file.")

    # (K, 2F) -> (2F, K), the orientation every scorer works in.
    ref_in = np.asarray(splits["ref_unlearn_membership"]).T.astype(bool)
    num_ref_models = ref_in.shape[1]
    forget_size = len(splits["forget_indices"])

    expected = (ref_in.shape[0], REF_COL_START + num_ref_models)
    if signals.shape != expected:
        raise ValueError(
            f"Signal matrix {signals.shape} disagrees with the splits: expected "
            f"{expected}. The signal cache is stale -- delete it and re-run."
        )
    if base_signals.shape != ref_in.shape:
        raise ValueError(
            f"Base signal matrix {base_signals.shape} disagrees with the splits: "
            f"expected {ref_in.shape}. Delete {base_path} and re-run."
        )

    return {
        "signals": signals,
        "base_signals": base_signals,
        "ref_in": ref_in,
        "forget_size": forget_size,
        "num_ref_models": num_ref_models,
    }


def world_states(phi: np.ndarray, base_phi: np.ndarray, ref_in: np.ndarray) -> dict:
    """Per-point Gaussians for the three worlds, from references only.

    Args:
        phi (np.ndarray): Logit signals, shape (num_audit, 3 + K).
        base_phi (np.ndarray): Pre-unlearn reference logit signals, (num_audit, K).
        ref_in (np.ndarray): Bool IN mask, shape (num_audit, K).

    Returns:
        dict: ``trained``, ``unlearned``, ``never`` each ``{"mean", "std"}``,
            plus ``collateral`` (num_audit,), the never-world shift caused by
            unlearning other points.
    """
    ref_phi = phi[:, REF_COL_START:]
    mean_tr, std_tr = _masked_moments(base_phi, ref_in)
    mean_un, std_un = _masked_moments(ref_phi, ref_in)
    mean_nv, std_nv = _masked_moments(ref_phi, ~ref_in)
    mean_nv_base, _ = _masked_moments(base_phi, ~ref_in)
    return {
        "trained": {"mean": mean_tr, "std": std_tr},
        "unlearned": {"mean": mean_un, "std": std_un},
        "never": {"mean": mean_nv, "std": std_nv},
        "collateral": mean_nv - mean_nv_base,
    }


def compute_three_way(
    signals: np.ndarray,
    base_signals: np.ndarray,
    ref_in: np.ndarray,
    forget_size: int,
    n_sigma: float = 2.0,
) -> dict:
    """Classify every audit point of every target into one of the three worlds.

    Args:
        signals (np.ndarray): (2F, 3 + K) true-class confidences, columns
            ``[original, unlearned, retrained, ref_0..]``, rows F then U.
        base_signals (np.ndarray): (2F, K) confidences under the pre-unlearn
            reference checkpoints, same column order as the refs.
        ref_in (np.ndarray): (2F, K) bool, True where the point was
            trained-then-unlearned in that reference.
        forget_size (int): Number of forget points (rows 0..forget_size-1).
        n_sigma (float): Never-world stds below the mean that count as advertised.

    Returns:
        dict: ``collateral`` diagnostics and, per target role, per-group world
            fractions, mean posteriors, ``member_rate`` (assigned to either
            trained world) and ``advertised``, plus ``binary_balanced_acc`` of
            the member-vs-never reduction on forget vs unseen.
    """
    phi = _logit(signals)
    base_phi = _logit(base_signals)
    worlds = world_states(phi, base_phi, ref_in)
    # three_way_posterior / advertised_mask take the two reference worlds in the
    # IN/OUT naming of three_way_analysis.
    two_way = {
        "mean_in": worlds["unlearned"]["mean"],
        "std_in": worlds["unlearned"]["std"],
        "mean_out": worlds["never"]["mean"],
        "std_out": worlds["never"]["std"],
    }
    never_col = HYPOTHESES.index("never")
    f_sl, u_sl = slice(0, forget_size), slice(forget_size, None)
    coll = worlds["collateral"]

    result = {
        "forget_size": int(forget_size),
        "unseen_size": int(phi.shape[0] - forget_size),
        "num_ref_models": int(ref_in.shape[1]),
        "n_sigma": float(n_sigma),
        "collateral": {
            "mean": float(coll.mean()),
            "forget": float(coll[f_sl].mean()),
            "unseen": float(coll[u_sl].mean()),
        },
        "targets": {},
    }
    for col, role in enumerate(TARGET_ROLES):
        target_phi = phi[:, col]
        post = three_way_posterior(target_phi, two_way, worlds["trained"])
        label = post.argmax(axis=1)
        member = label != never_col
        adv = advertised_mask(target_phi, two_way, n_sigma)
        entry = {
            "binary_balanced_acc": float(
                0.5 * (member[f_sl].mean() + (~member[u_sl]).mean())
            )
        }
        for group, sl in (("forget", f_sl), ("unseen", u_sl)):
            entry[group] = {
                "frac": {h: float((label[sl] == i).mean()) for i, h in enumerate(HYPOTHESES)},
                "mean_posterior": {
                    h: float(post[sl, i].mean()) for i, h in enumerate(HYPOTHESES)
                },
                "member_rate": float(member[sl].mean()),
                "advertised": float(adv[sl].mean()),
            }
        result["targets"][role] = entry
    return result


def format_three_way_rows(analysis: dict) -> list:
    """Render the analysis as log lines, matching the U-RMIA summary style.

    Args:
        analysis (dict): Output of :func:`compute_three_way`.

    Returns:
        list[str]: Lines to log or print.
    """
    c = analysis["collateral"]
    lines = [
        "Three-way test (Hayes et al. 2024 App. D.2): trained / unlearned / never "
        "from %d references per point, no labels"
        % analysis["num_ref_models"],
        f"  collateral (never after unlearning others - never before): "
        f"mean {c['mean']:+.4f}  forget {c['forget']:+.4f}  unseen {c['unseen']:+.4f}",
        f"{'target':<12} {'group':<8} {'trained':>9} {'unlearned':>10} {'never':>9} "
        f"{'member':>8} {'advert.':>9}",
    ]
    for role, res in analysis["targets"].items():
        for group in GROUPS:
            g = res[group]
            frac = g["frac"]
            lines.append(
                f"{role:<12} {group:<8} {frac['trained']:>9.3f} "
                f"{frac['unlearned']:>10.3f} {frac['never']:>9.3f} "
                f"{g['member_rate']:>8.3f} {g['advertised']:>9.3f}"
            )
    acc = "  ".join(
        f"{role} {res['binary_balanced_acc']:.3f}"
        for role, res in analysis["targets"].items()
    )
    lines += [
        f"  binary balanced acc (member = trained or unlearned, forget vs unseen): {acc}",
        "Reading it:",
        "  collateral must be ~0 and agree across groups, or the never world is "
        "not never-trained.",
        "  original/forget -> trained, original/unseen and retrained/* -> never: "
        "controls; if they fail, stop.",
        "  unlearned/forget: never = success, trained = under-unlearned, "
        "unlearned = deletion is a distinguishable third state.",
        "  member on 'unlearned' vs the two-way ulira row: the three-way test "
        "must not lose membership accuracy.",
        "  advert. = pushed below the never baseline; the two-way scorers file "
        "these as non-members.",
    ]
    return lines


def self_test(seed: int = 0) -> None:
    """Synthesise a run with three planted worlds and check they are recovered.

    Args:
        seed (int): RNG seed.

    Raises:
        AssertionError: If any target/group cell lands in the wrong world, the
            collateral control drifts, or the binary reduction fails.
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
    is_forget = np.arange(num_audit) < forget_size

    def draw(mu, shape):
        return mu + difficulty[:, None] + rng.normal(0.0, sigma, size=shape)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    ref_phi = np.where(
        ref_in,
        draw(mu_unlearned, (num_audit, num_ref)),
        draw(mu_never, (num_audit, num_ref)),
    )
    base_phi = np.where(
        ref_in,
        draw(mu_trained, (num_audit, num_ref)),
        draw(mu_never, (num_audit, num_ref)),
    )
    original = np.where(
        is_forget,
        draw(mu_trained, (num_audit, 1))[:, 0],
        draw(mu_never, (num_audit, 1))[:, 0],
    )
    unlearned = np.where(
        is_forget,
        draw(mu_unlearned, (num_audit, 1))[:, 0],
        draw(mu_never, (num_audit, 1))[:, 0],
    )
    retrained = draw(mu_never, (num_audit, 1))[:, 0]

    signals = sigmoid(np.column_stack([original, unlearned, retrained, ref_phi]))
    analysis = compute_three_way(signals, sigmoid(base_phi), ref_in, forget_size)
    print("\n".join(format_three_way_rows(analysis)))

    checks = [
        ("original", "forget", "trained"),
        ("original", "unseen", "never"),
        ("unlearned", "forget", "unlearned"),
        ("unlearned", "unseen", "never"),
        ("retrained", "forget", "never"),
        ("retrained", "unseen", "never"),
    ]
    for role, group, expected in checks:
        frac = analysis["targets"][role][group]["frac"]
        winner = max(frac, key=frac.get)
        assert winner == expected, (
            f"{role}/{group}: expected majority '{expected}', got '{winner}' {frac}"
        )
        assert frac[expected] > 0.5, f"{role}/{group}: {expected} only {frac[expected]}"

    c = analysis["collateral"]
    assert abs(c["mean"]) < 0.2, c
    assert abs(c["forget"] - c["unseen"]) < 0.2, c
    acc = {r: analysis["targets"][r]["binary_balanced_acc"] for r in TARGET_ROLES}
    # The planted unlearned/never worlds sit 2.5 sigma apart, so ~11% of each
    # is misread as the other: the Bayes rate caps 'unlearned' near 0.89 while
    # 'trained' stays cleanly separable -- the Hayes App. D.2 error structure.
    assert acc["original"] > 0.9, acc
    assert acc["unlearned"] > 0.8, acc
    assert acc["retrained"] < 0.6, acc
    print("self-test passed: all six cells, collateral and binary reduction recovered.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir", nargs="?", help="A finished level-3 run made with audit.three_way"
    )
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=2.0,
        help="Never-world stds below the mean to count as advertised (default: 2)",
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
    analysis = compute_three_way(
        run["signals"], run["base_signals"], run["ref_in"], run["forget_size"], args.n_sigma
    )
    print(f"\nRun: {args.log_dir}")
    print(
        f"  {run['forget_size']} forget / {run['forget_size']} unseen points, "
        f"K = {run['num_ref_models']} references\n"
    )
    print("\n".join(format_three_way_rows(analysis)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(analysis, f, indent=4)
        print(f"\nWritten to {args.json}\n")


if __name__ == "__main__":
    main()

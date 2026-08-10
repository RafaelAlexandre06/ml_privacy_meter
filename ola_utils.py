"""Helpers for the OLA (Outer Layer Attenuation) audit.

Paper: Prabhu, Sasy, Davidson and Ghodsi, *No More Tears: Privacy and Utility
Without the Onion* (``pdfs/aaai27-submission.pdf``).

OLA is a **training-time defense**, not an unlearner, so it does not touch
``unlearners.py`` and the game it plays is ordinary membership inference: is this
point in the training set? The pipeline here is therefore built on the standard
shadow-model structure that ``models/utils.py:split_dataset_for_training``
already produces -- models trained in pairs on complementary halves, so every
point is IN for exactly half of them -- rather than on the U-RMIA level-2/3
reference sets.

Three stages live here:

1. :func:`compute_asr` / :func:`select_outer_layer` -- find the points a strong
   attack succeeds on most often (paper Eq. 1), which become the outer layer.
2. :func:`prepare_ola_models` -- train the three targets: ``vanilla`` (leakage
   ceiling), ``ola`` (the defense), and ``onion`` (the Privacy Onion baseline,
   i.e. the outer layer deleted and the model retrained -- the thing OLA claims
   to beat).
3. :func:`log_ola_summary` -- report per scorer, per target, split by outer
   layer / inner-layer members / non-members, because the paper's whole claim is
   about what removal does to the *inner* layer.

Nothing here modifies existing files; ``get_model``, ``load_models``,
``train_models``, ``get_model_signals``, ``compute_attack_results`` and the
scorers in ``modules/mia/attacks/urmia.py`` are imported and reused.
"""

import gc
import json
import logging
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from audit import compute_attack_results
from dataset.utils import get_dataloader
from models.ola import ExampleTiedDropout
from models.utils import get_model
from trainers.default_trainer import inference, train
from trainers.ola_trainer import IndexedSubset, train_ola

OLA_SPLITS_FILE = "ola_splits.npz"
OLA_METADATA_FILE = "ola_models_metadata.json"

# vanilla = leakage ceiling and per-scorer positive control; onion = the baseline
# OLA is arguing against (delete the outer layer, retrain).
TARGET_ROLES = ["vanilla", "ola", "onion"]

# Audit subsets the summary reports separately. "outer" is what the defense
# targets; "inner" is where the Privacy Onion effect shows up.
AUDIT_SUBSETS = ["outer", "inner", "all_members"]


def check_ola_configs(configs: dict, dataset_size: int) -> None:
    """Validate the OLA config block.

    Args:
        configs (dict): Full config dictionary.
        dataset_size (int): Size of the training pool.

    Raises:
        ValueError: On any setting that would produce a meaningless run.
    """
    ola = configs["ola"]
    n_outer = int(ola["outer_size"])
    p_gen, p_mem = float(ola["p_gen"]), float(ola["p_mem"])
    num_shadow = int(configs["audit"]["num_shadow_models"])

    if num_shadow % 4 != 0:
        raise ValueError(
            f"num_shadow_models must be divisible by 4 (models are trained in "
            f"pairs, then split into selection and evaluation halves), got {num_shadow}"
        )
    if num_shadow < 8:
        raise ValueError(
            f"num_shadow_models={num_shadow} leaves too few references per half "
            "for stable per-sample Gaussians; use at least 8 (paper used 256)."
        )
    if not 0.0 < p_gen <= 1.0:
        raise ValueError(f"ola.p_gen must be in (0, 1], got {p_gen}")
    if not 0.0 <= p_mem <= 1.0:
        raise ValueError(f"ola.p_mem must be in [0, 1], got {p_mem}")
    if n_outer >= dataset_size // 2:
        raise ValueError(
            f"ola.outer_size={n_outer} must be well below half the pool "
            f"({dataset_size // 2}), since it is drawn from the target's members."
        )
    if configs["audit"].get("data_size") is not None:
        logging.getLogger("time_analysis").warning(
            "audit.data_size is set. The low-FPR resolution of this audit comes "
            "from the number of non-members; downsampling throws it away."
        )


def get_or_create_ola_splits(
    models_dir: str, dataset_size: int, configs: dict, logger: logging.Logger
) -> dict:
    """Load or draw the target model's training split.

    The target split is drawn **independently of the shadow pool** so that no
    shadow model is trained on exactly the target's data. Reusing a shadow
    model's split would hand the attack a perfect surrogate for the target and
    inflate every number in the report.

    Args:
        models_dir (str): ``<log_dir>/models``.
        dataset_size (int): Size of the training pool.
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object.

    Returns:
        dict: With ``target_train`` (member indices) and ``target_out``
            (non-member indices).
    """
    splits_path = f"{models_dir}/{OLA_SPLITS_FILE}"
    if os.path.exists(splits_path):
        data = np.load(splits_path)
        if int(data["dataset_size"]) != dataset_size:
            raise ValueError(
                f"Persisted OLA splits at {splits_path} were built for "
                f"dataset_size={int(data['dataset_size'])}, not {dataset_size}. "
                "Use a fresh log_dir."
            )
        logger.info("Loaded OLA splits from disk.")
        return {"target_train": data["target_train"], "target_out": data["target_out"]}

    # A dedicated stream so the target split does not move when the shadow pool
    # or the mask seed changes.
    rng = np.random.default_rng(int(configs["run"]["random_seed"]) + 9001)
    perm = rng.permutation(dataset_size)
    half = dataset_size // 2
    target_train = np.sort(perm[:half]).astype(np.int64)
    target_out = np.sort(perm[half:]).astype(np.int64)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        splits_path,
        target_train=target_train,
        target_out=target_out,
        dataset_size=np.int32(dataset_size),
        seed=np.int32(configs["run"]["random_seed"]),
    )
    logger.info(
        "Created OLA splits: |members|=%d, |non-members|=%d. At this many "
        "non-members the 0.1%%-FPR operating point is %d false positives.",
        len(target_train),
        len(target_out),
        max(1, int(round(0.001 * len(target_out)))),
    )
    return {"target_train": target_train, "target_out": target_out}


def compute_asr(
    signals: np.ndarray,
    memberships: np.ndarray,
    model_cols: np.ndarray,
    fpr_target: float,
    logger: logging.Logger,
) -> np.ndarray:
    """Per-point attack success rate over a set of shadow models (paper Eq. 1).

    ``ASR(x) = Pr[A(x, f_s) = 1 | x in D_s]``: for every shadow model that
    trained on x, run the attack against that model and record whether it called
    x a member. The attack is LiRA fitted **leave-one-out** -- the model being
    attacked is excluded from its own reference set, or its signal would sit
    inside the IN Gaussian by construction and every point would look like a
    member.

    The decision threshold is calibrated per model on that model's own
    non-members, so ``fpr_target`` is an actual false-positive rate rather than
    an arbitrary cut.

    Args:
        signals (np.ndarray): (num_samples, num_models) true-class confidences.
        memberships (np.ndarray): (num_models, num_samples) bool membership.
        model_cols (np.ndarray): Columns of ``signals`` to use (the selection
            half of the shadow pool).
        fpr_target (float): False-positive rate the threshold is set to.
        logger (logging.Logger): Logger object.

    Returns:
        np.ndarray: ASR per sample, shape (num_samples,). Samples that are a
            member of no selection model get NaN.
    """
    from modules.mia.attacks.urmia import run_ulira

    model_cols = np.asarray(model_cols)
    num_samples = signals.shape[0]
    successes = np.zeros(num_samples, dtype=float)
    trials = np.zeros(num_samples, dtype=float)

    for col in model_cols:
        ref_cols = model_cols[model_cols != col]
        ref_in = memberships[ref_cols].T  # (num_samples, K-1)
        scores = run_ulira(signals[:, col], signals[:, ref_cols], ref_in)

        is_member = memberships[col]
        non_member_scores = scores[~is_member]
        # Highest-scoring non-members define the cut: at fpr_target, this many
        # of them are allowed above it.
        threshold = np.quantile(non_member_scores, 1.0 - fpr_target)

        successes[is_member] += scores[is_member] > threshold
        trials[is_member] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        asr = successes / trials
    asr[trials == 0] = np.nan

    logger.info(
        "ASR over %d selection models at %.1f%% FPR: mean %.4f, p50 %.4f, "
        "p99 %.4f, max %.4f",
        len(model_cols),
        100 * fpr_target,
        np.nanmean(asr),
        np.nanpercentile(asr, 50),
        np.nanpercentile(asr, 99),
        np.nanmax(asr),
    )
    return asr


def select_outer_layer(
    asr: np.ndarray,
    target_train: np.ndarray,
    n_outer: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Take the ``n_outer`` most attackable of the target's training points.

    Args:
        asr (np.ndarray): Per-sample ASR from :func:`compute_asr`.
        target_train (np.ndarray): The target model's member indices.
        n_outer (int): Size of the outer layer.
        logger (logging.Logger): Logger object.

    Returns:
        np.ndarray: Sorted outer-layer indices into the full dataset.
    """
    member_asr = np.nan_to_num(asr[target_train], nan=-1.0)
    order = np.argsort(-member_asr, kind="stable")
    outer = np.sort(target_train[order[:n_outer]])

    cut = member_asr[order[n_outer - 1]]
    logger.info(
        "Outer layer: %d of %d members, ASR >= %.4f (member mean %.4f). "
        "Outer mean ASR %.4f vs inner mean %.4f.",
        n_outer,
        len(target_train),
        cut,
        member_asr.mean(),
        member_asr[order[:n_outer]].mean(),
        member_asr[order[n_outer:]].mean(),
    )
    if cut <= member_asr.mean():
        logger.warning(
            "The outer-layer cut is at or below the mean ASR: the attack is not "
            "separating points, so there is no meaningful outer layer to attenuate. "
            "Check the shadow pool trained correctly before spending more compute."
        )
    return outer


class _OlaMetadataStore:
    """Per-role checkpoint store with resume, mirroring the U-RMIA pattern.

    Writes metadata after **each** model rather than once at the end, so a run
    killed midway resumes instead of retraining everything (the same reason
    ``models/utils.py:prepare_models`` was changed in this fork).

    For the ``ola`` role it also records the mask fingerprint. Masks are rebuilt
    from the seed on load, and a mismatch means the loaded weights were trained
    through channels the inference-time drop no longer switches off -- invisible
    to any accuracy check, hence the hard failure.
    """

    def __init__(self, models_dir: str, configs: dict):
        self.models_dir = models_dir
        self.configs = configs
        self.path = f"{models_dir}/{OLA_METADATA_FILE}"
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        self.metadata = {}
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.metadata = json.load(f)

    def has(self, role: str) -> bool:
        return role in self.metadata and os.path.exists(f"{self.models_dir}/{role}.pkl")

    def load_weights(self, role: str) -> dict:
        with open(f"{self.models_dir}/{role}.pkl", "rb") as f:
            return pickle.load(f)

    def save(self, role: str, model, info: dict) -> None:
        with open(f"{self.models_dir}/{role}.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)
        self.metadata[role] = info
        with open(self.path, "w") as f:
            json.dump(self.metadata, f, indent=4)


def _build_ola_model(configs: dict, dataset_size: int, outer_indices: np.ndarray):
    """Instantiate the masked model for the given config and outer layer."""
    base = get_model(
        configs["train"]["model_name"], configs["data"]["dataset"], configs["train"]
    )
    return ExampleTiedDropout(
        base,
        dataset_size=dataset_size,
        outer_indices=outer_indices,
        p_gen=float(configs["ola"]["p_gen"]),
        p_mem=float(configs["ola"]["p_mem"]),
        seed=int(configs["ola"].get("mask_seed", configs["run"]["random_seed"])),
    )


def _subset_accuracy(model, dataset, indices, configs) -> float:
    """Eval-mode accuracy on a subset, batched. Empty subset returns NaN."""
    if len(indices) == 0:
        return float("nan")
    loader = get_dataloader(
        Subset(dataset, indices),
        batch_size=int(configs["train"]["batch_size"]),
        shuffle=False,
    )
    _, acc = inference(model, loader, configs["train"]["device"])
    return float(acc)


def prepare_ola_models(
    models_dir: str,
    dataset,
    splits: dict,
    outer_indices: np.ndarray,
    configs: dict,
    logger: logging.Logger,
) -> list:
    """Train (or load) the vanilla / ola / onion targets. Resumable.

    Following the convention in ``urmia_online_utils.py:211``, "test accuracy"
    here means accuracy on a slice of the target's **non-members** -- this
    pipeline has no separate held-out set, and non-members are held out by
    construction.

    Args:
        models_dir (str): ``<log_dir>/models``.
        dataset (torch.utils.data.Dataset): The training pool.
        splits (dict): Output of :func:`get_or_create_ola_splits`.
        outer_indices (np.ndarray): The outer layer.
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object.

    Returns:
        list: Models in ``TARGET_ROLES`` order, in eval mode on CPU.
    """
    store = _OlaMetadataStore(models_dir, configs)
    dataset_size = len(dataset)
    batch_size = int(configs["train"]["batch_size"])

    target_train = splits["target_train"]
    inner = np.setdiff1d(target_train, outer_indices, assume_unique=False)
    role_indices = {
        "vanilla": target_train,
        "ola": target_train,
        "onion": inner,  # the Privacy Onion baseline: outer layer deleted
    }

    test_indices = splits["target_out"][: int(configs["ola"].get("test_size", 5000))]
    test_loader = get_dataloader(
        Subset(dataset, test_indices), batch_size=batch_size, shuffle=False
    )
    models = []

    for role in TARGET_ROLES:
        if role == "ola":
            model = _build_ola_model(configs, dataset_size, outer_indices)
            fingerprint = model.mask_fingerprint()
        else:
            model = get_model(
                configs["train"]["model_name"],
                configs["data"]["dataset"],
                configs["train"],
            )
            fingerprint = None

        if store.has(role):
            model.load_state_dict(store.load_weights(role))
            if role == "ola":
                cached = store.metadata[role].get("mask_fingerprint")
                if cached != fingerprint:
                    raise ValueError(
                        f"Cached 'ola' model was trained with mask fingerprint "
                        f"{cached}, but this config rebuilds {fingerprint}. The "
                        "inference-time channel drop would not match the trained "
                        "network. Change log_dir, or restore p_gen/p_mem/mask_seed "
                        "and the outer layer to what produced the checkpoint."
                    )
            logger.info("Loaded target '%s' from disk.", role)
        else:
            indices = role_indices[role]
            logger.info("Training target '%s' on %d samples.", role, len(indices))
            if role == "ola":
                loader = get_dataloader(
                    IndexedSubset(dataset, indices), batch_size=batch_size, shuffle=True
                )
                model = train_ola(model, loader, configs["train"], test_loader)
            else:
                loader = get_dataloader(
                    Subset(dataset, indices), batch_size=batch_size, shuffle=True
                )
                model = train(model, loader, configs["train"], test_loader)

            info = {
                "role": role,
                "train_size": int(len(indices)),
                # test_acc == accuracy on held-out non-members; see the docstring.
                "test_acc": _subset_accuracy(model, dataset, test_indices, configs),
                "outer_acc": _subset_accuracy(model, dataset, outer_indices, configs),
                "inner_acc": _subset_accuracy(model, dataset, inner, configs),
            }
            if fingerprint is not None:
                info["mask_fingerprint"] = fingerprint
                info["p_gen"] = float(configs["ola"]["p_gen"])
                info["p_mem"] = float(configs["ola"]["p_mem"])
            store.save(role, model, info)
            logger.info(
                "Target '%s': test(non-member) %.4f | outer %.4f | inner %.4f",
                role,
                info["test_acc"],
                info["outer_acc"],
                info["inner_acc"],
            )

        model.eval().to("cpu")
        models.append(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return models


def get_ola_signals(
    models_list: list, dataset, configs: dict, tag: str, logger, is_population=False
) -> np.ndarray:
    """``get_model_signals`` with a cache filename that will not collide.

    This audit computes signals in **two** passes -- the shadow pool first (its
    signals are what pick the outer layer) and the three targets afterwards,
    since they cannot exist until the outer layer is known.
    ``get_signals.py:150`` derives the cache filename from
    ``audit.algorithm`` and only invalidates it on a row-count mismatch. Both
    passes cover the full pool, so the row counts agree and the second pass
    would silently load the first pass's signals.

    Passing a per-pass ``tag`` through ``audit.algorithm`` gives each pass its
    own file. Nothing else in ``get_model_signals`` reads that field.

    Args:
        models_list (list): Models to compute signals for.
        dataset: Dataset to compute them over.
        configs (dict): Full config dictionary; not mutated.
        tag (str): Cache key, e.g. ``"ola_shadow"`` or ``"ola_target"``.
        logger (logging.Logger): Logger object.
        is_population (bool): Passed through; appends ``_pop`` to the filename.

    Returns:
        np.ndarray: (num_samples, num_models) signals.
    """
    from get_signals import get_model_signals

    tagged = dict(configs)
    tagged["audit"] = dict(configs["audit"])
    tagged["audit"]["algorithm"] = tag
    return get_model_signals(models_list, dataset, tagged, logger, is_population)


def report_ola_attack(
    report_dir: str, name: str, scores: np.ndarray, memberships: np.ndarray, logger
) -> dict:
    """Compute and persist one attack result, keeping the raw scores.

    Saving ``scores``/``memberships`` is what lets ``analyze_tpr.py`` re-analyse
    a finished run with bootstrap CIs and no retraining.

    Args:
        report_dir (str): ``<log_dir>/report/exp``.
        name (str): ``<role>_<subset>_<scorer>``.
        scores (np.ndarray): MIA scores, higher = more likely member.
        memberships (np.ndarray): 1 for members, 0 for non-members.
        logger (logging.Logger): Logger object.

    Returns:
        dict: Output of ``audit.compute_attack_results``.
    """
    result = compute_attack_results(scores, memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        f"{report_dir}/attack_result_{name}.npz",
        scores=scores,
        memberships=memberships,
        fpr=result["fpr"],
        tpr=result["tpr"],
        auc=result["auc"],
    )
    logger.info(
        "%s: AUC %.4f, TPR@1%%FPR %.4f, TPR@0.1%%FPR %.4f, TPR@0%%FPR %.4f",
        name,
        result["auc"],
        result["one_fpr"],
        result["one_tenth_fpr"],
        result["zero_fpr"],
    )
    return result


def log_ola_summary(
    report_dir: str,
    results: dict,
    metadata: dict,
    configs: dict,
    n_nonmembers: int,
    logger: logging.Logger,
) -> None:
    """Log and save the outer/inner comparison across targets and scorers.

    Args:
        report_dir (str): ``<log_dir>/report``.
        results (dict): ``results[role][subset][scorer]`` -> attack result.
        metadata (dict): Per-role accuracies from the metadata store.
        configs (dict): Full config dictionary.
        n_nonmembers (int): Number of non-members each AUC is measured against.
        logger (logging.Logger): Logger object.
    """
    scorers = list(next(iter(next(iter(results.values())).values())).keys())
    summary = {"ola": configs["ola"], "roles": {}}

    for subset in AUDIT_SUBSETS:
        logger.info("")
        logger.info("=== %s ===", subset.upper())
        header = f"{'role':<10}"
        for scorer in scorers:
            header += f" {scorer + '_auc':>18} {scorer + '_tpr@.1%':>18}"
        logger.info(header)
        for role in TARGET_ROLES:
            line = f"{role:<10}"
            for scorer in scorers:
                res = results[role][subset][scorer]
                line += f" {res['auc']:>18.4f} {res['one_tenth_fpr']:>18.4f}"
            logger.info(line)

    for role in TARGET_ROLES:
        summary["roles"][role] = {
            "accuracy": metadata.get(role, {}),
            "attacks": {
                subset: {
                    scorer: {
                        "auc": float(results[role][subset][scorer]["auc"]),
                        "one_fpr": float(results[role][subset][scorer]["one_fpr"]),
                        "one_tenth_fpr": float(
                            results[role][subset][scorer]["one_tenth_fpr"]
                        ),
                        "zero_fpr": float(results[role][subset][scorer]["zero_fpr"]),
                    }
                    for scorer in scorers
                }
                for subset in AUDIT_SUBSETS
            },
        }

    # Positive control, carried over from the U-RMIA audit (findings 6.3): a
    # scorer that cannot detect membership on the undefended model is broken for
    # this run, and its 'ola' number is a false negative, not a privacy win.
    null_se = math.sqrt((2 * n_nonmembers + 1) / (12.0 * n_nonmembers * n_nonmembers))
    dead_cut = 0.5 + 2.0 * null_se
    dead = [
        s
        for s in scorers
        if max(
            results["vanilla"]["all_members"][s]["auc"],
            1.0 - results["vanilla"]["all_members"][s]["auc"],
        )
        < dead_cut
    ]
    if dead:
        logger.warning(
            "POSITIVE CONTROL FAILED: %s read chance on the 'vanilla' model, "
            "which has no defense applied and must leak. Discard their 'ola' "
            "numbers rather than reading them as a privacy win.",
            ", ".join(dead),
        )
    else:
        logger.info("Positive control passed: every scorer detects 'vanilla'.")
    summary["positive_control"] = {"dead_scorers": dead, "null_se": float(null_se)}

    logger.info("")
    logger.info(
        "Read the OUTER block for whether the defense works ('vanilla' vs 'ola'), "
        "and the INNER block for the Privacy Onion effect: if 'onion' inner is "
        "above 'vanilla' inner, deleting the outer layer promoted the next one, "
        "and 'ola' inner should stay at or below 'vanilla'. Check the accuracies "
        "first -- a collapsed model leaks nothing because it knows nothing."
    )

    with open(f"{report_dir}/ola_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

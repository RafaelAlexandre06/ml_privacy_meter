"""Helper functions for U-RMIA (RMIA adapted to machine-unlearning evaluation).

U-RMIA measures whether an unlearning algorithm actually removes the influence
of a forget set F, by running the RMIA membership inference attack to
distinguish:

- forget samples F  (were trained on, then unlearned)  -> label 1
- unseen samples U  (never trained on)                 -> label 0

If unlearning works, the attack should be no better than random. We attack three
models for comparison:

- ``original``  : trained on R + F (attack should succeed -> upper bound)
- ``unlearned`` : original after the unlearning step        (the measurement)
- ``retrained`` : trained from scratch on R only    (gold standard -> ~random)

Reference models are each trained on a random half of the retain set R, so both
F and U are OUT of every reference model. This is the offline RMIA setting with
all-False reference memberships.

This module mirrors the conventions of the existing pipeline (``run_mia.py``,
``models/utils.py``, ``get_signals.py``) but keeps everything under U-RMIA
specific filenames so it never collides with the paired-model MIA caches. It
does not modify any existing module.
"""

import copy
import gc
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from audit import compute_attack_results
from dataset.utils import get_dataloader, load_dataset_subsets
from get_signals import get_softmax
from models.utils import get_model
from trainers.default_trainer import train, inference
from unlearners import get_unlearner
from visualize import plot_roc, plot_roc_log

# Signal-matrix column layout: three fixed target columns then the references.
TARGET_ROLES = ["original", "unlearned", "retrained"]
REF_COL_START = len(TARGET_ROLES)  # references occupy columns REF_COL_START .. REF_COL_START+K-1

SPLITS_FILE = "urmia_splits.npz"
METADATA_FILE = "urmia_models_metadata.json"


def check_urmia_configs(configs: dict, dataset_size: int) -> None:
    """Validate the U-RMIA specific configuration.

    Should be called after ``util.check_configs`` (which validates the shared
    ``audit`` fields).

    Args:
        configs (dict): Full config dictionary.
        dataset_size (int): Number of samples in the training-pool dataset.

    Raises:
        NotImplementedError: For unsupported attack algorithms or models.
        ValueError: For invalid sizes or parameters.
        KeyError: If the ``unlearn`` block is missing.
    """
    algorithm = configs["audit"]["algorithm"]
    if algorithm != "RMIA":
        raise NotImplementedError(
            f"U-RMIA only supports the RMIA attack, got '{algorithm}'."
        )

    if "unlearn" not in configs:
        raise KeyError("Config is missing the 'unlearn' section required by U-RMIA.")

    # Raises NotImplementedError with the supported list if unknown.
    get_unlearner(configs["unlearn"]["algorithm"])

    forget_size = int(configs["unlearn"]["forget_size"])
    if forget_size < 1:
        raise ValueError("unlearn.forget_size must be at least 1.")
    if 2 * forget_size > dataset_size - 2:
        raise ValueError(
            f"unlearn.forget_size={forget_size} is too large: need "
            f"2*forget_size <= dataset_size-2 (dataset_size={dataset_size}) so that "
            f"the retain set R is non-empty and splittable into halves."
        )

    num_ref_models = configs["audit"]["num_ref_models"]
    if num_ref_models is None or num_ref_models < 1:
        raise ValueError("audit.num_ref_models must be at least 1 for U-RMIA.")

    offline_a = configs["audit"].get("offline_a", 0.3)
    if not 0.0 <= offline_a <= 1.0:
        raise ValueError(f"audit.offline_a must be in [0, 1], got {offline_a}.")

    model_name = configs["train"]["model_name"]
    if model_name in ("speedyresnet", "gpt2"):
        raise NotImplementedError(
            f"U-RMIA v1 does not support model '{model_name}' (its training and "
            f"signal paths diverge from the default trainer)."
        )


def get_or_create_splits(
    models_dir: str,
    dataset_size: int,
    pop_len: int,
    configs: dict,
    logger: logging.Logger,
) -> dict:
    """Load persisted F/U/R and reference splits, or generate and persist them.

    The splits are saved to ``<models_dir>/urmia_splits.npz`` and reloaded on
    subsequent runs so that a rerun with the same ``log_dir`` is consistent
    (this is the tool's log_dir-as-cache convention). Regenerating requires a
    fresh ``log_dir``.

    Args:
        models_dir (str): ``<log_dir>/models`` directory.
        dataset_size (int): Number of samples in the training-pool dataset.
        pop_len (int): Number of samples in the population dataset.
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object for the current run.

    Returns:
        dict: Keys ``forget_indices``, ``unseen_indices``, ``retain_indices``,
            ``ref_memberships`` (K, dataset_size) bool, ``population_indices``.

    Raises:
        ValueError: If a persisted split disagrees with the current config, or
            the requested population subsample is larger than the population.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    seed = int(configs["run"]["random_seed"])
    forget_size = int(configs["unlearn"]["forget_size"])
    num_ref_models = int(configs["audit"]["num_ref_models"])
    population_size = int(configs["audit"].get("population_size", pop_len))
    if population_size > pop_len:
        raise ValueError(
            f"audit.population_size={population_size} exceeds the population size "
            f"({pop_len})."
        )

    splits_path = f"{models_dir}/{SPLITS_FILE}"
    if os.path.exists(splits_path):
        data = np.load(splits_path)
        stored = {
            "seed": int(data["seed"]),
            "forget_size": int(data["forget_size"]),
            "dataset_size": int(data["dataset_size"]),
            "num_ref_models": int(data["num_ref_models"]),
            "population_size": int(data["population_size"]),
        }
        expected = {
            "seed": seed,
            "forget_size": forget_size,
            "dataset_size": dataset_size,
            "num_ref_models": num_ref_models,
            "population_size": population_size,
        }
        if stored != expected:
            raise ValueError(
                f"Persisted U-RMIA splits at {splits_path} disagree with the current "
                f"config (stored {stored} vs config {expected}). Change run.log_dir to "
                f"start a fresh experiment."
            )
        logger.info("Loaded U-RMIA splits from disk.")
        return {
            "forget_indices": data["forget_indices"],
            "unseen_indices": data["unseen_indices"],
            "retain_indices": data["retain_indices"],
            "ref_memberships": data["ref_memberships"],
            "population_indices": data["population_indices"],
        }

    # Warn if this directory already holds paired-scheme MIA caches.
    for stale in ("models_metadata.json", "memberships.npy"):
        if os.path.exists(f"{models_dir}/{stale}"):
            logger.warning(
                "Found %s in %s: this looks like a paired-model MIA log_dir. "
                "U-RMIA uses a different split scheme; consider a fresh log_dir.",
                stale,
                models_dir,
            )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(dataset_size)
    forget_indices = perm[:forget_size]
    unseen_indices = perm[forget_size : 2 * forget_size]
    retain_indices = perm[2 * forget_size :]

    ref_memberships = np.zeros((num_ref_models, dataset_size), dtype=bool)
    half = len(retain_indices) // 2
    for k in range(num_ref_models):
        chosen = rng.permutation(retain_indices)[:half]
        ref_memberships[k, chosen] = True

    population_indices = rng.choice(pop_len, size=population_size, replace=False)

    np.savez(
        splits_path,
        forget_indices=forget_indices,
        unseen_indices=unseen_indices,
        retain_indices=retain_indices,
        ref_memberships=ref_memberships,
        population_indices=population_indices,
        seed=seed,
        forget_size=forget_size,
        dataset_size=dataset_size,
        num_ref_models=num_ref_models,
        population_size=population_size,
    )
    logger.info(
        "Created U-RMIA splits: |F|=%d, |U|=%d, |R|=%d, K=%d, |pop|=%d",
        len(forget_indices),
        len(unseen_indices),
        len(retain_indices),
        num_ref_models,
        population_size,
    )
    return {
        "forget_indices": forget_indices,
        "unseen_indices": unseen_indices,
        "retain_indices": retain_indices,
        "ref_memberships": ref_memberships,
        "population_indices": population_indices,
    }


def _role_train_indices(splits: dict) -> dict:
    """Map each trained-from-scratch role to its training indices."""
    forget = splits["forget_indices"]
    retain = splits["retain_indices"]
    ref_memberships = splits["ref_memberships"]

    roles = {
        "original": np.concatenate([retain, forget]),
        "retrained": retain,
    }
    for k in range(ref_memberships.shape[0]):
        roles[f"reference_{k}"] = np.where(ref_memberships[k])[0]
    return roles


def _train_one(role, train_idx, dataset, dataset_size, configs, logger):
    """Train a single model on ``train_idx`` and evaluate it on the complement."""
    model_name = configs["train"]["model_name"]
    dataset_name = configs["data"]["dataset"]
    batch_size = configs["train"]["batch_size"]
    device = configs["train"]["device"]

    test_idx = np.setdiff1d(np.arange(dataset_size), train_idx)
    train_loader = get_dataloader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
    )
    test_loader = get_dataloader(Subset(dataset, test_idx), batch_size=batch_size)

    logger.info(
        "Training %s: train size %d, test size %d",
        role,
        len(train_idx),
        len(test_idx),
    )
    model = train(
        get_model(model_name, dataset_name, configs),
        train_loader,
        configs["train"],
        test_loader,
    )
    train_loss, train_acc = inference(model, train_loader, device)
    test_loss, test_acc = inference(model, test_loader, device)
    logger.info(
        "%s: train acc %.4f, test acc %.4f", role, train_acc, test_acc
    )
    metadata = {
        "role": role,
        "num_train": int(len(train_idx)),
        "optimizer": configs["train"]["optimizer"],
        "batch_size": batch_size,
        "epochs": configs["train"]["epochs"],
        "model_name": model_name,
        "learning_rate": configs["train"]["learning_rate"],
        "weight_decay": configs["train"]["weight_decay"],
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "dataset": dataset_name,
    }
    return model, metadata


def prepare_urmia_models(
    models_dir: str,
    dataset,
    splits: dict,
    configs: dict,
    logger: logging.Logger,
) -> dict:
    """Train (or load) all U-RMIA models, resuming from disk where possible.

    Follows the fork's resumable-training convention: skip any role whose ``.pkl``
    and metadata entry already exist, write metadata after each model, and free
    GPU memory between models.

    Args:
        models_dir (str): ``<log_dir>/models`` directory.
        dataset: The training-pool dataset.
        splits (dict): Output of :func:`get_or_create_splits`.
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object for the current run.

    Returns:
        dict: Mapping role name -> model, plus the accumulated metadata dict under
            the special key ``"__metadata__"``.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    metadata_path = f"{models_dir}/{METADATA_FILE}"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
    else:
        metadata_dict = {}

    model_name = configs["train"]["model_name"]
    dataset_name = configs["data"]["dataset"]
    device = configs["train"]["device"]
    dataset_size = len(dataset)

    def _persist(role, model, metadata):
        with open(f"{models_dir}/{role}.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)
        metadata_dict[role] = metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=4)

    def _load(role):
        model = get_model(model_name, dataset_name, configs)
        with open(f"{models_dir}/{role}.pkl", "rb") as f:
            model.load_state_dict(pickle.load(f))
        return model

    models = {}
    role_indices = _role_train_indices(splits)

    # 1) original, retrained, reference_k (trained from scratch).
    for role, train_idx in role_indices.items():
        pkl_path = f"{models_dir}/{role}.pkl"
        if role in metadata_dict and os.path.exists(pkl_path):
            logger.info("Model %s already trained, loading from disk", role)
            models[role] = _load(role)
            continue
        model, metadata = _train_one(
            role, train_idx, dataset, dataset_size, configs, logger
        )
        # Gold-standard forget accuracy on the retrained model, for the summary.
        if role == "retrained":
            forget_loader = get_dataloader(
                Subset(dataset, splits["forget_indices"]),
                batch_size=configs["train"]["batch_size"],
            )
            forget_loss, forget_acc = inference(model, forget_loader, device)
            metadata["forget_acc"] = forget_acc
            metadata["forget_loss"] = forget_loss
        _persist(role, model, metadata)
        models[role] = copy.deepcopy(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 2) unlearned (derived from original).
    role = "unlearned"
    pkl_path = f"{models_dir}/{role}.pkl"
    current_params = configs["unlearn"].get("params", {})
    if role in metadata_dict and os.path.exists(pkl_path):
        stored_params = metadata_dict[role].get("unlearn_params", {})
        if stored_params != current_params:
            logger.warning(
                "Cached unlearned model was produced with params %s but config has "
                "%s. Reusing the cached model; delete %s (and its signals) or change "
                "log_dir to regenerate.",
                stored_params,
                current_params,
                pkl_path,
            )
        logger.info("Model %s already trained, loading from disk", role)
        models[role] = _load(role)
    else:
        forget_loader = get_dataloader(
            Subset(dataset, splits["forget_indices"]),
            batch_size=configs["train"]["batch_size"],
            shuffle=True,
        )
        retain_loader = get_dataloader(
            Subset(dataset, splits["retain_indices"]),
            batch_size=configs["train"]["batch_size"],
            shuffle=True,
        )
        unlearn_configs = dict(configs["unlearn"])
        unlearn_configs.setdefault("seed", configs["run"]["random_seed"])
        unlearner = get_unlearner(configs["unlearn"]["algorithm"])
        model = unlearner(
            models["original"],
            forget_loader,
            retain_loader,
            unlearn_configs,
            configs["train"],
            logger,
        )
        # Diagnostics that make the summary interpretable.
        eval_forget = get_dataloader(
            Subset(dataset, splits["forget_indices"]),
            batch_size=configs["train"]["batch_size"],
        )
        eval_retain = get_dataloader(
            Subset(dataset, splits["retain_indices"]),
            batch_size=configs["train"]["batch_size"],
        )
        eval_unseen = get_dataloader(
            Subset(dataset, splits["unseen_indices"]),
            batch_size=configs["train"]["batch_size"],
        )
        forget_loss, forget_acc = inference(model, eval_forget, device)
        retain_loss, retain_acc = inference(model, eval_retain, device)
        unseen_loss, unseen_acc = inference(model, eval_unseen, device)
        logger.info(
            "%s: forget acc %.4f, retain acc %.4f, unseen acc %.4f",
            role,
            forget_acc,
            retain_acc,
            unseen_acc,
        )
        metadata = {
            "role": role,
            "model_name": model_name,
            "dataset": dataset_name,
            "unlearn_algorithm": configs["unlearn"]["algorithm"],
            "unlearn_params": current_params,
            "forget_acc": forget_acc,
            "forget_loss": forget_loss,
            "retain_acc": retain_acc,
            "retain_loss": retain_loss,
            "test_acc": unseen_acc,
            "test_loss": unseen_loss,
        }
        _persist(role, model, metadata)
        models[role] = copy.deepcopy(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    models["__metadata__"] = metadata_dict
    return models


def get_urmia_signals(
    models_list: list,
    subset,
    configs: dict,
    logger: logging.Logger,
    is_population: bool = False,
) -> np.ndarray:
    """Compute true-class softmax signals for every model on ``subset``.

    Cached to ``<log_dir>/signals/urmia_signals[.npy|_pop.npy]``. The cache is
    reused only when both the row count (number of audited samples) and the
    column count (3 + K models) match, so bumping ``num_ref_models`` correctly
    forces a recompute.

    Args:
        models_list (list): Ordered models [original, unlearned, retrained, ref_0..].
        subset: Dataset subset to evaluate (audit samples or population subsample).
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object for the current run.
        is_population (bool): Whether these are population signals.

    Returns:
        np.ndarray: Signals of shape (len(subset), len(models_list)).
    """
    signal_dir = f"{configs['run']['log_dir']}/signals"
    Path(signal_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_pop.npy" if is_population else ".npy"
    signal_path = f"{signal_dir}/urmia_signals{suffix}"

    expected_rows = len(subset)
    expected_cols = len(models_list)
    if os.path.exists(signal_path):
        cached = np.load(signal_path)
        if cached.shape == (expected_rows, expected_cols):
            logger.info("U-RMIA signals loaded from disk (%s).", signal_path)
            return cached
        logger.warning(
            "Cached signals shape %s does not match expected (%d, %d); recomputing.",
            cached.shape,
            expected_rows,
            expected_cols,
        )

    batch_size = configs["audit"]["batch_size"]
    model_name = configs["train"]["model_name"]
    device = configs["audit"]["device"]

    data, targets = load_dataset_subsets(
        subset, np.arange(len(subset)), model_name, batch_size, device
    )

    logger.info("Computing U-RMIA signals for %d models.", expected_cols)
    signals = [
        get_softmax(model, data, targets, batch_size, device) for model in models_list
    ]
    signals = np.concatenate(signals, axis=1)
    np.save(signal_path, signals)
    logger.info("U-RMIA signals saved to disk (%s).", signal_path)
    return signals


def report_urmia_attack(
    report_dir: str,
    role: str,
    mia_scores: np.ndarray,
    memberships: np.ndarray,
    logger: logging.Logger,
) -> dict:
    """Compute and save the attack report for a single target role.

    Mirrors ``audit.get_audit_results`` but names artifacts by role and logs with
    a string identifier (the shared helper formats the id with ``%d``).

    Args:
        report_dir (str): Folder for saving ROC plots and the result archive.
        role (str): Target role name (original/unlearned/retrained).
        mia_scores (np.ndarray): RMIA scores for the audited samples.
        memberships (np.ndarray): Ground-truth labels (1 forget, 0 unseen).
        logger (logging.Logger): Logger object for the current run.

    Returns:
        dict: Attack results (fpr, tpr, auc, one_fpr, one_tenth_fpr, zero_fpr).
    """
    attack_result = compute_attack_results(mia_scores, memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Target %s: AUC %.4f, TPR@1%%FPR %.4f, TPR@0.1%%FPR %.4f, TPR@0.0%%FPR %.4f",
        role,
        attack_result["auc"],
        attack_result["one_fpr"],
        attack_result["one_tenth_fpr"],
        attack_result["zero_fpr"],
    )

    plot_roc(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_{role}.png",
    )
    plot_roc_log(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_log_{role}.png",
    )
    np.savez(
        f"{report_dir}/attack_result_{role}",
        fpr=attack_result["fpr"],
        tpr=attack_result["tpr"],
        auc=attack_result["auc"],
        one_fpr=attack_result["one_fpr"],
        one_tenth_fpr=attack_result["one_tenth_fpr"],
        zero_fpr=attack_result["zero_fpr"],
        scores=mia_scores.ravel(),
        memberships=memberships.ravel(),
    )
    return attack_result


def log_urmia_summary(
    report_dir: str,
    results: dict,
    metadata: dict,
    configs: dict,
    logger: logging.Logger,
) -> None:
    """Log and save the comparison across original / unlearned / retrained.

    Args:
        report_dir (str): ``<log_dir>/report`` directory.
        results (dict): Role -> attack-result dict from :func:`report_urmia_attack`.
        metadata (dict): The models metadata dict (accuracies per role).
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object for the current run.
    """
    header = (
        f"{'role':<10} {'AUC':>8} {'2sided':>8} {'TPR@1%':>8} {'TPR@.1%':>8} "
        f"{'TPR@0%':>8} {'test_acc':>9} {'forget_acc':>11}"
    )
    logger.info("U-RMIA summary:")
    logger.info(header)

    summary = {"unlearn": configs["unlearn"], "roles": {}}
    for role in TARGET_ROLES:
        res = results[role]
        meta = metadata.get(role, {})
        test_acc = meta.get("test_acc")
        forget_acc = meta.get("forget_acc")
        # Two-sided AUC: AUC < 0.5 means the attack separates the classes in the
        # inverted direction (the over-unlearning signature). max(auc, 1-auc)
        # measures leakage regardless of direction.
        two_sided = max(res["auc"], 1.0 - res["auc"])
        logger.info(
            "%-10s %8.4f %8.4f %8.4f %8.4f %8.4f %9s %11s%s",
            role,
            res["auc"],
            two_sided,
            res["one_fpr"],
            res["one_tenth_fpr"],
            res["zero_fpr"],
            f"{test_acc:.4f}" if test_acc is not None else "n/a",
            f"{forget_acc:.4f}" if forget_acc is not None else "n/a",
            "  <-- AUC<0.5: likely over-unlearning" if res["auc"] < 0.45 else "",
        )
        summary["roles"][role] = {
            "auc": float(res["auc"]),
            "two_sided_auc": float(two_sided),
            "one_fpr": float(res["one_fpr"]),
            "one_tenth_fpr": float(res["one_tenth_fpr"]),
            "zero_fpr": float(res["zero_fpr"]),
            "test_acc": test_acc,
            "forget_acc": forget_acc,
        }

    logger.info(
        "Interpretation: 'original' AUC is the attack upper bound; 'retrained' AUC is "
        "the floor (~0.5); 'unlearned' between them measures residual leakage. Read AUC "
        "two-sided: AUC well below 0.5 also means leakage (over-unlearning gives forget "
        "points an abnormally-low-confidence signature). And watch test_acc: an "
        "unlearner that only moves AUC toward 0.5 by wrecking accuracy is not "
        "protecting privacy. NOTE: this offline (level-2) attack is one-sided and "
        "poorly calibrated for over-unlearning; the level-3 attack is the honest test."
    )

    with open(f"{report_dir}/urmia_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

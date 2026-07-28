"""Helpers for level-3 (strong, unlearning-aware) U-RMIA.

Level 2 (``urmia_utils.py``) trains reference models on halves of the retain set
R, so every audited point is OUT of every reference and the "member" side is
guessed via ``offline_a``. That attack is one-sided and can be fooled by
over-unlearning.

Level 3 instead builds **train-then-unlearn** reference models: each audited point
is trained-then-unlearned (IN) for some references and never-trained (OUT) for
others, giving the attack an empirical picture of what an unlearned point actually
looks like. This module builds those references (plus the same three target models
as level 2) and the online summary. Scoring lives in
``modules/mia/attacks/urmia.py``.

Nothing here modifies existing files; level-2 public helpers are imported and
reused (``get_urmia_signals``, ``report_urmia_attack``, ``check_urmia_configs``,
``TARGET_ROLES``, ``REF_COL_START``).
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

from dataset.utils import get_dataloader
from models.utils import get_model
from trainers.default_trainer import train, inference
from unlearners import get_unlearner
from urmia_utils import TARGET_ROLES, check_urmia_configs

ONLINE_SPLITS_FILE = "urmia_online_splits.npz"
ONLINE_METADATA_FILE = "urmia_online_models_metadata.json"

# Attack labels used for report artifacts and the summary.
ATTACKS = ["offline", "rmia_online", "ulira"]


def check_online_configs(configs: dict, dataset_size: int) -> None:
    """Validate config for the strong run (reuses the level-2 checks).

    Args:
        configs (dict): Full config dictionary.
        dataset_size (int): Size of the training-pool dataset.
    """
    check_urmia_configs(configs, dataset_size)
    num_ref_models = int(configs["audit"]["num_ref_models"])
    if num_ref_models < 2:
        raise ValueError(
            "Level-3 U-RMIA needs num_ref_models >= 2 (each audited sample must be "
            "IN for some references and OUT for others)."
        )
    if num_ref_models < 8:
        logging.getLogger("time_analysis").warning(
            "num_ref_models=%d is small for the level-3 attack; the U-LiRA Gaussian "
            "fits will be noisy. K>=8 is recommended (paper used 128).",
            num_ref_models,
        )


def get_or_create_online_splits(
    models_dir: str,
    dataset_size: int,
    pop_len: int,
    configs: dict,
    logger: logging.Logger,
) -> dict:
    """Load or generate the level-3 splits and per-reference IN/OUT assignment.

    Persisted to ``<models_dir>/urmia_online_splits.npz``. F/U/R are drawn first
    (so they match the level-2 seed scheme). Each reference gets a random half of
    R plus a ~50/50 IN/OUT assignment over the audit pool A = [F; U], with at
    least one IN and one OUT guaranteed.

    Returns:
        dict: forget_indices, unseen_indices, retain_indices, population_indices,
            audit_indices (A, ordered F then U), ref_retain_membership (K, N) bool,
            ref_unlearn_membership (K, 2*forget_size) bool.

    Raises:
        ValueError: On config/split mismatch, or population_size > pop_len.
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

    splits_path = f"{models_dir}/{ONLINE_SPLITS_FILE}"
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
                f"Persisted online splits at {splits_path} disagree with the current "
                f"config (stored {stored} vs config {expected}). Change run.log_dir to "
                f"start a fresh experiment."
            )
        logger.info("Loaded level-3 U-RMIA splits from disk.")
        return {
            "forget_indices": data["forget_indices"],
            "unseen_indices": data["unseen_indices"],
            "retain_indices": data["retain_indices"],
            "population_indices": data["population_indices"],
            "audit_indices": data["audit_indices"],
            "ref_retain_membership": data["ref_retain_membership"],
            "ref_unlearn_membership": data["ref_unlearn_membership"],
        }

    rng = np.random.default_rng(seed)
    perm = rng.permutation(dataset_size)
    forget_indices = perm[:forget_size]
    unseen_indices = perm[forget_size : 2 * forget_size]
    retain_indices = perm[2 * forget_size :]
    audit_indices = np.concatenate([forget_indices, unseen_indices])
    audit_size = len(audit_indices)

    ref_retain_membership = np.zeros((num_ref_models, dataset_size), dtype=bool)
    ref_unlearn_membership = np.zeros((num_ref_models, audit_size), dtype=bool)
    half = len(retain_indices) // 2
    for k in range(num_ref_models):
        chosen = rng.permutation(retain_indices)[:half]
        ref_retain_membership[k, chosen] = True
        assignment = rng.random(audit_size) < 0.5
        # Guarantee at least one IN and one OUT so both Gaussians/means exist.
        if not assignment.any():
            assignment[rng.integers(audit_size)] = True
        if assignment.all():
            assignment[rng.integers(audit_size)] = False
        ref_unlearn_membership[k, :] = assignment

    population_indices = rng.choice(pop_len, size=population_size, replace=False)

    np.savez(
        splits_path,
        forget_indices=forget_indices,
        unseen_indices=unseen_indices,
        retain_indices=retain_indices,
        population_indices=population_indices,
        audit_indices=audit_indices,
        ref_retain_membership=ref_retain_membership,
        ref_unlearn_membership=ref_unlearn_membership,
        seed=seed,
        forget_size=forget_size,
        dataset_size=dataset_size,
        num_ref_models=num_ref_models,
        population_size=population_size,
    )
    logger.info(
        "Created level-3 splits: |F|=%d, |U|=%d, |R|=%d, K=%d, |pop|=%d",
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
        "population_indices": population_indices,
        "audit_indices": audit_indices,
        "ref_retain_membership": ref_retain_membership,
        "ref_unlearn_membership": ref_unlearn_membership,
    }


def _train_model(role, train_idx, dataset, dataset_size, configs, logger):
    """Train one model on ``train_idx`` and evaluate it on the complement."""
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
        "Training %s: train size %d, test size %d", role, len(train_idx), len(test_idx)
    )
    model = train(
        get_model(model_name, dataset_name, configs),
        train_loader,
        configs["train"],
        test_loader,
    )
    train_loss, train_acc = inference(model, train_loader, device)
    test_loss, test_acc = inference(model, test_loader, device)
    logger.info("%s: train acc %.4f, test acc %.4f", role, train_acc, test_acc)
    metadata = {
        "role": role,
        "num_train": int(len(train_idx)),
        "model_name": model_name,
        "epochs": configs["train"]["epochs"],
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "dataset": dataset_name,
    }
    return model, metadata


class _MetadataStore:
    """Small helper for the resumable per-model metadata + pkl on disk."""

    def __init__(self, models_dir, model_name, dataset_name, configs):
        self.models_dir = models_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.configs = configs
        self.path = f"{models_dir}/{ONLINE_METADATA_FILE}"
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

    def has(self, role):
        return role in self.meta and os.path.exists(f"{self.models_dir}/{role}.pkl")

    def load(self, role):
        model = get_model(self.model_name, self.dataset_name, self.configs)
        with open(f"{self.models_dir}/{role}.pkl", "rb") as f:
            model.load_state_dict(pickle.load(f))
        return model

    def persist(self, role, model, metadata):
        with open(f"{self.models_dir}/{role}.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)
        self.meta[role] = metadata
        with open(self.path, "w") as f:
            json.dump(self.meta, f, indent=4)


def _unlearn(model, forget_idx, retain_idx, dataset, configs, logger):
    """Apply the configured unlearner to ``model`` given forget/retain indices."""
    batch_size = configs["train"]["batch_size"]
    forget_loader = get_dataloader(
        Subset(dataset, forget_idx), batch_size=batch_size, shuffle=True
    )
    retain_loader = get_dataloader(
        Subset(dataset, retain_idx), batch_size=batch_size, shuffle=True
    )
    unlearn_configs = dict(configs["unlearn"])
    unlearn_configs.setdefault("seed", configs["run"]["random_seed"])
    unlearner = get_unlearner(configs["unlearn"]["algorithm"])
    return unlearner(
        model, forget_loader, retain_loader, unlearn_configs, configs["train"], logger
    )


def prepare_online_target_models(models_dir, dataset, splits, configs, logger):
    """Train (or load) original / retrained / unlearned target models. Resumable."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    store = _MetadataStore(
        models_dir,
        configs["train"]["model_name"],
        configs["data"]["dataset"],
        configs,
    )
    device = configs["train"]["device"]
    dataset_size = len(dataset)
    forget = splits["forget_indices"]
    retain = splits["retain_indices"]

    role_indices = {
        "original": np.concatenate([retain, forget]),
        "retrained": retain,
    }

    models = {}
    for role, train_idx in role_indices.items():
        if store.has(role):
            logger.info("Model %s already trained, loading from disk", role)
            models[role] = store.load(role)
            continue
        model, metadata = _train_model(
            role, train_idx, dataset, dataset_size, configs, logger
        )
        if role == "retrained":
            forget_loader = get_dataloader(
                Subset(dataset, forget), batch_size=configs["train"]["batch_size"]
            )
            metadata["forget_loss"], metadata["forget_acc"] = inference(
                model, forget_loader, device
            )
        store.persist(role, model, metadata)
        models[role] = copy.deepcopy(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # unlearned: derived from original.
    role = "unlearned"
    if store.has(role):
        logger.info("Model %s already trained, loading from disk", role)
        models[role] = store.load(role)
    else:
        model = _unlearn(models["original"], forget, retain, dataset, configs, logger)
        eval_bs = configs["train"]["batch_size"]
        f_loss, f_acc = inference(
            model, get_dataloader(Subset(dataset, forget), batch_size=eval_bs), device
        )
        r_loss, r_acc = inference(
            model, get_dataloader(Subset(dataset, retain), batch_size=eval_bs), device
        )
        u_loss, u_acc = inference(
            model,
            get_dataloader(Subset(dataset, splits["unseen_indices"]), batch_size=eval_bs),
            device,
        )
        logger.info(
            "unlearned: forget acc %.4f, retain acc %.4f, unseen acc %.4f",
            f_acc,
            r_acc,
            u_acc,
        )
        metadata = {
            "role": role,
            "model_name": configs["train"]["model_name"],
            "dataset": configs["data"]["dataset"],
            "unlearn_algorithm": configs["unlearn"]["algorithm"],
            "unlearn_params": configs["unlearn"].get("params", {}),
            "forget_acc": f_acc,
            "forget_loss": f_loss,
            "retain_acc": r_acc,
            "retain_loss": r_loss,
            "test_acc": u_acc,
            "test_loss": u_loss,
        }
        store.persist(role, model, metadata)
        models[role] = copy.deepcopy(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return models, store.meta


def prepare_online_reference_models(models_dir, dataset, splits, configs, logger):
    """Train-then-unlearn each reference model. Resumable.

    Each reference k trains on its retain chunk plus the audit points assigned IN
    for k, then unlearns those IN points with the same unlearner as the target.

    Returns:
        list: The K unlearned reference models, ordered by index.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    store = _MetadataStore(
        models_dir,
        configs["train"]["model_name"],
        configs["data"]["dataset"],
        configs,
    )
    dataset_size = len(dataset)
    audit_indices = splits["audit_indices"]
    ref_retain = splits["ref_retain_membership"]
    ref_unlearn = splits["ref_unlearn_membership"]
    num_ref_models = ref_retain.shape[0]

    refs = []
    for k in range(num_ref_models):
        role = f"online_reference_{k}"
        if store.has(role):
            logger.info("Model %s already trained, loading from disk", role)
            refs.append(store.load(role))
            continue

        retain_idx = np.where(ref_retain[k])[0]
        in_idx = audit_indices[ref_unlearn[k]]
        train_idx = np.concatenate([retain_idx, in_idx])

        model, metadata = _train_model(
            role, train_idx, dataset, dataset_size, configs, logger
        )
        logger.info(
            "%s: unlearning %d IN audit points (train-then-unlearn)", role, len(in_idx)
        )
        model = _unlearn(model, in_idx, retain_idx, dataset, configs, logger)
        metadata["num_in_unlearned"] = int(len(in_idx))
        metadata["unlearn_algorithm"] = configs["unlearn"]["algorithm"]
        store.persist(role, model, metadata)
        refs.append(copy.deepcopy(model))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return refs


def log_online_summary(report_dir, results, metadata, configs, logger):
    """Log and save the naive-vs-strong comparison across all targets and scorers.

    Args:
        report_dir (str): ``<log_dir>/report`` directory.
        results (dict): results[role][attack] -> attack-result dict.
        metadata (dict): Target-model metadata (accuracies).
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object.
    """
    logger.info("Level-3 U-RMIA summary (AUC | two-sided AUC per scorer):")
    header = f"{'role':<10}"
    for attack in ATTACKS:
        header += f" {attack + '_auc':>16} {attack + '_2s':>10}"
    header += f" {'test_acc':>9} {'forget_acc':>11}"
    logger.info(header)

    summary = {"unlearn": configs["unlearn"], "roles": {}}
    for role in TARGET_ROLES:
        meta = metadata.get(role, {})
        test_acc = meta.get("test_acc")
        forget_acc = meta.get("forget_acc")
        line = f"{role:<10}"
        role_summary = {"test_acc": test_acc, "forget_acc": forget_acc, "attacks": {}}
        flagged = False
        for attack in ATTACKS:
            res = results[role][attack]
            auc = res["auc"]
            two_sided = max(auc, 1.0 - auc)
            if attack == "offline" and auc < 0.45:
                flagged = True
            line += f" {auc:>16.4f} {two_sided:>10.4f}"
            role_summary["attacks"][attack] = {
                "auc": float(auc),
                "two_sided_auc": float(two_sided),
                "one_fpr": float(res["one_fpr"]),
                "one_tenth_fpr": float(res["one_tenth_fpr"]),
                "zero_fpr": float(res["zero_fpr"]),
            }
        line += f" {test_acc:>9.4f}" if test_acc is not None else f" {'n/a':>9}"
        line += f" {forget_acc:>11.4f}" if forget_acc is not None else f" {'n/a':>11}"
        if flagged:
            line += "  <-- offline AUC<0.45: over-unlearning"
        logger.info(line)
        summary["roles"][role] = role_summary

    logger.info(
        "Interpretation: on 'unlearned', if the offline (naive) AUC sags toward/below "
        "0.5 while rmia_online and ulira two-sided AUC stay elevated, the naive attack "
        "is giving a false sense of privacy that the strong attacks see through. Read "
        "alongside test_acc: targeted unlearners keep accuracy high (unlike global "
        "weight noise)."
    )

    with open(f"{report_dir}/urmia_online_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

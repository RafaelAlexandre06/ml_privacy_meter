"""Entry point for U-RMIA: RMIA adapted to machine-unlearning evaluation.

U-RMIA audits whether an unlearning algorithm actually removes the influence of
a forget set. It runs the RMIA membership inference attack to distinguish
forget samples (trained then unlearned) from unseen samples (never trained), and
compares three target models: the original, the unlearned, and a
retrain-from-scratch baseline. See ``urmia_utils.py`` for details.

Usage:
    python run_urmia.py --cf configs/urmia/cifar10.yaml
"""

import argparse
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from get_signals import get_softmax  # noqa: F401  (kept for parity/imports check)
from modules.mia.attacks.rmia import run_rmia
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
)
from urmia_utils import (
    TARGET_ROLES,
    REF_COL_START,
    check_urmia_configs,
    get_or_create_splits,
    prepare_urmia_models,
    get_urmia_signals,
    report_urmia_attack,
    log_urmia_summary,
)
from retain_leakage import (
    select_retain_audit,
    retain_ref_in,
    compute_retain_leakage,
)

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True


def main():
    print(20 * "-")
    print("U-RMIA: Unlearning Membership Inference (RMIA)")
    print(20 * "-")

    parser = argparse.ArgumentParser(description="Run U-RMIA unlearning audit.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/urmia/cifar10.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    check_configs(configs)
    initialize_seeds(configs["run"]["random_seed"])

    log_dir = configs["run"]["log_dir"]
    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "signal_dir": f"{log_dir}/signals",
        "models_dir": f"{log_dir}/models",
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    start_time = time.time()

    # Stage 1: dataset (and population pool).
    baseline_time = time.time()
    dataset, population = load_dataset(configs, directories["data_dir"], logger)
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)

    check_urmia_configs(configs, len(dataset))

    # Stage 2: F/U/R and reference splits.
    splits = get_or_create_splits(
        directories["models_dir"], len(dataset), len(population), configs, logger
    )

    # Stage 3: train or load all models.
    baseline_time = time.time()
    models = prepare_urmia_models(
        directories["models_dir"], dataset, splits, configs, logger
    )
    metadata = models.pop("__metadata__")
    num_ref_models = int(configs["audit"]["num_ref_models"])
    ordered_models = [models[role] for role in TARGET_ROLES] + [
        models[f"reference_{k}"] for k in range(num_ref_models)
    ]
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    # Stage 4: signals. Audit rows are F (label 1) then U (label 0).
    baseline_time = time.time()
    forget_indices = splits["forget_indices"]
    unseen_indices = splits["unseen_indices"]
    audit_subset = Subset(
        dataset, np.concatenate([forget_indices, unseen_indices])
    )
    memberships = np.concatenate(
        [np.ones(len(forget_indices), dtype=bool), np.zeros(len(unseen_indices), dtype=bool)]
    )
    population_subset = Subset(population, splits["population_indices"])

    signals = get_urmia_signals(ordered_models, audit_subset, configs, logger)
    pop_signals = get_urmia_signals(
        ordered_models, population_subset, configs, logger, is_population=True
    )
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Stage 5: attack each target and report.
    baseline_time = time.time()
    offline_a = configs["audit"].get("offline_a", 0.3)
    ref_slice = slice(REF_COL_START, REF_COL_START + num_ref_models)
    ref_signals = signals[:, ref_slice]
    # Forget and unseen samples are OUT of every reference model.
    ref_memberships = np.zeros_like(ref_signals, dtype=bool)
    z_ref_signals = pop_signals[:, ref_slice]

    report_dir_exp = f"{directories['report_dir']}/exp"
    results = {}
    for col, role in enumerate(TARGET_ROLES):
        mia_scores = run_rmia(
            signals[:, col],
            ref_signals,
            ref_memberships,
            pop_signals[:, col],
            z_ref_signals,
            offline_a,
            num_reference_models=num_ref_models,
        )
        results[role] = report_urmia_attack(
            report_dir_exp, role, mia_scores, memberships, logger
        )
    logger.info("Auditing took %0.1f seconds", time.time() - baseline_time)

    # Stage 6 (optional): retain-set leakage (Hayes et al. 2024). Reuses the same
    # models and references; needs only one signal pass over a retain sample.
    retain_analysis = None
    retain_audit_size = configs["audit"].get("retain_audit_size")
    if retain_audit_size:
        baseline_time = time.time()
        retain_audit_idx = select_retain_audit(
            splits["retain_indices"], retain_audit_size, configs["run"]["random_seed"]
        )
        retain_signals = get_urmia_signals(
            ordered_models,
            Subset(dataset, retain_audit_idx),
            configs,
            logger,
            signal_tag="_retain",
        )
        ref_in = retain_ref_in(splits, retain_audit_idx)
        unseen_block = signals[len(forget_indices):]
        retain_analysis = compute_retain_leakage(
            retain_signals, ref_in, unseen_block, pop_signals, offline_a
        )
        logger.info(
            "Retain-leakage analysis took %0.1f seconds", time.time() - baseline_time
        )

    log_urmia_summary(
        directories["report_dir"], results, metadata, configs, logger, retain_analysis
    )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()

"""Entry point for level-3 (strong, unlearning-aware) U-RMIA.

Builds train-then-unlearn reference models and attacks the original / unlearned /
retrained targets with three scorers on a single shared reference set:

- ``offline``     : naive one-sided RMIA (per-sample OUT masking) — the level-2
                    style attack that can be fooled by over-unlearning.
- ``rmia_online`` : RMIA-family strong attack with empirical p(x).
- ``ulira``       : LiRA-style two-sided likelihood attack (the paper's method).

The summary shows all three side by side so the naive-vs-strong contrast is
visible in one run. Does not modify any existing file.

Usage:
    python run_urmia_online.py --cf configs/urmia/cifar10_online.yaml
"""

import argparse
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from modules.mia.attacks.urmia import (
    run_rmia_offline_masked,
    run_urmia_online,
    run_ulira,
)
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
)
from urmia_utils import TARGET_ROLES, REF_COL_START, get_urmia_signals, report_urmia_attack
from urmia_online_utils import (
    check_online_configs,
    get_or_create_online_splits,
    prepare_online_target_models,
    prepare_online_reference_models,
    log_online_summary,
)

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True


def main():
    print(20 * "-")
    print("U-RMIA level 3: strong, unlearning-aware attack")
    print(20 * "-")

    parser = argparse.ArgumentParser(description="Run strong (level-3) U-RMIA.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/urmia/cifar10_online.yaml",
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

    # Stage 1: dataset + population pool.
    baseline_time = time.time()
    dataset, population = load_dataset(configs, directories["data_dir"], logger)
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)

    check_online_configs(configs, len(dataset))

    # Stage 2: level-3 splits (F/U/R + per-reference IN/OUT assignment).
    splits = get_or_create_online_splits(
        directories["models_dir"], len(dataset), len(population), configs, logger
    )

    # Stage 3: target models + train-then-unlearn references.
    baseline_time = time.time()
    targets, target_metadata = prepare_online_target_models(
        directories["models_dir"], dataset, splits, configs, logger
    )
    online_refs = prepare_online_reference_models(
        directories["models_dir"], dataset, splits, configs, logger
    )
    ordered_models = [targets[role] for role in TARGET_ROLES] + online_refs
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    # Stage 4: signals. Audit rows are F (label 1) then U (label 0).
    baseline_time = time.time()
    forget_indices = splits["forget_indices"]
    unseen_indices = splits["unseen_indices"]
    audit_subset = Subset(dataset, np.concatenate([forget_indices, unseen_indices]))
    memberships = np.concatenate(
        [np.ones(len(forget_indices), dtype=bool), np.zeros(len(unseen_indices), dtype=bool)]
    )
    population_subset = Subset(population, splits["population_indices"])

    signals = get_urmia_signals(ordered_models, audit_subset, configs, logger)
    pop_signals = get_urmia_signals(
        ordered_models, population_subset, configs, logger, is_population=True
    )
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Stage 5: attack every target with all three scorers.
    baseline_time = time.time()
    num_ref_models = int(configs["audit"]["num_ref_models"])
    offline_a = configs["audit"].get("offline_a", 0.3)
    ref_slice = slice(REF_COL_START, REF_COL_START + num_ref_models)
    ref_signals = signals[:, ref_slice]
    z_ref_signals = pop_signals[:, ref_slice]
    # (num_audit_samples, K): True = sample was trained-then-unlearned in that ref.
    ref_in = splits["ref_unlearn_membership"].T

    report_dir_exp = f"{directories['report_dir']}/exp"
    results = {}
    for col, role in enumerate(TARGET_ROLES):
        target = signals[:, col]
        z_target = pop_signals[:, col]
        scored = {
            "offline": run_rmia_offline_masked(
                target, ref_signals, ref_in, z_target, z_ref_signals, offline_a
            ),
            "rmia_online": run_urmia_online(
                target, ref_signals, ref_in, z_target, z_ref_signals, offline_a
            ),
            "ulira": run_ulira(target, ref_signals, ref_in),
        }
        results[role] = {
            attack: report_urmia_attack(
                report_dir_exp, f"{role}_{attack}", mia_scores, memberships, logger
            )
            for attack, mia_scores in scored.items()
        }
    logger.info("Auditing took %0.1f seconds", time.time() - baseline_time)

    log_online_summary(
        directories["report_dir"], results, target_metadata, configs, logger
    )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()

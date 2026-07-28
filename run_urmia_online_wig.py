"""Entry point for level-3 U-RMIA with the WIG unlearner (Lee et al., PoPETs 2026).

Identical pipeline to ``run_urmia_online.py`` -- level-3 splits, targets,
train-then-unlearn references, six scorers -- but defaults to the WIG cluster
config. WIG (see ``wig_unlearner.py``) re-initializes the weights whose
gradient distributions overlap most between retain and forget data, then
fine-tunes on the retain set. It is *structural* rather than output-targeting,
so the paper predicts uniformly small attack gaps across statistics, unlike
``random_label``/``neggrad_plus`` which defend one property and leak through
another. With ``audit.entropy_signal: true`` the summary shows exactly that
contrast: compare each scorer's ``unlearned`` AUC against the ``retrained``
control row --

- biased unlearner: small gap on its defended statistic (e.g. ``ulira`` for
  random_label), large gap on the other (``ulira_mentr``);
- WIG working as claimed: small gaps everywhere, with ``test_acc`` near the
  original confirming the fine-tune repaired utility.

Scorers per target (original / unlearned / retrained):

- ``offline``      : naive one-sided RMIA (per-sample OUT masking).
- ``rmia_online``  : RMIA-family strong attack with empirical p(x).
- ``ulira``        : LiRA-style two-sided likelihood attack.
- ``conf_global``  : global threshold on true-class confidence.
- ``mentr_global`` : global threshold on modified entropy.
- ``ulira_mentr``  : U-LiRA on modified entropy.

Usage:
    python run_urmia_online_wig.py --cf configs/urmia/cifar10_online_wig.yaml
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
    run_global_threshold,
    log_mentr,
)
from urmia_entropy_signals import get_urmia_entropy_signals
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
)
from urmia_utils import TARGET_ROLES, REF_COL_START, get_urmia_signals, report_urmia_attack
from urmia_online_utils import (
    ATTACKS,
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
    print("U-RMIA level 3 with the WIG unlearner")
    print(20 * "-")

    parser = argparse.ArgumentParser(description="Run strong (level-3) U-RMIA with WIG.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/urmia/cifar10_online_wig.yaml",
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

    # Stage 3: target models + train-then-unlearn references. WIG runs K+1
    # times here (once per reference, once for the unlearned target): each run
    # is one gradient-stats pass plus params.epochs of retain fine-tuning.
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

    # Optional second statistic: Song & Mittal modified entropy. Needs its own
    # forward pass (the confidence cache keeps only the true-class column), but
    # only over the audit subset -- neither scorer that consumes it uses the
    # population z-samples.
    use_entropy = bool(configs["audit"].get("entropy_signal", False))
    mentr_signals = (
        get_urmia_entropy_signals(ordered_models, audit_subset, configs, logger)
        if use_entropy
        else None
    )
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Stage 5: attack every target with all scorers.
    baseline_time = time.time()
    num_ref_models = int(configs["audit"]["num_ref_models"])
    offline_a = configs["audit"].get("offline_a", 0.3)
    ref_slice = slice(REF_COL_START, REF_COL_START + num_ref_models)
    ref_signals = signals[:, ref_slice]
    z_ref_signals = pop_signals[:, ref_slice]
    # (num_audit_samples, K): True = sample was trained-then-unlearned in that ref.
    ref_in = splits["ref_unlearn_membership"].T

    attacks = list(ATTACKS)
    if use_entropy:
        # conf_global is free (no new signals) and is the control that separates
        # "better statistic" from "better calibration" in the summary.
        attacks += ["conf_global", "mentr_global", "ulira_mentr"]
        mentr_ref_signals = mentr_signals[:, ref_slice]

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
        if use_entropy:
            mentr_target = mentr_signals[:, col]
            scored["conf_global"] = run_global_threshold(target, higher_is_member=True)
            # Lower modified entropy = more member-like, hence the flip.
            scored["mentr_global"] = run_global_threshold(
                mentr_target, higher_is_member=False
            )
            scored["ulira_mentr"] = run_ulira(
                mentr_target, mentr_ref_signals, ref_in, transform=log_mentr
            )
        results[role] = {
            attack: report_urmia_attack(
                report_dir_exp, f"{role}_{attack}", mia_scores, memberships, logger
            )
            for attack, mia_scores in scored.items()
        }
    logger.info("Auditing took %0.1f seconds", time.time() - baseline_time)

    log_online_summary(
        directories["report_dir"], results, target_metadata, configs, logger, attacks
    )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()

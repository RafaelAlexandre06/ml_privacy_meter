"""Entry point for the OLA (Outer Layer Attenuation) audit.

Implements and audits Prabhu, Sasy, Davidson and Ghodsi, *No More Tears: Privacy
and Utility Without the Onion* (``pdfs/aaai27-submission.pdf``).

OLA is a training-time defense against the **Privacy Onion Effect**: deleting the
most attackable training points and retraining just promotes the next layer of
points to being the most attackable. OLA instead keeps them in training but
routes their memorization into channels that are switched off at inference, so
the trace they leave in the shared channels still shields their neighbours.

Five stages::

    python run_ola.py --cf configs/ola/cifar10.yaml

1. **Shadow pool** -- reuses ``split_dataset_for_training`` / ``train_models``
   unchanged. Pairs on complementary halves, so every point is IN for exactly
   half the pool, which is the online structure both LiRA and RMIA want. Split
   into a selection half and a disjoint evaluation half: picking the outer layer
   with the same models that later score it would bias toward points that merely
   look attackable to that particular reference set.
2. **Outer layer** -- per-point attack success rate (paper Eq. 1) over the
   selection half; top ``ola.outer_size`` among the target's members.
3. **Targets** -- ``vanilla`` (undefended ceiling), ``ola`` (the defense),
   ``onion`` (outer layer deleted and retrained: the baseline OLA argues against).
4. **Signals** -- one pass over the whole pool for the evaluation half and the
   three targets.
5. **Audit** -- four scorers, each reported on the outer layer, the inner-layer
   members and all members, always against the full non-member set.

The scorers are imported from ``modules/mia/attacks/urmia.py``. Their names carry
a ``urmia_`` prefix there because that module was written for the unlearning
audit, but the maths is generic: ``run_rmia_offline_masked`` is offline RMIA over
each point's OUT references, ``run_urmia_online`` is RMIA with p(x) measured from
the IN and OUT reference means, and ``run_ulira`` is LiRA. In an ordinary
membership game "IN" simply means "was a training member", which is what
``memberships`` already encodes, so they apply directly and are aliased to
neutral names below.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from models.utils import load_models, split_dataset_for_training, train_models
from modules.mia.attacks.urmia import (
    run_global_threshold as score_conf_global,
    run_rmia_offline_masked as score_rmia_offline,
    run_ulira as score_lira,
    run_urmia_online as score_rmia_online,
)
from ola_utils import (
    TARGET_ROLES,
    check_ola_configs,
    compute_asr,
    get_ola_signals,
    get_or_create_ola_splits,
    log_ola_summary,
    prepare_ola_models,
    report_ola_attack,
    select_outer_layer,
)
from util import (
    create_directories,
    initialize_seeds,
    load_dataset,
    setup_log,
)

torch.backends.cudnn.benchmark = True

OUTER_LAYER_FILE = "ola_outer_layer.npz"


def main():
    parser = argparse.ArgumentParser(description="Run the OLA privacy audit.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/ola/cifar10.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    initialize_seeds(configs["run"]["random_seed"])

    log_dir = configs["run"]["log_dir"]
    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "signal_dir": f"{log_dir}/signals",
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )
    models_dir = f"{log_dir}/models"
    report_exp = f"{directories['report_dir']}/exp"

    start_time = time.time()

    dataset, population = load_dataset(configs, directories["data_dir"], logger)
    check_ola_configs(configs, len(dataset))
    dataset_size = len(dataset)

    # ---- Stage 0: shadow pool -------------------------------------------------
    num_shadow = int(configs["audit"]["num_shadow_models"])
    baseline_time = time.time()
    shadow_models, memberships = load_models(
        log_dir, dataset, num_shadow, configs, logger
    )
    if shadow_models is None:
        data_splits, memberships = split_dataset_for_training(
            dataset_size, num_shadow // 2
        )
        shadow_models = train_models(
            log_dir, dataset, data_splits, memberships, configs, logger
        )
    shadow_models = shadow_models[:num_shadow]
    memberships = memberships[:num_shadow]
    logger.info("Shadow pool ready in %0.1f s", time.time() - baseline_time)

    # Disjoint halves. Pairs are adjacent (2k, 2k+1) and share a split, so cut on
    # a pair boundary to keep each half self-contained.
    half = num_shadow // 2
    select_cols = np.arange(half)
    eval_cols = np.arange(half, num_shadow)

    splits = get_or_create_ola_splits(models_dir, dataset_size, configs, logger)

    # ---- Stage 1: outer layer -------------------------------------------------
    outer_path = f"{models_dir}/{OUTER_LAYER_FILE}"
    baseline_time = time.time()
    shadow_signals = get_ola_signals(
        shadow_models, dataset, configs, "ola_shadow", logger
    )
    logger.info("Shadow signals took %0.1f s", time.time() - baseline_time)

    if os.path.exists(outer_path):
        cached = np.load(outer_path)
        outer_indices = cached["outer_indices"]
        asr = cached["asr"]
        logger.info("Loaded outer layer (%d points) from disk.", len(outer_indices))
    else:
        asr = compute_asr(
            shadow_signals,
            memberships,
            select_cols,
            float(configs["ola"].get("selection_fpr", 0.01)),
            logger,
        )
        outer_indices = select_outer_layer(
            asr, splits["target_train"], int(configs["ola"]["outer_size"]), logger
        )
        np.savez(
            outer_path,
            outer_indices=outer_indices,
            asr=asr,
            selection_models=select_cols,
            selection_fpr=np.float64(configs["ola"].get("selection_fpr", 0.01)),
            seed=np.int32(configs["run"]["random_seed"]),
        )

    del shadow_signals  # the selection pass is done; free it before training

    # ---- Stages 2-4: targets --------------------------------------------------
    baseline_time = time.time()
    targets = prepare_ola_models(
        models_dir, dataset, splits, outer_indices, configs, logger
    )
    logger.info("Target models ready in %0.1f s", time.time() - baseline_time)

    # ---- Stage 5: signals and audit ------------------------------------------
    population = Subset(
        population,
        np.random.choice(
            len(population),
            min(
                int(configs["audit"].get("population_size", len(population))),
                len(population),
            ),
            replace=False,
        ),
    )

    baseline_time = time.time()
    eval_models = [shadow_models[c] for c in eval_cols]
    # The evaluation half's audit signals are recomputed here rather than sliced
    # out of the "ola_shadow" cache. That is a few minutes of forward passes
    # against a job whose training stage is measured in hours, and it keeps the
    # target pass a single self-contained array whose column order matches
    # TARGET_ROLES + eval_cols. Slicing two caches together is where an
    # off-by-one would silently attack the wrong model.
    all_models = targets + eval_models
    signals = get_ola_signals(all_models, dataset, configs, "ola_target", logger)
    pop_signals = get_ola_signals(
        all_models, population, configs, "ola_target", logger, is_population=True
    )
    logger.info("Signals took %0.1f s", time.time() - baseline_time)

    n_targets = len(TARGET_ROLES)
    ref_signals = signals[:, n_targets:]
    z_ref_signals = pop_signals[:, n_targets:]
    # (num_samples, K_eval): True where the sample was a training member.
    ref_in = memberships[eval_cols].T

    is_member = np.zeros(dataset_size, dtype=int)
    is_member[splits["target_train"]] = 1
    outer_mask = np.zeros(dataset_size, dtype=bool)
    outer_mask[outer_indices] = True
    non_members = is_member == 0
    subset_rows = {
        "outer": outer_mask | non_members,
        "inner": ((is_member == 1) & ~outer_mask) | non_members,
        "all_members": np.ones(dataset_size, dtype=bool),
    }

    offline_a = float(configs["audit"]["offline_a"])
    baseline_time = time.time()
    results = {}
    for col, role in enumerate(TARGET_ROLES):
        target, z_target = signals[:, col], pop_signals[:, col]
        scored = {
            "rmia_offline": score_rmia_offline(
                target, ref_signals, ref_in, z_target, z_ref_signals, offline_a
            ),
            "rmia_online": score_rmia_online(
                target, ref_signals, ref_in, z_target, z_ref_signals, offline_a
            ),
            "lira": score_lira(target, ref_signals, ref_in),
            "conf_global": score_conf_global(target, higher_is_member=True),
        }
        results[role] = {
            subset: {
                scorer: report_ola_attack(
                    report_exp,
                    f"{role}_{subset}_{scorer}",
                    scores[rows],
                    is_member[rows],
                    logger,
                )
                for scorer, scores in scored.items()
            }
            for subset, rows in subset_rows.items()
        }
    logger.info("Auditing took %0.1f s", time.time() - baseline_time)

    with open(f"{models_dir}/ola_models_metadata.json", "r") as f:
        metadata = json.load(f)

    log_ola_summary(
        directories["report_dir"],
        results,
        metadata,
        configs,
        int(non_members.sum()),
        logger,
    )
    logger.info("Total runtime: %0.1f s", time.time() - start_time)


if __name__ == "__main__":
    main()

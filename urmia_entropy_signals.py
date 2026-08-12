"""Modified-entropy signals for U-RMIA.

The default U-RMIA signal (``urmia_utils.get_urmia_signals`` -> ``get_softmax``)
keeps only the true-class softmax column and discards the rest of the output
distribution. That is all RMIA needs, but it throws away information that is
potentially useful when auditing *unlearning* specifically: an unlearner
optimizes the true-class signal (gradient ascent on the forget set, or a loss
cap aimed at the retrained model's forget loss) and leaves the shape of the
remaining probability mass unconstrained. That residual shape is therefore an
un-optimized side channel.

This module computes the **modified prediction entropy** of Song & Mittal
(2021), "Systematic Evaluation of Privacy Risks of Machine Learning Models":

    Mentr(p, y) = -(1 - p_y) * log(p_y) - sum_{i != y} p_i * log(1 - p_i)

Unlike plain entropy ``H(p) = -sum p_i log p_i``, this is **label-aware**: a
confidently *wrong* prediction gets a high (non-member-like) score, whereas plain
entropy would score it as low/member-like. Lower Mentr means "more member-like"
-- the opposite direction from confidence, which the scorers account for.

Computed on the **audit subset only**. The scorers that consume it (U-LiRA and
the global-threshold baseline) do not use population z-samples, so there is no
reason to pay for population inference.

Cached to ``<log_dir>/signals/urmia_mentr_signals.npy`` with the same
(rows, cols) validation as the confidence cache, so bumping ``num_ref_models``
correctly forces a recompute.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel

from dataset.utils import load_dataset_subsets
from unlearner_registry import signal_cache_matches, write_signal_fingerprint

_EPS = 1e-12


def get_mentr(
    model: torch.nn.Module,
    samples: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Compute Song & Mittal modified prediction entropy for each sample.

    Requires the full output distribution, so unlike ``get_signals.get_softmax``
    this cannot be derived from the cached true-class confidences -- it needs its
    own forward pass.

    Args:
        model (torch.nn.Module): Model to evaluate.
        samples (torch.Tensor): Model input.
        labels (torch.Tensor): True class indices.
        batch_size (int): Batch size for inference.
        device (str): Device used for computing signals.

    Raises:
        NotImplementedError: If given a HuggingFace ``PreTrainedModel``; the
            sequence-level aggregation used for those models has no direct
            per-example Mentr analogue.

    Returns:
        np.ndarray: Shape (num_samples, 1). Lower = more member-like.
    """
    if isinstance(model, PreTrainedModel):
        raise NotImplementedError(
            "Modified entropy is not defined for sequence models in this pipeline; "
            "the transformer signal path aggregates per-token probabilities."
        )

    model.to(device)
    model.eval()
    mentr_list = []
    with torch.no_grad():
        for x, y in zip(
            torch.split(samples, batch_size), torch.split(labels, batch_size)
        ):
            x = x.to(device)
            y = y.to(device).long().reshape(-1, 1)

            log_p = torch.log_softmax(model(x), dim=1)
            p = log_p.exp()
            # log(1 - p_i), stable for p_i near 0; clamp keeps p_i = 1 finite.
            log_1mp = torch.log1p(-p.clamp(max=1.0 - 1e-6))

            p_y = p.gather(1, y).squeeze(1)
            log_p_y = log_p.gather(1, y).squeeze(1)
            true_term = -(1.0 - p_y) * log_p_y

            # sum over i != y of -p_i * log(1 - p_i): total minus the true-class entry.
            per_class = -p * log_1mp
            other_term = per_class.sum(dim=1) - per_class.gather(1, y).squeeze(1)

            mentr_list.append((true_term + other_term).to("cpu").reshape(-1, 1))
    model.to("cpu")
    return torch.cat(mentr_list).numpy()


def get_urmia_entropy_signals(
    models_list: list,
    subset,
    configs: dict,
    logger: logging.Logger,
) -> np.ndarray:
    """Compute cached modified-entropy signals for every model on ``subset``.

    Mirrors ``urmia_utils.get_urmia_signals`` (same ordering convention:
    columns 0..2 are the target roles, then the K references) but caches to a
    distinct filename so the two signal types never collide.

    Args:
        models_list (list): Ordered models [original, unlearned, retrained, ref_0..].
        subset: Audit subset to evaluate.
        configs (dict): Full config dictionary.
        logger (logging.Logger): Logger object for the current run.

    Returns:
        np.ndarray: Signals of shape (len(subset), len(models_list)).
    """
    signal_dir = f"{configs['run']['log_dir']}/signals"
    Path(signal_dir).mkdir(parents=True, exist_ok=True)
    signal_path = f"{signal_dir}/urmia_mentr_signals.npy"

    expected_rows = len(subset)
    expected_cols = len(models_list)
    if os.path.exists(signal_path):
        cached = np.load(signal_path)
        if cached.shape != (expected_rows, expected_cols):
            logger.warning(
                "Cached entropy signals shape %s does not match expected (%d, %d); "
                "recomputing.",
                cached.shape,
                expected_rows,
                expected_cols,
            )
        # Same blind spot as the softmax cache: the unlearner changes values, not
        # dimensions. See unlearner_registry.signal_cache_matches.
        elif signal_cache_matches(signal_path, configs, logger):
            logger.info("U-RMIA entropy signals loaded from disk (%s).", signal_path)
            write_signal_fingerprint(signal_path, configs)
            return cached

    batch_size = configs["audit"]["batch_size"]
    model_name = configs["train"]["model_name"]
    device = configs["audit"]["device"]

    data, targets = load_dataset_subsets(
        subset, np.arange(len(subset)), model_name, batch_size, device
    )

    logger.info("Computing U-RMIA entropy signals for %d models.", expected_cols)
    signals = np.concatenate(
        [get_mentr(model, data, targets, batch_size, device) for model in models_list],
        axis=1,
    )
    np.save(signal_path, signals)
    write_signal_fingerprint(signal_path, configs)
    logger.info("U-RMIA entropy signals saved to disk (%s).", signal_path)
    return signals

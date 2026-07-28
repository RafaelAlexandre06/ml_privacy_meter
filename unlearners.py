"""Unlearning algorithms for U-RMIA.

This module holds the machine-unlearning step used by ``run_urmia.py``. Every
unlearner shares a single functional interface so that the entry point can swap
algorithms via the ``unlearn.algorithm`` config field without touching the
pipeline:

    def <algorithm>(
        model,            # trained original model (never mutated in place)
        forget_loader,    # DataLoader over the forget set F
        retain_loader,    # DataLoader over the retain set R
        unlearn_configs,  # configs["unlearn"] (algorithm, forget_size, params, seed)
        train_configs,    # configs["train"] (device, optimizer hyperparams)
        logger,
    ) -> torch.nn.Module  # the unlearned model, returned on CPU

The interface intentionally receives both the forget and retain loaders even
when an algorithm ignores them (Gaussian noise ignores both), so that
gradient-based methods can be added later without changing the signature.

Currently implemented:
- ``gaussian_noise``: add calibrated Gaussian noise to the model weights.

Future algorithms would slot straight into ``UNLEARNERS`` following the same
signature, for example:
- ``neggrad``: gradient *ascent* on ``forget_loader`` for ``params.epochs`` using
  ``trainers.default_trainer.get_optimizer`` (backprop ``(-loss)``).
- ``graddesc``: gradient *descent* on ``retain_loader`` (catastrophic forgetting).
"""

import copy
import logging

import torch


def gaussian_noise(
    model: torch.nn.Module,
    forget_loader: torch.utils.data.DataLoader,
    retain_loader: torch.utils.data.DataLoader,
    unlearn_configs: dict,
    train_configs: dict,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Unlearn by adding Gaussian noise to the model parameters.

    This is the simplest possible unlearning baseline: it perturbs every
    trainable weight with independent Gaussian noise of standard deviation
    ``params.noise_std`` and takes no gradient step, so it ignores the forget
    and retain loaders. BatchNorm buffers (running mean/var) are left untouched,
    as they are not returned by ``model.parameters()``; perturbing running
    statistics is a different intervention and is out of scope here.

    Args:
        model (torch.nn.Module): Trained original model. Deep-copied, not mutated.
        forget_loader (torch.utils.data.DataLoader): Forget set loader (unused).
        retain_loader (torch.utils.data.DataLoader): Retain set loader (unused).
        unlearn_configs (dict): The ``unlearn`` config block. Uses
            ``params.noise_std`` and an optional ``seed`` for reproducibility.
        train_configs (dict): The ``train`` config block (unused here).
        logger (logging.Logger): Logger object for the current run.

    Returns:
        torch.nn.Module: The unlearned model, on CPU.
    """
    params = unlearn_configs.get("params", {}) or {}
    noise_std = float(params.get("noise_std", 0.0))
    seed = unlearn_configs.get("seed", None)

    unlearned = copy.deepcopy(model).to("cpu")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))

    num_perturbed = 0
    with torch.no_grad():
        for param in unlearned.parameters():
            if not param.requires_grad:
                continue
            noise = torch.randn(param.shape, generator=generator) * noise_std
            param.add_(noise)
            num_perturbed += param.numel()

    logger.info(
        "Gaussian-noise unlearning: perturbed %d parameters with noise_std=%.6g",
        num_perturbed,
        noise_std,
    )
    return unlearned


UNLEARNERS = {
    "gaussian_noise": gaussian_noise,
}


def get_unlearner(name: str):
    """Look up an unlearning algorithm by name.

    Args:
        name (str): The ``unlearn.algorithm`` config value.

    Raises:
        NotImplementedError: If the algorithm is not registered.

    Returns:
        Callable: The unlearner function following the module's interface.
    """
    if name not in UNLEARNERS:
        raise NotImplementedError(
            f"Unlearning algorithm '{name}' is not implemented. "
            f"Supported: {list(UNLEARNERS)}"
        )
    return UNLEARNERS[name]

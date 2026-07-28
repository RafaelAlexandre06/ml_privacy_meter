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
- ``gaussian_noise``: add calibrated Gaussian noise to all model weights. Blunt and
  data-independent: it does not target the forget set and tends to degrade the
  whole model.
- ``neggrad``: gradient *ascent* on the forget set (maximize its loss).
- ``random_label``: fine-tune the forget images toward random *wrong* labels
  ("fit garbage" on the forget set). Amnesiac-style relabeling.
- ``neggrad_plus``: gradient descent on the retain set *and* ascent on the forget
  set together, so accuracy on the retain set is preserved (Kurmanji et al.).

The targeted methods (``neggrad``, ``random_label``, ``neggrad_plus``) are far
gentler on overall accuracy than ``gaussian_noise`` because only the relevant data
drives the update, but they tend to *over-unlearn*: forget points end up with
abnormally low confidence, a signature a two-sided attack (level-3 U-RMIA) can
still detect. That is the expected, useful behavior to study, not a bug.
"""

import copy
import logging

import torch
from torch import nn

from trainers.default_trainer import get_optimizer


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


def _optimizer_config(unlearn_configs: dict, train_configs: dict) -> dict:
    """Build the optimizer config for an unlearning fine-tune.

    Reads hyperparameters from ``unlearn.params`` and falls back to the training
    config, so unlearning can use its own (typically smaller) learning rate.
    """
    params = unlearn_configs.get("params", {}) or {}
    return {
        "optimizer": params.get("optimizer", train_configs.get("optimizer", "SGD")),
        "learning_rate": params.get("learning_rate", 0.01),
        "weight_decay": params.get("weight_decay", train_configs.get("weight_decay", 0.0)),
        "momentum": params.get("momentum", train_configs.get("momentum", 0.0)),
    }


def _cycle(loader):
    """Yield batches from ``loader`` forever, re-iterating (and re-shuffling) it."""
    while True:
        for batch in loader:
            yield batch


def neggrad(
    model: torch.nn.Module,
    forget_loader: torch.utils.data.DataLoader,
    retain_loader: torch.utils.data.DataLoader,
    unlearn_configs: dict,
    train_configs: dict,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Unlearn by gradient ascent on the forget set (maximize its loss).

    Targets the forget data directly: it pushes predictions away from the correct
    label. Ascent is unbounded, so use a small learning rate and few epochs
    (``params.learning_rate``, ``params.epochs``) to avoid destroying the model.
    Ignores the retain loader.
    """
    params = unlearn_configs.get("params", {}) or {}
    epochs = int(params.get("epochs", 5))
    device = train_configs.get("device", "cpu")

    unlearned = copy.deepcopy(model).to(device)
    unlearned.train()
    optimizer = get_optimizer(unlearned, _optimizer_config(unlearn_configs, train_configs))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(unlearned(x), y)
            (-loss).backward()  # ascent: move to increase the forget-set loss
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            "neggrad epoch %d/%d: mean forget loss %.4f",
            epoch + 1,
            epochs,
            total_loss / max(len(forget_loader), 1),
        )

    return unlearned.to("cpu")


def random_label(
    model: torch.nn.Module,
    forget_loader: torch.utils.data.DataLoader,
    retain_loader: torch.utils.data.DataLoader,
    unlearn_configs: dict,
    train_configs: dict,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Unlearn by fine-tuning the forget images toward random *wrong* labels.

    This is the "fit garbage on the forget set" idea (Amnesiac-style relabeling):
    keep each forget image but train it toward a randomly chosen incorrect class,
    breaking the memorized image->true-label association. Ignores the retain loader.
    """
    params = unlearn_configs.get("params", {}) or {}
    epochs = int(params.get("epochs", 5))
    device = train_configs.get("device", "cpu")
    seed = unlearn_configs.get("seed", None)

    unlearned = copy.deepcopy(model).to(device)
    unlearned.train()
    optimizer = get_optimizer(unlearned, _optimizer_config(unlearn_configs, train_configs))
    criterion = nn.CrossEntropyLoss()

    # Number of classes = model output dimension (model-agnostic).
    with torch.no_grad():
        sample_x = next(iter(forget_loader))[0][:1].to(device)
        num_classes = unlearned(sample_x).shape[1]

    generator = torch.Generator()  # CPU generator for reproducible label draws
    if seed is not None:
        generator.manual_seed(int(seed))

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device).long()
            # Offset in [1, num_classes-1] guarantees a label different from y.
            offset = torch.randint(
                1, num_classes, y.shape, generator=generator
            ).to(device)
            wrong_y = (y + offset) % num_classes
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(unlearned(x), wrong_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            "random_label epoch %d/%d: mean wrong-label loss %.4f",
            epoch + 1,
            epochs,
            total_loss / max(len(forget_loader), 1),
        )

    return unlearned.to("cpu")


def neggrad_plus(
    model: torch.nn.Module,
    forget_loader: torch.utils.data.DataLoader,
    retain_loader: torch.utils.data.DataLoader,
    unlearn_configs: dict,
    train_configs: dict,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Unlearn by ascent on the forget set and descent on the retain set together.

    NegGrad+ (Kurmanji et al.): each step combines
    ``retain_coef * loss_retain - forget_coef * loss_forget``. Iterating over the
    retain loader (cycling the smaller forget loader) preserves retain accuracy
    while still pushing the forget set away, which is what makes it far gentler on
    utility than plain ascent or global weight noise.
    """
    params = unlearn_configs.get("params", {}) or {}
    epochs = int(params.get("epochs", 5))
    forget_coef = float(params.get("forget_coef", 1.0))
    retain_coef = float(params.get("retain_coef", 1.0))
    device = train_configs.get("device", "cpu")

    unlearned = copy.deepcopy(model).to(device)
    unlearned.train()
    optimizer = get_optimizer(unlearned, _optimizer_config(unlearn_configs, train_configs))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        forget_iter = _cycle(forget_loader)
        total_r, total_f = 0.0, 0.0
        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_iter)
            x_r, y_r = x_r.to(device), y_r.to(device).long()
            x_f, y_f = x_f.to(device), y_f.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            loss_r = criterion(unlearned(x_r), y_r)
            loss_f = criterion(unlearned(x_f), y_f)
            loss = retain_coef * loss_r - forget_coef * loss_f
            loss.backward()
            optimizer.step()
            total_r += loss_r.item()
            total_f += loss_f.item()
        n = max(len(retain_loader), 1)
        logger.info(
            "neggrad_plus epoch %d/%d: mean retain loss %.4f, mean forget loss %.4f",
            epoch + 1,
            epochs,
            total_r / n,
            total_f / n,
        )

    return unlearned.to("cpu")


UNLEARNERS = {
    "gaussian_noise": gaussian_noise,
    "neggrad": neggrad,
    "random_label": random_label,
    "neggrad_plus": neggrad_plus,
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

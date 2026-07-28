"""Weight Initialization based on Gradient similarity (WIG) unlearner.

Lee et al., "Weight initialization based on gradient similarity for versatile
machine unlearning", PoPETs 2026(3). Reference code: github.com/Ldoun/WIG
(``Unlearn_Reinit/``).

Unlike the output-targeting unlearners in ``unlearners.py`` (ascent raises the
forget loss, random labels lower the forget confidence), WIG is *structural*
and property-unbiased: it never optimizes any output statistic of the forget
set. Instead it

1. fits a per-weight Gaussian to the per-batch cross-entropy gradients of the
   retain data and another to those of the forget data,
2. scores each weight by the overlapping coefficient (OVL) of the two Gaussians
   -- high overlap means the weight carries features *shared* between retain
   and forget data, exactly the weights that would let forget-set information
   resurface during fine-tuning,
3. re-initializes the top ``init_ratio`` fraction per layer with Kaiming
   normal noise (the paper's N(0, 2/n_{l+1})), and
4. fine-tunes on the retain set, which repairs utility while catastrophic
   forgetting erases the forget set's contribution.

Because no single property (loss / confidence / entropy) is pushed, the paper
predicts uniformly small attack gaps across statistics -- the contrast to
``random_label``/``neggrad_plus``, which defend one property and leak through
others. That prediction is what the level-3 6-scorer grid tests.

Deviations from the reference implementation, both validated by the paper:

- The train-side gradient distribution is estimated on the retain loader (D^R)
  rather than the full train set D. The paper's Sec. 4.3 / Fig. 9 ablation
  shows the two are empirically equivalent: D^F's own contribution to D only
  adds a uniform self-similarity bias that top-p% selection ignores.
- The classifier head's ``nn.Linear`` weight is scored and re-initialized
  alongside ``nn.Conv2d`` weights (the reference scores Conv2d for CNNs and
  Linear for ViTs; wrn28-1 has both). Biases and BatchNorm parameters are
  never touched, matching the reference.

Config block::

    unlearn:
      algorithm: wig
      forget_size: 500
      params:
        init_ratio: 0.3        # fraction of weights re-initialized per layer
        grad_batch_size: 32    # forget-gradient batch size (paper Sec. 5.1)
        epochs: 5              # retain fine-tuning epochs
        learning_rate: 0.01
"""

import copy
import logging
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from trainers.default_trainer import get_optimizer

# Added to every gradient std so OVL stays finite when a weight's gradient is
# (numerically) constant across batches; mirrors the reference's +1e-9.
_STD_EPS = 1e-9


def _target_weight_modules(model: torch.nn.Module):
    """Yield ``(name, module)`` for every Conv2d/Linear whose weight WIG scores."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
            yield name, module


def _fan_out(weight: torch.Tensor) -> int:
    """Kaiming fan_out: out_features for Linear, out_channels * kH * kW for conv."""
    receptive = 1
    for size in weight.shape[2:]:
        receptive *= size
    return weight.shape[0] * receptive


def _grad_gaussian_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    logger: logging.Logger,
    tag: str,
):
    """Fit a per-weight Gaussian to the per-batch cross-entropy gradients.

    One backward pass per batch; per-weight mean and (sample) std are
    accumulated with Welford's online algorithm, so peak extra memory is two
    weight-sized tensors per scored layer regardless of batch count.

    The model is evaluated in ``eval()`` mode: gradients are still exact, but
    BatchNorm running statistics are not perturbed by the stats pass.

    Returns:
        dict: ``{module_name: (mean, std)}`` with tensors shaped like the weight.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    targets = list(_target_weight_modules(model))
    mean = {name: torch.zeros_like(m.weight) for name, m in targets}
    m2 = {name: torch.zeros_like(m.weight) for name, m in targets}

    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        count += 1
        with torch.no_grad():
            for name, module in targets:
                grad = module.weight.grad
                delta = grad - mean[name]
                mean[name] += delta / count
                m2[name] += delta * (grad - mean[name])
    model.zero_grad(set_to_none=True)

    if count < 2:
        logger.warning(
            "WIG %s stats: only %d gradient batch(es); std estimates are "
            "degenerate. Lower params.grad_batch_size or enlarge the set.",
            tag,
            count,
        )
    stats = {}
    with torch.no_grad():
        for name, _ in targets:
            var = m2[name] / max(count - 1, 1)
            stats[name] = (mean[name], torch.sqrt(var) + _STD_EPS)
    logger.info("WIG %s stats: %d gradient batches over %d layers", tag, count, len(targets))
    return stats


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _gaussian_ovl(
    mean_r: torch.Tensor,
    std_r: torch.Tensor,
    mean_f: torch.Tensor,
    std_f: torch.Tensor,
) -> torch.Tensor:
    """Elementwise overlapping coefficient of N(mean_r, std_r) and N(mean_f, std_f).

    Paper Eq. 4-5: solve for the (up to two) intersection points of the two
    pdfs, then sum min(CDF mass) over the three regions they delimit. Equal
    variances collapse both intersections to the midpoint, which the same
    region sum handles (the middle region gets zero mass), reproducing the
    analytic 2*Phi(-|mu_r - mu_f| / (2*sigma)).
    """
    var_r, var_f = std_r * std_r, std_f * std_f
    var_diff = var_f - var_r
    equal = torch.isclose(std_r, std_f, rtol=1e-5, atol=1e-12)

    # Intersection points (quadratic in x); clamp guards tiny negative values
    # from rounding, safe_diff guards the equal-variance rows overwritten below.
    sqrt_term = torch.sqrt(
        torch.clamp((mean_r - mean_f) ** 2 + var_diff * torch.log(var_f / var_r), min=0.0)
    )
    num_base = mean_r * var_f - mean_f * var_r
    safe_diff = torch.where(equal, torch.ones_like(var_diff), var_diff)
    x1 = (num_base - std_r * std_f * sqrt_term) / safe_diff
    x2 = (num_base + std_r * std_f * sqrt_term) / safe_diff
    lo, hi = torch.minimum(x1, x2), torch.maximum(x1, x2)
    mid = (mean_r + mean_f) / 2.0
    lo = torch.where(equal, mid, lo)
    hi = torch.where(equal, mid, hi)

    r_lo = _normal_cdf((lo - mean_r) / std_r)
    r_hi = _normal_cdf((hi - mean_r) / std_r)
    f_lo = _normal_cdf((lo - mean_f) / std_f)
    f_hi = _normal_cdf((hi - mean_f) / std_f)
    overlap = (
        torch.minimum(r_lo, f_lo)
        + torch.minimum(r_hi - r_lo, f_hi - f_lo)
        + torch.minimum(1.0 - r_hi, 1.0 - f_hi)
    )
    return torch.clamp(overlap, 0.0, 1.0)


def wig(
    model: torch.nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    unlearn_configs: dict,
    train_configs: dict,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Unlearn via gradient-similarity weight re-initialization + retain fine-tuning.

    Args:
        model (torch.nn.Module): Trained original model. Deep-copied, not mutated.
        forget_loader (DataLoader): Forget set loader. Its dataset is re-batched
            at ``params.grad_batch_size`` for the stats pass so small forget
            sets still yield enough batch gradients for a std estimate.
        retain_loader (DataLoader): Retain set loader; used for both the
            train-side gradient distribution and the fine-tuning phase.
        unlearn_configs (dict): The ``unlearn`` config block (params, seed).
        train_configs (dict): The ``train`` config block (device, optimizer
            fallbacks).
        logger (logging.Logger): Logger object for the current run.

    Returns:
        torch.nn.Module: The unlearned model, on CPU.
    """
    # Imported at call time: unlearners.py imports this module to register wig.
    from unlearners import _optimizer_config

    params = unlearn_configs.get("params", {}) or {}
    init_ratio = float(params.get("init_ratio", 0.3))
    grad_batch_size = int(params.get("grad_batch_size", 32))
    epochs = int(params.get("epochs", 5))
    device = train_configs.get("device", "cpu")
    seed = unlearn_configs.get("seed", None)

    if not 0.0 < init_ratio < 1.0:
        raise ValueError(f"wig: params.init_ratio must be in (0, 1), got {init_ratio}")

    unlearned = copy.deepcopy(model).to(device)

    # Phase 1: per-weight gradient Gaussians. Retain-side uses the loader as
    # given; forget-side is re-batched small (paper uses 32) so that e.g. 250
    # forget points give ~8 batch gradients instead of 1.
    forget_stats_loader = DataLoader(
        forget_loader.dataset, batch_size=grad_batch_size, shuffle=False
    )
    retain_stats = _grad_gaussian_stats(unlearned, retain_loader, device, logger, "retain")
    forget_stats = _grad_gaussian_stats(
        unlearned, forget_stats_loader, device, logger, "forget"
    )

    # Phase 2: per layer, re-initialize the top init_ratio fraction by OVL with
    # Kaiming normal noise N(0, 2/fan_out); everything else keeps its weights.
    generator = torch.Generator()  # CPU generator, matching gaussian_noise
    if seed is not None:
        generator.manual_seed(int(seed))

    total_masked, total_weights = 0, 0
    with torch.no_grad():
        for name, module in _target_weight_modules(unlearned):
            weight = module.weight
            mean_r, std_r = retain_stats[name]
            mean_f, std_f = forget_stats[name]
            ovl = _gaussian_ovl(mean_r, std_r, mean_f, std_f)

            # Exact-k index selection rather than a value threshold: tied OVL
            # values (e.g. dead weights with zero gradient on both sets all
            # score exactly 1.0) would otherwise over-select past init_ratio.
            k = max(1, int(round(init_ratio * weight.numel())))
            mask = torch.zeros_like(ovl, dtype=torch.bool)
            mask.view(-1)[torch.topk(ovl.flatten(), k).indices] = True

            init_std = math.sqrt(2.0 / _fan_out(weight))
            init = (torch.randn(weight.shape, generator=generator) * init_std).to(
                weight.device
            )
            weight.data = torch.where(mask, init, weight.data)

            masked = int(mask.sum().item())
            total_masked += masked
            total_weights += weight.numel()
            logger.info(
                "WIG re-init %s: %d/%d weights (%.1f%%), init std %.4g",
                name,
                masked,
                weight.numel(),
                100.0 * masked / weight.numel(),
                init_std,
            )
    logger.info(
        "WIG re-initialized %d/%d scored weights (%.1f%%) at init_ratio=%.2f",
        total_masked,
        total_weights,
        100.0 * total_masked / max(total_weights, 1),
        init_ratio,
    )

    # Phase 3: fine-tune on the retain set to repair utility while catastrophic
    # forgetting removes the forget set's shared-feature contribution.
    unlearned.train()
    optimizer = get_optimizer(unlearned, _optimizer_config(unlearn_configs, train_configs))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(unlearned(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            "wig fine-tune epoch %d/%d: mean retain loss %.4f",
            epoch + 1,
            epochs,
            total_loss / max(len(retain_loader), 1),
        )

    return unlearned.to("cpu")

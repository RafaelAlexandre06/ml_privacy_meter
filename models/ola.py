"""Example-tied dropout for Outer Layer Attenuation (OLA).

Paper: Prabhu, Sasy, Davidson and Ghodsi, *No More Tears: Privacy and Utility
Without the Onion* (``pdfs/aaai27-submission.pdf``), building on Example-Tied
Dropout (Maini et al., ICML 2023).

The idea in one paragraph. Split the channels of every layer into a ``p_gen``
fraction of **generalization channels**, active for every example, and a
remainder of **memorization channels**. Only the *outer layer* -- the points a
membership-inference attack finds most vulnerable -- gets memorization channels,
and each such point gets its own random subset. At inference every memorization
channel is switched off, so an outer-layer point's private pathway disappears
while the trace it left in the shared channels remains, which is what keeps its
neighbours from becoming the next outer layer.

Two consequences drive the design here:

- **Training needs sample identity.** A per-example mask cannot be derived from
  ``(data, target)``, so the forward pass takes dataset indices and
  ``trainers/ola_trainer.py`` feeds them in. See :class:`ExampleTiedDropout`.
- **Inference does not.** With every memorization channel dropped, the mask is
  the same for all inputs, so ``model(x)`` behaves like an ordinary module. That
  is what lets ``get_signals.py:get_softmax`` consume an OLA model unchanged.

Masks are *not* parameters and deliberately do not appear in ``state_dict`` --
:meth:`ExampleTiedDropout.state_dict` delegates to the wrapped model so an OLA
checkpoint round-trips through ``models/utils.py:get_model`` +
``load_state_dict`` like any other. They are instead regenerated from
``(p_gen, p_mem, seed, outer_indices, architecture)``, and
:meth:`ExampleTiedDropout.mask_fingerprint` exists so a resumed run can *prove*
it rebuilt the same ones. A silently different mask would leave the
inference-time drop switched off for channels the model actually trained
through, which no accuracy check would catch.

Only ``WideResNet`` is supported, matching the paper (they use WRN28-2).
"""

import hashlib

import numpy as np
import torch
from torch import nn

from models.wide_resnet import WideResNetGeneral, WRNBlock


def maskable_layers(model: nn.Module) -> list:
    """Modules whose output channels OLA masks, in forward order.

    For a WideResNet these are the stem convolution and each residual block --
    the points where a whole channel can be switched off without breaking the
    residual algebra. Masking inside a block (between ``conv_1`` and ``conv_2``)
    would leave the skip connection carrying the signal around the mask.

    Args:
        model (nn.Module): A ``WideResNetGeneral`` (or subclass) instance.

    Returns:
        list: ``(module, out_channels)`` pairs in forward order.

    Raises:
        NotImplementedError: For any other architecture.
    """
    if not isinstance(model, WideResNetGeneral):
        raise NotImplementedError(
            f"OLA masking is only implemented for WideResNet, got {type(model).__name__}"
        )

    layers = []
    for module in model.layers:
        if isinstance(module, nn.Conv2d):  # the stem; blocks hold their own convs
            layers.append((module, module.out_channels))
        elif isinstance(module, WRNBlock):
            layers.append((module, module.conv_2.out_channels))
    return layers


class ExampleTiedDropout(nn.Module):
    """Wrap a WideResNet with OLA's generalization/memorization channel split.

    Args:
        base (nn.Module): The WideResNet to wrap. Used in place, not copied.
        dataset_size (int): Size of the index space ``forward`` will be given.
        outer_indices (np.ndarray): Dataset indices of the outer layer. These are
            the only samples that get memorization channels.
        p_gen (float): Fraction of each layer's channels that are generalization
            channels. Selected exactly (``round(p_gen * C)``), not by coin flip,
            so the shared capacity is identical across layers of the same width.
        p_mem (float): Probability that each *remaining* channel is active for a
            given outer-layer sample. Independent Bernoulli, as the paper
            specifies -- so an outer-layer point's capacity varies, which is the
            point: the pathways must not collide.
        seed (int): Seeds both mask draws.
        device (str): Device the masks live on.
    """

    def __init__(
        self,
        base: nn.Module,
        dataset_size: int,
        outer_indices: np.ndarray,
        p_gen: float,
        p_mem: float,
        seed: int,
        device: str = "cpu",
    ):
        super().__init__()
        if not 0.0 < p_gen <= 1.0:
            raise ValueError(f"p_gen must be in (0, 1], got {p_gen}")
        if not 0.0 <= p_mem <= 1.0:
            raise ValueError(f"p_mem must be in [0, 1], got {p_mem}")

        self.base = base
        self.p_gen = float(p_gen)
        self.p_mem = float(p_mem)
        self.seed = int(seed)

        layers = maskable_layers(base)
        self.channel_counts = [count for _, count in layers]
        self.total_channels = int(sum(self.channel_counts))

        # Flat layout: one (N, total_channels) mask per batch, sliced per layer.
        offsets, cursor = [], 0
        for count in self.channel_counts:
            offsets.append(cursor)
            cursor += count
        self._spans = {
            id(module): (offsets[i], self.channel_counts[i])
            for i, (module, _) in enumerate(layers)
        }

        rng = np.random.default_rng(seed)
        gen_mask = self._draw_gen_mask(rng)
        mem_masks = self._draw_mem_masks(rng, gen_mask, len(outer_indices))

        # Buffers so .to(device) carries them; excluded from state_dict below.
        self.register_buffer(
            "gen_mask", torch.from_numpy(gen_mask).to(device), persistent=False
        )
        self.register_buffer(
            "mem_masks", torch.from_numpy(mem_masks).to(device), persistent=False
        )
        # Dataset index -> row in mem_masks, or -1 for inner-layer/non-member.
        row_of = np.full(dataset_size, -1, dtype=np.int64)
        row_of[np.asarray(outer_indices, dtype=np.int64)] = np.arange(
            len(outer_indices), dtype=np.int64
        )
        self.register_buffer(
            "row_of", torch.from_numpy(row_of).to(device), persistent=False
        )

        self._batch_mask = None
        self._handles = [
            module.register_forward_hook(self._apply_mask) for module, _ in layers
        ]

    def _draw_gen_mask(self, rng: np.random.Generator) -> np.ndarray:
        """Exactly ``round(p_gen * C)`` generalization channels per layer."""
        mask = np.zeros(self.total_channels, dtype=bool)
        cursor = 0
        for count in self.channel_counts:
            n_gen = int(round(self.p_gen * count))
            chosen = rng.choice(count, size=n_gen, replace=False)
            mask[cursor + chosen] = True
            cursor += count
        return mask

    def _draw_mem_masks(
        self, rng: np.random.Generator, gen_mask: np.ndarray, n_outer: int
    ) -> np.ndarray:
        """Per-outer-sample Bernoulli(p_mem) over the non-generalization channels."""
        draws = rng.random((n_outer, self.total_channels)) < self.p_mem
        # A generalization channel is always on; never double-count it as memory.
        return np.logical_and(draws, ~gen_mask[None, :])

    def mask_fingerprint(self) -> str:
        """Stable hash of the full masking state, for verifying a resumed run.

        Covers ``row_of`` as well as the masks themselves. The masks alone are a
        function of ``(seed, p_gen, p_mem, channel counts, len(outer))`` and say
        nothing about *which* points are in the outer layer -- so hashing only
        them would give an identical digest to a run that attenuated a
        completely different set of the same size, and the resume guard would
        wave through weights trained on the wrong points.

        Returns:
            str: 16-hex-character digest.
        """
        digest = hashlib.sha256()
        digest.update(self.gen_mask.cpu().numpy().tobytes())
        digest.update(self.mem_masks.cpu().numpy().tobytes())
        digest.update(self.row_of.cpu().numpy().tobytes())
        return digest.hexdigest()[:16]

    def _apply_mask(self, module, inputs, output):
        """Forward hook: zero the channels this sample is not allowed to use."""
        offset, width = self._spans[id(module)]
        if self._batch_mask is None:  # eval: generalization channels only
            mask = self.gen_mask[offset : offset + width].view(1, width, 1, 1)
        else:
            mask = self._batch_mask[:, offset : offset + width].view(-1, width, 1, 1)
        return output * mask.to(output.dtype)

    def compose_mask(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-sample channel mask: generalization channels plus own memory ones.

        Args:
            idx (torch.Tensor): Dataset indices of the batch, shape (N,).

        Returns:
            torch.Tensor: Bool mask of shape (N, total_channels).
        """
        idx = idx.to(self.row_of.device)
        mask = self.gen_mask.unsqueeze(0).repeat(idx.shape[0], 1)
        rows = self.row_of[idx]
        outer = rows >= 0
        if bool(outer.any()):
            mask[outer] |= self.mem_masks[rows[outer]]
        return mask

    def forward(self, x: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """Run the wrapped model under the appropriate channel mask.

        Args:
            x (torch.Tensor): Input batch.
            idx (torch.Tensor): Dataset indices, required in training mode.

        Returns:
            torch.Tensor: Model output.

        Raises:
            ValueError: If called in training mode without indices -- silently
                falling back to the eval mask would train the model with its
                memorization channels permanently off.
        """
        if self.training:
            if idx is None:
                raise ValueError(
                    "ExampleTiedDropout needs dataset indices in training mode; "
                    "use trainers/ola_trainer.py, which supplies them."
                )
            self._batch_mask = self.compose_mask(idx)
        else:
            self._batch_mask = None
        try:
            return self.base(x)
        finally:
            self._batch_mask = None

    def state_dict(self, *args, **kwargs):
        """Delegate to the wrapped model so checkpoints stay architecture-plain.

        An OLA checkpoint is loadable by ``get_model(...)`` + ``load_state_dict``
        with no knowledge of OLA. The masks are excluded by construction and are
        rebuilt from the seed instead; see the module docstring.
        """
        return self.base.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Counterpart to :meth:`state_dict`; loads into the wrapped model."""
        return self.base.load_state_dict(state_dict, strict=strict)

    def remove_hooks(self):
        """Detach the forward hooks, leaving the bare unmasked model behind."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

"""Training loop for Outer Layer Attenuation (OLA).

``trainers/default_trainer.py:train`` iterates ``(data, target)``, which carries
no sample identity, so it cannot drive the per-example channel masks in
``models/ola.py``. This module is the same loop with dataset indices threaded
through: :class:`IndexedSubset` attaches them and :func:`train_ola` forwards them
into :meth:`ExampleTiedDropout.forward`.

Optimizer, cosine schedule and reporting mirror ``default_trainer`` exactly --
including its ``step * 256`` / ``initial_lr=0.1`` schedule constants -- so an OLA
model's utility is directly comparable with the ``vanilla`` baseline trained by
the default path. Do not "fix" the schedule here without retraining the baseline.
"""

import time
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler

from trainers.default_trainer import get_optimizer, inference, lr_update


class IndexedSubset(torch.utils.data.Dataset):
    """A ``Subset`` that also returns each sample's index in the *base* dataset.

    OLA masks are keyed on the position of a sample in the full training pool,
    not on its position within the subset a given model trains on, so the index
    returned here is the base-dataset one.

    Args:
        dataset (torch.utils.data.Dataset): The full training pool.
        indices (np.ndarray): Base-dataset indices making up this subset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        """Returns ``(data, target, base_index)``."""
        base_index = int(self.indices[position])
        data, target = self.dataset[base_index]
        return data, target, base_index


def train_ola(
    model,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
):
    """Train an :class:`~models.ola.ExampleTiedDropout` model.

    Args:
        model (ExampleTiedDropout): The wrapped model. Its masks must already be
            built for the outer layer this run uses.
        train_loader (DataLoader): Must yield ``(data, target, index)``, i.e. be
            built over an :class:`IndexedSubset`.
        configs (dict): The ``train`` config block.
        test_loader (DataLoader, optional): Plain ``(data, target)`` loader for
            per-epoch test accuracy. Evaluated in eval mode, so it measures the
            model *as audited* -- memorization channels dropped.

    Returns:
        ExampleTiedDropout: The trained model, moved back to CPU.

    Raises:
        ValueError: If ``train_loader`` does not yield indices, which would
            otherwise surface as a confusing unpacking error mid-epoch.
    """
    device = configs.get("device", "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)

    epochs = configs.get("epochs", 1)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_update(
            step * 256, epochs, len(train_loader) * 256, 0.1
        ),
    )

    for epoch_idx in range(epochs):
        start_time = time.time()
        total_loss, correct_predictions = 0, 0
        model.train()

        for batch in train_loader:
            if len(batch) != 3:
                raise ValueError(
                    "train_ola needs a loader over IndexedSubset yielding "
                    f"(data, target, index); got a {len(batch)}-tuple."
                )
            data, target, idx = batch
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).long(),
            )
            idx = idx.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output = model(data, idx)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct_predictions += output.argmax(dim=1).eq(target).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / len(train_loader.dataset)
        print(
            f"Epoch [{epoch_idx + 1}/{epochs}] | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        if test_loader:
            # Eval mode drops the memorization channels, so this is the accuracy
            # of the model that actually gets deployed and audited -- not of the
            # wider network that was just trained.
            test_loss, test_acc = inference(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        print(f"Epoch {epoch_idx + 1} took {time.time() - start_time:.2f} seconds")

    model.to("cpu")
    return model

"""
Optimized trainer for Vision Transformer (Self-contained)
Includes: Label Smoothing, Mixup/CutMix, EMA, CosineAnnealingWarmRestarts,
Gradient Clipping, Early Stopping, Best Model Saving.
All with PyTorch native DataLoader.
"""

import os
import time
import yaml
import json
import copy
import argparse
from typing import Dict, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# ---------- DataLoader (PyTorch) ----------
def get_dataloaders_pytorch(
    dataset_name: str,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create PyTorch DataLoader for training and validation.
    """
    if dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
        ])
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        num_classes = 10
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader, num_classes

# ---------- Mixup / CutMix ----------
class Mixup:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class CutMix:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())
        else:
            lam = 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        W, H = x.size(2), x.size(3)
        cut_rat = (1. - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
# ---------- EMA ----------
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# ---------- Evaluation ----------
def evaluate_optimized(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# ---------- Main training ----------
def train_optimized(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    configs: Dict,
) -> Tuple[nn.Module, Dict[str, float]]:
    device = torch.device(configs["train"].get("device", "cuda:0"))
    model = model.to(device)

    # Unpack parameters
    epochs = int(configs["train"]["epochs"])
    batch_size = int(configs["train"]["batch_size"])
    lr = float(configs["train"]["learning_rate"])
    wd = float(configs["train"]["weight_decay"])
    opt_name = configs["train"].get("optimizer", "AdamW")
    use_amp = bool(configs["train"].get("use_amp", True))
    label_smoothing = float(configs["train"].get("label_smoothing", 0.1))
    mixup_alpha = float(configs["train"].get("mixup_alpha", 0.0))
    cutmix_alpha = float(configs["train"].get("cutmix_alpha", 0.0))
    ema_decay = float(configs["train"].get("ema_decay", 0.999))
    grad_clip_norm = float(configs["train"].get("grad_clip_norm", 1.0))
    early_stop_patience = int(configs["train"].get("early_stop_patience", 0))
    warmup_epochs = int(configs["train"].get("warmup_epochs", 0))

    # Optimizer
    if opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # Scheduler with safe T_0
    T_0 = max(1, epochs // 3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=lr * 0.01)

    # Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Augmentations
    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if use_mixup and use_cutmix:
        print("Warning: Both Mixup and CutMix enabled. Using CutMix only.")
        use_mixup = False
    mixup_fn = Mixup(alpha=mixup_alpha) if use_mixup else None
    cutmix_fn = CutMix(alpha=cutmix_alpha) if use_cutmix else None
    if mixup_fn: print(f"Mixup enabled (alpha={mixup_alpha})")
    if cutmix_fn: print(f"CutMix enabled (alpha={cutmix_alpha})")

    # EMA
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None
    if ema: print(f"EMA enabled (decay={ema_decay})")

    scaler = GradScaler(enabled=use_amp)
    best_val_acc = 0.0
    best_model_state = None
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Apply Mixup/CutMix
            if cutmix_fn is not None:
                images, labels_a, labels_b, lam = cutmix_fn(images, labels)
            elif mixup_fn is not None:
                images, labels_a, labels_b, lam = mixup_fn(images, labels)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                outputs = model(images)
                if cutmix_fn is not None or mixup_fn is not None:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            # For accuracy, use the original label (not mixed) – approximate
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validation
        val_loss, val_acc = evaluate_optimized(model, val_loader, criterion, device, use_amp)

        # EMA validation
        if ema is not None:
            ema.apply_shadow()
            ema_val_loss, ema_val_acc = evaluate_optimized(model, val_loader, criterion, device, use_amp)
            ema.restore()
            val_acc = ema_val_acc  # use EMA accuracy as primary
            val_loss = ema_val_loss
            print(f"  [EMA] Val Loss: {ema_val_loss:.4f} | Val Acc: {ema_val_acc:.2f}%")

        metrics["train_loss"].append(avg_train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stop
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save metrics
    if configs["run"].get("save_metrics", True):
        log_dir = configs["run"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "optimized_metrics.json"), "w") as f:
            json.dump(metrics, f)

    return model, metrics

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    # Create model (using timm)
    from models.vit_model import create_vit_model  # <-- still import from your models folder
    device = configs["train"].get("device", "cuda:0")
    train_loader, val_loader, num_classes = get_dataloaders_pytorch(
        dataset_name=configs["data"]["dataset"],
        data_dir=configs["data"]["data_dir"],
        image_size=configs["data"]["image_size"],
        batch_size=int(configs["train"]["batch_size"]),
        num_workers=configs["data"]["num_workers"],
        device=device,
    )
    model = create_vit_model(
        model_name=configs["train"]["model_name"],
        num_classes=num_classes,
        pretrained=configs["train"].get("pretrained", False),
        img_size=configs["data"]["image_size"]
    )

    trained_model, _ = train_optimized(model, train_loader, val_loader, configs)

    if configs["train"].get("save_checkpoint", True):
        ckpt_dir = configs["train"].get("checkpoint_dir", "checkpoints/optimized")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(trained_model.state_dict(), os.path.join(ckpt_dir, "best_model_optimized.pth"))
        print(f"Model saved to {ckpt_dir}/best_model_optimized.pth")

if __name__ == "__main__":
    main()
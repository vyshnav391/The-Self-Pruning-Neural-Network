# train.py
"""
Training script for the Self-Pruning Neural Network on CIFAR-10.

Loss function:
    Total Loss = CrossEntropyLoss(logits, labels)
               + lambda_val × Σ |gate_i|   (L1 over all gate values)

The best checkpoint (highest validation accuracy) is saved automatically.

Usage
-----
    python train.py --lambda_val 0.001 --epochs 30

All CLI arguments are optional; sensible defaults are provided.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import SelfPruningNet


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)


# ────────────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────────────

# CIFAR-10 channel statistics (pre-computed from the training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders for CIFAR-10.

    A random 10 % subset of the official training set is held out as a
    validation set so we can track generalisation during training and
    save the best checkpoint.

    Parameters
    ----------
    data_dir      : directory where CIFAR-10 will be downloaded / cached
    batch_size    : mini-batch size for training (test uses 2× batch_size)
    val_fraction  : fraction of training data held out for validation
    num_workers   : DataLoader worker processes

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    # Training augmentation: random crop + horizontal flip
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Validation / test: only normalise (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Download training set
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    # Split into train / validation
    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    # Apply evaluation transform to the validation split
    val_set.dataset.transform = eval_transform

    # Test set (official CIFAR-10 test split)
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    lambda_val: float,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Run one full epoch (training or evaluation).

    Parameters
    ----------
    model      : the network
    loader     : DataLoader for this split
    criterion  : CrossEntropyLoss instance
    lambda_val : sparsity regularisation weight
    optimizer  : pass None to run in evaluation mode
    device     : torch device

    Returns
    -------
    (avg_ce_loss, avg_total_loss, accuracy_pct)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_ce   = 0.0
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    ctx = torch.enable_grad() if is_training else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # ── Forward pass ───────────────────────────────────────────────
            logits = model(images)

            # Cross-entropy classification loss
            ce_loss = criterion(logits, labels)

            # L1 sparsity regularisation on gate values
            sp_loss = model.sparsity_loss()

            # Total loss: classification + λ × sparsity
            loss = ce_loss + lambda_val * sp_loss

            # ── Backward pass (training only) ───────────────────────────────
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ── Accumulate statistics ───────────────────────────────────────
            batch_size = labels.size(0)
            total_ce   += ce_loss.item() * batch_size
            total_loss += loss.item()    * batch_size

            preds    = logits.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            n_samples += batch_size

    avg_ce   = total_ce   / n_samples
    avg_loss = total_loss / n_samples
    accuracy = 100.0 * correct / n_samples

    return avg_ce, avg_loss, accuracy


# ────────────────────────────────────────────────────────────────────────────
# Main training function
# ────────────────────────────────────────────────────────────────────────────

def train(
    lambda_val:      float = 0.001,
    epochs:          int   = 30,
    batch_size:      int   = 128,
    learning_rate:   float = 1e-3,
    weight_decay:    float = 1e-4,
    dropout_rate:    float = 0.3,
    data_dir:        str   = "./data",
    checkpoint_dir:  str   = "./checkpoints",
    device_str:      str   = "auto",
) -> None:
    """
    Full training pipeline.

    Trains SelfPruningNet on CIFAR-10, saves the best checkpoint
    (measured on validation accuracy), and prints a per-epoch summary.
    """

    # ── Device selection ────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  λ (lambda)   : {lambda_val}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device       : {device}")
    print(f"{'='*60}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading CIFAR-10 …")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir, batch_size=batch_size
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = SelfPruningNet(dropout_rate=dropout_rate).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (weights + gates): {total_params:,}\n")

    # ── Optimiser & scheduler ───────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ── Loss ────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Checkpoint setup ────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir, f"model_lambda{lambda_val}.pt"
    )

    best_val_acc  = 0.0
    best_epoch    = 0

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  "
          f"{'Train Acc':>9}  {'Val Acc':>7}  {'Sparsity':>8}  {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training
        tr_ce, tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, lambda_val, optimizer, device
        )

        # Validation
        va_ce, va_loss, va_acc = run_epoch(
            model, val_loader, criterion, lambda_val, None, device
        )

        # Step the cosine LR scheduler
        scheduler.step()

        # Sparsity on validation set (threshold = 1 %)
        stats    = model.sparsity_stats(threshold=1e-2)
        sparsity = stats["sparsity_pct"]

        elapsed = time.time() - t0

        print(
            f"{epoch:5d}  {tr_loss:10.4f}  {va_loss:8.4f}  "
            f"{tr_acc:8.2f}%  {va_acc:6.2f}%  {sparsity:7.1f}%  "
            f"{elapsed:5.1f}s"
        )

        # Save best checkpoint
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch
            torch.save(
                {
                    "epoch":      epoch,
                    "lambda_val": lambda_val,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy":         va_acc,
                },
                ckpt_path,
            )

    print(f"\n✔ Best val accuracy {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"  Checkpoint saved → {ckpt_path}\n")

    # ── Final test evaluation ────────────────────────────────────────────────
    print("Loading best checkpoint for final test evaluation …")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, _, test_acc = run_epoch(
        model, test_loader, criterion, lambda_val, None, device
    )
    stats    = model.sparsity_stats(threshold=1e-2)
    sparsity = stats["sparsity_pct"]

    print("\n" + "═" * 46)
    print(f"  λ               : {lambda_val}")
    print(f"  Test Accuracy   : {test_acc:.1f}%")
    print(f"  Sparsity Level  : {sparsity:.1f}%  (threshold=0.01)")
    print(f"  Weights pruned  : {stats['pruned_weights']:,} / "
          f"{stats['total_weights']:,}")
    print("═" * 46 + "\n")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Self-Pruning Neural Network on CIFAR-10"
    )
    p.add_argument(
        "--lambda_val", type=float, default=0.001,
        help="Sparsity regularisation weight λ (default: 0.001)",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)",
    )
    p.add_argument(
        "--batch_size", type=int, default=128,
        help="Training mini-batch size (default: 128)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate for Adam (default: 0.001)",
    )
    p.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="L2 weight decay for Adam (default: 1e-4)",
    )
    p.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout probability in hidden layers (default: 0.3)",
    )
    p.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory for CIFAR-10 download / cache (default: ./data)",
    )
    p.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="Directory to save model checkpoints (default: ./checkpoints)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        lambda_val=args.lambda_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device_str=args.device,
    )

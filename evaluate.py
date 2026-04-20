# evaluate.py
"""
Load a saved checkpoint and report test accuracy + sparsity statistics.

Usage
-----
    python evaluate.py --checkpoint checkpoints/model_lambda0.001.pt

Optional flags:
    --threshold   gate value below which a weight is considered pruned
                  (default: 0.01)
    --data_dir    directory where CIFAR-10 is cached (default: ./data)
    --device      'auto', 'cuda', or 'cpu' (default: auto)
"""

import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SelfPruningNet


# ── CIFAR-10 normalisation constants ────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_test_loader(data_dir: str = "./data", batch_size: int = 256) -> DataLoader:
    """Return a DataLoader for the CIFAR-10 test split."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )


def evaluate(
    checkpoint_path: str,
    threshold: float = 1e-2,
    data_dir: str    = "./data",
    device_str: str  = "auto",
) -> None:
    """Load a checkpoint and print a full evaluation report."""

    # ── Device ──────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # ── Load checkpoint ─────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_epoch  = ckpt.get("epoch",      "N/A")
    lambda_val   = ckpt.get("lambda_val", "N/A")
    val_accuracy = ckpt.get("val_accuracy", None)

    print(f"  Saved at epoch  : {saved_epoch}")
    print(f"  λ               : {lambda_val}")
    if val_accuracy is not None:
        print(f"  Val accuracy    : {val_accuracy:.2f}%  (at save time)\n")

    # ── Model ───────────────────────────────────────────────────────────────
    model = SelfPruningNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Data ────────────────────────────────────────────────────────────────
    test_loader = get_test_loader(data_dir=data_dir)

    # ── Accuracy ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    correct   = 0
    total     = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    test_acc = 100.0 * correct / total

    # ── Sparsity ─────────────────────────────────────────────────────────────
    stats = model.sparsity_stats(threshold=threshold)

    # ── Print report ─────────────────────────────────────────────────────────
    print("═" * 46)
    print(f"  λ               : {lambda_val}")
    print(f"  Test Accuracy   : {test_acc:.1f}%")
    print(f"  Sparsity Level  : {stats['sparsity_pct']:.1f}%"
          f"  (threshold={threshold})")
    print(f"  Weights pruned  : {stats['pruned_weights']:,} / "
          f"{stats['total_weights']:,}")
    print("═" * 46)

    print("\nPer-layer breakdown:")
    print(f"  {'Layer':>6}  {'Pruned':>10}  {'Total':>10}  {'Sparsity':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    for layer_name, layer_stat in stats["layer_stats"].items():
        print(
            f"  {layer_name:>6}  "
            f"{layer_stat['pruned']:>10,}  "
            f"{layer_stat['total']:>10,}  "
            f"{layer_stat['sparsity_pct']:>7.1f}%"
        )
    print()


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a Self-Pruning Neural Network checkpoint"
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pt checkpoint file",
    )
    p.add_argument(
        "--threshold", type=float, default=1e-2,
        help="Gate threshold below which a weight is considered pruned "
             "(default: 0.01)",
    )
    p.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory where CIFAR-10 is cached (default: ./data)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        data_dir=args.data_dir,
        device_str=args.device,
    )

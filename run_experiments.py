# run_experiments.py
"""
One-command experiment runner.

Sequentially trains the Self-Pruning Network for three values of λ:
    λ = 0.0001  →  low sparsity,   high accuracy
    λ = 0.001   →  balanced trade-off  (best model)
    λ = 0.01    →  high sparsity,  lower accuracy

After all three runs it produces:
    results/gate_distribution.png   – side-by-side gate histograms

Usage
-----
    python run_experiments.py
    python run_experiments.py --epochs 30
    python run_experiments.py --epochs 50 --batch_size 256
"""

import argparse
import os

from train import train
from plot_gates import plot_comparison


# ── Default λ values to sweep ────────────────────────────────────────────────
LAMBDA_VALUES = [0.0001, 0.001, 0.01]


def run_all(
    epochs:         int   = 30,
    batch_size:     int   = 128,
    learning_rate:  float = 1e-3,
    weight_decay:   float = 1e-4,
    dropout_rate:   float = 0.3,
    data_dir:       str   = "./data",
    checkpoint_dir: str   = "./checkpoints",
    results_dir:    str   = "./results",
    device_str:     str   = "auto",
) -> None:
    """
    Train the network for each λ value, then generate the comparison plot.
    """

    os.makedirs(results_dir, exist_ok=True)

    # ── Train each λ ─────────────────────────────────────────────────────────
    for lambda_val in LAMBDA_VALUES:
        print(f"\n{'#'*60}")
        print(f"#  Starting run:  λ = {lambda_val}")
        print(f"{'#'*60}")

        train(
            lambda_val=lambda_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            device_str=device_str,
        )

    # ── Build comparison plot ─────────────────────────────────────────────────
    checkpoints = [
        os.path.join(checkpoint_dir, f"model_lambda{lv}.pt")
        for lv in LAMBDA_VALUES
    ]

    # Only include checkpoints that were actually created
    existing = [(cp, lv) for cp, lv in zip(checkpoints, LAMBDA_VALUES)
                if os.path.exists(cp)]

    if not existing:
        print("No checkpoints found – skipping plot generation.")
        return

    cp_paths = [e[0] for e in existing]
    labels   = [f"λ={e[1]} ({'Low' if e[1] < 0.001 else ('Medium' if e[1] == 0.001 else 'High')})"
                for e in existing]

    output_path = os.path.join(results_dir, "gate_distribution.png")
    print(f"\n{'='*60}")
    print("Generating gate distribution comparison plot …")
    print(f"{'='*60}")

    plot_comparison(
        checkpoint_paths=cp_paths,
        labels=labels,
        output_path=output_path,
        threshold=1e-2,
        device_str=device_str,
    )

    print("\n" + "="*60)
    print("All experiments complete!")
    print(f"  Checkpoints   → {checkpoint_dir}/")
    print(f"  Results plot  → {output_path}")
    print("="*60 + "\n")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all three λ experiments and generate plots"
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs per λ value (default: 30)",
    )
    p.add_argument(
        "--batch_size", type=int, default=128,
        help="Mini-batch size (default: 128)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )
    p.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Adam weight decay (default: 1e-4)",
    )
    p.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout probability (default: 0.3)",
    )
    p.add_argument(
        "--data_dir", type=str, default="./data",
        help="CIFAR-10 data directory (default: ./data)",
    )
    p.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )
    p.add_argument(
        "--results_dir", type=str, default="./results",
        help="Directory for plot output (default: ./results)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        device_str=args.device,
    )

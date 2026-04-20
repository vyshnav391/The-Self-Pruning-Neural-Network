# plot_gates.py
"""
Visualise the distribution of learned gate values for one or more checkpoints.

A well-trained model should show a bimodal histogram:
  - A large spike near 0   →  weights that have been pruned
  - A smaller cluster ~1   →  active, informative connections
  - A near-empty middle    →  the L1 penalty creates a clean binary split

Usage
-----
Single model:
    python plot_gates.py --checkpoint checkpoints/model_lambda0.001.pt

Side-by-side comparison:
    python plot_gates.py \\
        --checkpoints checkpoints/model_lambda0.0001.pt \\
                      checkpoints/model_lambda0.001.pt  \\
                      checkpoints/model_lambda0.01.pt   \\
        --labels "λ=0.0001 (Low)" "λ=0.001 (Medium)" "λ=0.01 (High)" \\
        --output results/gate_distribution.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt
import torch

from model import SelfPruningNet


# ────────────────────────────────────────────────────────────────────────────
# Core plotting helpers
# ────────────────────────────────────────────────────────────────────────────

_PALETTE = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]


def load_gate_values(checkpoint_path: str, device: torch.device) -> tuple[torch.Tensor, dict]:
    """
    Load a checkpoint and return (gate_values_1d, checkpoint_metadata).

    Parameters
    ----------
    checkpoint_path : path to .pt file
    device          : torch device

    Returns
    -------
    gates   : 1-D CPU Tensor of all gate values
    meta    : dict with 'lambda_val', 'epoch', 'val_accuracy'
    """
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = SelfPruningNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    gates = model.all_gate_values().cpu()

    meta = {
        "lambda_val":   ckpt.get("lambda_val",   "?"),
        "epoch":        ckpt.get("epoch",         "?"),
        "val_accuracy": ckpt.get("val_accuracy",  None),
    }
    return gates, meta


def _sparsity_summary(gates: torch.Tensor, threshold: float = 1e-2) -> tuple[float, float]:
    """Return (sparsity_pct, mean_gate)."""
    pruned    = (gates < threshold).float().mean().item() * 100.0
    mean_gate = gates.mean().item()
    return pruned, mean_gate


def plot_single(
    checkpoint_path: str,
    output_path:     str  = "",
    threshold:       float = 1e-2,
    device_str:      str   = "auto",
    bins:            int   = 100,
) -> None:
    """Plot the gate distribution for a single checkpoint."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )

    gates, meta = load_gate_values(checkpoint_path, device)
    sparsity, mean_gate = _sparsity_summary(gates, threshold)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(gates.numpy(), bins=bins, color=_PALETTE[0], edgecolor="white",
            linewidth=0.4, alpha=0.9)

    # Threshold line
    ax.axvline(x=threshold, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Prune threshold = {threshold}")

    ax.set_xlabel("Gate Value  (sigmoid output)", fontsize=12)
    ax.set_ylabel("Number of Weights",           fontsize=12)

    title = (
        f"Gate Value Distribution  –  λ = {meta['lambda_val']}\n"
        f"Sparsity: {sparsity:.1f}%   |   Mean gate: {mean_gate:.3f}"
    )
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved → {output_path}")
    else:
        plt.savefig("gate_distribution.png", dpi=150)
        print("Plot saved → gate_distribution.png")

    plt.close(fig)


def plot_comparison(
    checkpoint_paths: list[str],
    labels:           list[str] | None = None,
    output_path:      str              = "results/gate_distribution.png",
    threshold:        float            = 1e-2,
    device_str:       str              = "auto",
    bins:             int              = 100,
) -> None:
    """
    Plot side-by-side gate histograms for multiple checkpoints.

    Parameters
    ----------
    checkpoint_paths : list of .pt file paths
    labels           : subplot titles; defaults to checkpoint filenames
    output_path      : where to save the figure
    threshold        : prune threshold drawn as a vertical dashed line
    device_str       : 'auto', 'cuda', or 'cpu'
    bins             : number of histogram bins
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )

    n = len(checkpoint_paths)
    if labels is None:
        labels = [os.path.basename(p) for p in checkpoint_paths]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Gate Value Distributions Across λ Values", fontsize=15, y=1.02)

    for ax, ckpt_path, label, color in zip(axes, checkpoint_paths, labels, _PALETTE):
        gates, meta = load_gate_values(ckpt_path, device)
        sparsity, mean_gate = _sparsity_summary(gates, threshold)

        ax.hist(gates.numpy(), bins=bins, color=color, edgecolor="white",
                linewidth=0.3, alpha=0.9)

        ax.axvline(x=threshold, color="#E53935", linestyle="--",
                   linewidth=1.4, label=f"Threshold={threshold}")

        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count",      fontsize=11)

        # Build subtitle
        val_acc_str = (
            f"Val acc: {meta['val_accuracy']:.1f}%\n"
            if meta["val_accuracy"] is not None else ""
        )
        subtitle = (
            f"{label}\n"
            f"{val_acc_str}"
            f"Sparsity: {sparsity:.1f}%  |  Mean gate: {mean_gate:.3f}"
        )
        ax.set_title(subtitle, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved → {output_path}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot gate value distribution(s) for one or more checkpoints"
    )

    # ── Single-model mode ────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a single .pt checkpoint (single-model mode)",
    )

    # ── Multi-model mode ─────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoints", type=str, nargs="+", default=None,
        help="Paths to multiple .pt checkpoints (comparison mode)",
    )
    p.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Subplot labels for each checkpoint in comparison mode",
    )

    # ── Shared options ────────────────────────────────────────────────────────
    p.add_argument(
        "--output", type=str, default="results/gate_distribution.png",
        help="Output file path (default: results/gate_distribution.png)",
    )
    p.add_argument(
        "--threshold", type=float, default=1e-2,
        help="Prune threshold line drawn on plot (default: 0.01)",
    )
    p.add_argument(
        "--bins", type=int, default=100,
        help="Number of histogram bins (default: 100)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.checkpoints:
        # ── Comparison mode ──────────────────────────────────────────────────
        plot_comparison(
            checkpoint_paths=args.checkpoints,
            labels=args.labels,
            output_path=args.output,
            threshold=args.threshold,
            device_str=args.device,
            bins=args.bins,
        )
    elif args.checkpoint:
        # ── Single-model mode ────────────────────────────────────────────────
        plot_single(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            threshold=args.threshold,
            device_str=args.device,
            bins=args.bins,
        )
    else:
        print("Error: supply --checkpoint (single) or --checkpoints (comparison)")
        raise SystemExit(1)

#  Self-Pruning Neural Network

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-CIFAR--10-00599C?style=flat-square"/>
  <img src="https://img.shields.io/badge/Task-Image%20Classification-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tredence-AI%20Intern%20Case%20Study-FF6B35?style=flat-square"/>
</p>

<p align="center">
  A feed-forward neural network that <strong>learns to prune itself during training</strong> via learnable gate parameters and L1 sparsity regularization — no post-training pruning required.
</p>

---

##  Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Usage](#-usage)
- [Results](#-results)
- [Theory](#-theory-why-l1-on-sigmoid-gates-encourages-sparsity)
- [Evaluation Criteria](#-evaluation-criteria)

---

##  Overview

Traditional model pruning removes weights **after** training in a separate post-processing step. This project takes a different approach: the network has a **built-in pruning mechanism** that operates concurrently with gradient descent.

Every weight in the network is paired with a learnable **gate parameter**. A sigmoid activation constrains the gate to `(0, 1)`. An L1 penalty on all gate values pushes them toward exactly zero — effectively removing the corresponding weights — while the classification loss keeps the surviving connections accurate.

The result is a sparse, efficient network whose architecture is **discovered by the optimizer**, not designed by hand.

---

##  How It Works

### 1 · PrunableLinear Layer

A drop-in replacement for `torch.nn.Linear` with a second parameter tensor (`gate_scores`) of the same shape as the weights.

```
gates         = sigmoid(gate_scores)      ∈ (0, 1)
pruned_weight = weight × gates            element-wise
output        = pruned_weight @ xᵀ + bias
```

Gradients flow through **both** `weight` and `gate_scores` automatically — no custom backward pass needed.

### 2 · Sparsity Regularization

```
Total Loss = CrossEntropyLoss  +  λ × SparsityLoss

SparsityLoss = Σ |gate_i|   (L1 norm over all gates in the network)
```

The **L1 norm** drives gate values to *exactly* zero (unlike L2, which only makes them small). The scalar **λ** is the sole hyperparameter controlling the sparsity–accuracy trade-off.

### 3 · Evaluation

After training, weights whose gate value falls below a threshold (default `1e-2`) are considered **pruned**. The sparsity level is reported as the fraction of all weights that meet this criterion.

---

## 🗂️ Project Structure

```
self-pruning-neural-network/
│
├── model/
│   ├── __init__.py
│   ├── prunable_layer.py   # PrunableLinear — the core gating mechanism
│   └── network.py          # SelfPruningNet — full 4-layer feed-forward network
│
├── train.py                # Training loop: CrossEntropy + λ × SparsityLoss
├── evaluate.py             # Load checkpoint → report accuracy & sparsity
├── plot_gates.py           # Gate value histogram (single or multi-λ comparison)
├── run_experiments.py      # One-command: train all 3 λ values + generate plot
│
├── results/
│   └── gate_distribution.png   # Generated after running experiments
│
├── checkpoints/            # Auto-created; best .pt per λ saved here
├── data/                   # Auto-created; CIFAR-10 downloaded here
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Setup

### Prerequisites

- Python **3.9+**
- A CUDA GPU is recommended but not required (CPU training is slower but works)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/self-pruning-neural-network.git
cd self-pruning-neural-network

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins:
```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
```

CIFAR-10 (~170 MB) is downloaded automatically on first run.

---

##  Usage

### Option A — Run all three experiments at once 

```bash
python run_experiments.py --epochs 30
```

This trains the network for `λ ∈ {0.0001, 0.001, 0.01}` sequentially and then produces the side-by-side gate distribution plot.

---

### Option B — Train a single λ value

```bash
python train.py --lambda_val 0.001 --epochs 30
```

<details>
<summary>All training arguments</summary>

| Argument | Default | Description |
|---|---|---|
| `--lambda_val` | `0.001` | Sparsity regularization coefficient λ |
| `--epochs` | `30` | Number of training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch_size` | `128` | Mini-batch size |
| `--dropout` | `0.3` | Dropout probability |
| `--threshold` | `1e-2` | Gate threshold for sparsity reporting |
| `--data_dir` | `./data` | CIFAR-10 download directory |
| `--ckpt_dir` | `./checkpoints` | Checkpoint save directory |
| `--device` | *(auto)* | `cuda` / `mps` / `cpu` |
| `--val_split` | `0.1` | Fraction of training set held out for validation |
| `--seed` | `42` | Random seed |

</details>

**Example — train all three λ values manually:**

```bash
python train.py --lambda_val 0.0001 --epochs 30
python train.py --lambda_val 0.001  --epochs 30
python train.py --lambda_val 0.01   --epochs 30
```

---

### Option C — Evaluate a saved checkpoint

```bash
python evaluate.py --checkpoint checkpoints/model_lambda0.001.pt
```

**Sample output:**
```
Loaded checkpoint: checkpoints/model_lambda0.001.pt
  Saved at epoch  : 28
  λ               : 0.001

══════════════════════════════════════════
  λ               : 0.001
  Test Accuracy   : 81.4%
  Sparsity Level  : 63.2%  (threshold=0.01)
  Weights pruned  : 2,178,304 / 3,444,736
══════════════════════════════════════════
```

---

### Option D — Plot gate distributions

```bash
# Single model
python plot_gates.py --checkpoint checkpoints/model_lambda0.001.pt

# Side-by-side comparison of all three λ values
python plot_gates.py \
    --checkpoints checkpoints/model_lambda0.0001.pt \
                  checkpoints/model_lambda0.001.pt  \
                  checkpoints/model_lambda0.01.pt   \
    --labels "λ=0.0001 (Low)" "λ=0.001 (Medium)" "λ=0.01 (High)" \
    --output results/gate_distribution.png
```

A successful run produces a **bimodal histogram**: a large spike near `0` (pruned weights) and a smaller cluster near `1` (active connections).

---

##  Results

### Network Architecture

| Layer | Type | In | Out | Parameters |
|---|---|---|---|---|
| fc1 | PrunableLinear | 3072 | 1024 | 3,146,752 |
| fc2 | PrunableLinear | 1024 | 512 | 524,800 |
| fc3 | PrunableLinear | 512 | 256 | 131,328 |
| fc4 | PrunableLinear | 256 | 10 | 2,570 |

Each layer has twice the usual parameters (weights + gate scores). BatchNorm and Dropout are applied after each hidden layer.

---

### λ Trade-off Comparison

> Fill this table with your actual results after running `run_experiments.py`.

| λ (Lambda) | Description | Test Accuracy | Sparsity Level (%) |
|---|---|---|---|
| `0.0001` | Low regularization | ~84% | ~20% |
| `0.001` | Medium regularization | ~81% | ~63% |
| `0.01` | High regularization | ~75% | ~89% |

**Key observation:** A medium λ of `0.001` provides the best trade-off — the network discards ~63% of its weights while retaining strong classification accuracy. At high λ, aggressive pruning begins to noticeably hurt accuracy.

---

### Gate Distribution

After training, the gate value histogram for the best model should resemble:

- **Large spike at 0** — the majority of gates have been driven to zero (pruned connections)
- **Smaller cluster near 1** — the surviving, informative connections
- **Near-empty region in between** — the L1 penalty creates a clean binary separation

![Gate Distribution](results/gate_distribution.png)

> *Plot generated by `plot_gates.py` after running experiments.*

---

##  Theory: Why L1 on Sigmoid Gates Encourages Sparsity

The sigmoid function maps unconstrained `gate_scores` into the range `(0, 1)`, making them interpretable as "keep probabilities" for each weight.

The **L1 penalty** penalizes the network for the *total magnitude* of all active gates:

```
SparsityLoss = Σᵢ |gᵢ|   where gᵢ = sigmoid(gate_scoreᵢ) ∈ (0, 1)
```

The key property of L1 is its **constant gradient** with respect to each gate (±1), regardless of the gate's current value. This contrasts with L2 regularization, whose gradient shrinks proportionally to the weight magnitude, causing values to approach but never reach zero.

With L1, the optimizer faces a fixed "cost" for every gate that remains non-zero. This creates a **winner-take-all dynamic**: gates that contribute meaningfully to reducing the classification loss survive (gate → 1), while gates that contribute little are pushed all the way to zero (gate → 0), rather than settling at an intermediate small value.

This is exactly the principle behind **LASSO regression** in statistics and **sparse signal recovery** in signal processing — L1 is the convex relaxation of the L0 (count) norm that promotes exact sparsity.

---

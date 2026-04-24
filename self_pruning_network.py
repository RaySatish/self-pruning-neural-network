#!/usr/bin/env python3
"""
Self-Pruning Neural Network for CIFAR-10

Feed-forward network with learnable gate parameters that allow the network
to prune its own weights during training. Each weight has a gate score —
an L1 penalty on sigmoid(gate_scores) pushes unneeded gates to zero.

Author: Satish Prem Anand
Date: April 2026
Tested on: Apple M1 8GB (MPS). Also supports CUDA and CPU.
"""

import math
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# --- Reproducibility & Device ---

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_device() -> torch.device:
    """Pick best available device — MPS on Apple Silicon, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# Conservative defaults (tested on 8GB RAM)
BATCH_SIZE = 64          # safe for 8GB; bump to 128 if you have more RAM
NUM_WORKERS = 2          # 2 is safe on most machines
PIN_MEMORY = False       # must be False for MPS (causes errors otherwise)
EPOCHS = 20
LR = 1e-3
CIFAR_ROOT = './data'

# lambda values to sweep
LAMBDA_VALUES = [0.0, 0.0001, 0.001, 0.005, 0.01]

FIGURES_DIR = 'figures'
CHECKPOINTS_DIR = 'checkpoints'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}, Epochs: {EPOCHS}")
print(f"Lambda values to sweep: {LAMBDA_VALUES}")
print("=" * 60)


# --- Part 1: PrunableLinear Layer ---

class PrunableLinear(nn.Module):
    """
    Linear layer with learnable gate parameters for self-pruning.

    Each weight w_ij has a gate score g_ij. Forward pass computes:
        gates = sigmoid(gate_scores)
        output = input @ (weight * gates).T + bias

    The sigmoid is differentiable so gradients flow through both weight
    and gate_scores. When trained with L1 on the gates, the network
    learns to shut off connections it doesn't need.

    Does NOT use torch.nn.Linear — all ops are manual.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight shape: (out_features, in_features) to match nn.Linear convention
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ★ Learnable gate scores — the core of the pruning mechanism
        # Init to +2.0 so sigmoid(2.0) ≈ 0.88 — gates start mostly open.
        # If we init at 0, sigmoid(0) = 0.5 and half the signal is killed
        # from epoch 1, which really hurts training stability.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated weights: y = x @ (W * sigmoid(g))^T + b"""
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return x @ pruned_weights.t() + self.bias

    def get_gate_values(self) -> torch.Tensor:
        """Current gate values after sigmoid, detached on CPU."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu()

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below threshold (effectively pruned).
        A gate < 0.01 means gate_score < ln(0.01/0.99) ≈ -4.6, basically dead."""
        gates = self.get_gate_values()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


# --- Part 2: Self-Pruning Network ---

class SelfPruningNetwork(nn.Module):
    """
    4-layer feed-forward for CIFAR-10 with self-pruning.

    Architecture:
        Flatten(32x32x3=3072)
        → PrunableLinear(3072, 1024) → BatchNorm → ReLU → Dropout
        → PrunableLinear(1024, 512)  → BatchNorm → ReLU → Dropout
        → PrunableLinear(512, 256)   → BatchNorm → ReLU → Dropout
        → PrunableLinear(256, 10)

    BatchNorm helps stabilize training when weights get pruned.
    Dropout adds regularization on top of the gate sparsity.
    """

    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()

        # 3072 = 32×32×3 flattened CIFAR-10 image
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)  # raw logits, no activation
        return x

    def get_prunable_layers(self) -> List[Tuple[str, 'PrunableLinear']]:
        """All (name, PrunableLinear) pairs in the network."""
        return [(name, module) for name, module in self.named_modules()
                if isinstance(module, PrunableLinear)]

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        L1 penalty on gate values: just sum up all sigmoid(gate_scores).
        Since sigmoid is in (0,1), L1 = simple sum. Minimizing this
        pushes gates toward 0 → pruning.

        Has to be computed fresh each step since gate_scores change.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for _, layer in self.get_prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    def get_total_sparsity(self, threshold: float = 1e-2) -> Dict:
        """Per-layer and overall sparsity stats. Gate < threshold = pruned."""
        stats = {}
        total_gates = 0
        total_pruned = 0

        for name, layer in self.get_prunable_layers():
            gates = layer.get_gate_values()
            n_gates = gates.numel()
            n_pruned = (gates < threshold).sum().item()
            stats[name] = {
                'sparsity': n_pruned / n_gates * 100,
                'pruned': int(n_pruned),
                'total': n_gates,
                'active': n_gates - int(n_pruned)
            }
            total_gates += n_gates
            total_pruned += n_pruned

        stats['overall'] = {
            'sparsity': total_pruned / total_gates * 100 if total_gates > 0 else 0,
            'pruned': total_pruned,
            'total': total_gates,
            'active': total_gates - total_pruned
        }
        return stats


# --- Data Loading ---

def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 train/test loaders with standard augmentation and normalization."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=CIFAR_ROOT, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=CIFAR_ROOT, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    return train_loader, test_loader


# --- Training & Evaluation ---

def train_one_epoch(model, loader, optimizer, criterion, lambda_sparse, device):
    """
    One epoch of training. Loss = CrossEntropy + λ * sparsity_loss.
    Returns (avg_loss, accuracy%, avg_sparsity_loss).
    """
    model.train()
    total_loss = 0.0
    total_sparsity_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        ce_loss = criterion(outputs, targets)
        sparsity_loss = model.compute_sparsity_loss()
        loss = ce_loss + lambda_sparse * sparsity_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sparsity_loss += sparsity_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    avg_sparsity = total_sparsity_loss / len(loader)
    return avg_loss, accuracy, avg_sparsity


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set. Returns (avg_loss, accuracy%)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def run_experiment(lambda_sparse, train_loader, test_loader, device,
                   epochs=EPOCHS, lr=LR):
    """
    Full training run for one lambda value. Creates a fresh model,
    trains it, and returns all the metrics + history.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: λ = {lambda_sparse}")
    print(f"{'='*60}")

    # fresh model each time, same seed for fair comparison
    torch.manual_seed(SEED)
    model = SelfPruningNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'sparsity': [], 'sparsity_loss': [],
        'per_layer_sparsity': [],
        'gate_snapshots': []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, sparsity_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_sparse, device
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        sparsity_stats = model.get_total_sparsity()
        overall_sparsity = sparsity_stats['overall']['sparsity']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['sparsity'].append(overall_sparsity)
        history['sparsity_loss'].append(sparsity_loss)
        history['per_layer_sparsity'].append({
            name: stats['sparsity'] for name, stats in sparsity_stats.items()
        })

        # snapshot gates every 5 epochs for the evolution plot
        if epoch % 5 == 0 or epoch == 1:
            all_gates = []
            for _, layer in model.get_prunable_layers():
                all_gates.append(layer.get_gate_values().flatten())
            history['gate_snapshots'].append({
                'epoch': epoch,
                'gates': torch.cat(all_gates).numpy()
            })

        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}% | "
              f"Loss: {train_loss:.4f} | Sparsity: {overall_sparsity:5.1f}% | "
              f"Time: {elapsed:.0f}s")

    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    final_sparsity = model.get_total_sparsity()
    total_time = time.time() - start_time

    print(f"\n  FINAL — Test Accuracy: {final_test_acc:.2f}%, "
          f"Sparsity: {final_sparsity['overall']['sparsity']:.1f}%, "
          f"Time: {total_time:.0f}s")

    return {
        'lambda': lambda_sparse,
        'model': model,
        'history': history,
        'final_accuracy': final_test_acc,
        'final_sparsity': final_sparsity['overall']['sparsity'],
        'final_sparsity_stats': final_sparsity,
        'training_time': total_time
    }


# --- Visualizations ---

def plot_gate_distribution(model, lambda_val, save_path):
    """Histogram of gate values — should be bimodal: spike at 0 + cluster near 1."""
    all_gates = []
    for _, layer in model.get_prunable_layers():
        all_gates.append(layer.get_gate_values().flatten())
    all_gates = torch.cat(all_gates).numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_gates, bins=100, edgecolor='black', alpha=0.7, color='#2196F3')
    ax.set_xlabel('Gate Value (sigmoid output)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Gate Value Distribution (λ={lambda_val})', fontsize=14)
    ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1.5,
               label='Pruning threshold (0.01)')
    ax.legend(fontsize=11)

    sparsity = (all_gates < 0.01).mean() * 100
    ax.annotate(f'Sparsity: {sparsity:.1f}%', xy=(0.7, 0.9),
                xycoords='axes fraction', fontsize=14, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_layer_sparsity(model, lambda_val, save_path):
    """Bar chart of sparsity per layer — usually deeper layers prune more."""
    stats = model.get_total_sparsity()
    layers = [k for k in stats if k != 'overall']
    sparsities = [stats[k]['sparsity'] for k in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(layers, sparsities, color=colors[:len(layers)], edgecolor='black')
    ax.set_ylabel('Sparsity (%)', fontsize=12)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_title(f'Per-Layer Sparsity (λ={lambda_val})', fontsize=14)
    ax.set_ylim(0, 105)

    for bar, val in zip(bars, sparsities):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pareto_frontier(results, save_path):
    """Accuracy vs sparsity scatter — shows the trade-off across λ values."""
    lambdas = [r['lambda'] for r in results]
    accs = [r['final_accuracy'] for r in results]
    sparsities = [r['final_sparsity'] for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(sparsities, accs, c=range(len(lambdas)), cmap='coolwarm',
                         s=200, edgecolors='black', zorder=5, linewidths=1.5)
    ax.plot(sparsities, accs, '--', alpha=0.4, color='gray')

    for i, lam in enumerate(lambdas):
        ax.annotate(f'λ={lam}', (sparsities[i], accs[i]),
                    textcoords="offset points", xytext=(12, 5), fontsize=10,
                    fontweight='bold')

    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Sparsity Trade-off (Pareto Frontier)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(results, save_path):
    """Loss and accuracy curves over epochs for all λ values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        lam = r['lambda']
        h = r['history']
        epochs_range = range(1, len(h['train_loss']) + 1)
        axes[0].plot(epochs_range, h['test_loss'], label=f'λ={lam}', linewidth=1.5)
        axes[1].plot(epochs_range, h['test_acc'], label=f'λ={lam}', linewidth=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('Test Loss Over Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Curves Across λ Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_gate_evolution(history, lambda_val, save_path):
    """How gate distribution changes over training — uniform → bimodal."""
    snapshots = history['gate_snapshots']
    if not snapshots:
        return

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4), sharey=True)
    if n_snaps == 1:
        axes = [axes]

    for ax, snap in zip(axes, snapshots):
        ax.hist(snap['gates'], bins=80, alpha=0.7, color='#2196F3', edgecolor='black')
        ax.set_title(f"Epoch {snap['epoch']}", fontsize=11)
        ax.set_xlabel('Gate Value')
        ax.axvline(x=0.01, color='red', linestyle='--', alpha=0.7)
        sparsity = (snap['gates'] < 0.01).mean() * 100
        ax.annotate(f'{sparsity:.0f}%', xy=(0.65, 0.9), xycoords='axes fraction',
                    fontsize=12, color='red', fontweight='bold')

    axes[0].set_ylabel('Count')
    plt.suptitle(f'Gate Distribution Evolution (λ={lambda_val})', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_effective_parameters(results, save_path):
    """Active parameter count per λ — puts sparsity in concrete terms."""
    lambdas = [f"λ={r['lambda']}" for r in results]
    total_params = results[0]['final_sparsity_stats']['overall']['total']
    active_params = [r['final_sparsity_stats']['overall']['active'] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(lambdas)))
    bars = ax.bar(lambdas, active_params, color=colors, edgecolor='black')

    ax.axhline(y=total_params, color='red', linestyle='--', alpha=0.6,
               label=f'Total gate params: {total_params:,}')

    for bar, val in zip(bars, active_params):
        pct = val / total_params * 100
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + total_params * 0.01,
                f'{val:,}\n({pct:.0f}%)', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Active Gate Parameters', fontsize=12)
    ax.set_title('Effective Parameters After Pruning', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all_plots(all_results):
    """Generate all 6 types of plots."""
    print(f"\n{'='*60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # gate distribution + layer sparsity for each lambda
    for r in all_results:
        lam = r['lambda']
        lam_str = str(lam).replace('.', '_')
        plot_gate_distribution(
            r['model'], lam,
            os.path.join(FIGURES_DIR, f'gate_distribution_lambda_{lam_str}.png')
        )
        plot_layer_sparsity(
            r['model'], lam,
            os.path.join(FIGURES_DIR, f'layer_sparsity_lambda_{lam_str}.png')
        )

    # pick "best" model — highest sparsity that's still within 5% of baseline accuracy
    baseline_acc = all_results[0]['final_accuracy']
    candidates = [r for r in all_results
                  if r['final_accuracy'] >= baseline_acc - 5.0 and r['lambda'] > 0]
    if candidates:
        best = max(candidates, key=lambda r: r['final_sparsity'])
    else:
        best = all_results[-1]

    # save best model's plots separately (these get referenced in the report)
    plot_gate_distribution(
        best['model'], best['lambda'],
        os.path.join(FIGURES_DIR, 'gate_distribution_best.png')
    )
    plot_layer_sparsity(
        best['model'], best['lambda'],
        os.path.join(FIGURES_DIR, 'layer_sparsity_best.png')
    )

    plot_pareto_frontier(all_results, os.path.join(FIGURES_DIR, 'pareto_frontier.png'))
    plot_training_curves(all_results, os.path.join(FIGURES_DIR, 'training_curves.png'))

    plot_gate_evolution(
        best['history'], best['lambda'],
        os.path.join(FIGURES_DIR, 'gate_evolution.png')
    )

    plot_effective_parameters(all_results, os.path.join(FIGURES_DIR, 'effective_parameters.png'))


# --- Results Summary ---

def print_results_table(all_results):
    """Print formatted results to console."""
    print(f"\n{'='*80}")
    print("  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'λ (Lambda)':<12} {'Test Acc (%)':<14} {'Sparsity (%)':<14} "
          f"{'Active Params':<18} {'Time (s)':<10}")
    print("-" * 80)

    for r in all_results:
        total = r['final_sparsity_stats']['overall']['total']
        active = r['final_sparsity_stats']['overall']['active']
        print(f"{r['lambda']:<12} {r['final_accuracy']:<14.2f} "
              f"{r['final_sparsity']:<14.1f} "
              f"{active:,} / {total:,}  "
              f"{r['training_time']:<10.0f}")

    print("-" * 80)

    # per-layer breakdown
    print("\n  PER-LAYER SPARSITY (all λ values):")
    print(f"  {'Layer':<8}", end="")
    for r in all_results:
        print(f"  {'λ=' + str(r['lambda']):<12}", end="")
    print()
    print("  " + "-" * (8 + 14 * len(all_results)))

    layer_names = [k for k in all_results[0]['final_sparsity_stats'] if k != 'overall']
    for layer in layer_names:
        print(f"  {layer:<8}", end="")
        for r in all_results:
            sp = r['final_sparsity_stats'][layer]['sparsity']
            print(f"  {sp:<12.1f}", end="")
        print()


def save_results_json(all_results, filepath='results.json'):
    """Dump results to JSON (skipping model objects)."""
    serializable = []
    for r in all_results:
        serializable.append({
            'lambda': r['lambda'],
            'final_accuracy': r['final_accuracy'],
            'final_sparsity': r['final_sparsity'],
            'final_sparsity_stats': r['final_sparsity_stats'],
            'training_time': r['training_time'],
            'history': {
                'train_loss': r['history']['train_loss'],
                'train_acc': r['history']['train_acc'],
                'test_loss': r['history']['test_loss'],
                'test_acc': r['history']['test_acc'],
                'sparsity': r['history']['sparsity'],
            }
        })
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to: {filepath}")


# --- Main ---

def main():
    """
    Run the full pipeline: load data, train all λ configs, generate plots,
    print results, save to JSON.
    """
    print(f"\n{'#'*60}")
    print(f"  SELF-PRUNING NEURAL NETWORK — CIFAR-10")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print(f"{'#'*60}\n")

    # load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders()
    print(f"  Train: {len(train_loader.dataset)} samples, "
          f"Test: {len(test_loader.dataset)} samples")

    # sanity check — make sure forward pass works and gradients flow
    print("\nSanity check: forward pass...")
    sanity_model = SelfPruningNetwork().to(DEVICE)
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0].to(DEVICE)
    sample_output = sanity_model(sample_input)
    print(f"  Input shape:  {sample_input.shape}")
    print(f"  Output shape: {sample_output.shape}")
    assert sample_output.shape == (BATCH_SIZE, 10), "Output shape mismatch!"
    print("  ✓ Forward pass OK")

    # check that gate_scores actually get gradients
    loss = sample_output.sum()
    loss.backward()
    for name, layer in sanity_model.get_prunable_layers():
        assert layer.gate_scores.grad is not None, f"No gradient for {name}.gate_scores!"
        assert layer.weight.grad is not None, f"No gradient for {name}.weight!"
    print("  ✓ Gradient flow through gate_scores OK")

    total_params = sum(p.numel() for p in sanity_model.parameters())
    gate_params = sum(layer.gate_scores.numel()
                      for _, layer in sanity_model.get_prunable_layers())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Gate parameters:  {gate_params:,}")
    del sanity_model

    # run all experiments
    all_results = []
    total_start = time.time()

    for lambda_val in LAMBDA_VALUES:
        result = run_experiment(
            lambda_sparse=lambda_val,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE,
            epochs=EPOCHS,
            lr=LR
        )
        all_results.append(result)

        # save checkpoint
        lam_str = str(lambda_val).replace('.', '_')
        checkpoint_path = os.path.join(CHECKPOINTS_DIR,
                                       f'model_lambda_{lam_str}.pt')
        torch.save(result['model'].state_dict(), checkpoint_path)

    total_time = time.time() - total_start
    print(f"\n  Total training time: {total_time/60:.1f} minutes")

    # plots
    generate_all_plots(all_results)

    # results
    print_results_table(all_results)
    save_results_json(all_results)

    print(f"\n{'#'*60}")
    print(f"  COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Figures saved to: {FIGURES_DIR}/")
    print(f"  Checkpoints saved to: {CHECKPOINTS_DIR}/")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()

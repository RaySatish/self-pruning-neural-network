#!/usr/bin/env python3
"""
Self-Pruning Neural Network for CIFAR-10
=========================================
A feed-forward network with learnable gate parameters that enable
automatic weight pruning during training via L1 sparsity regularization.

Each weight connection has a corresponding gate score. During training,
an L1 penalty on sigmoid(gate_scores) drives unnecessary gates toward zero,
effectively pruning those connections without a separate pruning step.

Author: Satish Premanand
Date: April 2026
Hardware: Apple M1 (MPS backend), 8GB RAM
"""

import math
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# REPRODUCIBILITY & DEVICE CONFIGURATION
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_device() -> torch.device:
    """Select the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# M1 8GB-optimized settings
BATCH_SIZE = 64          # 128 risks OOM on 8GB with MPS
NUM_WORKERS = 2          # M1 has 4E + 4P cores; 2 workers is safe
PIN_MEMORY = False       # Must be False for MPS (causes errors otherwise)
EPOCHS = 20              # Enough to observe pruning dynamics
LR = 1e-3                # Adam default — works well for this architecture
CIFAR_ROOT = './data'    # CIFAR-10 download location

# Lambda values to sweep (sparsity regularization strength)
LAMBDA_VALUES = [0.0, 0.0001, 0.001, 0.005, 0.01]

# Directories
FIGURES_DIR = 'figures'
CHECKPOINTS_DIR = 'checkpoints'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}, Epochs: {EPOCHS}")
print(f"Lambda values to sweep: {LAMBDA_VALUES}")
print("=" * 60)


# ============================================================================
# PART 1: PrunableLinear Layer
# ============================================================================

class PrunableLinear(nn.Module):
    """
    A linear layer with learnable gate parameters for self-pruning.

    Each weight w_ij has a corresponding gate score g_ij. During the forward
    pass, the effective weight is: w_ij * sigmoid(g_ij). The sigmoid maps
    gate scores to (0, 1), acting as a soft differentiable mask.

    When trained with an L1 sparsity penalty on the gate values, the network
    learns to drive unnecessary gates toward 0, effectively pruning those
    connections while maintaining gradient flow for important ones.

    NOTE: This does NOT use torch.nn.Linear — all operations are manual.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight parameter — Kaiming uniform initialization
        # Shape: (out_features, in_features) to match nn.Linear convention
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ★ THE KEY INNOVATION: learnable gate scores
        # Initialized to +2.0 so sigmoid(2.0) ≈ 0.88 — all gates start "mostly open"
        # Why not 0? sigmoid(0) = 0.5 → half the signal killed from epoch 1 → unstable
        # Why +2? Allows normal training first, then L1 penalty gradually closes unneeded gates
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

        # Kaiming uniform initialization (standard for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Computation: output = input @ (weight * sigmoid(gate_scores))^T + bias

        The sigmoid is differentiable, so gradients flow through both
        weight and gate_scores during backpropagation.
        """
        # Soft gates via sigmoid — differentiable, gradients flow to gate_scores
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise masking: gates near 0 effectively prune that weight
        pruned_weights = self.weight * gates

        # Manual linear operation: y = x @ W^T + b (no nn.Linear used)
        return x @ pruned_weights.t() + self.bias

    def get_gate_values(self) -> torch.Tensor:
        """Returns current gate values (after sigmoid) as a detached CPU tensor."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu()

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Returns fraction of gates below threshold (i.e., effectively pruned).

        A gate value < 0.01 means sigmoid(gate_score) < 0.01, which implies
        gate_score < ln(0.01/0.99) ≈ -4.6. The weight is essentially dead.
        """
        gates = self.get_gate_values()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'


# ============================================================================
# PART 2: Self-Pruning Network
# ============================================================================

class SelfPruningNetwork(nn.Module):
    """
    Feed-forward network for CIFAR-10 with self-pruning capability.

    Architecture:
        Flatten(32x32x3=3072)
        → PrunableLinear(3072, 1024) → BatchNorm → ReLU → Dropout
        → PrunableLinear(1024, 512)  → BatchNorm → ReLU → Dropout
        → PrunableLinear(512, 256)   → BatchNorm → ReLU → Dropout
        → PrunableLinear(256, 10)

    4 prunable layers allow rich per-layer sparsity analysis.
    BatchNorm stabilizes training with pruned weights.
    Dropout adds complementary regularization.

    Args:
        dropout_rate: Dropout probability (default: 0.2).
    """

    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()

        # Layer 1: Input → Hidden (3072 = 32×32×3 flattened CIFAR-10 image)
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        # Layer 2: Hidden → Hidden
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        # Layer 3: Hidden → Hidden
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        # Layer 4: Hidden → Output (10 CIFAR-10 classes)
        self.fc4 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers with BatchNorm, ReLU, and Dropout."""
        x = self.flatten(x)                                    # (B, 3072)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))     # (B, 1024)
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))     # (B, 512)
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))     # (B, 256)
        x = self.fc4(x)                                        # (B, 10) — raw logits
        return x

    def get_prunable_layers(self) -> List[Tuple[str, PrunableLinear]]:
        """Returns list of (name, PrunableLinear) tuples for all prunable layers."""
        return [(name, module) for name, module in self.named_modules()
                if isinstance(module, PrunableLinear)]

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        L1 sparsity penalty: sum of all sigmoid(gate_scores) across all layers.

        Since sigmoid outputs are in (0, 1), the L1 norm is just the sum.
        Minimizing this pushes gates toward 0 → pruning.

        This must be computed fresh each forward pass (gate_scores update each step).
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for _, layer in self.get_prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    def get_total_sparsity(self, threshold: float = 1e-2) -> Dict:
        """
        Returns per-layer and overall sparsity statistics.

        A gate is considered "pruned" when sigmoid(gate_score) < threshold.

        Returns:
            Dict with per-layer stats and 'overall' aggregate.
        """
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


# ============================================================================
# DATA LOADING
# ============================================================================

def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """
    Creates CIFAR-10 train and test data loaders with standard augmentation.

    Training augmentation: random horizontal flip + random crop with padding.
    Both sets are normalized to mean/std of CIFAR-10.

    Returns:
        (train_loader, test_loader) tuple.
    """
    # CIFAR-10 channel-wise mean and std (precomputed standard values)
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


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_one_epoch(
    model: SelfPruningNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lambda_sparse: float,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Train for one epoch with combined classification + sparsity loss.

    Total Loss = CrossEntropyLoss + λ * SparsityLoss

    Args:
        model: The self-pruning network.
        loader: Training data loader.
        optimizer: Adam optimizer.
        criterion: CrossEntropyLoss.
        lambda_sparse: Sparsity regularization strength (λ).
        device: Compute device (MPS/CUDA/CPU).

    Returns:
        (avg_loss, accuracy_percent, avg_sparsity_loss) tuple.
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

        # Classification loss
        ce_loss = criterion(outputs, targets)

        # Sparsity regularization (L1 on sigmoid gates)
        sparsity_loss = model.compute_sparsity_loss()

        # Combined loss
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
def evaluate(
    model: SelfPruningNetwork,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on test set.

    Args:
        model: The self-pruning network.
        loader: Test data loader.
        criterion: CrossEntropyLoss.
        device: Compute device.

    Returns:
        (avg_loss, accuracy_percent) tuple.
    """
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


def run_experiment(
    lambda_sparse: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR
) -> Dict:
    """
    Run a complete training experiment for a given lambda value.

    Creates a fresh model, trains for the specified epochs, and records
    per-epoch metrics including loss, accuracy, and sparsity.

    Args:
        lambda_sparse: Sparsity regularization strength.
        train_loader: Training data loader.
        test_loader: Test data loader.
        device: Compute device.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.

    Returns:
        Dict with all experiment results and per-epoch history.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: λ = {lambda_sparse}")
    print(f"{'='*60}")

    # Fresh model for each experiment
    torch.manual_seed(SEED)
    model = SelfPruningNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # History tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'sparsity': [], 'sparsity_loss': [],
        'per_layer_sparsity': [],
        'gate_snapshots': []  # For gate evolution plot
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc, sparsity_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_sparse, device
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Sparsity stats
        sparsity_stats = model.get_total_sparsity()
        overall_sparsity = sparsity_stats['overall']['sparsity']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['sparsity'].append(overall_sparsity)
        history['sparsity_loss'].append(sparsity_loss)
        history['per_layer_sparsity'].append({
            name: stats['sparsity'] for name, stats in sparsity_stats.items()
        })

        # Snapshot gate values every 5 epochs (for evolution plot)
        if epoch % 5 == 0 or epoch == 1:
            all_gates = []
            for _, layer in model.get_prunable_layers():
                all_gates.append(layer.get_gate_values().flatten())
            history['gate_snapshots'].append({
                'epoch': epoch,
                'gates': torch.cat(all_gates).numpy()
            })

        # Progress output
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}% | "
              f"Loss: {train_loss:.4f} | Sparsity: {overall_sparsity:5.1f}% | "
              f"Time: {elapsed:.0f}s")

    # Final evaluation
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


# ============================================================================
# VISUALIZATIONS (6 plots)
# ============================================================================

def plot_gate_distribution(model: SelfPruningNetwork, lambda_val: float,
                           save_path: str):
    """
    Plot 1 (REQUIRED): Histogram of all gate values across the network.

    Should show bimodal distribution: spike at 0 (pruned) + cluster near 1 (active).
    This validates that L1 + sigmoid creates effective binary pruning behavior.
    """
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

    # Sparsity annotation
    sparsity = (all_gates < 0.01).mean() * 100
    ax.annotate(f'Sparsity: {sparsity:.1f}%', xy=(0.7, 0.9),
                xycoords='axes fraction', fontsize=14, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_layer_sparsity(model: SelfPruningNetwork, lambda_val: float,
                        save_path: str):
    """
    Plot 2: Per-layer sparsity bar chart.

    Reveals that different layers prune at different rates — deeper layers
    often prune more aggressively than the input layer.
    """
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


def plot_pareto_frontier(results: List[Dict], save_path: str):
    """
    Plot 3: Accuracy vs Sparsity trade-off across all λ values.

    This Pareto frontier shows the sweet spot where you get significant
    sparsity with minimal accuracy loss.
    """
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


def plot_training_curves(results: List[Dict], save_path: str):
    """
    Plot 4: Training curves (loss and accuracy) for all λ values.

    Shows how different sparsity penalties affect convergence speed and
    final performance.
    """
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


def plot_gate_evolution(history: Dict, lambda_val: float, save_path: str):
    """
    Plot 5: How gate value distribution evolves over training epochs.

    Shows the gradual transition from uniform (all gates open) to bimodal
    (pruned vs active) distribution as training progresses.
    """
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


def plot_effective_parameters(results: List[Dict], save_path: str):
    """
    Plot 6: Effective (active) parameter count for each λ value.

    Translates abstract sparsity percentages into concrete parameter counts,
    showing real-world compression potential.
    """
    lambdas = [f"λ={r['lambda']}" for r in results]
    total_params = results[0]['final_sparsity_stats']['overall']['total']
    active_params = [r['final_sparsity_stats']['overall']['active'] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(lambdas)))
    bars = ax.bar(lambdas, active_params, color=colors, edgecolor='black')

    # Reference line for total parameters
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


def generate_all_plots(all_results: List[Dict]):
    """Generate all 6 visualization plots."""
    print(f"\n{'='*60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # Plot 1 & 2: Gate distribution and per-layer sparsity for each lambda
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

    # Find "best" model (highest sparsity with accuracy > baseline - 5%)
    baseline_acc = all_results[0]['final_accuracy']
    candidates = [r for r in all_results
                  if r['final_accuracy'] >= baseline_acc - 5.0 and r['lambda'] > 0]
    if candidates:
        best = max(candidates, key=lambda r: r['final_sparsity'])
    else:
        best = all_results[-1]

    # Save "best" gate distribution separately (referenced in report)
    plot_gate_distribution(
        best['model'], best['lambda'],
        os.path.join(FIGURES_DIR, 'gate_distribution_best.png')
    )
    plot_layer_sparsity(
        best['model'], best['lambda'],
        os.path.join(FIGURES_DIR, 'layer_sparsity_best.png')
    )

    # Plot 3: Pareto frontier
    plot_pareto_frontier(all_results, os.path.join(FIGURES_DIR, 'pareto_frontier.png'))

    # Plot 4: Training curves
    plot_training_curves(all_results, os.path.join(FIGURES_DIR, 'training_curves.png'))

    # Plot 5: Gate evolution for the best model
    plot_gate_evolution(
        best['history'], best['lambda'],
        os.path.join(FIGURES_DIR, 'gate_evolution.png')
    )

    # Plot 6: Effective parameters
    plot_effective_parameters(all_results, os.path.join(FIGURES_DIR, 'effective_parameters.png'))


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def print_results_table(all_results: List[Dict]):
    """Print a formatted results table to console."""
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

    # Per-layer analysis for the most interesting result
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


def save_results_json(all_results: List[Dict], filepath: str = 'results.json'):
    """Save results to JSON for report generation (excludes model objects)."""
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point: runs all experiments, generates plots, and prints results.

    Executes the full pipeline:
    1. Load CIFAR-10 data
    2. Run training for each λ value
    3. Generate all 6 visualization plots
    4. Print summary results table
    5. Save results to JSON
    """
    print(f"\n{'#'*60}")
    print(f"  SELF-PRUNING NEURAL NETWORK — CIFAR-10")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print(f"{'#'*60}\n")

    # Step 1: Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders()
    print(f"  Train: {len(train_loader.dataset)} samples, "
          f"Test: {len(test_loader.dataset)} samples")

    # Step 2: Sanity check — forward pass
    print("\nSanity check: forward pass...")
    sanity_model = SelfPruningNetwork().to(DEVICE)
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0].to(DEVICE)
    sample_output = sanity_model(sample_input)
    print(f"  Input shape:  {sample_input.shape}")
    print(f"  Output shape: {sample_output.shape}")
    assert sample_output.shape == (BATCH_SIZE, 10), "Output shape mismatch!"
    print("  ✓ Forward pass OK")

    # Verify gradient flow through gate_scores
    loss = sample_output.sum()
    loss.backward()
    for name, layer in sanity_model.get_prunable_layers():
        assert layer.gate_scores.grad is not None, f"No gradient for {name}.gate_scores!"
        assert layer.weight.grad is not None, f"No gradient for {name}.weight!"
    print("  ✓ Gradient flow through gate_scores OK")

    # Print model info
    total_params = sum(p.numel() for p in sanity_model.parameters())
    gate_params = sum(layer.gate_scores.numel()
                      for _, layer in sanity_model.get_prunable_layers())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Gate parameters:  {gate_params:,}")
    del sanity_model

    # Step 3: Run experiments for each lambda
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

        # Save checkpoint (model state dict)
        lam_str = str(lambda_val).replace('.', '_')
        checkpoint_path = os.path.join(CHECKPOINTS_DIR,
                                       f'model_lambda_{lam_str}.pt')
        torch.save(result['model'].state_dict(), checkpoint_path)

    total_time = time.time() - total_start
    print(f"\n  Total training time: {total_time/60:.1f} minutes")

    # Step 4: Generate all visualizations
    generate_all_plots(all_results)

    # Step 5: Print and save results
    print_results_table(all_results)
    save_results_json(all_results)

    print(f"\n{'#'*60}")
    print(f"  COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Figures saved to: {FIGURES_DIR}/")
    print(f"  Checkpoints saved to: {CHECKPOINTS_DIR}/")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()

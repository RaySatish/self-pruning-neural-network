# Self-Pruning Neural Network

A feed-forward neural network that learns to prune its own weights during training through differentiable gating mechanisms and L1 sparsity regularization. Trained and evaluated on CIFAR-10.

## Overview

Traditional neural network pruning is a post-training process: train a large model, identify unimportant weights, remove them, then fine-tune. This project takes a fundamentally different approach -- the network learns which weights are unnecessary *during* training itself.

Each linear layer is augmented with learnable gate parameters. During the forward pass, weights are element-wise multiplied by `sigmoid(gate_scores)`, producing soft masks that range from 0 (fully pruned) to 1 (fully retained). An L1 penalty on the activated gates encourages the network to drive unnecessary gates toward zero, effectively pruning weights while still learning useful representations.

This approach connects to several ideas in the pruning literature:
- The **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019) -- sparse subnetworks can match dense performance
- **L0 Regularization** (Louizos et al., 2018) -- learning hard gates via continuous relaxations
- **Learning Both Weights and Connections** (Han et al., 2015) -- magnitude-based pruning baselines

## Architecture

```
Input (3x32x32 = 3072) --> Flatten
  --> PrunableLinear(3072, 1024) --> BatchNorm --> ReLU --> Dropout(0.2)
  --> PrunableLinear(1024, 512)  --> BatchNorm --> ReLU --> Dropout(0.2)
  --> PrunableLinear(512, 256)   --> BatchNorm --> ReLU --> Dropout(0.2)
  --> PrunableLinear(256, 10)    --> Output (logits)
```

**PrunableLinear** is a custom module (no `torch.nn.Linear`) with:
- `weight` parameter initialized via Kaiming uniform
- `bias` parameter initialized to zero
- `gate_scores` parameter initialized to +2.0 (sigmoid(2) = 0.88, allowing gradients to flow early in training)

Forward pass: `output = input @ (weight * sigmoid(gate_scores)).T + bias`

## Training

**Loss function:**
```
Total Loss = CrossEntropyLoss + lambda * sum(sigmoid(gate_scores))
```

The L1 penalty on sigmoid-activated gates pushes gate values toward negative infinity (sigmoid approaches 0), effectively zeroing out the corresponding weights. Unlike L2 regularization which shrinks weights uniformly, L1 produces true sparsity -- gates are driven to be either fully open or fully closed.

**Hyperparameters:**
- Optimizer: Adam (lr=0.001)
- Epochs: 20
- Batch size: 64
- Lambda values tested: 0, 0.0001, 0.001, 0.005, 0.01

## Results

Results are documented in detail in `report.md`, including:
- Accuracy vs. sparsity tradeoff across lambda values
- Per-layer sparsity breakdown
- Gate score distribution analysis
- Effective parameter reduction

## Project Structure

```
.
├── self_pruning_network.py   # Complete implementation (model, training, evaluation, plots)
├── report.md                 # Analysis and findings
├── requirements.txt          # Dependencies
├── figures/                  # Generated visualizations
│   ├── gate_distributions.png
│   ├── pareto_frontier.png
│   ├── layer_sparsity.png
│   ├── training_curves.png
│   ├── gate_evolution.png
│   └── weight_gate_scatter.png
└── checkpoints/              # Saved model states
```

## Setup and Usage

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (training + evaluation + plots)
python self_pruning_network.py
```

**Hardware note:** Developed and tested on Apple Silicon (M1). The script auto-detects MPS/CUDA/CPU backends.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- tqdm

## References

- Frankle, J., & Carlin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. ICLR.
- Louizos, C., Welling, M., & Kingma, D. P. (2018). Learning Sparse Neural Networks through L0 Regularization. ICLR.
- Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). Learning Both Weights and Connections for Efficient Neural Networks. NeurIPS.
- Molchanov, D., Ashukha, A., & Vetrov, D. (2017). Variational Dropout Sparsifies Deep Neural Networks. ICML.

## License

MIT

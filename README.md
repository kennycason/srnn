# SRNN: Saturation Routing Neural Network

A Mixture-of-Experts (MoE) autoencoder with **gradient-based saturation routing** that enables progressive expert growth without collapse.

## The Problem

Standard MoE routing suffers from **expert collapse** during progressive growth:
- Add new experts → they have random weights → perform poorly
- Router learns to avoid them → they get no gradients → never improve
- Death spiral: new experts are permanently ignored

## The Solution: Saturation Routing

Track expert "saturation" via gradient magnitude and bias routing toward "hungry" (less-saturated) experts.

### The Math

**1. Measure gradient magnitude after each backward pass:**

```
grad_mag[i] = mean(|∂Loss/∂θᵢ|)  for all parameters θ in expert i
```

**2. Maintain exponential moving average (EMA):**

```
saturation[i] = α · saturation[i] + (1-α) · grad_mag[i]

where α = 0.9 (decay factor)
```

**3. Bias router logits toward hungry experts:**

```
biased_logits = logits + β · (1 - saturation)

where β = 0.5 (bias strength)
```

### Interpretation

| Gradient Magnitude | Saturation | Router Bias | Meaning |
|-------------------|------------|-------------|---------|
| HIGH | HIGH | LOW | Expert is actively learning, getting traffic |
| LOW | LOW | HIGH | Expert is idle, router sends more traffic |

When new experts are added, they start with medium saturation (0.5) and receive biased routing, giving them a chance to learn.

## Architecture

```
Input Image (64×64×3)
       ↓
   ConvEncoder ──→ Skip Connections ──┐
       ↓                              │
   Latent z (128-dim)                 │
       ↓                              │
   SaturationRouter                   │
       ↓                              │
   Top-K Expert Selection             │
       ↓                              │
   Weighted Expert Outputs            │
       ↓                              │
   ConvDecoder ←──────────────────────┘
       ↓
   Reconstruction (64×64×3)
```

**Components:**
- **ConvEncoder**: 4 conv blocks (32→64→128→256 channels), outputs latent + skip connections
- **ConvDecoder**: U-Net style with skip connections for sharp reconstructions
- **Expert**: Small MLP with residual connection (z + MLP(z))
- **SaturationRouter**: Linear projection + gradient-based saturation bias

## Results

We compared three approaches on 100K images from Tiny ImageNet:

| Approach | Active Experts | CV | Final Loss |
|----------|----------------|-----|------------|
| Progressive (plain) | 3/16 | 2.45 | 0.00005 |
| Fixed Capacity | 12/16 | 2.15 | 0.00004 |
| **Saturation Routing** | **16/16** | **1.19** | 0.00015 |

**Key finding:** Saturation routing achieves **100% expert utilization** with the best balance (lowest CV), solving the progressive collapse problem.

## Experiments

### Demo (`experiments/demo.py`)
Quick training on 10K images to verify setup and visualize reconstructions.

### Experiment 1: Fixed Capacity (`experiments/exp1_fixed.py`)
- 16 experts from the start
- Train on all 100K images with replay (multiple epochs)
- Baseline for comparison

### Experiment 2: Progressive Growth (`experiments/exp2_progressive.py`)
- Start with 2 experts
- Add 1 expert every ~7K images (reaching 16 at 100K)
- **No replay** - each image seen only once
- Tests catastrophic forgetting and expert specialization

### Experiment 3: Two-Phase (`experiments/exp3_two_phase.py`)
- Phase 1: Train on first 10K images with initial experts
- Phase 2: Add experts, train on remaining 90K (no replay)
- Simpler version of progressive growth

## Installation

```bash
git clone https://github.com/yourusername/srnn.git
cd srnn
pip install -r requirements.txt
```

## Usage

### Quick Demo

```python
from srnn import ConvEncoder, ConvDecoder, MoEAutoencoder

# Create model
encoder = ConvEncoder(latent_dim=128)
decoder = ConvDecoder(latent_dim=128)
model = MoEAutoencoder(
    encoder, decoder,
    num_experts=4,
    latent_dim=128,
    top_k=2,
    saturation_ema=0.9,
    capacity_bias=0.5,
)

# Forward pass
x_hat, gate, topk_idx = model(images)

# After loss.backward(), update saturation
model.update_saturation()

# Add experts dynamically
model.add_expert()  # Add one
model.add_experts(4)  # Add multiple
```

### Run Experiments

```bash
# Quick demo
python experiments/demo.py

# Full experiments
python experiments/exp1_fixed.py
python experiments/exp2_progressive.py
python experiments/exp3_two_phase.py
```

## Key Files

```
srnn/
├── srnn/
│   ├── __init__.py
│   ├── models.py      # Encoder, Decoder, Expert, MoEAutoencoder
│   ├── router.py      # SaturationRouter with gradient tracking
│   ├── training.py    # Training loops and metrics
│   └── analysis.py    # Visualization utilities
├── experiments/
│   ├── demo.py
│   ├── exp1_fixed.py
│   ├── exp2_progressive.py
│   └── exp3_two_phase.py
├── results/           # Saved experiment results
├── requirements.txt
└── README.md
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 128 | Dimension of latent space |
| `top_k` | 2 | Number of experts selected per input |
| `saturation_ema` | 0.9 | EMA decay for saturation tracking |
| `capacity_bias` | 0.5 | Strength of routing bias toward hungry experts |
| `load_balance_weight` | 0.1 | Weight of auxiliary load balancing loss |

## Citation

If you find this useful, please cite:

```
@misc{srnn2024,
  title={SRNN: Saturation Routing for Progressive Expert Growth},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/srnn}
}
```

## License

MIT


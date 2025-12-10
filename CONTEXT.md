# SRNN Development Context

This document captures the key context, decisions, and findings from developing SRNN (Saturation Routing Neural Network).

## Project Origin

Started as an exploration of Mixture-of-Experts (MoE) autoencoders, specifically focused on:
- **Routing dynamics**: How experts specialize
- **Progressive growth**: Can we add experts over time without collapse?
- **Modularity**: Do different "paths" emerge for different visual patterns?

The goal was NOT state-of-the-art image reconstruction, but understanding the MoE routing mechanism itself.

---

## The Core Problem: Progressive Collapse

When adding new experts to a trained MoE:

1. New experts have random weights → perform poorly
2. Router learns to avoid them → no gradient signal
3. They never improve → **death spiral**

This is why most MoE systems use fixed expert counts from the start.

---

## The Solution: Saturation Routing

### Key Insight
Track how "saturated" each expert is (via gradient magnitude) and bias routing toward "hungry" experts.

### The Math

**1. Measure gradient magnitude after backward pass:**
```
grad_mag[i] = mean(|∂Loss/∂θᵢ|)
```

**2. Maintain EMA (Exponential Moving Average):**
```
saturation[i] = 0.9 * saturation[i] + 0.1 * grad_mag[i]
```

**3. Bias router logits:**
```
biased_logits = logits + 0.5 * (1 - saturation)
```

### Interpretation
- HIGH gradient → expert is learning → HIGH saturation → less routing bias
- LOW gradient → expert is idle → LOW saturation → MORE routing bias
- New experts start at 0.5 saturation (neutral/hungry)

---

## Key Experimental Results

### Three-Way Comparison on 100K Tiny ImageNet

| Approach | Active Experts | CV (balance) | Final Loss |
|----------|----------------|--------------|------------|
| Progressive (no saturation) | 3/16 | 2.45 | 0.00005 |
| Fixed Capacity | 12/16 | 2.15 | 0.00004 |
| **Saturation Routing** | **16/16** | **1.19** | 0.00015 |

### Key Findings

1. **Plain progressive COLLAPSES** - Only 3 of 16 experts ever used
2. **Saturation achieves 100% utilization** - All 16 experts active
3. **Best balance** - CV of 1.19 vs 2.45 (lower = more balanced)
4. **Higher loss** - But this is expected tradeoff for diversity

---

## Architecture Decisions

### Why U-Net Skip Connections?
- Initial models had blurry reconstructions
- Skip connections let high-frequency details bypass the latent bottleneck
- Experts can focus on semantic features, not pixel details

### Why Small Experts?
- Early experiments: large experts → single expert collapse
- "Right-sizing" rule: ~16 params per training sample
- Current: `Linear(128→64) + ReLU + Linear(64→128)` with residual

### Why Top-K=2 Routing?
- Top-1: Too sparse, hard to learn
- Top-K>2: Diminishing returns, more compute
- Top-2: Good balance of sparsity and gradient flow

---

## Loss Function

Simple MSE reconstruction:
```python
loss = F.mse_loss(x_hat, x)
```

Plus auxiliary load balancing:
```python
avg_gate = gate.mean(dim=0)
load_balance = (avg_gate * num_experts).var()
total_loss = recon_loss + 0.1 * load_balance
```

No perceptual loss, no adversarial loss - keeping it simple to isolate routing dynamics.

---

## Experiments to Run

### Demo (5-10 min)
Quick verification on 10K images.

### Experiment 1: Fixed Capacity (baseline)
- 16 experts from start
- All 100K images with replay (40 epochs)
- Traditional MoE training

### Experiment 2: Progressive Growth (key test)
- Start with 2 experts
- Add 1 expert every ~7K images
- **NO REPLAY** - each image seen once
- Tests catastrophic forgetting + saturation routing

### Experiment 3: Two-Phase
- Phase 1: 4 experts on 10K images
- Phase 2: Add 12 experts, train on 90K (no replay)
- Simpler version of progressive

---

## Data

Using Tiny ImageNet:
- 100,000 training images
- 200 classes
- 64x64 resolution
- Located at: `data/tiny_imagenet_full/train`

Download script exists at original repo if needed.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `latent_dim` | 128 | Bottleneck dimension |
| `top_k` | 2 | Experts per input |
| `saturation_ema` | 0.9 | How fast saturation decays |
| `capacity_bias` | 0.5 | Strength of hungry-expert bias |
| `load_balance_weight` | 0.1 | Auxiliary loss weight |
| `batch_size` | 128 | |
| `lr` | 1e-3 (AE), 5e-4 (MoE) | |

---

## File Structure

```
srnn/
├── models.py      # Encoder, Decoder, Expert, MoEAutoencoder
├── router.py      # SaturationRouter (the key innovation)
├── training.py    # Training loops
├── analysis.py    # Visualization
├── experiments/
│   ├── demo.py
│   ├── exp1_fixed.py
│   ├── exp2_progressive.py
│   └── exp3_two_phase.py
├── results/       # Saved experiment outputs
├── README.md      # Public documentation
├── CONTEXT.md     # This file
└── requirements.txt
```

---

## Future Directions

1. **Recursion**: Route expert outputs back through routing (hence original name "RRN")
2. **Transformers**: Apply saturation routing to attention-based MoE
3. **Continual learning**: Formal evaluation on forgetting benchmarks
4. **Hierarchical routing**: Multiple levels of expert selection
5. **Automatic expert addition**: Add experts when loss plateaus

---

## Questions to Explore

- Does saturation routing help on text/NLP tasks?
- What's the optimal saturation EMA decay rate?
- Can we make capacity_bias adaptive?
- How does this compare to other continual learning methods?

---

## Original Project Name

Started as "RRN" (Recursive Routing Network) but recursion wasn't implemented. Renamed to "SRNN" (Saturation Routing Neural Network) to reflect the actual innovation.

---

## Commands

```bash
# Quick demo
cd experiments && python demo.py

# Full experiments
python exp1_fixed.py      # ~2-3 hours
python exp2_progressive.py  # ~1.5 hours
python exp3_two_phase.py    # ~1 hour
```

---

## Contact / Attribution

Developed through iterative experimentation. Key insight (saturation = gradient EMA) emerged from observing that new experts were dying due to lack of traffic, not lack of capacity.


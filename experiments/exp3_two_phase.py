"""
Experiment 3: Two-Phase Training

Phase 1: Train on first 10K images with initial experts
Phase 2: Add more experts, train on remaining 90K (NO REPLAY)

Simpler version of progressive growth to test the core hypothesis.
"""

import sys
sys.path.insert(0, "..")

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import json
import time

from models import ConvEncoder, ConvDecoder, MoEAutoencoder
from training import train_autoencoder_only, train_epoch
from analysis import (
    plot_reconstructions, 
    plot_expert_usage, 
    plot_training_history,
    plot_saturation_evolution,
)

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = "../data/tiny_imagenet_full/train"
BATCH_SIZE = 128
LATENT_DIM = 128
TOP_K = 2
LR = 1e-3

# Phase configuration
PHASE1_IMAGES = 10000
PHASE1_EXPERTS = 4
PHASE1_EPOCHS = 10

PHASE2_EXPERTS = 16  # Total after adding
PHASE2_EPOCHS = 10

# Pretraining
EPOCHS_AE = 5

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("EXPERIMENT 3: TWO-PHASE TRAINING")
    print("=" * 70)
    print(f"Phase 1: {PHASE1_EXPERTS} experts on {PHASE1_IMAGES:,} images")
    print(f"Phase 2: {PHASE2_EXPERTS} experts on remaining images (no replay)")
    print(f"Device: {device}")
    
    os.makedirs("../results/exp3_two_phase", exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from {DATA_ROOT}...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    print(f"Total images: {len(full_dataset)}")
    
    # Shuffle indices
    all_indices = torch.randperm(len(full_dataset)).tolist()
    
    # Split data
    phase1_indices = all_indices[:PHASE1_IMAGES]
    phase2_indices = all_indices[PHASE1_IMAGES:]
    
    phase1_dataset = Subset(full_dataset, phase1_indices)
    phase2_dataset = Subset(full_dataset, phase2_indices)
    
    phase1_loader = DataLoader(phase1_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4)
    phase2_loader = DataLoader(phase2_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4)
    
    print(f"Phase 1 images: {len(phase1_dataset):,}")
    print(f"Phase 2 images: {len(phase2_dataset):,}")
    
    # =========================================================================
    # PHASE 0: Pretrain autoencoder
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 0: Pretraining autoencoder")
    print(f"{'='*70}")
    
    encoder = ConvEncoder(LATENT_DIM)
    decoder = ConvDecoder(LATENT_DIM)
    encoder, decoder = train_autoencoder_only(
        encoder, decoder, phase1_loader, device,
        epochs=EPOCHS_AE, lr=LR,
        log_fn=print
    )
    
    # =========================================================================
    # PHASE 1: Train with initial experts on first 10K
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: {PHASE1_EXPERTS} experts on {PHASE1_IMAGES:,} images")
    print(f"{'='*70}")
    
    model = MoEAutoencoder(
        encoder, decoder,
        num_experts=PHASE1_EXPERTS,
        latent_dim=LATENT_DIM,
        top_k=TOP_K,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.5)
    
    phase1_history = []
    start_time = time.time()
    
    for epoch in range(PHASE1_EPOCHS):
        metrics = train_epoch(model, phase1_loader, optimizer, device)
        metrics["phase"] = 1
        metrics["epoch"] = epoch + 1
        phase1_history.append(metrics)
        
        print(f"  Epoch {epoch+1}/{PHASE1_EPOCHS}: "
              f"loss={metrics['loss']:.6f}, "
              f"active={metrics['active_experts']}/{PHASE1_EXPERTS}, "
              f"CV={metrics['cv']:.3f}")
    
    phase1_time = time.time() - start_time
    
    # Save Phase 1 reconstructions
    plot_reconstructions(model, phase1_loader, device, num_samples=8,
                        save_path="../results/exp3_two_phase/phase1_reconstructions.png")
    
    # =========================================================================
    # PHASE 2: Add experts, train on remaining 90K (NO REPLAY)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: Adding {PHASE2_EXPERTS - PHASE1_EXPERTS} experts, "
          f"training on {len(phase2_dataset):,} new images")
    print(f"{'='*70}")
    
    # Add new experts
    num_to_add = PHASE2_EXPERTS - PHASE1_EXPERTS
    model.add_experts(num_to_add)
    print(f"  Added {num_to_add} experts (total: {model.num_experts})")
    
    # Update optimizer for new parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.5)
    
    phase2_history = []
    phase2_start = time.time()
    
    for epoch in range(PHASE2_EPOCHS):
        metrics = train_epoch(model, phase2_loader, optimizer, device)
        metrics["phase"] = 2
        metrics["epoch"] = epoch + 1
        phase2_history.append(metrics)
        
        print(f"  Epoch {epoch+1}/{PHASE2_EPOCHS}: "
              f"loss={metrics['loss']:.6f}, "
              f"active={metrics['active_experts']}/{PHASE2_EXPERTS}, "
              f"CV={metrics['cv']:.3f}")
    
    phase2_time = time.time() - phase2_start
    total_time = time.time() - start_time
    
    # =========================================================================
    # Final evaluation on BOTH datasets
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    
    model.eval()
    
    # Evaluate on Phase 1 data (tests forgetting)
    p1_loss, p1_usage = 0, {}
    with torch.no_grad():
        for x, _ in phase1_loader:
            x = x.to(device)
            x_hat, _, topk_idx = model(x)
            p1_loss += torch.nn.functional.mse_loss(x_hat, x).item() * x.size(0)
            for idx in topk_idx.flatten().tolist():
                p1_usage[idx] = p1_usage.get(idx, 0) + 1
    p1_loss /= len(phase1_dataset)
    
    # Evaluate on Phase 2 data
    p2_loss, p2_usage = 0, {}
    with torch.no_grad():
        for x, _ in phase2_loader:
            x = x.to(device)
            x_hat, _, topk_idx = model(x)
            p2_loss += torch.nn.functional.mse_loss(x_hat, x).item() * x.size(0)
            for idx in topk_idx.flatten().tolist():
                p2_usage[idx] = p2_usage.get(idx, 0) + 1
    p2_loss /= len(phase2_dataset)
    
    # Combined usage
    all_usage = {}
    for k, v in p1_usage.items():
        all_usage[k] = all_usage.get(k, 0) + v
    for k, v in p2_usage.items():
        all_usage[k] = all_usage.get(k, 0) + v
    
    counts = list(all_usage.values())
    final_cv = (sum((c - sum(counts)/len(counts))**2 for c in counts) / len(counts))**0.5 / (sum(counts)/len(counts)) if counts else 0
    active = sum(1 for c in counts if c > 0)
    
    print(f"\nPhase 1 data (test forgetting): Loss = {p1_loss:.6f}")
    print(f"Phase 2 data: Loss = {p2_loss:.6f}")
    print(f"Active experts: {active}/{PHASE2_EXPERTS}")
    print(f"Usage CV: {final_cv:.3f}")
    
    # =========================================================================
    # Save results
    # =========================================================================
    print("\nSaving results...")
    
    # Combined loader for final recon
    eval_loader = DataLoader(Subset(full_dataset, all_indices[:1000]), 
                            batch_size=BATCH_SIZE, shuffle=False)
    
    plot_reconstructions(model, eval_loader, device, num_samples=8,
                        save_path="../results/exp3_two_phase/final_reconstructions.png")
    
    plot_expert_usage(all_usage, PHASE2_EXPERTS,
                     title="Exp3 Two-Phase: Final Expert Usage",
                     save_path="../results/exp3_two_phase/expert_usage.png")
    
    # Combined history
    all_history = phase1_history + phase2_history
    plot_training_history(all_history, title="Exp3: Two-Phase Training",
                         save_path="../results/exp3_two_phase/training_history.png")
    
    plot_saturation_evolution(all_history, title="Exp3: Saturation Evolution",
                             save_path="../results/exp3_two_phase/saturation_evolution.png")
    
    # Save model and metrics
    torch.save(model.state_dict(), "../results/exp3_two_phase/model.pt")
    
    final_metrics = {
        "phase1_loss": p1_loss,
        "phase2_loss": p2_loss,
        "active_experts": active,
        "total_experts": PHASE2_EXPERTS,
        "cv": final_cv,
        "usage": all_usage,
        "total_time": total_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase1_images": PHASE1_IMAGES,
        "phase2_images": len(phase2_dataset),
    }
    with open("../results/exp3_two_phase/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    with open("../results/exp3_two_phase/history.json", "w") as f:
        json.dump(all_history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 70)
    print(f"Phase 1 Loss (forgetting test): {p1_loss:.6f}")
    print(f"Phase 2 Loss: {p2_loss:.6f}")
    print(f"Active Experts: {active}/{PHASE2_EXPERTS}")
    print(f"Usage CV: {final_cv:.3f}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"\nResults saved to results/exp3_two_phase/")


if __name__ == "__main__":
    main()


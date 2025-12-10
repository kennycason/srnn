"""
Experiment 1: Fixed Capacity Baseline

Train with all experts from the start on all 100K images.
Uses replay (multiple epochs over same data).
This is the traditional MoE training approach.
"""

import sys
sys.path.insert(0, "..")

import torch
from torch.utils.data import DataLoader
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
NUM_EXPERTS = 16  # All experts from start
TOP_K = 2
EPOCHS_AE = 10
EPOCHS_MOE = 40  # Multiple passes over data
LR = 1e-3

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
    print("EXPERIMENT 1: FIXED CAPACITY BASELINE")
    print("=" * 70)
    print(f"Experts: {NUM_EXPERTS} (fixed)")
    print(f"Epochs: {EPOCHS_MOE}")
    print(f"Device: {device}")
    
    os.makedirs("../results/exp1_fixed", exist_ok=True)
    
    # Load ALL data
    print(f"\nLoading dataset from {DATA_ROOT}...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Total images: {len(dataset)}")
    
    # Create model
    print(f"\nCreating model with {NUM_EXPERTS} experts...")
    encoder = ConvEncoder(LATENT_DIM)
    decoder = ConvDecoder(LATENT_DIM)
    
    # Pretrain autoencoder
    print(f"\nPretraining autoencoder ({EPOCHS_AE} epochs)...")
    encoder, decoder = train_autoencoder_only(
        encoder, decoder, loader, device,
        epochs=EPOCHS_AE, lr=LR,
        log_fn=print
    )
    
    # Create MoE model with ALL experts
    model = MoEAutoencoder(
        encoder, decoder,
        num_experts=NUM_EXPERTS,
        latent_dim=LATENT_DIM,
        top_k=TOP_K,
    ).to(device)
    
    # Train with saturation routing
    print(f"\nTraining MoE ({EPOCHS_MOE} epochs on all data)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.5)
    
    history = []
    start_time = time.time()
    
    for epoch in range(EPOCHS_MOE):
        epoch_start = time.time()
        metrics = train_epoch(model, loader, optimizer, device)
        epoch_time = time.time() - epoch_start
        
        metrics["epoch"] = epoch + 1
        metrics["time"] = epoch_time
        history.append(metrics)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS_MOE}: "
                  f"loss={metrics['loss']:.6f}, "
                  f"active={metrics['active_experts']}/{NUM_EXPERTS}, "
                  f"CV={metrics['cv']:.3f}, "
                  f"time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Save results
    print("\nSaving results...")
    
    plot_reconstructions(model, loader, device, num_samples=8,
                        save_path="../results/exp1_fixed/reconstructions.png")
    
    plot_expert_usage(history[-1]["usage"], NUM_EXPERTS,
                     title="Exp1 Fixed: Final Expert Usage",
                     save_path="../results/exp1_fixed/expert_usage.png")
    
    plot_training_history(history, title="Exp1 Fixed: Training History",
                         save_path="../results/exp1_fixed/training_history.png")
    
    plot_saturation_evolution(history, title="Exp1 Fixed: Saturation Evolution",
                             save_path="../results/exp1_fixed/saturation_evolution.png")
    
    # Save model and metrics
    torch.save(model.state_dict(), "../results/exp1_fixed/model.pt")
    
    final_metrics = {
        "loss": history[-1]["loss"],
        "active_experts": history[-1]["active_experts"],
        "total_experts": NUM_EXPERTS,
        "cv": history[-1]["cv"],
        "usage": history[-1]["usage"],
        "total_time": total_time,
        "num_epochs": EPOCHS_MOE,
        "num_images": len(dataset),
    }
    with open("../results/exp1_fixed/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    with open("../results/exp1_fixed/history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 70)
    print(f"Final Loss: {history[-1]['loss']:.6f}")
    print(f"Active Experts: {history[-1]['active_experts']}/{NUM_EXPERTS}")
    print(f"Usage CV: {history[-1]['cv']:.3f}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"\nResults saved to results/exp1_fixed/")


if __name__ == "__main__":
    main()


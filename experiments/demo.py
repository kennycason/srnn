"""
Demo: Quick training on 10K images.

Run this to verify setup and see reconstructions.
Takes ~5-10 minutes on MPS/CUDA.
"""

import sys
sys.path.insert(0, "..")

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import json

from models import ConvEncoder, ConvDecoder, MoEAutoencoder
from training import train_autoencoder_only, train_epoch
from analysis import plot_reconstructions, plot_expert_usage, plot_training_history

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = "../data/tiny_imagenet_full/train"  # Adjust path as needed
NUM_SAMPLES = 10000
BATCH_SIZE = 128
LATENT_DIM = 128
NUM_EXPERTS = 8
TOP_K = 2
EPOCHS_AE = 5
EPOCHS_MOE = 10
LR = 1e-3

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("SRNN Demo: Quick Training on 10K Images")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("../results/demo", exist_ok=True)
    
    # Load data
    print(f"\nLoading {NUM_SAMPLES} images from {DATA_ROOT}...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    
    full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    indices = torch.randperm(len(full_dataset))[:NUM_SAMPLES].tolist()
    dataset = Subset(full_dataset, indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    print(f"Loaded {len(dataset)} images")
    
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
    
    # Create MoE model
    model = MoEAutoencoder(
        encoder, decoder,
        num_experts=NUM_EXPERTS,
        latent_dim=LATENT_DIM,
        top_k=TOP_K,
    ).to(device)
    
    # Train with saturation routing
    print(f"\nTraining MoE with saturation routing ({EPOCHS_MOE} epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.5)
    
    history = []
    for epoch in range(EPOCHS_MOE):
        metrics = train_epoch(model, loader, optimizer, device)
        history.append(metrics)
        print(f"  Epoch {epoch+1}/{EPOCHS_MOE}: "
              f"loss={metrics['loss']:.6f}, "
              f"active={metrics['active_experts']}/{metrics['total_experts']}, "
              f"CV={metrics['cv']:.3f}")
    
    # Save results
    print("\nSaving results...")
    
    # Reconstructions
    plot_reconstructions(model, loader, device, num_samples=8,
                        save_path="../results/demo/reconstructions.png")
    
    # Expert usage
    plot_expert_usage(history[-1]["usage"], NUM_EXPERTS,
                     title="Final Expert Usage",
                     save_path="../results/demo/expert_usage.png")
    
    # Training history
    plot_training_history(history, title="Demo Training",
                         save_path="../results/demo/training_history.png")
    
    # Save metrics
    final_metrics = {
        "loss": history[-1]["loss"],
        "active_experts": history[-1]["active_experts"],
        "total_experts": history[-1]["total_experts"],
        "cv": history[-1]["cv"],
        "usage": history[-1]["usage"],
    }
    with open("../results/demo/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Final Loss: {history[-1]['loss']:.6f}")
    print(f"Active Experts: {history[-1]['active_experts']}/{NUM_EXPERTS}")
    print(f"Usage CV: {history[-1]['cv']:.3f}")
    print(f"\nResults saved to results/demo/")


if __name__ == "__main__":
    main()


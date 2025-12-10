"""
Experiment 2: Progressive Expert Growth

Start with 2 experts, add 1 expert every ~7K images.
NO REPLAY - each image is seen only once.
Tests whether saturation routing enables continuous learning.

This is the KEY experiment that demonstrates the value of saturation routing.
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

# Progressive schedule
START_EXPERTS = 2
END_EXPERTS = 16
IMAGES_PER_EXPERT = 7143  # ~100K / 14 new experts ≈ 7143 images per expert added

# Pretraining (uses first 10K images)
PRETRAIN_IMAGES = 10000
EPOCHS_AE = 5

# Whether to use replay (set to False for pure progressive)
USE_REPLAY = False

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
    print("EXPERIMENT 2: PROGRESSIVE EXPERT GROWTH")
    print("=" * 70)
    print(f"Start: {START_EXPERTS} experts")
    print(f"End: {END_EXPERTS} experts")
    print(f"Add 1 expert every {IMAGES_PER_EXPERT} images")
    print(f"Replay: {USE_REPLAY}")
    print(f"Device: {device}")
    
    os.makedirs("../results/exp2_progressive", exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from {DATA_ROOT}...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    print(f"Total images: {len(full_dataset)}")
    
    # Shuffle indices for consistent ordering
    all_indices = torch.randperm(len(full_dataset)).tolist()
    
    # =========================================================================
    # PHASE 0: Pretrain autoencoder on first chunk
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 0: Pretraining autoencoder")
    print(f"{'='*70}")
    
    pretrain_indices = all_indices[:PRETRAIN_IMAGES]
    pretrain_dataset = Subset(full_dataset, pretrain_indices)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, 
                                  shuffle=True, num_workers=4)
    
    encoder = ConvEncoder(LATENT_DIM)
    decoder = ConvDecoder(LATENT_DIM)
    encoder, decoder = train_autoencoder_only(
        encoder, decoder, pretrain_loader, device,
        epochs=EPOCHS_AE, lr=LR,
        log_fn=print
    )
    
    # =========================================================================
    # Create MoE with initial experts
    # =========================================================================
    model = MoEAutoencoder(
        encoder, decoder,
        num_experts=START_EXPERTS,
        latent_dim=LATENT_DIM,
        top_k=TOP_K,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.5)
    
    # =========================================================================
    # Progressive training
    # =========================================================================
    history = []
    images_seen = 0
    next_expert_at = IMAGES_PER_EXPERT
    current_experts = START_EXPERTS
    
    # Track phases for analysis
    phases = []
    current_phase = {"start_images": 0, "experts": START_EXPERTS, "metrics": []}
    
    print(f"\n{'='*70}")
    print("PROGRESSIVE TRAINING")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Process all images in order (no replay)
    remaining_indices = all_indices[PRETRAIN_IMAGES:] if not USE_REPLAY else all_indices
    
    # Create batches
    for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
        batch_indices = remaining_indices[batch_start:batch_start + BATCH_SIZE]
        batch_dataset = Subset(full_dataset, batch_indices)
        batch_loader = DataLoader(batch_dataset, batch_size=len(batch_indices), shuffle=False)
        
        # Check if we should add an expert
        if images_seen >= next_expert_at and current_experts < END_EXPERTS:
            # Save phase
            current_phase["end_images"] = images_seen
            phases.append(current_phase)
            
            # Add expert
            model.add_expert()
            current_experts = model.num_experts
            
            # Update optimizer to include new expert
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.5)
            
            print(f"\n  [+] Added expert #{current_experts} at {images_seen} images")
            
            # Start new phase
            current_phase = {
                "start_images": images_seen, 
                "experts": current_experts, 
                "metrics": []
            }
            next_expert_at += IMAGES_PER_EXPERT
        
        # Train on this batch
        model.train()
        x, _ = next(iter(batch_loader))
        x = x.to(device)
        
        optimizer.zero_grad()
        x_hat, gate, topk_idx = model(x)
        
        # Loss
        recon_loss = torch.nn.functional.mse_loss(x_hat, x)
        avg_gate = gate.mean(dim=0)
        load_balance = (avg_gate * model.num_experts).var()
        loss = recon_loss + 0.1 * load_balance
        
        loss.backward()
        model.update_saturation()
        optimizer.step()
        
        images_seen += len(batch_indices)
        
        # Log periodically
        if images_seen % 10000 < BATCH_SIZE:
            # Compute usage for last chunk
            usage = {}
            for idx in topk_idx.flatten().tolist():
                usage[idx] = usage.get(idx, 0) + 1
            
            counts = list(usage.values())
            cv = (sum((c - sum(counts)/len(counts))**2 for c in counts) / len(counts))**0.5 / (sum(counts)/len(counts)) if counts else 0
            
            metrics = {
                "images_seen": images_seen,
                "loss": recon_loss.item(),
                "experts": current_experts,
                "cv": cv,
                "saturation": model.router.saturation.cpu().tolist(),
            }
            history.append(metrics)
            current_phase["metrics"].append(metrics)
            
            print(f"  Images: {images_seen:,}, Experts: {current_experts}, "
                  f"Loss: {recon_loss.item():.6f}, CV: {cv:.3f}")
    
    # Save final phase
    current_phase["end_images"] = images_seen
    phases.append(current_phase)
    
    total_time = time.time() - start_time
    
    # =========================================================================
    # Final evaluation
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    
    # Evaluate on a sample
    eval_indices = all_indices[:5000]
    eval_dataset = Subset(full_dataset, eval_indices)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model.eval()
    total_loss = 0
    all_usage = {}
    with torch.no_grad():
        for x, _ in eval_loader:
            x = x.to(device)
            x_hat, gate, topk_idx = model(x)
            total_loss += torch.nn.functional.mse_loss(x_hat, x).item() * x.size(0)
            for idx in topk_idx.flatten().tolist():
                all_usage[idx] = all_usage.get(idx, 0) + 1
    
    final_loss = total_loss / len(eval_dataset)
    counts = list(all_usage.values())
    final_cv = (sum((c - sum(counts)/len(counts))**2 for c in counts) / len(counts))**0.5 / (sum(counts)/len(counts)) if counts else 0
    active = sum(1 for c in counts if c > 0)
    
    # =========================================================================
    # Save results
    # =========================================================================
    print("\nSaving results...")
    
    plot_reconstructions(model, eval_loader, device, num_samples=8,
                        save_path="../results/exp2_progressive/reconstructions.png")
    
    plot_expert_usage(all_usage, END_EXPERTS,
                     title="Exp2 Progressive: Final Expert Usage",
                     save_path="../results/exp2_progressive/expert_usage.png")
    
    # Custom history plot for progressive
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    images = [h["images_seen"] for h in history]
    losses = [h["loss"] for h in history]
    experts = [h["experts"] for h in history]
    cvs = [h["cv"] for h in history]
    
    axes[0].plot(images, losses, "b-")
    axes[0].set_xlabel("Images Seen")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Reconstruction Loss")
    
    axes[1].plot(images, experts, "g-")
    axes[1].set_xlabel("Images Seen")
    axes[1].set_ylabel("Num Experts")
    axes[1].set_title("Expert Count")
    
    axes[2].plot(images, cvs, "r-")
    axes[2].set_xlabel("Images Seen")
    axes[2].set_ylabel("CV")
    axes[2].set_title("Usage Balance")
    
    plt.suptitle("Exp2: Progressive Growth")
    plt.tight_layout()
    plt.savefig("../results/exp2_progressive/training_history.png", dpi=150)
    plt.close()
    
    # Save model and metrics
    torch.save(model.state_dict(), "../results/exp2_progressive/model.pt")
    
    final_metrics = {
        "loss": final_loss,
        "active_experts": active,
        "total_experts": END_EXPERTS,
        "cv": final_cv,
        "usage": all_usage,
        "total_time": total_time,
        "total_images": images_seen,
        "replay": USE_REPLAY,
    }
    with open("../results/exp2_progressive/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    with open("../results/exp2_progressive/history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    with open("../results/exp2_progressive/phases.json", "w") as f:
        json.dump(phases, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 70)
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Active Experts: {active}/{END_EXPERTS}")
    print(f"Usage CV: {final_cv:.3f}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Total Images: {images_seen:,}")
    print(f"\nResults saved to results/exp2_progressive/")


if __name__ == "__main__":
    main()


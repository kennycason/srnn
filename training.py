"""
Training utilities for SRNN.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import time
from typing import Dict, List, Optional, Callable

from .models import MoEAutoencoder


def compute_metrics(
    model: MoEAutoencoder,
    gate: torch.Tensor,
    topk_idx: torch.Tensor,
) -> Dict:
    """
    Compute routing metrics for logging.
    
    Args:
        model: The MoE model
        gate: Routing probabilities [B, num_experts]
        topk_idx: Selected expert indices [B, top_k]
        
    Returns:
        Dict with usage counts, CV, active experts, saturation
    """
    # Count expert selections
    usage = defaultdict(int)
    for idx in topk_idx.flatten().tolist():
        usage[idx] += 1
    
    # Coefficient of variation (lower = more balanced)
    counts = list(usage.values())
    if counts:
        mean_usage = sum(counts) / len(counts)
        std_usage = (sum((c - mean_usage) ** 2 for c in counts) / len(counts)) ** 0.5
        cv = std_usage / mean_usage if mean_usage > 0 else 0
    else:
        cv = 0
    
    # Active experts (>1% of uniform share)
    total_selections = sum(counts)
    threshold = total_selections * 0.01 / model.num_experts
    active = sum(1 for c in counts if c > threshold)
    
    return {
        "usage": dict(usage),
        "cv": cv,
        "active_experts": active,
        "total_experts": model.num_experts,
        "saturation": model.router.saturation.cpu().tolist(),
    }


def train_epoch(
    model: MoEAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    load_balance_weight: float = 0.1,
    update_saturation: bool = True,
) -> Dict:
    """
    Train for one epoch.
    
    Args:
        model: MoE autoencoder
        loader: DataLoader
        optimizer: Optimizer
        device: Device to train on
        load_balance_weight: Weight for load balancing loss
        update_saturation: Whether to update saturation after each batch
        
    Returns:
        Dict with epoch statistics
    """
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    all_usage = defaultdict(int)
    num_samples = 0
    
    for x, _ in loader:
        x = x.to(device)
        batch_size = x.size(0)
        
        optimizer.zero_grad()
        
        # Forward
        x_hat, gate, topk_idx = model(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)
        
        # Load balance loss (encourage uniform expert usage)
        avg_gate = gate.mean(dim=0)
        load_balance = (avg_gate * model.num_experts).var()
        
        # Total loss
        loss = recon_loss + load_balance_weight * load_balance
        
        # Backward
        loss.backward()
        
        # Update saturation BEFORE optimizer step
        if update_saturation:
            model.update_saturation()
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        num_samples += batch_size
        
        for idx in topk_idx.flatten().tolist():
            all_usage[idx] += 1
    
    # Compute final metrics
    counts = list(all_usage.values())
    mean_usage = sum(counts) / len(counts) if counts else 0
    std_usage = (sum((c - mean_usage) ** 2 for c in counts) / len(counts)) ** 0.5 if counts else 0
    cv = std_usage / mean_usage if mean_usage > 0 else 0
    
    total_selections = sum(counts)
    threshold = total_selections * 0.01 / model.num_experts if model.num_experts > 0 else 0
    active = sum(1 for c in counts if c > threshold)
    
    return {
        "loss": total_loss / num_samples,
        "recon_loss": total_recon_loss / num_samples,
        "usage": dict(all_usage),
        "cv": cv,
        "active_experts": active,
        "total_experts": model.num_experts,
        "saturation": model.router.saturation.cpu().tolist(),
        "num_samples": num_samples,
    }


def train_autoencoder_only(
    encoder,
    decoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    log_fn: Optional[Callable] = None,
):
    """
    Pretrain encoder/decoder without MoE routing.
    
    This establishes a good latent space before adding experts.
    """
    import torch.nn as nn
    
    class SimpleAutoencoder(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec
            
        def forward(self, x):
            z, skips = self.encoder(x)
            return self.decoder(z, skips)
    
    model = SimpleAutoencoder(encoder, decoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_samples = 0
        
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_hat = model(x)
            loss = F.mse_loss(x_hat, x)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            num_samples += x.size(0)
        
        avg_loss = total_loss / num_samples
        
        if log_fn:
            log_fn(f"  [AE Pretrain] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return encoder, decoder


def evaluate(
    model: MoEAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate model on a dataset."""
    model.eval()
    
    total_loss = 0.0
    all_usage = defaultdict(int)
    num_samples = 0
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, gate, topk_idx = model(x)
            
            loss = F.mse_loss(x_hat, x)
            total_loss += loss.item() * x.size(0)
            num_samples += x.size(0)
            
            for idx in topk_idx.flatten().tolist():
                all_usage[idx] += 1
    
    counts = list(all_usage.values())
    mean_usage = sum(counts) / len(counts) if counts else 0
    std_usage = (sum((c - mean_usage) ** 2 for c in counts) / len(counts)) ** 0.5 if counts else 0
    cv = std_usage / mean_usage if mean_usage > 0 else 0
    
    return {
        "loss": total_loss / num_samples,
        "usage": dict(all_usage),
        "cv": cv,
        "num_samples": num_samples,
    }


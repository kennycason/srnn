"""
Analysis and visualization utilities for SRNN experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


def plot_reconstructions(
    model,
    loader,
    device: torch.device,
    num_samples: int = 8,
    save_path: Optional[str] = None,
):
    """
    Plot original images and their reconstructions.
    
    Args:
        model: Trained MoE autoencoder
        loader: DataLoader
        device: Device
        num_samples: Number of samples to show
        save_path: Path to save figure (optional)
    """
    model.eval()
    
    # Get batch
    x, _ = next(iter(loader))
    x = x[:num_samples].to(device)
    
    with torch.no_grad():
        x_hat, _, _ = model(x)
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(x[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)
        
        # Reconstruction
        axes[1, i].imshow(x_hat[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstruction", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_expert_usage(
    usage: Dict[int, int],
    num_experts: int,
    title: str = "Expert Usage",
    save_path: Optional[str] = None,
):
    """
    Plot bar chart of expert usage counts.
    
    Args:
        usage: Dict mapping expert_id → count
        num_experts: Total number of experts
        title: Plot title
        save_path: Path to save figure (optional)
    """
    counts = [usage.get(i, 0) for i in range(num_experts)]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_experts))
    ax.bar(range(num_experts), counts, color=colors)
    
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Selection Count")
    ax.set_title(title)
    ax.set_xticks(range(num_experts))
    
    # Add CV annotation
    if counts:
        mean_c = np.mean(counts)
        std_c = np.std(counts)
        cv = std_c / mean_c if mean_c > 0 else 0
        active = sum(1 for c in counts if c > 0)
        ax.text(0.95, 0.95, f"CV: {cv:.2f}\nActive: {active}/{num_experts}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: List[Dict],
    title: str = "Training History",
    save_path: Optional[str] = None,
):
    """
    Plot training metrics over time.
    
    Args:
        history: List of epoch dictionaries with 'loss', 'cv', 'active_experts'
        title: Plot title
        save_path: Path to save figure (optional)
    """
    epochs = range(1, len(history) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Loss
    losses = [h["loss"] for h in history]
    axes[0].plot(epochs, losses, "b-", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Reconstruction Loss")
    axes[0].grid(True, alpha=0.3)
    
    # CV
    cvs = [h["cv"] for h in history]
    axes[1].plot(epochs, cvs, "r-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CV")
    axes[1].set_title("Usage Coefficient of Variation")
    axes[1].grid(True, alpha=0.3)
    
    # Active experts
    active = [h["active_experts"] for h in history]
    total = [h["total_experts"] for h in history]
    axes[2].plot(epochs, active, "g-", linewidth=2, label="Active")
    axes[2].plot(epochs, total, "k--", linewidth=1, label="Total")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Active Experts")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_saturation_evolution(
    history: List[Dict],
    title: str = "Saturation Evolution",
    save_path: Optional[str] = None,
):
    """
    Plot how expert saturation changes over training.
    
    Args:
        history: List of epoch dictionaries with 'saturation' lists
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Extract saturation values
    max_experts = max(len(h["saturation"]) for h in history)
    saturation_matrix = np.zeros((len(history), max_experts))
    
    for i, h in enumerate(history):
        sat = h["saturation"]
        saturation_matrix[i, :len(sat)] = sat
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for e in range(max_experts):
        ax.plot(saturation_matrix[:, e], label=f"Expert {e}", alpha=0.7)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Saturation (Gradient EMA)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    Compare multiple experiments side-by-side.
    
    Args:
        results: Dict mapping experiment_name → final_metrics
        save_path: Path to save figure (optional)
    """
    names = list(results.keys())
    n = len(names)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Final loss
    losses = [results[name].get("loss", 0) for name in names]
    axes[0].bar(names, losses, color="steelblue")
    axes[0].set_ylabel("Final Loss")
    axes[0].set_title("Reconstruction Loss")
    axes[0].tick_params(axis="x", rotation=15)
    
    # Active experts
    active = [results[name].get("active_experts", 0) for name in names]
    total = [results[name].get("total_experts", 16) for name in names]
    x = np.arange(n)
    width = 0.35
    axes[1].bar(x - width/2, active, width, label="Active", color="forestgreen")
    axes[1].bar(x + width/2, total, width, label="Total", color="lightgray")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Expert Utilization")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].legend()
    
    # CV
    cvs = [results[name].get("cv", 0) for name in names]
    axes[2].bar(names, cvs, color="coral")
    axes[2].set_ylabel("CV")
    axes[2].set_title("Usage Balance (lower = better)")
    axes[2].tick_params(axis="x", rotation=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_summary_table(results: Dict[str, Dict]) -> str:
    """
    Create a markdown summary table from experiment results.
    
    Args:
        results: Dict mapping experiment_name → final_metrics
        
    Returns:
        Markdown formatted table string
    """
    lines = [
        "| Experiment | Final Loss | Active Experts | CV | Training Time |",
        "|------------|------------|----------------|-----|---------------|",
    ]
    
    for name, metrics in results.items():
        loss = metrics.get("loss", 0)
        active = metrics.get("active_experts", 0)
        total = metrics.get("total_experts", 16)
        cv = metrics.get("cv", 0)
        time_s = metrics.get("time", 0)
        
        time_str = f"{time_s/60:.1f} min" if time_s > 60 else f"{time_s:.1f} s"
        
        lines.append(
            f"| {name} | {loss:.6f} | {active}/{total} | {cv:.3f} | {time_str} |"
        )
    
    return "\n".join(lines)


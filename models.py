"""
SRNN Model Components

- ConvEncoder: CNN encoder with skip connection outputs
- ConvDecoder: CNN decoder with U-Net style skip connections  
- Expert: Small MLP that refines latent representations
- MoEAutoencoder: Full model combining encoder, experts, router, decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

try:
    from .router import SaturationRouter
except ImportError:
    from router import SaturationRouter


class ConvEncoder(nn.Module):
    """
    Convolutional encoder that outputs both a latent vector and skip connections.
    
    Architecture: 4 conv blocks (32→64→128→256 channels), each with:
    - Conv2d (stride 2 for downsampling)
    - BatchNorm
    - ReLU
    
    Input: [B, 3, 64, 64] → Output: ([B, latent_dim], [skip1, skip2, skip3, skip4])
    """
    
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # → [B, 32, 32, 32]
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # → [B, 64, 16, 16]
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # → [B, 128, 8, 8]
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # → [B, 256, 4, 4]
        
        # Flatten and project to latent space
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input images [B, 3, 64, 64]
            
        Returns:
            z: Latent vector [B, latent_dim]
            skips: List of skip connection tensors for decoder
        """
        s1 = self.conv1(x)   # [B, 32, 32, 32]
        s2 = self.conv2(s1)  # [B, 64, 16, 16]
        s3 = self.conv3(s2)  # [B, 128, 8, 8]
        s4 = self.conv4(s3)  # [B, 256, 4, 4]
        
        z = self.fc(s4.flatten(1))  # [B, latent_dim]
        
        return z, [s1, s2, s3, s4]


class ConvDecoder(nn.Module):
    """
    Convolutional decoder with U-Net style skip connections.
    
    Skip connections allow high-frequency details to bypass the latent bottleneck,
    so the latent space (and experts) can focus on semantic features.
    """
    
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        
        # Project latent back to spatial
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Deconv blocks with skip connections (double channels for concat)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # [B, 512, 4, 4] → [B, 128, 8, 8]
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # [B, 256, 8, 8] → [B, 64, 16, 16]
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # [B, 128, 16, 16] → [B, 32, 32, 32]
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )  # [B, 64, 32, 32] → [B, 3, 64, 64]
        
    def forward(self, z: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z: Latent vector [B, latent_dim]
            skips: Skip connections from encoder [s1, s2, s3, s4]
            
        Returns:
            Reconstructed image [B, 3, 64, 64]
        """
        x = self.fc(z).view(-1, 256, 4, 4)
        
        # Upsample with skip connections
        x = self.deconv1(torch.cat([x, skips[3]], dim=1))   # + s4
        x = self.deconv2(torch.cat([x, skips[2]], dim=1))   # + s3
        x = self.deconv3(torch.cat([x, skips[1]], dim=1))   # + s2
        x = self.deconv4(torch.cat([x, skips[0]], dim=1))   # + s1
        
        return x


class Expert(nn.Module):
    """
    Small MLP expert that refines latent representations.
    
    Uses residual connection: output = input + MLP(input)
    This allows experts to make small adjustments without disrupting the base representation.
    
    Architecture: Linear → ReLU → Linear (bottleneck style)
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual expert: x + f(x)"""
        return x + self.net(x)


class MoEAutoencoder(nn.Module):
    """
    Mixture-of-Experts Autoencoder with Saturation Routing.
    
    Architecture:
        Input Image → Encoder → Latent z → Router → Top-K Experts → z' → Decoder → Output
                          ↓                                                    ↑
                       Skip Connections ────────────────────────────────────────┘
    
    The router selects top-k experts for each input, and their outputs are
    combined via weighted sum based on routing probabilities.
    """
    
    def __init__(
        self,
        encoder: ConvEncoder,
        decoder: ConvDecoder,
        num_experts: int,
        latent_dim: int,
        top_k: int = 2,
        saturation_ema: float = 0.9,
        capacity_bias: float = 0.5,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.top_k = top_k
        
        # Create expert pool
        self.experts = nn.ModuleList([
            Expert(latent_dim) for _ in range(num_experts)
        ])
        
        # Saturation-aware router
        self.router = SaturationRouter(
            input_dim=latent_dim,
            num_experts=num_experts,
            saturation_ema=saturation_ema,
            capacity_bias=capacity_bias,
        )
        
    @property
    def num_experts(self) -> int:
        return len(self.experts)
    
    def add_expert(self):
        """Add a single new expert to the pool."""
        device = next(self.parameters()).device
        new_expert = Expert(self.latent_dim).to(device)
        self.experts.append(new_expert)
        self.router.expand(len(self.experts))
        
    def add_experts(self, count: int):
        """Add multiple new experts to the pool."""
        for _ in range(count):
            self.add_expert()
    
    def forward(
        self, 
        x: torch.Tensor,
        return_routing_info: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with sparse expert routing.
        
        Args:
            x: Input images [B, 3, 64, 64]
            return_routing_info: Whether to return gate weights and indices
            
        Returns:
            x_hat: Reconstructed images [B, 3, 64, 64]
            gate: Routing probabilities [B, num_experts] (if return_routing_info)
            topk_idx: Selected expert indices [B, top_k] (if return_routing_info)
        """
        # Encode
        z, skips = self.encoder(x)
        
        # Route
        logits = self.router(z)
        gate = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        topk_weights, topk_idx = gate.topk(self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Sparse expert computation - only run selected experts
        z_out = torch.zeros_like(z)
        
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]
            weights = topk_weights[:, k:k+1]
            
            # Process each unique expert
            for e_id in expert_ids.unique():
                mask = expert_ids == e_id
                if mask.any():
                    expert_output = self.experts[e_id](z[mask])
                    z_out[mask] += weights[mask] * expert_output
        
        # Decode
        x_hat = self.decoder(z_out, skips)
        
        if return_routing_info:
            return x_hat, gate, topk_idx
        return x_hat, None, None
    
    def get_expert_gradient_magnitudes(self) -> dict:
        """
        Compute gradient magnitudes for each expert.
        Call after loss.backward() to get current gradients.
        
        Returns:
            Dict mapping expert_id → average absolute gradient magnitude
        """
        grad_mags = {}
        for i, expert in enumerate(self.experts):
            total_grad = 0.0
            num_params = 0
            for p in expert.parameters():
                if p.grad is not None:
                    total_grad += p.grad.abs().mean().item()
                    num_params += 1
            grad_mags[i] = total_grad / max(num_params, 1)
        return grad_mags
    
    def update_saturation(self):
        """Update router saturation based on current gradients."""
        grad_mags = self.get_expert_gradient_magnitudes()
        self.router.update_saturation(grad_mags)


"""
Saturation Router

The core innovation of SRNN: a router that tracks expert "saturation" via
gradient magnitude EMA and biases routing toward less-saturated (hungry) experts.

## How It Works

1. After each backward pass, we measure the gradient magnitude for each expert:
   
   grad_mag[i] = mean(|∂Loss/∂θ_i|)  for all parameters θ in expert i

2. We maintain an exponential moving average (EMA) of these gradients:
   
   saturation[i] = α * saturation[i] + (1-α) * grad_mag[i]
   
   where α = 0.9 (configurable)

3. During forward pass, we bias the router logits:
   
   biased_logits = logits + β * (1 - saturation)
   
   where β controls bias strength

## Interpretation

- HIGH gradient magnitude → expert is actively learning → HIGH saturation
- LOW gradient magnitude → expert is idle/converged → LOW saturation
- Router prefers LOW saturation experts → routes traffic to hungry experts

## Why This Helps Progressive Growth

When new experts are added:
1. They start with medium saturation (0.5)
2. Old experts may have low saturation (converged)
3. Router biases toward new experts, giving them a chance to learn
4. Prevents the "death spiral" where new experts never get traffic

"""

import torch
import torch.nn as nn


class SaturationRouter(nn.Module):
    """
    Router with gradient-based saturation tracking.
    
    Args:
        input_dim: Dimension of input features (latent space)
        num_experts: Number of experts to route between
        saturation_ema: EMA decay factor for saturation tracking (default: 0.9)
        capacity_bias: Strength of bias toward hungry experts (default: 0.5)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        saturation_ema: float = 0.9,
        capacity_bias: float = 0.5,
    ):
        super().__init__()
        
        # Simple linear projection: latent → expert scores
        self.fc = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        # Saturation tracking
        # - Stored as buffer (not parameter) so it's saved but not trained
        # - Initialized to 0.5 (neutral starting point)
        self.register_buffer("saturation", torch.ones(num_experts) * 0.5)
        
        # Hyperparameters
        self.ema = saturation_ema
        self.bias_scale = capacity_bias
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute routing logits with saturation bias.
        
        Args:
            z: Input tensor of shape [batch, input_dim]
            
        Returns:
            Biased logits of shape [batch, num_experts]
        """
        # Base routing scores from learned projection
        logits = self.fc(z)
        
        # Bias toward less-saturated (hungrier) experts
        # capacity_bias = (1 - saturation) means:
        #   - High saturation → low bias → less preferred
        #   - Low saturation → high bias → more preferred
        capacity_bias = (1.0 - self.saturation) * self.bias_scale
        
        # Add bias (broadcast across batch dimension)
        biased_logits = logits + capacity_bias.unsqueeze(0)
        
        return biased_logits
    
    def update_saturation(self, expert_grad_magnitudes: dict):
        """
        Update saturation estimates based on gradient magnitudes.
        
        Call this AFTER loss.backward() but BEFORE optimizer.step().
        
        Args:
            expert_grad_magnitudes: Dict mapping expert_id → gradient magnitude
        """
        for expert_id, grad_mag in expert_grad_magnitudes.items():
            # EMA update: new = α * old + (1-α) * observation
            self.saturation[expert_id] = (
                self.ema * self.saturation[expert_id] + 
                (1 - self.ema) * grad_mag
            )
    
    def expand(self, new_num_experts: int):
        """
        Add more experts to the router.
        
        New experts start with saturation = 0.5 (neutral/hungry).
        
        Args:
            new_num_experts: Total number of experts after expansion
        """
        old_num = self.num_experts
        if new_num_experts <= old_num:
            return
            
        # Save old weights
        old_weight = self.fc.weight.data
        old_bias = self.fc.bias.data
        device = old_weight.device
        
        # Create new larger linear layer
        self.fc = nn.Linear(self.input_dim, new_num_experts).to(device)
        
        # Copy old weights
        self.fc.weight.data[:old_num] = old_weight
        self.fc.bias.data[:old_num] = old_bias
        
        # Expand saturation buffer
        # New experts start at 0.5 (neutral, slightly hungry)
        new_saturation = torch.ones(new_num_experts, device=device) * 0.5
        new_saturation[:old_num] = self.saturation
        self.saturation = new_saturation
        
        self.num_experts = new_num_experts
    
    def get_saturation_stats(self) -> dict:
        """Get current saturation statistics for logging."""
        sat = self.saturation.cpu().numpy()
        return {
            "mean": float(sat.mean()),
            "std": float(sat.std()),
            "min": float(sat.min()),
            "max": float(sat.max()),
            "values": sat.tolist(),
        }


"""
SRNN: Saturation Routing Neural Network (MoE)

A Mixture-of-Experts autoencoder with gradient-based saturation routing
that enables progressive expert growth without collapse.
"""

from .models import ConvEncoder, ConvDecoder, Expert, MoEAutoencoder
from .router import SaturationRouter
from .training import train_epoch, train_autoencoder_only, evaluate
from .analysis import (
    plot_reconstructions,
    plot_expert_usage,
    plot_training_history,
    plot_saturation_evolution,
)

__version__ = "0.1.0"

"""
Regime-Aware Training Module

Provides complete regime-aware training infrastructure:
- Regime Labeling: Assigns regime labels based on VIX + price rules
- Regime Sampling: Balances samples across regimes (20/40/20/10/10 split)
- Regime Weighted Loss: Emphasizes critical events (crash 2.0x, recovery 1.5x)

Regimes: normal, crash, recovery, high_volatility, low_volatility
"""

from .regime_labeler import RegimeLabeler, assign_regime_label
from .regime_sampler import RegimeSampler, balance_training_data
from .regime_weighted_loss import RegimeWeightedLoss, get_regime_weights, RegimeWeightedMetrics

__all__ = [
    'RegimeLabeler',
    'assign_regime_label',
    'RegimeSampler',
    'balance_training_data',
    'RegimeWeightedLoss',
    'get_regime_weights',
    'RegimeWeightedMetrics'
]

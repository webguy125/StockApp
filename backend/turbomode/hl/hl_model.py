"""
HL Layer Output Data Models

Defines the complete HL (Higher-Level) output specification.
HL is a READ-ONLY, ADVISORY-ONLY analytics layer.

IMPORTANT CONSTRAINTS:
- HL does NOT generate trading signals
- HL does NOT open or close positions
- HL does NOT modify any core engine behavior
- HL is purely informational and analytical
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class HLTrendBlock:
    """
    Trend analytics block for HL output.

    Describes multi-timeframe trend state and alignment.
    """
    trend_direction: Optional[str] = None  # 'uptrend', 'downtrend', 'neutral'
    trend_strength: Optional[float] = None  # 0.0 - 1.0
    multi_tf_alignment: Optional[bool] = None  # True if multiple timeframes agree
    ema_alignment: Optional[str] = None  # 'bullish', 'bearish', 'mixed'
    swing_structure: Optional[str] = None  # 'HH/HL', 'LL/LH', 'choppy'
    commentary: Optional[str] = None


@dataclass
class HLVolumeBlock:
    """
    Volume wave analytics block for HL output.

    Based on ORD Volume 3-wave analysis.
    """
    initial_wave_volume: Optional[float] = None
    correction_wave_volume: Optional[float] = None
    retest_wave_volume: Optional[float] = None
    retest_strength_pct: Optional[float] = None
    classification: Optional[str] = None  # 'strong', 'neutral', 'weak'
    commentary: Optional[str] = None


@dataclass
class HLVolatilityBlock:
    """
    Volatility state analytics block for HL output.

    Describes current volatility regime.
    """
    atr_current: Optional[float] = None
    atr_percentile: Optional[float] = None  # 0.0 - 1.0 (current ATR vs historical range)
    volatility_state: Optional[str] = None  # 'expanding', 'contracting', 'stable'
    regime: Optional[str] = None  # 'low', 'normal', 'elevated', 'extreme'
    commentary: Optional[str] = None


@dataclass
class HLProbabilityDriftBlock:
    """
    Probability drift analytics block for HL output.

    Compares initial model probability to live-adjusted probability.
    """
    initial_probability: Optional[float] = None  # 0.0 - 1.0 (original model output)
    live_probability: Optional[float] = None  # 0.0 - 1.0 (adjusted for live conditions)
    delta: Optional[float] = None  # live_probability - initial_probability
    drift_direction: Optional[str] = None  # 'strengthening', 'weakening', 'stable'
    commentary: Optional[str] = None


@dataclass
class HLStructureBlock:
    """
    Market structure analytics block for HL output.

    Describes key price levels and structure state.
    """
    key_support: Optional[float] = None
    key_resistance: Optional[float] = None
    current_price: Optional[float] = None
    distance_to_support_pct: Optional[float] = None
    distance_to_resistance_pct: Optional[float] = None
    structure_state: Optional[str] = None  # 'holding', 'breaking', 'retesting'
    commentary: Optional[str] = None


@dataclass
class HLOutput:
    """
    Complete HL (Higher-Level) output for a symbol.

    READ-ONLY, ADVISORY-ONLY analytics layer.
    Does NOT generate signals or trigger trades.
    """
    symbol: str
    timestamp: str

    # Core HL fields
    hl_bias: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    hl_confidence: Optional[float] = None  # 0.0 - 1.0

    # Analytics blocks
    trend: Optional[HLTrendBlock] = None
    volume: Optional[HLVolumeBlock] = None
    volatility: Optional[HLVolatilityBlock] = None
    probability_drift: Optional[HLProbabilityDriftBlock] = None
    structure: Optional[HLStructureBlock] = None

    # Summary narrative
    hl_summary: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'hl_bias': self.hl_bias,
            'hl_confidence': self.hl_confidence,
            'trend': {
                'trend_direction': self.trend.trend_direction if self.trend else None,
                'trend_strength': self.trend.trend_strength if self.trend else None,
                'multi_tf_alignment': self.trend.multi_tf_alignment if self.trend else None,
                'ema_alignment': self.trend.ema_alignment if self.trend else None,
                'swing_structure': self.trend.swing_structure if self.trend else None,
                'commentary': self.trend.commentary if self.trend else None
            } if self.trend else None,
            'volume': {
                'initial_wave_volume': self.volume.initial_wave_volume if self.volume else None,
                'correction_wave_volume': self.volume.correction_wave_volume if self.volume else None,
                'retest_wave_volume': self.volume.retest_wave_volume if self.volume else None,
                'retest_strength_pct': self.volume.retest_strength_pct if self.volume else None,
                'classification': self.volume.classification if self.volume else None,
                'commentary': self.volume.commentary if self.volume else None
            } if self.volume else None,
            'volatility': {
                'atr_current': self.volatility.atr_current if self.volatility else None,
                'atr_percentile': self.volatility.atr_percentile if self.volatility else None,
                'volatility_state': self.volatility.volatility_state if self.volatility else None,
                'regime': self.volatility.regime if self.volatility else None,
                'commentary': self.volatility.commentary if self.volatility else None
            } if self.volatility else None,
            'probability_drift': {
                'initial_probability': self.probability_drift.initial_probability if self.probability_drift else None,
                'live_probability': self.probability_drift.live_probability if self.probability_drift else None,
                'delta': self.probability_drift.delta if self.probability_drift else None,
                'drift_direction': self.probability_drift.drift_direction if self.probability_drift else None,
                'commentary': self.probability_drift.commentary if self.probability_drift else None
            } if self.probability_drift else None,
            'structure': {
                'key_support': self.structure.key_support if self.structure else None,
                'key_resistance': self.structure.key_resistance if self.structure else None,
                'current_price': self.structure.current_price if self.structure else None,
                'distance_to_support_pct': self.structure.distance_to_support_pct if self.structure else None,
                'distance_to_resistance_pct': self.structure.distance_to_resistance_pct if self.structure else None,
                'structure_state': self.structure.structure_state if self.structure else None,
                'commentary': self.structure.commentary if self.structure else None
            } if self.structure else None,
            'hl_summary': self.hl_summary
        }

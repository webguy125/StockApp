"""
Regime Classification Engine
Classifies HOLD signals into regime states using drift, pressure, volatility, and quality score

RESTRICTIONS:
- ONLY for HOLD signals
- NO modifications to prediction engine
- NO modifications to BUY/SELL logic
- Purely descriptive classification (no predictions)
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def classify_regime(
    drift: float,
    pressure_index: str,
    volatility_regime: str,
    quality_score: int,
    pnl: Optional[float]
) -> str:
    """
    Classify HOLD signal into a regime state

    Regime States:
    - Stable Neutral: Centered, stable pressure, profitable volatility
    - Expanding Neutral: Moving away from center but still balanced
    - Breaking Neutral: Near breakout threshold
    - Bullish Drift: Drifting toward call strike
    - Bearish Drift: Drifting toward put strike
    - Volatility Compression: IV collapsing (good for sellers)
    - Volatility Expansion: IV spiking (risk for sellers)

    Args:
        drift: Distance from center strike
        pressure_index: Pressure classification (Stable, Bullish, Bearish, Break)
        volatility_regime: Volatility classification (Profitable, Collapsing, Breakout)
        quality_score: Composite quality score (0-100)
        pnl: Current P&L (can be None)

    Returns:
        Regime state string
    """
    try:
        # Check for breakout conditions first
        if pressure_index == "Break":
            return "Breaking Neutral"

        # Check for volatility expansion
        if volatility_regime == "Breakout":
            return "Volatility Expansion"

        # Check for volatility compression
        if volatility_regime == "Collapsing":
            return "Volatility Compression"

        # Check for directional drift
        # Drift > 2.0 is significant drift from center
        if abs(drift) > 2.0:
            if drift > 0:
                # Drifting upward (toward call strike)
                if pressure_index == "Bullish":
                    return "Bullish Drift"
                else:
                    return "Expanding Neutral"
            else:
                # Drifting downward (toward put strike)
                if pressure_index == "Bearish":
                    return "Bearish Drift"
                else:
                    return "Expanding Neutral"

        # Check for expanding neutral (moderate drift)
        if abs(drift) > 1.0:
            return "Expanding Neutral"

        # Default to stable neutral
        if pressure_index == "Stable" and quality_score >= 60:
            return "Stable Neutral"
        else:
            return "Stable Neutral"

    except Exception as e:
        logger.error(f"[REGIME] Error classifying regime: {e}")
        return "Unknown"


def get_regime_color(regime_state: str) -> str:
    """
    Get color code for regime state (for frontend display)

    Args:
        regime_state: Regime state string

    Returns:
        CSS color class name
    """
    color_map = {
        "Stable Neutral": "regime-stable",
        "Expanding Neutral": "regime-expanding",
        "Breaking Neutral": "regime-breaking",
        "Bullish Drift": "regime-bullish",
        "Bearish Drift": "regime-bearish",
        "Volatility Compression": "regime-compression",
        "Volatility Expansion": "regime-expansion",
        "Unknown": "regime-unknown"
    }
    return color_map.get(regime_state, "regime-unknown")


def get_regime_priority(regime_state: str) -> int:
    """
    Get priority score for regime state (for sorting/alerting)

    Higher priority = more attention required

    Args:
        regime_state: Regime state string

    Returns:
        Priority score (0-10)
    """
    priority_map = {
        "Breaking Neutral": 10,      # Immediate attention
        "Volatility Expansion": 9,   # High risk
        "Bearish Drift": 7,          # Directional risk
        "Bullish Drift": 7,          # Directional risk
        "Expanding Neutral": 5,      # Moderate risk
        "Volatility Compression": 3, # Good for sellers
        "Stable Neutral": 1,         # Ideal state
        "Unknown": 0
    }
    return priority_map.get(regime_state, 0)

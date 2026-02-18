"""
Options Analytics Engine for HOLD Signals
Computes advanced neutral-regime metrics:
- Drift Detection: Distance from ITM center
- Pressure Index: Call/put pressure classification
- Volatility Regime: IV behavior classification
- Quality Score: 0-100 composite score

RESTRICTIONS:
- ONLY for HOLD signals
- NO modifications to prediction engine
- NO modifications to existing P&L logic
- All metrics computed independently
"""

from typing import Optional, Dict
import logging
import math

logger = logging.getLogger(__name__)


def calculate_drift(
    current_price: float,
    short_call_strike: float,
    short_put_strike: float
) -> float:
    """
    Calculate drift from ITM center strike

    Drift = distance between current price and center of condor
    Center = midpoint between short call and short put strikes

    Args:
        current_price: Current stock price
        short_call_strike: Short call strike
        short_put_strike: Short put strike

    Returns:
        Drift value (positive = above center, negative = below center)
    """
    try:
        center = (short_call_strike + short_put_strike) / 2.0
        drift = current_price - center
        return round(drift, 2)
    except Exception as e:
        logger.error(f"[DRIFT] Error calculating drift: {e}")
        return 0.0


def calculate_pressure_index(
    calls: Dict[float, Dict],
    puts: Dict[float, Dict],
    short_call_strike: float,
    short_put_strike: float,
    current_price: float
) -> str:
    """
    Classify call/put pressure

    Classifications:
    - Stable: Balanced premium decay
    - Bullish: Call premium decaying faster (price moving up)
    - Bearish: Put premium decaying faster (price moving down)
    - Break: One side near zero (breakout imminent)

    Args:
        calls: Call option chain
        puts: Put option chain
        short_call_strike: Short call strike
        short_put_strike: Short put strike
        current_price: Current price

    Returns:
        Pressure classification string
    """
    try:
        if short_call_strike not in calls or short_put_strike not in puts:
            return "Unknown"

        call_premium = calls[short_call_strike].get('mid', 0)
        put_premium = puts[short_put_strike].get('mid', 0)

        if call_premium <= 0 or put_premium <= 0:
            return "Unknown"

        # Check for breakout (one side near zero)
        if call_premium < 0.10 or put_premium < 0.10:
            return "Break"

        # Calculate premium ratio
        ratio = call_premium / put_premium

        # Classify pressure
        if 0.80 <= ratio <= 1.20:
            return "Stable"
        elif ratio > 1.20:
            return "Bearish"  # Put decaying faster (price moving down)
        else:
            return "Bullish"  # Call decaying faster (price moving up)

    except Exception as e:
        logger.error(f"[PRESSURE] Error calculating pressure index: {e}")
        return "Unknown"


def calculate_volatility_regime(
    calls: Dict[float, Dict],
    puts: Dict[float, Dict],
    short_call_strike: float,
    short_put_strike: float
) -> str:
    """
    Classify IV behavior

    Classifications:
    - Profitable: IV stable, premiums decaying normally
    - Collapsing: IV dropping rapidly (good for short options)
    - Breakout: IV spiking (risk of loss)

    Uses implied volatility proxies from option premiums

    Args:
        calls: Call option chain
        puts: Put option chain
        short_call_strike: Short call strike
        short_put_strike: Short put strike

    Returns:
        Volatility regime string
    """
    try:
        if short_call_strike not in calls or short_put_strike not in puts:
            return "Unknown"

        call_premium = calls[short_call_strike].get('mid', 0)
        put_premium = puts[short_put_strike].get('mid', 0)

        if call_premium <= 0 or put_premium <= 0:
            return "Unknown"

        # Calculate average premium as IV proxy
        avg_premium = (call_premium + put_premium) / 2.0

        # Classify based on premium levels
        # Low premiums = collapsing IV (good for sellers)
        # High premiums = elevated IV (risk)
        if avg_premium < 0.50:
            return "Collapsing"
        elif avg_premium > 2.00:
            return "Breakout"
        else:
            return "Profitable"

    except Exception as e:
        logger.error(f"[VOLATILITY] Error calculating volatility regime: {e}")
        return "Unknown"


def calculate_quality_score(
    drift: float,
    pressure_index: str,
    volatility_regime: str,
    pnl: Optional[float],
    age_days: int,
    short_call_strike: float,
    short_put_strike: float,
    current_price: float
) -> int:
    """
    Calculate composite quality score (0-100)

    Components:
    - Drift penalty: Lower score if drifting far from center
    - Pressure bonus: Higher score for stable pressure
    - Volatility bonus: Higher score for profitable/collapsing IV
    - P&L bonus: Higher score for positive P&L
    - Time decay bonus: Higher score for older positions (theta decay)

    Args:
        drift: Drift from center
        pressure_index: Pressure classification
        volatility_regime: Volatility classification
        pnl: Current P&L (can be None)
        age_days: Signal age in days
        short_call_strike: Short call strike
        short_put_strike: Short put strike
        current_price: Current price

    Returns:
        Quality score 0-100
    """
    try:
        score = 50  # Base score

        # Component 1: Drift penalty (max ±20 points)
        # Penalize drift > 50% of wing width
        wing_width = (short_call_strike - short_put_strike) / 2.0
        drift_ratio = abs(drift) / wing_width if wing_width > 0 else 0

        if drift_ratio < 0.25:
            score += 20  # Excellent centering
        elif drift_ratio < 0.50:
            score += 10  # Good centering
        elif drift_ratio < 0.75:
            score -= 10  # Drifting
        else:
            score -= 20  # Dangerous drift

        # Component 2: Pressure bonus (±15 points)
        if pressure_index == "Stable":
            score += 15
        elif pressure_index in ["Bullish", "Bearish"]:
            score += 5
        elif pressure_index == "Break":
            score -= 15

        # Component 3: Volatility bonus (±15 points)
        if volatility_regime == "Collapsing":
            score += 15  # Best for short options
        elif volatility_regime == "Profitable":
            score += 10
        elif volatility_regime == "Breakout":
            score -= 15

        # Component 4: P&L bonus (±20 points)
        if pnl is not None:
            if pnl > 0:
                # Scale bonus by P&L magnitude
                pnl_bonus = min(20, int(pnl * 2))  # $10 P&L = +20 points
                score += pnl_bonus
            else:
                # Scale penalty by loss magnitude
                pnl_penalty = max(-20, int(pnl * 2))
                score += pnl_penalty

        # Component 5: Time decay bonus (±10 points)
        # Reward older positions (theta decay working)
        if age_days >= 7:
            score += 10
        elif age_days >= 3:
            score += 5
        elif age_days < 1:
            score -= 5  # Too new to evaluate

        # Clamp to 0-100
        score = max(0, min(100, score))

        return score

    except Exception as e:
        logger.error(f"[QUALITY] Error calculating quality score: {e}")
        return 50  # Return neutral score on error


def compute_analytics(
    symbol: str,
    current_price: float,
    calls: Dict[float, Dict],
    puts: Dict[float, Dict],
    short_call_strike: float,
    short_put_strike: float,
    pnl: Optional[float],
    age_days: int
) -> Dict:
    """
    Compute full analytics suite for HOLD signal

    Args:
        symbol: Stock ticker
        current_price: Current stock price
        calls: Call option chain
        puts: Put option chain
        short_call_strike: Short call strike
        short_put_strike: Short put strike
        pnl: Current P&L (can be None)
        age_days: Signal age in days

    Returns:
        Dictionary with all analytics metrics
    """
    try:
        # Calculate individual metrics
        drift = calculate_drift(current_price, short_call_strike, short_put_strike)

        pressure_index = calculate_pressure_index(
            calls, puts, short_call_strike, short_put_strike, current_price
        )

        volatility_regime = calculate_volatility_regime(
            calls, puts, short_call_strike, short_put_strike
        )

        quality_score = calculate_quality_score(
            drift, pressure_index, volatility_regime, pnl, age_days,
            short_call_strike, short_put_strike, current_price
        )

        logger.info(
            f"[ANALYTICS] {symbol}: drift={drift}, pressure={pressure_index}, "
            f"volatility={volatility_regime}, quality={quality_score}"
        )

        return {
            'drift': drift,
            'pressure_index': pressure_index,
            'volatility_regime': volatility_regime,
            'quality_score': quality_score
        }

    except Exception as e:
        logger.error(f"[ANALYTICS] Error computing analytics for {symbol}: {e}")
        return {
            'drift': 0.0,
            'pressure_index': 'Unknown',
            'volatility_regime': 'Unknown',
            'quality_score': 50
        }

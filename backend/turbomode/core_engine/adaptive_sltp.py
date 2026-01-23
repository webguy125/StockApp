
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Adaptive Stop Loss / Take Profit Calculator

Calculates dynamic stop loss and take profit levels based on:
- ATR (volatility)
- Model confidence
- Sector characteristics
- Prediction horizon
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# Sector volatility multipliers
SECTOR_VOLATILITY = {
    'technology': 1.3,
    'financials': 1.1,
    'healthcare': 1.1,
    'industrials': 1.0,
    'energy': 1.2,
    'consumer_discretionary': 1.2,
    'consumer_staples': 0.9,
    'utilities': 0.8,
    'materials': 1.0,
    'real_estate': 0.9,
    'communication_services': 1.1,
}

# Horizon scaling factors
HORIZON_MULTIPLIERS = {
    '1d': 1.0,
    '2d': 1.5,
    '5d': 2.0,
}

# Default reward ratio (target distance / stop distance)
DEFAULT_REWARD_RATIO = 2.5


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR).

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close
        period: ATR period (default: 14)

    Returns:
        ATR value (scalar)
    """
    if len(df) < period + 1:
        # Not enough data, return simple range
        return (df['high'].max() - df['low'].min()) / len(df)

    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR is the rolling mean of True Range
    atr = true_range.rolling(window=period).mean().iloc[-1]

    return float(atr)


def calculate_confidence_modifier(confidence: float) -> float:
    """
    Calculate confidence modifier based on model probability.

    Args:
        confidence: Model confidence (prob_buy or prob_sell), range [0, 1]

    Returns:
        Confidence modifier in range [0.8, 1.2]
    """
    # Map confidence from [0.5, 0.9] to [0.8, 1.2]
    # Lower confidence → tighter stops (0.8x)
    # Higher confidence → wider stops (1.2x)

    conf_min = 0.5
    conf_max = 0.9
    mod_min = 0.8
    mod_max = 1.2

    if confidence <= conf_min:
        return mod_min
    elif confidence >= conf_max:
        return mod_max
    else:
        # Linear interpolation
        t = (confidence - conf_min) / (conf_max - conf_min)
        return mod_min + t * (mod_max - mod_min)


def calculate_adaptive_sltp(
    entry_price: float,
    atr: float,
    sector: str,
    confidence: float,
    horizon: str,
    position_type: str,
    reward_ratio: float = DEFAULT_REWARD_RATIO
) -> Dict[str, float]:
    """
    Calculate adaptive stop loss and take profit levels.

    Args:
        entry_price: Entry price
        atr: Average True Range (volatility measure)
        sector: Sector name
        confidence: Model confidence (prob_buy for long, prob_sell for short)
        horizon: "1d", "2d", or "5d"
        position_type: "long" or "short"
        reward_ratio: Target distance / stop distance (default: 2.5)

    Returns:
        Dictionary with:
            - stop_price: Stop loss price
            - target_price: Take profit price
            - stop_distance: Stop distance from entry (1R)
            - target_distance: Target distance from entry
            - r1_price: +1R price level
            - r2_price: +2R price level
            - r3_price: +3R price level
    """
    # Get sector volatility multiplier
    sector_mult = SECTOR_VOLATILITY.get(sector, 1.0)

    # Get horizon multiplier
    horizon_mult = HORIZON_MULTIPLIERS.get(horizon, 1.0)

    # Get confidence modifier
    conf_mod = calculate_confidence_modifier(confidence)

    # Calculate stop distance (1R)
    stop_distance = atr * sector_mult * conf_mod * horizon_mult

    # Calculate target distance
    target_distance = stop_distance * reward_ratio

    # Calculate price levels
    if position_type == 'long':
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
        r1_price = entry_price + stop_distance  # +1R
        r2_price = entry_price + 2 * stop_distance  # +2R
        r3_price = entry_price + 3 * stop_distance  # +3R
    else:  # short
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance
        r1_price = entry_price - stop_distance  # +1R (profit for short)
        r2_price = entry_price - 2 * stop_distance  # +2R
        r3_price = entry_price - 3 * stop_distance  # +3R

    return {
        'stop_price': stop_price,
        'target_price': target_price,
        'stop_distance': stop_distance,
        'target_distance': target_distance,
        'r1_price': r1_price,
        'r2_price': r2_price,
        'r3_price': r3_price,
    }


def update_trailing_stop(
    position_type: str,
    entry_price: float,
    current_price: float,
    current_stop: float,
    stop_distance: float,
    partial_1R_done: bool,
    partial_2R_done: bool
) -> float:
    """
    Update trailing stop based on R-multiple progress.

    Rules:
    - Never widen stop (increase risk)
    - At +1R: move stop to breakeven
    - At +2R: trail stop more aggressively
    - Always respect the "never widen" rule

    Args:
        position_type: "long" or "short"
        entry_price: Entry price
        current_price: Current market price
        current_stop: Current stop price
        stop_distance: Initial stop distance (1R)
        partial_1R_done: Whether +1R partial exit happened
        partial_2R_done: Whether +2R partial exit happened

    Returns:
        Updated stop price
    """
    new_stop = current_stop

    if position_type == 'long':
        # Calculate R-multiples
        r_multiple = (current_price - entry_price) / stop_distance if stop_distance > 0 else 0

        if r_multiple >= 1.0 and partial_1R_done:
            # Move stop to breakeven or slightly above
            breakeven_stop = entry_price
            new_stop = max(current_stop, breakeven_stop)

        if r_multiple >= 2.0 and partial_2R_done:
            # Trail stop at +1R (lock in 1R profit)
            trailing_stop = entry_price + stop_distance
            new_stop = max(current_stop, trailing_stop)

    else:  # short
        # Calculate R-multiples (profit direction is reversed)
        r_multiple = (entry_price - current_price) / stop_distance if stop_distance > 0 else 0

        if r_multiple >= 1.0 and partial_1R_done:
            # Move stop to breakeven or slightly below
            breakeven_stop = entry_price
            new_stop = min(current_stop, breakeven_stop)

        if r_multiple >= 2.0 and partial_2R_done:
            # Trail stop at +1R (lock in 1R profit)
            trailing_stop = entry_price - stop_distance
            new_stop = min(current_stop, trailing_stop)

    return new_stop


def check_partial_profit_levels(
    position_type: str,
    entry_price: float,
    current_price: float,
    stop_distance: float,
    partial_1R_done: bool,
    partial_2R_done: bool,
    partial_3R_done: bool
) -> Dict[str, bool]:
    """
    Check which partial profit levels have been reached.

    Args:
        position_type: "long" or "short"
        entry_price: Entry price
        current_price: Current market price
        stop_distance: Initial stop distance (1R)
        partial_1R_done: Whether +1R already taken
        partial_2R_done: Whether +2R already taken
        partial_3R_done: Whether +3R already taken

    Returns:
        Dictionary with flags for each level:
            - take_1R: True if should take 50% at +1R
            - take_2R: True if should take 25% at +2R
            - take_3R: True if should take remaining at +3R
    """
    if stop_distance == 0:
        return {'take_1R': False, 'take_2R': False, 'take_3R': False}

    # Calculate R-multiple
    if position_type == 'long':
        r_multiple = (current_price - entry_price) / stop_distance
    else:  # short
        r_multiple = (entry_price - current_price) / stop_distance

    return {
        'take_1R': r_multiple >= 1.0 and not partial_1R_done,
        'take_2R': r_multiple >= 2.0 and not partial_2R_done and partial_1R_done,
        'take_3R': r_multiple >= 3.0 and not partial_3R_done and partial_2R_done,
    }

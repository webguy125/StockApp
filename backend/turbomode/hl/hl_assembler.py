"""
HL Assembler - Build HL Blocks from Analytics

Assembles HL blocks by calling existing analyzer functions or clients.

IMPORTANT CONSTRAINTS:
- All functions are pure and read-only
- NO database writes allowed
- NO modifications to analyzer modules
- Graceful degradation when data is missing
"""

from typing import Optional
from backend.turbomode.hl.hl_model import (
    HLTrendBlock,
    HLVolumeBlock,
    HLVolatilityBlock,
    HLProbabilityDriftBlock,
    HLStructureBlock
)


def build_trend_block(symbol: str) -> HLTrendBlock:
    """
    Build HLTrendBlock by calling existing trend analyzer.

    Args:
        symbol: Stock symbol

    Returns:
        HLTrendBlock with trend analytics

    Constraints:
        - Pure, read-only function
        - Degrades gracefully if data missing
    """
    try:
        from backend.turbomode.analyzer import analyze_trend

        # Use placeholder signal_type since HL is advisory-only
        trend_data = analyze_trend(symbol, 'BUY')

        return HLTrendBlock(
            trend_direction=trend_data.get('trend'),
            trend_strength=trend_data.get('trend_strength', 50) / 100.0,
            multi_tf_alignment=trend_data.get('with_trend') == 1,
            ema_alignment=_infer_ema_alignment(trend_data.get('trend')),
            swing_structure=_infer_swing_structure(trend_data.get('trend')),
            commentary=trend_data.get('trend_commentary', 'Trend analysis unavailable')
        )

    except Exception as e:
        print(f"[HL ASSEMBLER] Error building trend block for {symbol}: {e}")
        return HLTrendBlock(
            trend_direction='neutral',
            trend_strength=0.5,
            multi_tf_alignment=False,
            ema_alignment='mixed',
            swing_structure='choppy',
            commentary='Trend data unavailable - analysis error'
        )


def build_volume_block(symbol: str) -> HLVolumeBlock:
    """
    Build HLVolumeBlock by calling existing volume analyzer.

    Args:
        symbol: Stock symbol

    Returns:
        HLVolumeBlock with volume analytics

    Constraints:
        - Pure, read-only function
        - Degrades gracefully if data missing
    """
    try:
        from backend.turbomode.analyzer import analyze_ord_volume

        volume_data = analyze_ord_volume(symbol)

        return HLVolumeBlock(
            initial_wave_volume=volume_data.get('ord_initial_volume'),
            correction_wave_volume=volume_data.get('ord_correction_volume'),
            retest_wave_volume=volume_data.get('ord_retest_volume'),
            retest_strength_pct=volume_data.get('ord_retest_strength_pct'),
            classification=volume_data.get('ord_classification'),
            commentary=volume_data.get('ord_commentary', 'Volume analysis unavailable')
        )

    except Exception as e:
        print(f"[HL ASSEMBLER] Error building volume block for {symbol}: {e}")
        return HLVolumeBlock(
            initial_wave_volume=0,
            correction_wave_volume=0,
            retest_wave_volume=0,
            retest_strength_pct=0,
            classification='unavailable',
            commentary='Volume data unavailable - analysis error'
        )


def build_volatility_block(symbol: str) -> HLVolatilityBlock:
    """
    Build HLVolatilityBlock from market data.

    Args:
        symbol: Stock symbol

    Returns:
        HLVolatilityBlock with volatility analytics

    Constraints:
        - Pure, read-only function
        - Degrades gracefully if data missing
    """
    try:
        from master_market_data.market_data_api import get_market_data_api
        import pandas as pd
        import numpy as np

        market_data_api = get_market_data_api()
        df = market_data_api.get_candles(symbol, timeframe='1d')

        if df is None or len(df) < 20:
            raise ValueError("Insufficient market data")

        # Calculate ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        atr_current = df['tr'].rolling(window=14).mean().iloc[-1]

        # Calculate ATR percentile (current ATR vs 100-day range)
        atr_100d = df['tr'].rolling(window=14).mean().tail(100)
        atr_min = atr_100d.min()
        atr_max = atr_100d.max()

        if atr_max > atr_min:
            atr_percentile = (atr_current - atr_min) / (atr_max - atr_min)
        else:
            atr_percentile = 0.5

        # Determine volatility state (expanding/contracting/stable)
        atr_recent = df['tr'].rolling(window=14).mean().tail(10)
        atr_slope = np.polyfit(range(len(atr_recent)), atr_recent, 1)[0]

        if atr_slope > 0.05:
            volatility_state = 'expanding'
        elif atr_slope < -0.05:
            volatility_state = 'contracting'
        else:
            volatility_state = 'stable'

        # Determine regime
        if atr_percentile < 0.25:
            regime = 'low'
        elif atr_percentile < 0.75:
            regime = 'normal'
        elif atr_percentile < 0.90:
            regime = 'elevated'
        else:
            regime = 'extreme'

        commentary = f"Volatility is {regime} ({volatility_state}). ATR: ${atr_current:.2f} at {atr_percentile*100:.0f}th percentile."

        return HLVolatilityBlock(
            atr_current=float(atr_current),
            atr_percentile=float(atr_percentile),
            volatility_state=volatility_state,
            regime=regime,
            commentary=commentary
        )

    except Exception as e:
        print(f"[HL ASSEMBLER] Error building volatility block for {symbol}: {e}")
        return HLVolatilityBlock(
            atr_current=None,
            atr_percentile=0.5,
            volatility_state='stable',
            regime='normal',
            commentary='Volatility data unavailable - analysis error'
        )


def build_probability_drift_block(symbol: str, initial_probability: Optional[float] = None) -> HLProbabilityDriftBlock:
    """
    Build HLProbabilityDriftBlock by computing drift from live conditions.

    Args:
        symbol: Stock symbol
        initial_probability: Initial model probability (0.0-1.0)

    Returns:
        HLProbabilityDriftBlock with probability drift analytics

    Constraints:
        - Pure, read-only function
        - Degrades gracefully if data missing
    """
    try:
        from backend.turbomode.hl.probability_drift import compute_probability_drift

        # If no initial probability provided, use neutral
        if initial_probability is None:
            initial_probability = 0.5

        # Get live analytics
        trend_block = build_trend_block(symbol)
        volume_block = build_volume_block(symbol)
        volatility_block = build_volatility_block(symbol)

        # Compute drift
        drift_result = compute_probability_drift(
            initial_probability=initial_probability,
            trend_strength=trend_block.trend_strength if trend_block.trend_strength else 0.5,
            volume_classification=volume_block.classification if volume_block.classification else 'neutral',
            volatility_state=volatility_block.volatility_state if volatility_block.volatility_state else 'stable',
            structure_state='holding'  # Placeholder until structure block implemented
        )

        return HLProbabilityDriftBlock(
            initial_probability=drift_result['initial_probability'],
            live_probability=drift_result['live_probability'],
            delta=drift_result['delta'],
            drift_direction=drift_result['drift_direction'],
            commentary=drift_result['commentary']
        )

    except Exception as e:
        print(f"[HL ASSEMBLER] Error building probability drift block for {symbol}: {e}")
        return HLProbabilityDriftBlock(
            initial_probability=initial_probability if initial_probability else 0.5,
            live_probability=initial_probability if initial_probability else 0.5,
            delta=0.0,
            drift_direction='stable',
            commentary='Probability drift unavailable - analysis error'
        )


def build_structure_block(symbol: str) -> HLStructureBlock:
    """
    Build HLStructureBlock from market data (key levels, swing points).

    Args:
        symbol: Stock symbol

    Returns:
        HLStructureBlock with market structure analytics

    Constraints:
        - Pure, read-only function
        - Degrades gracefully if data missing
    """
    try:
        from master_market_data.market_data_api import get_market_data_api
        import numpy as np

        market_data_api = get_market_data_api()
        df = market_data_api.get_candles(symbol, timeframe='1d')

        if df is None or len(df) < 50:
            raise ValueError("Insufficient market data")

        current_price = float(df['close'].iloc[-1])

        # Find key support: lowest low in last 20 days
        key_support = float(df['low'].tail(20).min())

        # Find key resistance: highest high in last 20 days
        key_resistance = float(df['high'].tail(20).max())

        # Calculate distances
        dist_to_support = ((current_price - key_support) / key_support) * 100
        dist_to_resistance = ((key_resistance - current_price) / current_price) * 100

        # Determine structure state
        if current_price > key_resistance * 0.99:
            structure_state = 'breaking'  # Breaking resistance
        elif current_price < key_support * 1.01:
            structure_state = 'breaking'  # Breaking support
        elif abs(current_price - key_support) < abs(current_price - key_resistance):
            structure_state = 'retesting'  # Closer to support
        else:
            structure_state = 'holding'  # Holding structure

        commentary = f"Price ${current_price:.2f} is {dist_to_support:.1f}% above support (${key_support:.2f}) and {dist_to_resistance:.1f}% below resistance (${key_resistance:.2f}). Structure: {structure_state}."

        return HLStructureBlock(
            key_support=key_support,
            key_resistance=key_resistance,
            current_price=current_price,
            distance_to_support_pct=dist_to_support,
            distance_to_resistance_pct=dist_to_resistance,
            structure_state=structure_state,
            commentary=commentary
        )

    except Exception as e:
        print(f"[HL ASSEMBLER] Error building structure block for {symbol}: {e}")
        return HLStructureBlock(
            key_support=None,
            key_resistance=None,
            current_price=None,
            distance_to_support_pct=None,
            distance_to_resistance_pct=None,
            structure_state='holding',
            commentary='Structure data unavailable - analysis error'
        )


# Helper functions

def _infer_ema_alignment(trend: Optional[str]) -> str:
    """Infer EMA alignment from trend direction"""
    if trend == 'uptrend':
        return 'bullish'
    elif trend == 'downtrend':
        return 'bearish'
    else:
        return 'mixed'


def _infer_swing_structure(trend: Optional[str]) -> str:
    """Infer swing structure from trend direction"""
    if trend == 'uptrend':
        return 'HH/HL'  # Higher Highs / Higher Lows
    elif trend == 'downtrend':
        return 'LL/LH'  # Lower Lows / Lower Highs
    else:
        return 'choppy'

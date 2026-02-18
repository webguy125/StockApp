"""
HL Service - Pure Function for Building HL Output

Provides the core function to build an HLOutput from existing analytics inputs.

IMPORTANT CONSTRAINTS:
- Does NOT read or write to any database
- Does NOT trigger any trading actions
- Does NOT modify core engine behavior
- Accepts precomputed analytics via context dict
- Purely functional and read-only
"""

from datetime import datetime
from typing import Dict, Optional
from backend.turbomode.hl.hl_model import (
    HLOutput,
    HLTrendBlock,
    HLVolumeBlock,
    HLVolatilityBlock,
    HLProbabilityDriftBlock,
    HLStructureBlock
)


def build_hl_output(symbol: str, context: Dict) -> HLOutput:
    """
    Build a complete HLOutput from precomputed analytics context.

    Args:
        symbol: Stock symbol
        context: Dictionary containing precomputed analytics:
            - 'trend': Trend analytics dict
            - 'volume': Volume analytics dict
            - 'volatility': Volatility analytics dict
            - 'probability': Probability analytics dict
            - 'structure': Structure analytics dict

    Returns:
        HLOutput object with all blocks populated

    Constraints:
        - Must NOT read or write to any database
        - Must NOT trigger any trading actions
        - Must compute hl_bias and hl_confidence from context
        - Must handle missing context gracefully
    """

    timestamp = datetime.now().isoformat()

    # Build individual blocks from context
    trend_block = _build_trend_block_from_context(context.get('trend', {}))
    volume_block = _build_volume_block_from_context(context.get('volume', {}))
    volatility_block = _build_volatility_block_from_context(context.get('volatility', {}))
    probability_block = _build_probability_block_from_context(context.get('probability', {}))
    structure_block = _build_structure_block_from_context(context.get('structure', {}))

    # Compute hl_bias and hl_confidence based on alignment rules
    hl_bias, hl_confidence = _compute_hl_bias_and_confidence(
        trend_block, volume_block, volatility_block, probability_block, structure_block
    )

    # Synthesize hl_summary
    hl_summary = _synthesize_hl_summary(
        symbol, hl_bias, hl_confidence,
        trend_block, volume_block, volatility_block, probability_block, structure_block
    )

    return HLOutput(
        symbol=symbol,
        timestamp=timestamp,
        hl_bias=hl_bias,
        hl_confidence=hl_confidence,
        trend=trend_block,
        volume=volume_block,
        volatility=volatility_block,
        probability_drift=probability_block,
        structure=structure_block,
        hl_summary=hl_summary
    )


def _build_trend_block_from_context(trend_data: Dict) -> Optional[HLTrendBlock]:
    """Build HLTrendBlock from trend analytics context"""
    if not trend_data:
        return HLTrendBlock(commentary="Trend data unavailable")

    return HLTrendBlock(
        trend_direction=trend_data.get('trend'),
        trend_strength=trend_data.get('trend_strength', 0) / 100.0,  # Normalize to 0-1
        multi_tf_alignment=trend_data.get('multi_tf_alignment'),
        ema_alignment=trend_data.get('ema_alignment'),
        swing_structure=trend_data.get('swing_structure'),
        commentary=trend_data.get('trend_commentary', 'No trend commentary available')
    )


def _build_volume_block_from_context(volume_data: Dict) -> Optional[HLVolumeBlock]:
    """Build HLVolumeBlock from volume analytics context"""
    if not volume_data:
        return HLVolumeBlock(commentary="Volume data unavailable")

    return HLVolumeBlock(
        initial_wave_volume=volume_data.get('ord_initial_volume'),
        correction_wave_volume=volume_data.get('ord_correction_volume'),
        retest_wave_volume=volume_data.get('ord_retest_volume'),
        retest_strength_pct=volume_data.get('ord_retest_strength_pct'),
        classification=volume_data.get('ord_classification'),
        commentary=volume_data.get('ord_commentary', 'No volume commentary available')
    )


def _build_volatility_block_from_context(volatility_data: Dict) -> Optional[HLVolatilityBlock]:
    """Build HLVolatilityBlock from volatility analytics context"""
    if not volatility_data:
        return HLVolatilityBlock(commentary="Volatility data unavailable")

    atr = volatility_data.get('atr_current')
    percentile = volatility_data.get('atr_percentile', 0.5)

    # Classify volatility regime
    if percentile < 0.25:
        regime = 'low'
    elif percentile < 0.75:
        regime = 'normal'
    elif percentile < 0.90:
        regime = 'elevated'
    else:
        regime = 'extreme'

    return HLVolatilityBlock(
        atr_current=atr,
        atr_percentile=percentile,
        volatility_state=volatility_data.get('volatility_state', 'stable'),
        regime=regime,
        commentary=volatility_data.get('commentary', f'Volatility regime: {regime}')
    )


def _build_probability_block_from_context(probability_data: Dict) -> Optional[HLProbabilityDriftBlock]:
    """Build HLProbabilityDriftBlock from probability analytics context"""
    if not probability_data:
        return HLProbabilityDriftBlock(commentary="Probability data unavailable")

    initial_prob = probability_data.get('initial_probability', 0.5)
    live_prob = probability_data.get('live_probability', initial_prob)
    delta = live_prob - initial_prob

    # Classify drift direction
    if abs(delta) < 0.05:
        drift_direction = 'stable'
    elif delta > 0:
        drift_direction = 'strengthening'
    else:
        drift_direction = 'weakening'

    return HLProbabilityDriftBlock(
        initial_probability=initial_prob,
        live_probability=live_prob,
        delta=delta,
        drift_direction=drift_direction,
        commentary=probability_data.get('commentary', f'Probability {drift_direction} ({delta:+.1%})')
    )


def _build_structure_block_from_context(structure_data: Dict) -> Optional[HLStructureBlock]:
    """Build HLStructureBlock from structure analytics context"""
    if not structure_data:
        return HLStructureBlock(commentary="Structure data unavailable")

    current_price = structure_data.get('current_price')
    support = structure_data.get('key_support')
    resistance = structure_data.get('key_resistance')

    # Calculate distances
    dist_to_support = None
    dist_to_resistance = None
    if current_price and support:
        dist_to_support = ((current_price - support) / support) * 100
    if current_price and resistance:
        dist_to_resistance = ((resistance - current_price) / current_price) * 100

    return HLStructureBlock(
        key_support=support,
        key_resistance=resistance,
        current_price=current_price,
        distance_to_support_pct=dist_to_support,
        distance_to_resistance_pct=dist_to_resistance,
        structure_state=structure_data.get('structure_state', 'holding'),
        commentary=structure_data.get('commentary', 'Structure analysis unavailable')
    )


def _compute_hl_bias_and_confidence(
    trend: Optional[HLTrendBlock],
    volume: Optional[HLVolumeBlock],
    volatility: Optional[HLVolatilityBlock],
    probability: Optional[HLProbabilityDriftBlock],
    structure: Optional[HLStructureBlock]
) -> tuple[str, float]:
    """
    Compute hl_bias and hl_confidence based on alignment rules.

    Alignment logic:
    - hl_bias: Determined by majority vote of block signals
    - hl_confidence: Weighted average of block confidences and alignment

    Returns:
        (hl_bias, hl_confidence) tuple
    """

    # Default to neutral if no data
    if not any([trend, volume, volatility, probability, structure]):
        return ('neutral', 0.5)

    # Count bullish/bearish signals
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0

    # Trend signal
    if trend and trend.trend_direction:
        total_signals += 1
        if trend.trend_direction == 'uptrend':
            bullish_signals += 1
        elif trend.trend_direction == 'downtrend':
            bearish_signals += 1

    # Volume signal
    if volume and volume.classification:
        total_signals += 1
        if volume.classification == 'strong':
            # Strong volume supports current trend
            if trend and trend.trend_direction == 'uptrend':
                bullish_signals += 1
            elif trend and trend.trend_direction == 'downtrend':
                bearish_signals += 1

    # Probability signal
    if probability and probability.drift_direction:
        total_signals += 1
        if probability.drift_direction == 'strengthening' and probability.live_probability and probability.live_probability > 0.5:
            bullish_signals += 1
        elif probability.drift_direction == 'strengthening' and probability.live_probability and probability.live_probability < 0.5:
            bearish_signals += 1

    # Structure signal
    if structure and structure.structure_state:
        total_signals += 1
        if structure.structure_state == 'holding':
            # Structure holding supports continuation
            if trend and trend.trend_direction == 'uptrend':
                bullish_signals += 1
            elif trend and trend.trend_direction == 'downtrend':
                bearish_signals += 1
        elif structure.structure_state == 'breaking':
            # Structure breaking suggests reversal
            if trend and trend.trend_direction == 'uptrend':
                bearish_signals += 1
            elif trend and trend.trend_direction == 'downtrend':
                bullish_signals += 1

    # Determine bias by majority vote
    if bullish_signals > bearish_signals:
        hl_bias = 'bullish'
    elif bearish_signals > bullish_signals:
        hl_bias = 'bearish'
    else:
        hl_bias = 'neutral'

    # Compute confidence based on alignment
    if total_signals == 0:
        hl_confidence = 0.5
    else:
        alignment_ratio = max(bullish_signals, bearish_signals) / total_signals
        hl_confidence = min(0.95, max(0.1, alignment_ratio))

    return (hl_bias, hl_confidence)


def _synthesize_hl_summary(
    symbol: str,
    hl_bias: str,
    hl_confidence: float,
    trend: Optional[HLTrendBlock],
    volume: Optional[HLVolumeBlock],
    volatility: Optional[HLVolatilityBlock],
    probability: Optional[HLProbabilityDriftBlock],
    structure: Optional[HLStructureBlock]
) -> str:
    """
    Synthesize a concise narrative summary from HL blocks.

    Must be descriptive only - NO BUY/SELL commands.
    """

    parts = []

    # Lead with bias and confidence
    confidence_label = 'high' if hl_confidence > 0.75 else 'moderate' if hl_confidence > 0.5 else 'low'
    parts.append(f"{symbol} shows {confidence_label} confidence {hl_bias} bias.")

    # Add trend context
    if trend and trend.trend_direction:
        parts.append(f"Trend: {trend.trend_direction}.")

    # Add volume context
    if volume and volume.classification:
        parts.append(f"Volume: {volume.classification}.")

    # Add probability drift
    if probability and probability.drift_direction:
        parts.append(f"Probability: {probability.drift_direction}.")

    # Add structure context
    if structure and structure.structure_state:
        parts.append(f"Structure: {structure.structure_state}.")

    # Add volatility warning if elevated
    if volatility and volatility.regime in ['elevated', 'extreme']:
        parts.append(f"Volatility: {volatility.regime} - caution advised.")

    return " ".join(parts)

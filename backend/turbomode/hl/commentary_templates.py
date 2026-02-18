"""
HL Commentary Templates

Provides pure, deterministic, rule-based commentary generation for HL blocks.

IMPORTANT CONSTRAINTS:
- NO randomness in commentary generation
- NO external LLM calls
- Commentary generated from numeric and categorical analytics ONLY
- Deterministic and reproducible
- NO BUY/SELL commands - descriptive language only
"""

from backend.turbomode.hl.hl_model import (
    HLTrendBlock,
    HLVolumeBlock,
    HLVolatilityBlock,
    HLProbabilityDriftBlock,
    HLStructureBlock,
    HLOutput
)
from typing import Optional


def trend_commentary(trend_block: Optional[HLTrendBlock]) -> str:
    """
    Generate deterministic commentary for trend block.

    Args:
        trend_block: HLTrendBlock instance

    Returns:
        Human-readable commentary string

    Constraints:
        - Deterministic and rule-based
        - No BUY/SELL commands
        - Handles missing data gracefully
    """
    if not trend_block or not trend_block.trend_direction:
        return "Trend analysis unavailable."

    direction = trend_block.trend_direction
    strength = trend_block.trend_strength if trend_block.trend_strength else 0.5
    alignment = trend_block.multi_tf_alignment
    ema = trend_block.ema_alignment if trend_block.ema_alignment else 'mixed'
    swing = trend_block.swing_structure if trend_block.swing_structure else 'choppy'

    # Strength label
    if strength > 0.75:
        strength_label = "strong"
    elif strength > 0.5:
        strength_label = "moderate"
    else:
        strength_label = "weak"

    # Alignment label
    if alignment:
        alignment_label = "Multiple timeframes confirm"
    else:
        alignment_label = "Timeframes show divergence"

    # Build commentary
    if direction == 'uptrend':
        commentary = f"{alignment_label} {strength_label} uptrend. EMA alignment: {ema}. Swing structure: {swing}. Conditions favor upward continuation."
    elif direction == 'downtrend':
        commentary = f"{alignment_label} {strength_label} downtrend. EMA alignment: {ema}. Swing structure: {swing}. Conditions favor downward continuation."
    else:
        commentary = f"Market in neutral consolidation. EMA alignment: {ema}. Swing structure: {swing}. No clear directional bias."

    return commentary


def volume_commentary(volume_block: Optional[HLVolumeBlock]) -> str:
    """
    Generate deterministic commentary for volume block.

    Args:
        volume_block: HLVolumeBlock instance

    Returns:
        Human-readable commentary string

    Constraints:
        - Deterministic and rule-based
        - No BUY/SELL commands
        - Handles missing data gracefully
    """
    if not volume_block or not volume_block.classification:
        return "Volume analysis unavailable."

    classification = volume_block.classification
    retest_strength = volume_block.retest_strength_pct if volume_block.retest_strength_pct else 0
    initial_vol = volume_block.initial_wave_volume if volume_block.initial_wave_volume else 0
    retest_vol = volume_block.retest_wave_volume if volume_block.retest_wave_volume else 0

    # Build commentary based on classification
    if classification == 'strong':
        commentary = f"Retest volume ({retest_vol:,.0f}) is STRONG at {retest_strength:.1f}% of initial wave ({initial_vol:,.0f}). High conviction continuation likely. Participants demonstrating commitment."
    elif classification == 'neutral':
        commentary = f"Retest volume ({retest_vol:,.0f}) is NEUTRAL at {retest_strength:.1f}% of initial wave ({initial_vol:,.0f}). Moderate conviction. Watch for follow-through."
    elif classification == 'weak':
        commentary = f"Retest volume ({retest_vol:,.0f}) is WEAK at {retest_strength:.1f}% of initial wave ({initial_vol:,.0f}). Low conviction suggests caution. Participants showing hesitation."
    else:
        commentary = "Volume classification unavailable - insufficient data for wave analysis."

    return commentary


def volatility_commentary(vol_block: Optional[HLVolatilityBlock]) -> str:
    """
    Generate deterministic commentary for volatility block.

    Args:
        vol_block: HLVolatilityBlock instance

    Returns:
        Human-readable commentary string

    Constraints:
        - Deterministic and rule-based
        - No BUY/SELL commands
        - Handles missing data gracefully
    """
    if not vol_block or not vol_block.regime:
        return "Volatility analysis unavailable."

    regime = vol_block.regime
    state = vol_block.volatility_state if vol_block.volatility_state else 'stable'
    atr = vol_block.atr_current if vol_block.atr_current else 0
    percentile = vol_block.atr_percentile if vol_block.atr_percentile else 0.5

    # Build commentary based on regime and state
    if regime == 'extreme':
        commentary = f"Volatility EXTREME ({percentile*100:.0f}th percentile). ATR: ${atr:.2f}. State: {state}. Elevated risk - position sizing and stops require careful attention."
    elif regime == 'elevated':
        commentary = f"Volatility ELEVATED ({percentile*100:.0f}th percentile). ATR: ${atr:.2f}. State: {state}. Above-normal price swings expected - adjust risk parameters accordingly."
    elif regime == 'low':
        commentary = f"Volatility LOW ({percentile*100:.0f}th percentile). ATR: ${atr:.2f}. State: {state}. Compressed range suggests potential expansion ahead."
    else:
        commentary = f"Volatility NORMAL ({percentile*100:.0f}th percentile). ATR: ${atr:.2f}. State: {state}. Standard risk environment."

    # Add state context
    if state == 'expanding':
        commentary += " Volatility expanding - breakout conditions developing."
    elif state == 'contracting':
        commentary += " Volatility contracting - range compression occurring."

    return commentary


def probability_drift_commentary(prob_block: Optional[HLProbabilityDriftBlock]) -> str:
    """
    Generate deterministic commentary for probability drift block.

    Args:
        prob_block: HLProbabilityDriftBlock instance

    Returns:
        Human-readable commentary string

    Constraints:
        - Deterministic and rule-based
        - No BUY/SELL commands
        - Handles missing data gracefully
    """
    if not prob_block or prob_block.initial_probability is None:
        return "Probability drift analysis unavailable."

    initial = prob_block.initial_probability
    live = prob_block.live_probability if prob_block.live_probability else initial
    delta = prob_block.delta if prob_block.delta else 0
    direction = prob_block.drift_direction if prob_block.drift_direction else 'stable'

    # Format probabilities as percentages
    initial_pct = initial * 100
    live_pct = live * 100
    delta_pct = delta * 100

    # Build commentary based on drift direction and magnitude
    if direction == 'strengthening':
        if abs(delta) > 0.15:
            magnitude = "significantly"
        elif abs(delta) > 0.05:
            magnitude = "moderately"
        else:
            magnitude = "slightly"

        commentary = f"Initial probability {initial_pct:.1f}% has {magnitude} strengthened to {live_pct:.1f}% ({delta_pct:+.1f}%) based on live conditions. Model conviction increasing."

    elif direction == 'weakening':
        if abs(delta) > 0.15:
            magnitude = "significantly"
        elif abs(delta) > 0.05:
            magnitude = "moderately"
        else:
            magnitude = "slightly"

        commentary = f"Initial probability {initial_pct:.1f}% has {magnitude} weakened to {live_pct:.1f}% ({delta_pct:+.1f}%) based on live conditions. Model conviction decreasing."

    else:
        commentary = f"Probability stable at {live_pct:.1f}% (initial: {initial_pct:.1f}%). Live conditions align with initial model assessment."

    return commentary


def structure_commentary(struct_block: Optional[HLStructureBlock]) -> str:
    """
    Generate deterministic commentary for structure block.

    Args:
        struct_block: HLStructureBlock instance

    Returns:
        Human-readable commentary string

    Constraints:
        - Deterministic and rule-based
        - No BUY/SELL commands
        - Handles missing data gracefully
    """
    if not struct_block or struct_block.current_price is None:
        return "Structure analysis unavailable."

    current = struct_block.current_price
    support = struct_block.key_support
    resistance = struct_block.key_resistance
    state = struct_block.structure_state if struct_block.structure_state else 'holding'
    dist_support = struct_block.distance_to_support_pct
    dist_resistance = struct_block.distance_to_resistance_pct

    # Build commentary based on structure state and position
    if state == 'breaking':
        if dist_support and dist_support < 2:
            commentary = f"Price ${current:.2f} BREAKING support at ${support:.2f} ({dist_support:.1f}% above). Critical level under pressure - invalidation risk elevated."
        elif dist_resistance and dist_resistance < 2:
            commentary = f"Price ${current:.2f} BREAKING resistance at ${resistance:.2f} ({dist_resistance:.1f}% below). Breakout conditions developing - confirmation needed."
        else:
            commentary = f"Price ${current:.2f} breaking key level. Structure invalidation in progress."

    elif state == 'retesting':
        if dist_support and dist_support < 5:
            commentary = f"Price ${current:.2f} RETESTING support at ${support:.2f} ({dist_support:.1f}% above). Key level being tested - hold/break decision imminent."
        else:
            commentary = f"Price ${current:.2f} retesting structure. Key level interaction in progress."

    else:  # holding
        if support and resistance:
            commentary = f"Price ${current:.2f} HOLDING structure between support ${support:.2f} ({dist_support:.1f}% above) and resistance ${resistance:.2f} ({dist_resistance:.1f}% below). Range-bound conditions."
        else:
            commentary = f"Price ${current:.2f} holding structure. Key levels intact."

    return commentary


def hl_summary_commentary(hl_output: HLOutput) -> str:
    """
    Synthesize a concise narrative summary from all HL blocks.

    Args:
        hl_output: Complete HLOutput instance

    Returns:
        Comprehensive summary commentary string

    Constraints:
        - Deterministic and rule-based
        - No BUY/SELL commands
        - Uses hl_bias and hl_confidence plus key signals from each block
    """
    symbol = hl_output.symbol
    bias = hl_output.hl_bias if hl_output.hl_bias else 'neutral'
    confidence = hl_output.hl_confidence if hl_output.hl_confidence else 0.5

    # Confidence label
    if confidence > 0.75:
        confidence_label = "HIGH confidence"
    elif confidence > 0.5:
        confidence_label = "MODERATE confidence"
    else:
        confidence_label = "LOW confidence"

    # Start with bias and confidence
    summary_parts = [f"{symbol} shows {confidence_label} {bias.upper()} bias."]

    # Add trend context
    if hl_output.trend and hl_output.trend.trend_direction:
        trend_dir = hl_output.trend.trend_direction
        trend_strength = hl_output.trend.trend_strength if hl_output.trend.trend_strength else 0.5

        if trend_strength > 0.75:
            strength_word = "strong"
        elif trend_strength > 0.5:
            strength_word = "moderate"
        else:
            strength_word = "weak"

        summary_parts.append(f"Trend: {strength_word} {trend_dir}.")

    # Add volume context
    if hl_output.volume and hl_output.volume.classification:
        vol_class = hl_output.volume.classification
        summary_parts.append(f"Volume: {vol_class} retest conviction.")

    # Add probability drift context
    if hl_output.probability_drift and hl_output.probability_drift.drift_direction:
        drift_dir = hl_output.probability_drift.drift_direction
        summary_parts.append(f"Probability: {drift_dir}.")

    # Add structure context
    if hl_output.structure and hl_output.structure.structure_state:
        struct_state = hl_output.structure.structure_state
        summary_parts.append(f"Structure: {struct_state}.")

    # Add volatility warning if needed
    if hl_output.volatility and hl_output.volatility.regime in ['elevated', 'extreme']:
        vol_regime = hl_output.volatility.regime
        summary_parts.append(f"Volatility: {vol_regime.upper()} - heightened risk management required.")

    # Final assessment
    if bias == 'bullish' and confidence > 0.6:
        summary_parts.append("Conditions favor upward continuation with measured risk.")
    elif bias == 'bearish' and confidence > 0.6:
        summary_parts.append("Conditions favor downward continuation with measured risk.")
    elif confidence < 0.4:
        summary_parts.append("Mixed signals suggest patience and selective participation.")
    else:
        summary_parts.append("Monitor for clearer directional signals.")

    return " ".join(summary_parts)

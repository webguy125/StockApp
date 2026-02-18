"""
Probability Drift Formula

Defines a clear, deterministic formula for probability drift.
Compares initial model probability to live conditions.

IMPORTANT CONSTRAINTS:
- Does NOT modify the ML model or its training
- Does NOT change how initial_probability is computed
- Probability drift computed purely from existing probabilities and live analytics
- No feedback into ML or core engine - HL-only
"""

from typing import Dict


def compute_probability_drift(
    initial_probability: float,
    trend_strength: float,
    volume_classification: str,
    volatility_state: str,
    structure_state: str
) -> Dict:
    """
    Compute live_probability and delta based on initial_probability and live analytics.

    Formula:
    live_probability = initial_probability + trend_adjustment + volume_adjustment + volatility_adjustment + structure_adjustment
    Bounded within [0.0, 1.0]

    Args:
        initial_probability: Original model output (0.0 - 1.0)
        trend_strength: Trend strength normalized (0.0 - 1.0)
        volume_classification: 'strong', 'neutral', 'weak'
        volatility_state: 'expanding', 'contracting', 'stable'
        structure_state: 'holding', 'breaking', 'retesting'

    Returns:
        Dictionary with:
        - initial_probability: Input value
        - live_probability: Adjusted probability (0.0 - 1.0)
        - delta: live_probability - initial_probability
        - drift_direction: 'strengthening', 'weakening', 'stable'
        - commentary: Human-readable explanation

    Constraints:
        - Deterministic and transparent
        - live_probability always bounded [0.0, 1.0]
        - delta = live_probability - initial_probability
        - No feedback into ML or core engine
    """

    # Validate initial_probability
    if initial_probability < 0.0 or initial_probability > 1.0:
        initial_probability = max(0.0, min(1.0, initial_probability))

    # Initialize live_probability
    live_probability = initial_probability

    # Track adjustments for transparency
    adjustments = {
        'trend': 0.0,
        'volume': 0.0,
        'volatility': 0.0,
        'structure': 0.0
    }

    # -------------------------------------------------------------------------
    # TREND ADJUSTMENT
    # -------------------------------------------------------------------------
    # Logic: Strong trend strengthens conviction in trend direction
    # - If initial_probability > 0.5 (bullish model): strong trend strengthens
    # - If initial_probability < 0.5 (bearish model): strong trend strengthens
    # - Neutral trend (0.5) has no adjustment

    if trend_strength > 0.65:  # Strong trend
        # Strengthen conviction by moving away from 0.5
        if initial_probability > 0.5:
            adjustments['trend'] = (trend_strength - 0.5) * 0.15  # Up to +0.075 boost
        else:
            adjustments['trend'] = -(trend_strength - 0.5) * 0.15  # Up to -0.075 boost
    elif trend_strength < 0.35:  # Weak trend
        # Weaken conviction by moving toward 0.5
        if initial_probability > 0.5:
            adjustments['trend'] = -(0.5 - trend_strength) * 0.10  # Up to -0.05 penalty
        else:
            adjustments['trend'] = (0.5 - trend_strength) * 0.10  # Up to +0.05 penalty

    # -------------------------------------------------------------------------
    # VOLUME ADJUSTMENT
    # -------------------------------------------------------------------------
    # Logic: Strong volume confirms conviction, weak volume reduces it

    if volume_classification == 'strong':
        # Strong volume confirms direction
        if initial_probability > 0.5:
            adjustments['volume'] = +0.05  # Strengthen bullish
        else:
            adjustments['volume'] = -0.05  # Strengthen bearish
    elif volume_classification == 'weak':
        # Weak volume reduces conviction
        if initial_probability > 0.5:
            adjustments['volume'] = -0.05  # Weaken bullish
        else:
            adjustments['volume'] = +0.05  # Weaken bearish
    # 'neutral' has no adjustment

    # -------------------------------------------------------------------------
    # VOLATILITY ADJUSTMENT
    # -------------------------------------------------------------------------
    # Logic: Expanding volatility reduces confidence, contracting increases it

    if volatility_state == 'expanding':
        # Expanding volatility reduces confidence - move toward 0.5
        if initial_probability > 0.5:
            adjustments['volatility'] = -0.03
        else:
            adjustments['volatility'] = +0.03
    elif volatility_state == 'contracting':
        # Contracting volatility increases confidence - move away from 0.5
        if initial_probability > 0.5:
            adjustments['volatility'] = +0.03
        else:
            adjustments['volatility'] = -0.03
    # 'stable' has no adjustment

    # -------------------------------------------------------------------------
    # STRUCTURE ADJUSTMENT
    # -------------------------------------------------------------------------
    # Logic: Structure holding supports continuation, breaking suggests reversal

    if structure_state == 'holding':
        # Structure holding supports current direction
        if initial_probability > 0.5:
            adjustments['structure'] = +0.04  # Strengthen bullish
        else:
            adjustments['structure'] = -0.04  # Strengthen bearish
    elif structure_state == 'breaking':
        # Structure breaking suggests reversal
        if initial_probability > 0.5:
            adjustments['structure'] = -0.08  # Weaken bullish (breaking support)
        else:
            adjustments['structure'] = +0.08  # Weaken bearish (breaking resistance)
    elif structure_state == 'retesting':
        # Structure retest is neutral - small penalty for uncertainty
        if initial_probability > 0.5:
            adjustments['structure'] = -0.02
        else:
            adjustments['structure'] = +0.02

    # -------------------------------------------------------------------------
    # COMPUTE LIVE PROBABILITY
    # -------------------------------------------------------------------------

    total_adjustment = sum(adjustments.values())
    live_probability = initial_probability + total_adjustment

    # CRITICAL: Bound within [0.0, 1.0]
    live_probability = max(0.0, min(1.0, live_probability))

    # Compute delta
    delta = live_probability - initial_probability

    # Determine drift direction
    if abs(delta) < 0.05:
        drift_direction = 'stable'
    elif delta > 0:
        drift_direction = 'strengthening'
    else:
        drift_direction = 'weakening'

    # Generate commentary
    commentary = _generate_drift_commentary(
        initial_probability, live_probability, delta, drift_direction, adjustments
    )

    return {
        'initial_probability': initial_probability,
        'live_probability': live_probability,
        'delta': delta,
        'drift_direction': drift_direction,
        'commentary': commentary,
        'adjustments': adjustments  # For transparency and debugging
    }


def _generate_drift_commentary(
    initial_prob: float,
    live_prob: float,
    delta: float,
    direction: str,
    adjustments: Dict
) -> str:
    """
    Generate human-readable commentary explaining the drift.

    Args:
        initial_prob: Initial probability
        live_prob: Live adjusted probability
        delta: Difference
        direction: Drift direction
        adjustments: Dict of individual adjustments

    Returns:
        Commentary string
    """
    initial_pct = initial_prob * 100
    live_pct = live_prob * 100
    delta_pct = delta * 100

    if direction == 'stable':
        commentary = f"Probability stable at {live_pct:.1f}% (initial: {initial_pct:.1f}%). Live conditions align with model assessment."
    elif direction == 'strengthening':
        # Identify strongest contributor
        max_adj_key = max(adjustments, key=lambda k: adjustments[k] if adjustments[k] > 0 else 0)
        max_adj_val = adjustments[max_adj_key]

        if max_adj_val > 0.04:
            main_factor = max_adj_key
            commentary = f"Probability strengthened from {initial_pct:.1f}% to {live_pct:.1f}% ({delta_pct:+.1f}%). Primary driver: {main_factor} alignment."
        else:
            commentary = f"Probability strengthened from {initial_pct:.1f}% to {live_pct:.1f}% ({delta_pct:+.1f}%). Multiple factors support model conviction."
    else:  # weakening
        # Identify strongest detractor
        min_adj_key = min(adjustments, key=lambda k: adjustments[k] if adjustments[k] < 0 else 0)
        min_adj_val = adjustments[min_adj_key]

        if min_adj_val < -0.04:
            main_factor = min_adj_key
            commentary = f"Probability weakened from {initial_pct:.1f}% to {live_pct:.1f}% ({delta_pct:+.1f}%). Primary detractor: {main_factor} divergence."
        else:
            commentary = f"Probability weakened from {initial_pct:.1f}% to {live_pct:.1f}% ({delta_pct:+.1f}%). Multiple factors reduce model conviction."

    return commentary


# Testing and validation

if __name__ == '__main__':
    print("=" * 80)
    print("PROBABILITY DRIFT FORMULA - TEST CASES")
    print("=" * 80)

    test_cases = [
        {
            'name': 'Bullish Model + Strong Trend + Strong Volume',
            'initial_probability': 0.75,
            'trend_strength': 0.8,
            'volume_classification': 'strong',
            'volatility_state': 'stable',
            'structure_state': 'holding'
        },
        {
            'name': 'Bullish Model + Weak Trend + Weak Volume',
            'initial_probability': 0.65,
            'trend_strength': 0.3,
            'volume_classification': 'weak',
            'volatility_state': 'expanding',
            'structure_state': 'breaking'
        },
        {
            'name': 'Neutral Model + Mixed Signals',
            'initial_probability': 0.5,
            'trend_strength': 0.5,
            'volume_classification': 'neutral',
            'volatility_state': 'stable',
            'structure_state': 'holding'
        },
        {
            'name': 'Bearish Model + Strong Trend + Strong Volume',
            'initial_probability': 0.25,
            'trend_strength': 0.8,
            'volume_classification': 'strong',
            'volatility_state': 'stable',
            'structure_state': 'holding'
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['name']}")
        print("-" * 80)

        result = compute_probability_drift(
            initial_probability=test['initial_probability'],
            trend_strength=test['trend_strength'],
            volume_classification=test['volume_classification'],
            volatility_state=test['volatility_state'],
            structure_state=test['structure_state']
        )

        print(f"Initial Probability: {result['initial_probability']*100:.1f}%")
        print(f"Live Probability: {result['live_probability']*100:.1f}%")
        print(f"Delta: {result['delta']*100:+.1f}%")
        print(f"Direction: {result['drift_direction']}")
        print(f"Adjustments: {result['adjustments']}")
        print(f"Commentary: {result['commentary']}")

    print("\n" + "=" * 80)

"""
Regime Transition Detection Engine
Detects transitions between regime states and identifies strengthening/weakening signals

RESTRICTIONS:
- ONLY for HOLD signals
- NO modifications to prediction engine
- NO modifications to BUY/SELL logic
- Purely descriptive detection (no predictions)
"""

from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_transition(
    current_regime: str,
    previous_regime: Optional[str],
    drift: float,
    previous_drift: Optional[float],
    quality_score: int,
    previous_quality_score: Optional[int]
) -> Tuple[str, float]:
    """
    Detect regime transition and calculate transition strength

    Transition Types:
    - Neutral → Bullish Drift
    - Neutral → Bearish Drift
    - Neutral → Breaking
    - Compression → Expansion
    - Expansion → Compression
    - Strengthening Neutral
    - Weakening Neutral

    Args:
        current_regime: Current regime state
        previous_regime: Previous regime state (can be None)
        drift: Current drift value
        previous_drift: Previous drift value (can be None)
        quality_score: Current quality score
        previous_quality_score: Previous quality score (can be None)

    Returns:
        Tuple of (transition_state, transition_strength)
        transition_strength: 0.0-1.0 (0=no transition, 1=strong transition)
    """
    try:
        # If no previous state, no transition
        if previous_regime is None:
            return ("No Transition", 0.0)

        # If regime state unchanged
        if current_regime == previous_regime:
            # Check for strengthening/weakening within same regime
            if previous_quality_score is not None:
                quality_delta = quality_score - previous_quality_score

                if quality_delta >= 10:
                    strength = min(1.0, quality_delta / 20.0)
                    return ("Strengthening Neutral", strength)
                elif quality_delta <= -10:
                    strength = min(1.0, abs(quality_delta) / 20.0)
                    return ("Weakening Neutral", strength)

            return ("No Transition", 0.0)

        # Regime state changed - detect transition type
        transition_strength = calculate_transition_strength(
            current_regime, previous_regime, drift, previous_drift,
            quality_score, previous_quality_score
        )

        # Classify transition
        transition_type = classify_transition(current_regime, previous_regime)

        return (transition_type, transition_strength)

    except Exception as e:
        logger.error(f"[TRANSITION] Error detecting transition: {e}")
        return ("Unknown", 0.0)


def classify_transition(current_regime: str, previous_regime: str) -> str:
    """
    Classify the type of regime transition

    Args:
        current_regime: Current regime state
        previous_regime: Previous regime state

    Returns:
        Transition type string
    """
    # Neutral → Directional
    if "Neutral" in previous_regime and "Bullish Drift" in current_regime:
        return "Neutral → Bullish Drift"
    if "Neutral" in previous_regime and "Bearish Drift" in current_regime:
        return "Neutral → Bearish Drift"
    if "Neutral" in previous_regime and "Breaking" in current_regime:
        return "Neutral → Breaking"

    # Directional → Neutral
    if "Bullish Drift" in previous_regime and "Neutral" in current_regime:
        return "Bullish Drift → Neutral"
    if "Bearish Drift" in previous_regime and "Neutral" in current_regime:
        return "Bearish Drift → Neutral"

    # Volatility transitions
    if "Compression" in previous_regime and "Expansion" in current_regime:
        return "Compression → Expansion"
    if "Expansion" in previous_regime and "Compression" in current_regime:
        return "Expansion → Compression"

    # Volatility → Neutral
    if "Compression" in previous_regime and "Neutral" in current_regime:
        return "Compression → Neutral"
    if "Expansion" in previous_regime and "Neutral" in current_regime:
        return "Expansion → Neutral"

    # Neutral → Volatility
    if "Neutral" in previous_regime and "Compression" in current_regime:
        return "Neutral → Compression"
    if "Neutral" in previous_regime and "Expansion" in current_regime:
        return "Neutral → Expansion"

    # Expanding → Breaking
    if "Expanding" in previous_regime and "Breaking" in current_regime:
        return "Expanding → Breaking"

    # Directional switch
    if "Bullish" in previous_regime and "Bearish" in current_regime:
        return "Bullish → Bearish"
    if "Bearish" in previous_regime and "Bullish" in current_regime:
        return "Bearish → Bullish"

    # Generic transition
    return f"{previous_regime} → {current_regime}"


def calculate_transition_strength(
    current_regime: str,
    previous_regime: str,
    drift: float,
    previous_drift: Optional[float],
    quality_score: int,
    previous_quality_score: Optional[int]
) -> float:
    """
    Calculate strength of regime transition (0.0-1.0)

    Factors:
    - Magnitude of drift change
    - Magnitude of quality score change
    - Severity of regime change

    Args:
        current_regime: Current regime state
        previous_regime: Previous regime state
        drift: Current drift
        previous_drift: Previous drift
        quality_score: Current quality score
        previous_quality_score: Previous quality score

    Returns:
        Transition strength (0.0-1.0)
    """
    strength = 0.5  # Base strength for any regime change

    # Factor 1: Drift change
    if previous_drift is not None:
        drift_delta = abs(drift - previous_drift)
        # Normalize: 0-3 drift change = 0.0-0.3 strength contribution
        drift_strength = min(0.3, drift_delta / 10.0)
        strength += drift_strength

    # Factor 2: Quality score change
    if previous_quality_score is not None:
        quality_delta = abs(quality_score - previous_quality_score)
        # Normalize: 0-30 quality change = 0.0-0.3 strength contribution
        quality_strength = min(0.3, quality_delta / 100.0)
        strength += quality_strength

    # Factor 3: Severity of regime change
    severity_bonus = 0.0
    if "Breaking" in current_regime:
        severity_bonus = 0.4  # Breaking is severe
    elif "Expansion" in current_regime:
        severity_bonus = 0.3  # Expansion is concerning
    elif "Drift" in current_regime:
        severity_bonus = 0.2  # Drift is moderate
    elif "Compression" in current_regime:
        severity_bonus = -0.2  # Compression is favorable

    strength += severity_bonus

    # Clamp to 0.0-1.0
    strength = max(0.0, min(1.0, strength))

    return round(strength, 2)


def get_transition_severity(transition_type: str) -> str:
    """
    Get severity level of transition

    Args:
        transition_type: Transition type string

    Returns:
        Severity level: "Critical", "Warning", "Info", "Favorable"
    """
    if "Breaking" in transition_type:
        return "Critical"
    elif "Expansion" in transition_type:
        return "Warning"
    elif "Drift" in transition_type and "Neutral" not in transition_type.split("→")[1]:
        return "Warning"
    elif "Compression" in transition_type:
        return "Favorable"
    elif "Weakening" in transition_type:
        return "Warning"
    elif "Strengthening" in transition_type:
        return "Favorable"
    else:
        return "Info"

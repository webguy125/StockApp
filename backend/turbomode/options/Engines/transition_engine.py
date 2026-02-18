"""
Transition Engine - Detects regime transitions
"""


def detect_transition(current_regime, previous_regime, drift, previous_drift, quality_score, previous_quality_score):
    """
    Detect regime transitions and measure strength.

    Args:
        current_regime: Current regime state
        previous_regime: Previous regime state
        drift: Current drift
        previous_drift: Previous drift
        quality_score: Current quality score
        previous_quality_score: Previous quality score

    Returns:
        Tuple of (transition_state, transition_strength)
        transition_state: 'No Transition', 'Early', 'Active', 'Completed'
        transition_strength: 0.0 to 1.0
    """
    if not previous_regime:
        return ("No Transition", 0.0)

    # Check if regime changed
    if current_regime != previous_regime:
        # Calculate transition strength based on drift and quality changes
        drift_change = abs(drift - (previous_drift or 0.0))
        quality_change = abs(quality_score - (previous_quality_score or 0.0))

        strength = min(1.0, (drift_change + quality_change / 100.0) / 2.0)

        if strength > 0.7:
            return ("Active", strength)
        elif strength > 0.3:
            return ("Early", strength)
        else:
            return ("Completed", strength)

    return ("No Transition", 0.0)

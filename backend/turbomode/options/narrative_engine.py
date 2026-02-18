"""
Regime Narrative Engine
Generates plain-English explanations of regime state and transitions

RESTRICTIONS:
- ONLY for HOLD signals
- NO modifications to prediction engine
- Narratives must be DESCRIPTIVE not PREDICTIVE
- NO trade recommendations
- NO modification of model signals
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def generate_narrative(
    regime_state: str,
    transition_state: str,
    transition_strength: float,
    drift: float,
    pressure_index: str,
    volatility_regime: str,
    quality_score: int,
    pnl: Optional[float] = None
) -> str:
    """
    Generate plain-English narrative explaining current regime

    IMPORTANT: Narrative is purely descriptive, not predictive

    Args:
        regime_state: Current regime state
        transition_state: Current transition state
        transition_strength: Transition strength (0.0-1.0)
        drift: Current drift value
        pressure_index: Pressure classification
        volatility_regime: Volatility classification
        quality_score: Quality score
        pnl: Current P&L (optional)

    Returns:
        Descriptive narrative string
    """
    try:
        # Build narrative components
        parts = []

        # Part 1: Regime state description
        regime_description = describe_regime_state(regime_state, drift, quality_score)
        parts.append(regime_description)

        # Part 2: Transition description (if transitioning)
        if transition_state != "No Transition":
            transition_description = describe_transition(
                transition_state, transition_strength
            )
            parts.append(transition_description)

        # Part 3: Pressure and volatility context
        context_description = describe_market_context(
            pressure_index, volatility_regime
        )
        if context_description:
            parts.append(context_description)

        # Part 4: P&L context (if available)
        if pnl is not None:
            pnl_description = describe_pnl(pnl)
            parts.append(pnl_description)

        # Combine parts
        narrative = " ".join(parts)

        logger.debug(f"[NARRATIVE] Generated: {narrative}")
        return narrative

    except Exception as e:
        logger.error(f"[NARRATIVE] Error generating narrative: {e}")
        return "Unable to generate narrative."


def describe_regime_state(regime_state: str, drift: float, quality_score: int) -> str:
    """
    Generate description of regime state

    Args:
        regime_state: Regime state
        drift: Drift value
        quality_score: Quality score

    Returns:
        Regime state description
    """
    descriptions = {
        "Stable Neutral": (
            f"Position is well-centered with minimal drift ({drift:+.2f}). "
            f"Quality score of {quality_score}/100 indicates stable conditions."
        ),
        "Expanding Neutral": (
            f"Position is drifting from center ({drift:+.2f}), but remains within "
            f"neutral regime. Quality score: {quality_score}/100."
        ),
        "Breaking Neutral": (
            f"Position is approaching breakout threshold with drift of {drift:+.2f}. "
            f"One side showing critical pressure. Quality score: {quality_score}/100."
        ),
        "Bullish Drift": (
            f"Position drifting upward toward call strike ({drift:+.2f}). "
            f"Underlying showing directional momentum. Quality score: {quality_score}/100."
        ),
        "Bearish Drift": (
            f"Position drifting downward toward put strike ({drift:+.2f}). "
            f"Underlying showing directional momentum. Quality score: {quality_score}/100."
        ),
        "Volatility Compression": (
            f"Implied volatility is collapsing with drift of {drift:+.2f}. "
            f"Favorable for short option sellers. Quality score: {quality_score}/100."
        ),
        "Volatility Expansion": (
            f"Implied volatility is spiking with drift of {drift:+.2f}. "
            f"Increased risk for short positions. Quality score: {quality_score}/100."
        )
    }

    return descriptions.get(
        regime_state,
        f"Current regime: {regime_state}. Drift: {drift:+.2f}, Quality: {quality_score}/100."
    )


def describe_transition(transition_state: str, transition_strength: float) -> str:
    """
    Generate description of regime transition

    Args:
        transition_state: Transition type
        transition_strength: Strength of transition

    Returns:
        Transition description
    """
    strength_word = get_strength_word(transition_strength)

    descriptions = {
        "Neutral → Bullish Drift": (
            f"{strength_word.capitalize()} transition from neutral to bullish drift detected."
        ),
        "Neutral → Bearish Drift": (
            f"{strength_word.capitalize()} transition from neutral to bearish drift detected."
        ),
        "Neutral → Breaking": (
            f"{strength_word.capitalize()} transition toward breakout detected."
        ),
        "Compression → Expansion": (
            f"{strength_word.capitalize()} shift from volatility compression to expansion."
        ),
        "Expansion → Compression": (
            f"{strength_word.capitalize()} shift from volatility expansion to compression."
        ),
        "Strengthening Neutral": (
            f"Neutral position {strength_word} strengthening."
        ),
        "Weakening Neutral": (
            f"Neutral position {strength_word} weakening."
        )
    }

    # Check for partial matches
    for key, desc in descriptions.items():
        if key in transition_state:
            return desc

    # Default
    return f"{strength_word.capitalize()} regime transition: {transition_state}."


def describe_market_context(pressure_index: str, volatility_regime: str) -> str:
    """
    Generate description of market context

    Args:
        pressure_index: Pressure classification
        volatility_regime: Volatility classification

    Returns:
        Market context description
    """
    parts = []

    # Pressure context
    pressure_descriptions = {
        "Stable": "Options premiums are decaying in balance",
        "Bullish": "Call premiums decaying faster than puts",
        "Bearish": "Put premiums decaying faster than calls",
        "Break": "One side approaching zero premium"
    }

    if pressure_index in pressure_descriptions:
        parts.append(pressure_descriptions[pressure_index])

    # Volatility context
    volatility_descriptions = {
        "Profitable": "with stable implied volatility",
        "Collapsing": "with rapidly collapsing IV",
        "Breakout": "with spiking implied volatility"
    }

    if volatility_regime in volatility_descriptions:
        parts.append(volatility_descriptions[volatility_regime])

    if parts:
        return " ".join(parts) + "."
    else:
        return ""


def describe_pnl(pnl: float) -> str:
    """
    Generate description of P&L status

    Args:
        pnl: Current P&L value

    Returns:
        P&L description
    """
    if pnl > 5:
        return f"Position showing strong profit of ${pnl:.2f}."
    elif pnl > 2:
        return f"Position showing modest profit of ${pnl:.2f}."
    elif pnl > 0:
        return f"Position slightly profitable at ${pnl:.2f}."
    elif pnl > -2:
        return f"Position showing minor loss of ${pnl:.2f}."
    elif pnl > -5:
        return f"Position showing moderate loss of ${pnl:.2f}."
    else:
        return f"Position showing significant loss of ${pnl:.2f}."


def get_strength_word(strength: float) -> str:
    """
    Get descriptive word for transition strength

    Args:
        strength: Transition strength (0.0-1.0)

    Returns:
        Strength descriptor
    """
    if strength >= 0.8:
        return "strong"
    elif strength >= 0.6:
        return "moderate"
    elif strength >= 0.4:
        return "gradual"
    else:
        return "weak"


def generate_summary_narrative(
    regime_state: str,
    quality_score: int,
    timeline_summary: dict
) -> str:
    """
    Generate high-level summary narrative from timeline data

    Args:
        regime_state: Current regime state
        quality_score: Current quality score
        timeline_summary: Timeline summary statistics

    Returns:
        Summary narrative string
    """
    try:
        parts = []

        # Current state
        parts.append(f"Currently in {regime_state} regime with quality score {quality_score}/100.")

        # Regime stability
        if timeline_summary.get('regime_changes', 0) == 0:
            parts.append("Regime has remained stable since monitoring began.")
        elif timeline_summary.get('regime_changes', 0) == 1:
            parts.append("Regime transitioned once during monitoring period.")
        else:
            parts.append(f"Regime has transitioned {timeline_summary.get('regime_changes', 0)} times.")

        # Quality trend
        quality_trend = timeline_summary.get('quality_trend', 'Stable')
        if quality_trend == "Improving":
            parts.append("Quality trend is improving over time.")
        elif quality_trend == "Declining":
            parts.append("Quality trend is declining over time.")
        else:
            parts.append("Quality remains stable over time.")

        return " ".join(parts)

    except Exception as e:
        logger.error(f"[NARRATIVE] Error generating summary: {e}")
        return f"Current regime: {regime_state}, Quality: {quality_score}/100."

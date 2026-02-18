"""
Signal Intelligence Layer for BUY/SELL Signals
Enriches existing BUY and SELL predictions with options-based context, strength, risk, and confirmation

RESTRICTIONS:
- ONLY enriches existing BUY/SELL signals
- NO modifications to prediction engine
- NO modifications to model output
- NO modifications to signal generation logic
- Purely additive intelligence layer
"""

from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def compute_signal_strength(
    quality_score: int,
    drift: float,
    pressure_index: str,
    volatility_regime: str,
    model_confidence: float
) -> str:
    """
    Classify signal strength as Strong, Moderate, or Weak

    Combines:
    - Quality score (0-100)
    - Drift magnitude
    - Pressure index alignment
    - Volatility regime
    - Model confidence

    Args:
        quality_score: Quality score from analytics
        drift: Drift from center
        pressure_index: Pressure classification
        volatility_regime: Volatility classification
        model_confidence: Model confidence (0.0-1.0)

    Returns:
        "Strong", "Moderate", or "Weak"
    """
    try:
        score = 0

        # Component 1: Model confidence (0-40 points)
        if model_confidence >= 0.70:
            score += 40
        elif model_confidence >= 0.55:
            score += 30
        elif model_confidence >= 0.45:
            score += 20
        else:
            score += 10

        # Component 2: Quality score (0-30 points)
        if quality_score >= 80:
            score += 30
        elif quality_score >= 60:
            score += 20
        elif quality_score >= 40:
            score += 10
        else:
            score += 5

        # Component 3: Pressure alignment (0-20 points)
        if pressure_index in ["Stable", "Bullish", "Bearish"]:
            score += 20
        elif pressure_index == "Break":
            score += 5

        # Component 4: Volatility (0-10 points)
        if volatility_regime in ["Profitable", "Collapsing"]:
            score += 10
        elif volatility_regime == "Breakout":
            score += 0

        # Classify
        if score >= 75:
            return "Strong"
        elif score >= 50:
            return "Moderate"
        else:
            return "Weak"

    except Exception as e:
        logger.error(f"[SIGNAL_INTEL] Error computing signal strength: {e}")
        return "Moderate"


def compute_confirmation_status(
    base_signal: str,
    pressure_index: str,
    drift: float,
    regime_state: str,
    volatility_regime: str
) -> str:
    """
    Classify confirmation status as Confirmed, Unconfirmed, or Contradicted

    Compares model direction (BUY/SELL) with:
    - Options pressure direction
    - Drift direction
    - Regime state
    - Volatility behavior

    Args:
        base_signal: "BUY" or "SELL"
        pressure_index: Pressure classification
        drift: Drift value
        regime_state: Regime state
        volatility_regime: Volatility classification

    Returns:
        "Confirmed", "Unconfirmed", or "Contradicted"
    """
    try:
        confirmation_signals = 0
        contradiction_signals = 0

        # Check pressure alignment
        if base_signal == "BUY":
            if pressure_index == "Bullish":
                confirmation_signals += 2
            elif pressure_index == "Bearish":
                contradiction_signals += 2
            elif pressure_index == "Break":
                contradiction_signals += 1
        elif base_signal == "SELL":
            if pressure_index == "Bearish":
                confirmation_signals += 2
            elif pressure_index == "Bullish":
                contradiction_signals += 2
            elif pressure_index == "Break":
                contradiction_signals += 1

        # Check drift direction
        if base_signal == "BUY" and drift > 1.0:
            confirmation_signals += 1
        elif base_signal == "SELL" and drift < -1.0:
            confirmation_signals += 1
        elif base_signal == "BUY" and drift < -1.0:
            contradiction_signals += 1
        elif base_signal == "SELL" and drift > 1.0:
            contradiction_signals += 1

        # Check regime alignment
        if "Bullish" in regime_state and base_signal == "BUY":
            confirmation_signals += 1
        elif "Bearish" in regime_state and base_signal == "SELL":
            confirmation_signals += 1
        elif "Bullish" in regime_state and base_signal == "SELL":
            contradiction_signals += 1
        elif "Bearish" in regime_state and base_signal == "BUY":
            contradiction_signals += 1

        # Check volatility
        if volatility_regime == "Breakout":
            contradiction_signals += 1
        elif volatility_regime in ["Profitable", "Collapsing"]:
            confirmation_signals += 1

        # Classify
        if confirmation_signals >= 3 and contradiction_signals == 0:
            return "Confirmed"
        elif contradiction_signals >= 2:
            return "Contradicted"
        else:
            return "Unconfirmed"

    except Exception as e:
        logger.error(f"[SIGNAL_INTEL] Error computing confirmation status: {e}")
        return "Unconfirmed"


def compute_risk_level(
    volatility_regime: str,
    transition_state: str,
    transition_strength: float,
    quality_score: int,
    regime_state: str
) -> str:
    """
    Classify risk level as Low, Medium, or High

    Uses:
    - Volatility regime
    - Transition state and strength
    - Quality score
    - Regime state

    Args:
        volatility_regime: Volatility classification
        transition_state: Transition state
        transition_strength: Transition strength (0.0-1.0)
        quality_score: Quality score
        regime_state: Regime state

    Returns:
        "Low", "Medium", or "High"
    """
    try:
        risk_score = 0

        # Component 1: Volatility regime
        if volatility_regime == "Breakout":
            risk_score += 3
        elif volatility_regime == "Profitable":
            risk_score += 1
        elif volatility_regime == "Collapsing":
            risk_score += 0

        # Component 2: Transition risk
        if "Breaking" in transition_state:
            risk_score += 3
        elif "Weakening" in transition_state:
            risk_score += 2
        elif transition_strength >= 0.7:
            risk_score += 2
        elif transition_strength >= 0.4:
            risk_score += 1

        # Component 3: Regime state
        if "Breaking" in regime_state:
            risk_score += 2
        elif "Expansion" in regime_state:
            risk_score += 2
        elif "Drift" in regime_state:
            risk_score += 1

        # Component 4: Quality score (inverse)
        if quality_score < 40:
            risk_score += 2
        elif quality_score < 60:
            risk_score += 1

        # Classify
        if risk_score >= 6:
            return "High"
        elif risk_score >= 3:
            return "Medium"
        else:
            return "Low"

    except Exception as e:
        logger.error(f"[SIGNAL_INTEL] Error computing risk level: {e}")
        return "Medium"


def compute_regime_alignment(
    base_signal: str,
    regime_state: str,
    pressure_index: str,
    drift: float
) -> str:
    """
    Classify regime alignment as Aligned, Neutral, or Misaligned

    Compares base_signal direction with:
    - Regime state
    - Pressure index
    - Drift direction

    Args:
        base_signal: "BUY" or "SELL"
        regime_state: Regime state
        pressure_index: Pressure classification
        drift: Drift value

    Returns:
        "Aligned", "Neutral", or "Misaligned"
    """
    try:
        alignment_score = 0

        # Check regime state alignment
        if base_signal == "BUY":
            if "Bullish" in regime_state or "Compression" in regime_state:
                alignment_score += 2
            elif "Bearish" in regime_state or "Breaking" in regime_state:
                alignment_score -= 2
            elif "Stable" in regime_state:
                alignment_score += 0
        elif base_signal == "SELL":
            if "Bearish" in regime_state or "Expansion" in regime_state:
                alignment_score += 2
            elif "Bullish" in regime_state or "Compression" in regime_state:
                alignment_score -= 2
            elif "Stable" in regime_state:
                alignment_score += 0

        # Check pressure alignment
        if base_signal == "BUY":
            if pressure_index == "Bullish":
                alignment_score += 1
            elif pressure_index == "Bearish":
                alignment_score -= 1
        elif base_signal == "SELL":
            if pressure_index == "Bearish":
                alignment_score += 1
            elif pressure_index == "Bullish":
                alignment_score -= 1

        # Check drift alignment
        if base_signal == "BUY" and drift > 0.5:
            alignment_score += 1
        elif base_signal == "SELL" and drift < -0.5:
            alignment_score += 1
        elif base_signal == "BUY" and drift < -0.5:
            alignment_score -= 1
        elif base_signal == "SELL" and drift > 0.5:
            alignment_score -= 1

        # Classify
        if alignment_score >= 2:
            return "Aligned"
        elif alignment_score <= -2:
            return "Misaligned"
        else:
            return "Neutral"

    except Exception as e:
        logger.error(f"[SIGNAL_INTEL] Error computing regime alignment: {e}")
        return "Neutral"


def compute_timing_phase(
    transition_state: str,
    transition_strength: float,
    quality_score: int,
    age_days: int
) -> str:
    """
    Classify timing phase as Early, Normal, or Late

    Uses:
    - Transition state and strength
    - Quality score trends
    - Signal age

    Args:
        transition_state: Transition state
        transition_strength: Transition strength (0.0-1.0)
        quality_score: Quality score
        age_days: Signal age in days

    Returns:
        "Early", "Normal", or "Late"
    """
    try:
        # Check for fresh transitions
        if transition_strength >= 0.6 and age_days <= 2:
            return "Early"

        # Check for strengthening
        if "Strengthening" in transition_state:
            return "Early"

        # Check for weakening or late-stage signals
        if "Weakening" in transition_state or "Breaking" in transition_state:
            return "Late"

        # Check age
        if age_days >= 10:
            return "Late"
        elif age_days >= 5:
            return "Normal"
        elif age_days <= 2:
            return "Early"

        # Check quality degradation
        if quality_score < 40:
            return "Late"
        elif quality_score >= 70:
            return "Early"

        return "Normal"

    except Exception as e:
        logger.error(f"[SIGNAL_INTEL] Error computing timing phase: {e}")
        return "Normal"


def compute_signal_quality_score(
    signal_strength: str,
    confirmation_status: str,
    risk_level: str,
    regime_alignment: str,
    timing_phase: str
) -> int:
    """
    Compute composite signal quality score (0-100)

    Combines all intelligence factors into single score

    Args:
        signal_strength: Signal strength classification
        confirmation_status: Confirmation classification
        risk_level: Risk classification
        regime_alignment: Regime alignment classification
        timing_phase: Timing phase classification

    Returns:
        Quality score 0-100
    """
    try:
        score = 50  # Base score

        # Signal strength component
        if signal_strength == "Strong":
            score += 25
        elif signal_strength == "Moderate":
            score += 10
        elif signal_strength == "Weak":
            score -= 10

        # Confirmation component
        if confirmation_status == "Confirmed":
            score += 20
        elif confirmation_status == "Unconfirmed":
            score += 0
        elif confirmation_status == "Contradicted":
            score -= 20

        # Risk component (inverse)
        if risk_level == "Low":
            score += 15
        elif risk_level == "Medium":
            score += 0
        elif risk_level == "High":
            score -= 15

        # Regime alignment component
        if regime_alignment == "Aligned":
            score += 15
        elif regime_alignment == "Neutral":
            score += 0
        elif regime_alignment == "Misaligned":
            score -= 15

        # Timing component
        if timing_phase == "Early":
            score += 10
        elif timing_phase == "Normal":
            score += 5
        elif timing_phase == "Late":
            score -= 10

        # Clamp to 0-100
        score = max(0, min(100, score))

        return score

    except Exception as e:
        logger.error(f"[SIGNAL_INTEL] Error computing signal quality score: {e}")
        return 50


def enrich_signal(
    symbol: str,
    base_signal: str,
    model_confidence: float,
    model_probability: float,
    analytics: Dict,
    regime_data: Dict,
    age_days: int
) -> Dict:
    """
    Enrich a BUY or SELL signal with full intelligence layer

    Args:
        symbol: Stock ticker
        base_signal: "BUY" or "SELL"
        model_confidence: Model confidence
        model_probability: Model probability
        analytics: Analytics dictionary from analytics_engine
        regime_data: Regime data from regime endpoint
        age_days: Signal age in days

    Returns:
        Enriched signal dictionary
    """
    try:
        # Compute all intelligence metrics
        signal_strength = compute_signal_strength(
            quality_score=analytics.get('quality_score', 50),
            drift=analytics.get('drift', 0.0),
            pressure_index=analytics.get('pressure_index', 'Unknown'),
            volatility_regime=analytics.get('volatility_regime', 'Unknown'),
            model_confidence=model_confidence
        )

        confirmation_status = compute_confirmation_status(
            base_signal=base_signal,
            pressure_index=analytics.get('pressure_index', 'Unknown'),
            drift=analytics.get('drift', 0.0),
            regime_state=regime_data.get('regime_state', 'Unknown'),
            volatility_regime=analytics.get('volatility_regime', 'Unknown')
        )

        risk_level = compute_risk_level(
            volatility_regime=analytics.get('volatility_regime', 'Unknown'),
            transition_state=regime_data.get('transition_state', 'No Transition'),
            transition_strength=regime_data.get('transition_strength', 0.0),
            quality_score=analytics.get('quality_score', 50),
            regime_state=regime_data.get('regime_state', 'Unknown')
        )

        regime_alignment = compute_regime_alignment(
            base_signal=base_signal,
            regime_state=regime_data.get('regime_state', 'Unknown'),
            pressure_index=analytics.get('pressure_index', 'Unknown'),
            drift=analytics.get('drift', 0.0)
        )

        timing_phase = compute_timing_phase(
            transition_state=regime_data.get('transition_state', 'No Transition'),
            transition_strength=regime_data.get('transition_strength', 0.0),
            quality_score=analytics.get('quality_score', 50),
            age_days=age_days
        )

        signal_quality_score = compute_signal_quality_score(
            signal_strength=signal_strength,
            confirmation_status=confirmation_status,
            risk_level=risk_level,
            regime_alignment=regime_alignment,
            timing_phase=timing_phase
        )

        logger.info(
            f"[SIGNAL_INTEL] {symbol} ({base_signal}): strength={signal_strength}, "
            f"confirmation={confirmation_status}, risk={risk_level}, "
            f"alignment={regime_alignment}, timing={timing_phase}, quality={signal_quality_score}"
        )

        return {
            'symbol': symbol,
            'base_signal': base_signal,
            'model_confidence': round(model_confidence, 3),
            'model_probability': round(model_probability, 3),
            'signal_strength': signal_strength,
            'confirmation_status': confirmation_status,
            'risk_level': risk_level,
            'regime_alignment': regime_alignment,
            'timing_phase': timing_phase,
            'signal_quality_score': signal_quality_score
        }

    except Exception as e:
        logger.error(f"[SIGNAL_INTEL] Error enriching signal for {symbol}: {e}")
        return {
            'symbol': symbol,
            'base_signal': base_signal,
            'model_confidence': model_confidence,
            'model_probability': model_probability,
            'signal_strength': 'Unknown',
            'confirmation_status': 'Unknown',
            'risk_level': 'Unknown',
            'regime_alignment': 'Unknown',
            'timing_phase': 'Unknown',
            'signal_quality_score': 0
        }

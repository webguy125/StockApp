"""
Signal Intelligence Engine - Enriches signals with final intelligence classification
"""


def enrich_signal(symbol, base_signal, model_confidence, model_probability, analytics, regime_data, age_days):
    """
    Enrich signal with intelligence classifications.

    Args:
        symbol: Stock symbol
        base_signal: 'BUY' or 'SELL'
        model_confidence: Model confidence (0.0 to 1.0)
        model_probability: Model probability (0.0 to 1.0)
        analytics: Dictionary with drift, pressure_index, volatility_regime, quality_score
        regime_data: Dictionary with regime_state, transition_state, transition_strength
        age_days: Signal age in days

    Returns:
        Enriched signal dictionary with intelligence fields
    """
    # Extract analytics
    drift = analytics.get('drift', 0.0)
    quality_score = analytics.get('quality_score', 0.0)
    volatility_regime = analytics.get('volatility_regime', 'Unknown')

    # Extract regime data
    regime_state = regime_data.get('regime_state', 'Unknown')
    transition_state = regime_data.get('transition_state', 'No Transition')

    # Classify signal strength based on confidence and quality
    if model_confidence >= 0.8 and quality_score >= 70:
        signal_strength = "Strong"
    elif model_confidence >= 0.6 and quality_score >= 50:
        signal_strength = "Moderate"
    else:
        signal_strength = "Weak"

    # Classify confirmation status
    if model_probability >= 0.85 and quality_score >= 75:
        confirmation_status = "Confirmed"
    elif model_probability >= 0.65:
        confirmation_status = "Partial"
    else:
        confirmation_status = "Unconfirmed"

    # Classify risk level
    if volatility_regime == "High" or quality_score < 40:
        risk_level = "High"
    elif volatility_regime == "Medium" or quality_score < 60:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Classify regime alignment
    if base_signal == "BUY" and regime_state == "Bullish":
        regime_alignment = "Aligned"
    elif base_signal == "SELL" and regime_state == "Bearish":
        regime_alignment = "Aligned"
    elif regime_state in ["Bullish", "Bearish"]:
        regime_alignment = "Conflicting"
    else:
        regime_alignment = "Neutral"

    # Classify timing phase
    if age_days < 3 and confirmation_status == "Confirmed":
        timing_phase = "Entry"
    elif age_days > 10 or quality_score < 40:
        timing_phase = "Exit"
    else:
        timing_phase = "Hold"

    # Calculate final signal quality score (0-100)
    signal_quality_score = (
        model_confidence * 40 +
        (quality_score / 100) * 30 +
        model_probability * 20 +
        (1.0 if regime_alignment == "Aligned" else 0.5 if regime_alignment == "Neutral" else 0.0) * 10
    )

    return {
        "symbol": symbol,
        "base_signal": base_signal,
        "model_confidence": model_confidence,
        "model_probability": model_probability,
        "signal_strength": signal_strength,
        "confirmation_status": confirmation_status,
        "risk_level": risk_level,
        "regime_alignment": regime_alignment,
        "timing_phase": timing_phase,
        "signal_quality_score": round(signal_quality_score, 2)
    }

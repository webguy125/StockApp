"""
Trade Service Module
Trade idea generation and strategy detection
"""

import uuid
from services.indicator_service import calculate_rsi


def generate_trade_ideas(df, symbol, current_price):
    """Auto-generate trade ideas based on technical analysis"""
    # Calculate indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df = df.dropna()

    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    rsi = df['RSI'].iloc[-1]

    ideas = []

    # Strategy 1: Moving Average Crossover (Golden/Death Cross)
    golden_cross_ideas = detect_golden_cross(df, current_price, sma_20, sma_50)
    ideas.extend(golden_cross_ideas)

    death_cross_ideas = detect_death_cross(df, current_price, sma_20, sma_50)
    ideas.extend(death_cross_ideas)

    # Strategy 2: RSI Oversold/Overbought
    rsi_ideas = detect_rsi_signals(current_price, rsi)
    ideas.extend(rsi_ideas)

    # Strategy 3: Support/Resistance
    support_resistance_ideas = detect_support_resistance(df, current_price)
    ideas.extend(support_resistance_ideas)

    return {
        'symbol': symbol,
        'current_price': float(current_price),
        'ideas': ideas,
        'total_ideas': len(ideas),
        'market_condition': {
            'rsi': float(rsi),
            'trend': 'bullish' if sma_20 > sma_50 else 'bearish',
            'support': float(df['Low'].tail(20).min()),
            'resistance': float(df['High'].tail(20).max())
        }
    }


def detect_golden_cross(df, current_price, sma_20, sma_50):
    """Detect golden cross pattern"""
    ideas = []

    if sma_20 > sma_50 and df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]:
        ideas.append({
            'id': str(uuid.uuid4()),
            'type': 'BUY',
            'strategy': 'Golden Cross',
            'entry': float(current_price),
            'target': float(current_price * 1.10),
            'stop_loss': float(current_price * 0.95),
            'risk_reward': 2.0,
            'confidence': 0.75,
            'reason': 'SMA 20 crossed above SMA 50 (bullish signal)',
            'timeframe': '1-3 weeks'
        })

    return ideas


def detect_death_cross(df, current_price, sma_20, sma_50):
    """Detect death cross pattern"""
    ideas = []

    if sma_20 < sma_50 and df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]:
        ideas.append({
            'id': str(uuid.uuid4()),
            'type': 'SELL',
            'strategy': 'Death Cross',
            'entry': float(current_price),
            'target': float(current_price * 0.90),
            'stop_loss': float(current_price * 1.05),
            'risk_reward': 2.0,
            'confidence': 0.75,
            'reason': 'SMA 20 crossed below SMA 50 (bearish signal)',
            'timeframe': '1-3 weeks'
        })

    return ideas


def detect_rsi_signals(current_price, rsi):
    """Detect RSI oversold/overbought signals"""
    ideas = []

    if rsi < 30:
        ideas.append({
            'id': str(uuid.uuid4()),
            'type': 'BUY',
            'strategy': 'RSI Oversold',
            'entry': float(current_price),
            'target': float(current_price * 1.08),
            'stop_loss': float(current_price * 0.96),
            'risk_reward': 2.0,
            'confidence': 0.70,
            'reason': f'RSI at {rsi:.1f} indicates oversold conditions',
            'timeframe': '3-7 days'
        })
    elif rsi > 70:
        ideas.append({
            'id': str(uuid.uuid4()),
            'type': 'SELL',
            'strategy': 'RSI Overbought',
            'entry': float(current_price),
            'target': float(current_price * 0.92),
            'stop_loss': float(current_price * 1.04),
            'risk_reward': 2.0,
            'confidence': 0.70,
            'reason': f'RSI at {rsi:.1f} indicates overbought conditions',
            'timeframe': '3-7 days'
        })

    return ideas


def detect_support_resistance(df, current_price):
    """Detect support and resistance levels"""
    ideas = []

    recent_high = df['High'].tail(20).max()
    recent_low = df['Low'].tail(20).min()

    # Near support
    if abs(current_price - recent_low) / recent_low < 0.02:
        ideas.append({
            'id': str(uuid.uuid4()),
            'type': 'BUY',
            'strategy': 'Support Bounce',
            'entry': float(current_price),
            'target': float((recent_high + recent_low) / 2),
            'stop_loss': float(recent_low * 0.98),
            'risk_reward': 3.0,
            'confidence': 0.65,
            'reason': f'Price near support at ${recent_low:.2f}',
            'timeframe': '1-2 weeks'
        })

    # Near resistance
    if abs(current_price - recent_high) / recent_high < 0.02:
        ideas.append({
            'id': str(uuid.uuid4()),
            'type': 'SELL',
            'strategy': 'Resistance Rejection',
            'entry': float(current_price),
            'target': float((recent_high + recent_low) / 2),
            'stop_loss': float(recent_high * 1.02),
            'risk_reward': 3.0,
            'confidence': 0.65,
            'reason': f'Price near resistance at ${recent_high:.2f}',
            'timeframe': '1-2 weeks'
        })

    return ideas

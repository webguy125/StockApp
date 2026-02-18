"""
Wing Selector Engine - Selects optimal strikes for iron condor wings
"""


def select_wings(calls, puts, current_price, stop_upper, stop_lower):
    """
    Select short call and short put strikes for iron condor.

    Args:
        calls: Dictionary of call option data {strike: {bid, ask, mid}}
        puts: Dictionary of put option data {strike: {bid, ask, mid}}
        current_price: Current stock price
        stop_upper: Upper stop loss level
        stop_lower: Lower stop loss level

    Returns:
        Tuple of (short_call_strike, short_put_strike) or None if no valid wings
    """
    if not calls or not puts:
        return None

    # Get available strikes
    call_strikes = sorted([float(s) for s in calls.keys()])
    put_strikes = sorted([float(s) for s in puts.keys()])

    if not call_strikes or not put_strikes:
        return None

    # Simple placeholder logic: select strikes near stop levels
    # Real implementation would consider Greeks, IV, probability, etc.

    # Find call strike near upper stop
    short_call_strike = min(call_strikes, key=lambda s: abs(s - stop_upper)) if stop_upper else call_strikes[-1]

    # Find put strike near lower stop
    short_put_strike = min(put_strikes, key=lambda s: abs(s - stop_lower)) if stop_lower else put_strikes[0]

    return (short_call_strike, short_put_strike)

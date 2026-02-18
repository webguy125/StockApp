"""
Condor Pricing Engine - Calculates P&L for iron condor structures
"""


def calculate_condor_pnl(calls, puts, short_call_strike, short_put_strike):
    """
    Calculate P&L metrics for an iron condor.

    Args:
        calls: Dictionary of call option data {strike: {bid, ask, mid}}
        puts: Dictionary of put option data {strike: {bid, ask, mid}}
        short_call_strike: Short call strike price
        short_put_strike: Short put strike price

    Returns:
        Dictionary with max_credit, max_profit, max_loss, pnl
    """
    if not calls or not puts:
        return {"pnl": None}

    # Get option prices
    short_call = calls.get(short_call_strike, {})
    short_put = puts.get(short_put_strike, {})

    if not short_call or not short_put:
        return {"pnl": None}

    # Simple placeholder calculation
    # Real implementation would calculate full condor spreads with long wings
    call_premium = short_call.get('mid', short_call.get('bid', 0))
    put_premium = short_put.get('mid', short_put.get('bid', 0))

    max_credit = call_premium + put_premium
    max_profit = max_credit  # Simplified
    max_loss = max_credit * -2  # Simplified placeholder
    pnl = max_credit + max_loss  # Net P&L

    return {
        'max_credit': max_credit,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'pnl': pnl
    }

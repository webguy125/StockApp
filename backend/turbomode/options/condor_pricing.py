"""
Iron Condor Pricing Calculator
Computes entry mid-price, current mid-price, and P&L
If invalid returns None
"""

from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_condor_pnl(
    calls: Dict[float, Dict],
    puts: Dict[float, Dict],
    short_call_strike: float,
    short_put_strike: float,
    entry_price_call: Optional[float] = None,
    entry_price_put: Optional[float] = None
) -> Optional[Dict]:
    """
    Calculate Iron Condor P&L

    Iron Condor = Short Call + Short Put (both OTM)
    - Entry: Credit received = call_premium + put_premium
    - Current: Credit if closed now = call_mid + put_mid
    - P&L = entry_credit - current_credit (positive = profit)

    Args:
        calls: Dictionary of call strikes and prices
        puts: Dictionary of put strikes and prices
        short_call_strike: Strike of short call
        short_put_strike: Strike of short put
        entry_price_call: Entry premium for call (if known, else use current)
        entry_price_put: Entry premium for put (if known, else use current)

    Returns:
        Dictionary with:
        {
            'entry_credit': float,
            'current_cost': float,
            'pnl': float,
            'pnl_pct': float,
            'call_premium': float,
            'put_premium': float
        }

        Returns None if invalid pricing
    """
    try:
        # Get current call premium
        if short_call_strike not in calls:
            logger.warning(f"[PRICING] Call strike {short_call_strike} not in chain")
            return None

        call_data = calls[short_call_strike]
        if 'mid' not in call_data or call_data['mid'] <= 0:
            logger.warning(f"[PRICING] Invalid call mid price for strike {short_call_strike}")
            return None

        current_call_mid = call_data['mid']

        # Get current put premium
        if short_put_strike not in puts:
            logger.warning(f"[PRICING] Put strike {short_put_strike} not in chain")
            return None

        put_data = puts[short_put_strike]
        if 'mid' not in put_data or put_data['mid'] <= 0:
            logger.warning(f"[PRICING] Invalid put mid price for strike {short_put_strike}")
            return None

        current_put_mid = put_data['mid']

        # Use entry prices if provided, else use current prices
        entry_call = entry_price_call if entry_price_call is not None else current_call_mid
        entry_put = entry_price_put if entry_price_put is not None else current_put_mid

        # Calculate P&L
        entry_credit = entry_call + entry_put
        current_cost = current_call_mid + current_put_mid
        pnl = entry_credit - current_cost

        # Calculate P&L percentage (relative to entry credit)
        pnl_pct = (pnl / entry_credit * 100) if entry_credit > 0 else 0

        result = {
            'entry_credit': round(entry_credit, 2),
            'current_cost': round(current_cost, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'call_premium': round(current_call_mid, 2),
            'put_premium': round(current_put_mid, 2)
        }

        logger.info(f"[PRICING] Condor P&L: entry={result['entry_credit']}, current={result['current_cost']}, pnl={result['pnl']} ({result['pnl_pct']}%)")
        return result

    except Exception as e:
        logger.error(f"[PRICING] Error calculating condor P&L: {e}")
        return None

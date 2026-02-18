"""
Wing Selector for Iron Condors
Selects nearest ITM call above current price and nearest ITM put below current price
Center of neutral trade is based on nearest ITM strike to current price
If either missing returns None
"""

from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def select_wings(
    calls: Dict[float, Dict],
    puts: Dict[float, Dict],
    current_price: float,
    stop_upper: float,
    stop_lower: float
) -> Optional[Tuple[float, float]]:
    """
    Select Iron Condor wing strikes

    Strategy:
    - Short call: Nearest ITM call strike ABOVE current price
    - Short put: Nearest ITM put strike BELOW current price
    - Center is based on nearest ITM strike to underlying price
    - Both must exist and have valid pricing

    Args:
        calls: Dictionary of {strike: {'bid': ..., 'ask': ..., 'mid': ...}}
        puts: Dictionary of {strike: {'bid': ..., 'ask': ..., 'mid': ...}}
        current_price: Current stock price
        stop_upper: Upper stop bound for HOLD signal (not used in ITM selection)
        stop_lower: Lower stop bound for HOLD signal (not used in ITM selection)

    Returns:
        Tuple of (short_call_strike, short_put_strike), or None if invalid
    """
    if not calls or not puts:
        logger.warning("[WINGS] Empty calls or puts dictionary")
        return None

    try:
        # Find short call: nearest ITM call strike ABOVE current price
        # ITM for call means strike < current_price, but we want the one above price
        # So we select nearest strike > current_price from calls
        valid_calls = [strike for strike in calls.keys() if strike > current_price]

        if not valid_calls:
            logger.warning(f"[WINGS] No call strikes above current price ({current_price})")
            return None

        short_call_strike = min(valid_calls)

        # Find short put: nearest ITM put strike BELOW current price
        # ITM for put means strike > current_price, but we want the one below price
        # So we select nearest strike < current_price from puts
        valid_puts = [strike for strike in puts.keys() if strike < current_price]

        if not valid_puts:
            logger.warning(f"[WINGS] No put strikes below current price ({current_price})")
            return None

        short_put_strike = max(valid_puts)

        # Validate strikes make sense
        if short_put_strike >= short_call_strike:
            logger.warning(f"[WINGS] Invalid strikes: put={short_put_strike} >= call={short_call_strike}")
            return None

        if short_put_strike >= current_price or short_call_strike <= current_price:
            logger.warning(f"[WINGS] Strikes not properly positioned: put={short_put_strike}, price={current_price}, call={short_call_strike}")
            return None

        # Validate both strikes have valid pricing
        if 'mid' not in calls[short_call_strike] or 'mid' not in puts[short_put_strike]:
            logger.warning("[WINGS] Missing mid price for selected strikes")
            return None

        if calls[short_call_strike]['mid'] <= 0 or puts[short_put_strike]['mid'] <= 0:
            logger.warning("[WINGS] Invalid mid price (<=0) for selected strikes")
            return None

        logger.info(f"[WINGS] Selected ITM strikes: call={short_call_strike}, put={short_put_strike} (price={current_price}, centered on nearest ITM)")
        return (short_call_strike, short_put_strike)

    except Exception as e:
        logger.error(f"[WINGS] Error selecting wings: {e}")
        return None

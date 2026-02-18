"""
HOLD Iron Condor Engine
Orchestrates full pipeline for computing Iron Condor P&L on HOLD signals
Enforces market-hours rule (08:30-15:00 CST)
If any step fails returns None

RESTRICTIONS:
- ONLY for HOLD signals
- NO modifications to prediction engine
- NO modifications to existing P&L logic
- NO modifications to BUY/SELL behavior
"""

from datetime import datetime
import pytz
from typing import Optional
import logging

from .options_data_provider import get_chain, get_underlying_price
from .expiration_selector import select_expiration
from .wing_selector import select_wings
from .condor_pricing import calculate_condor_pnl
from .errors import SkipReason

logger = logging.getLogger(__name__)


def is_market_hours() -> bool:
    """
    Check if current time is during regular U.S. market hours

    Market hours: 08:30-15:00 CST (09:30-16:00 EST)
    No pre-market, no after-hours

    Returns:
        True if within market hours, False otherwise
    """
    try:
        cst = pytz.timezone('America/Chicago')
        now = datetime.now(cst)

        # Market is closed on weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check time range (08:30-15:00 CST)
        market_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    except Exception as e:
        logger.error(f"[MARKET HOURS] Error checking market hours: {e}")
        return False


def compute_hold_condor_pnl(
    symbol: str,
    current_price: float,
    stop_upper: float,
    stop_lower: float,
    ibkr_client: Optional[object] = None  # Deprecated: kept for backward compatibility
) -> Optional[float]:
    """
    Compute real-time Iron Condor P&L for a HOLD signal

    Pipeline:
    1. Check market hours (skip if closed)
    2. Fetch option chain from unified REST provider (Tradier → Yahoo fallback)
    3. Select expiration (2 weeks preferred)
    4. Select OTM wings (call above stop_upper, put below stop_lower)
    5. Calculate P&L

    Args:
        symbol: Stock ticker
        current_price: Current stock price
        stop_upper: Upper stop bound for HOLD signal
        stop_lower: Lower stop bound for HOLD signal
        ibkr_client: DEPRECATED - no longer used (REST-only provider)

    Returns:
        P&L as float (positive = profit), or None if any step fails

    Skip Reasons:
    - Market closed
    - No option chain (both Tradier and Yahoo failed)
    - No valid expiration
    - No OTM wings
    - Invalid pricing
    """
    # Step 1: Check market hours
    if not is_market_hours():
        logger.debug(f"[CONDOR] {symbol}: Skipping (market closed) - {SkipReason.MARKET_CLOSED}")
        return None

    # Validate inputs
    if not stop_upper or not stop_lower or stop_upper <= stop_lower:
        logger.warning(f"[CONDOR] {symbol}: Invalid stop bounds - {SkipReason.MISSING_STOP_BOUNDS}")
        return None

    # Step 2: Fetch option chain from unified REST provider (Tradier → Yahoo fallback)
    try:
        chain_data = get_chain(symbol)

        if not chain_data or 'expirations' not in chain_data or 'chains' not in chain_data:
            logger.warning(f"[CONDOR] {symbol}: No option chain - {SkipReason.NO_OPTION_CHAIN}")
            return None

        # Step 3: Select expiration
        expiration = select_expiration(chain_data['expirations'])

        if not expiration or expiration not in chain_data['chains']:
            logger.warning(f"[CONDOR] {symbol}: No valid expiration - {SkipReason.NO_VALID_EXPIRATION}")
            return None

        chain = chain_data['chains'][expiration]
        calls = chain.get('calls', {})
        puts = chain.get('puts', {})

        if not calls or not puts:
            logger.warning(f"[CONDOR] {symbol}: Empty calls or puts - {SkipReason.NO_OTM_WINGS}")
            return None

        # Step 4: Select wings
        wings = select_wings(calls, puts, current_price, stop_upper, stop_lower)

        if not wings:
            logger.warning(f"[CONDOR] {symbol}: No valid OTM wings - {SkipReason.NO_OTM_WINGS}")
            return None

        short_call_strike, short_put_strike = wings

        # Step 5: Calculate P&L
        pricing = calculate_condor_pnl(calls, puts, short_call_strike, short_put_strike)

        if not pricing or 'pnl' not in pricing:
            logger.warning(f"[CONDOR] {symbol}: Invalid pricing - {SkipReason.INVALID_MID_PRICE}")
            return None

        pnl = pricing['pnl']
        logger.info(f"[CONDOR] {symbol}: P&L = ${pnl:.2f} (call={short_call_strike}, put={short_put_strike})")

        return pnl

    except Exception as e:
        logger.error(f"[CONDOR] {symbol}: Unexpected error - {e}")
        return None

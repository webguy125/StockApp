"""
Unified REST-Only Options Data Provider
========================================

SINGLE SOURCE OF TRUTH for all options data in backend/turbomode/Options

Architecture:
- Tier 1 (Primary): Tradier REST API
- Tier 2 (Fallback): Yahoo Finance (yfinance)
- NO WebSocket, NO streaming, NO IBKR for data fetching

All modules must import from this file instead of calling Tradier or Yahoo directly.

Usage:
    from .options_data_provider import get_chain, get_expirations, get_underlying_price, get_greeks

Design Rules:
- REST-only (deterministic, reproducible)
- Normalized data structure (all sources return same format)
- Contamination-proof (detect malformed/NaN data and fallback)
- Session auto-renewal for 24/7 operation
"""

import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import math

# Import Tradier Options REST client (dedicated options client)
from .tradier_options_client import get_tradier_options_client

# Import yfinance for fallback
try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL CLIENTS (Singleton Pattern)
# ============================================================================

_tradier_client = None
_yahoo_available = yf is not None


def _get_tradier_client():
    """Get singleton Tradier Options REST client"""
    global _tradier_client
    if _tradier_client is None:
        try:
            _tradier_client = get_tradier_options_client()
            logger.info("[OPTIONS PROVIDER] Tradier Options REST client initialized (Tier 1)")
        except Exception as e:
            logger.error(f"[OPTIONS PROVIDER] Failed to initialize Tradier Options client: {e}")
            _tradier_client = None
    return _tradier_client


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

def is_malformed(chain: Optional[Dict]) -> bool:
    """
    Detect malformed option chain data

    Checks for:
    - None or empty chain
    - Missing required fields (calls, puts, expirations)
    - NaN values in critical fields (bid, ask, mid, strike)
    - Empty expirations list
    - Zero or negative prices

    Args:
        chain: Option chain dictionary

    Returns:
        True if malformed, False if valid
    """
    if not chain:
        return True

    # Check for required top-level keys
    if 'expirations' not in chain or 'chains' not in chain:
        logger.warning("[DATA QUALITY] Missing expirations or chains")
        return True

    # Check expirations list
    if not chain['expirations'] or len(chain['expirations']) == 0:
        logger.warning("[DATA QUALITY] Empty expirations list")
        return True

    # Check chains structure
    chains = chain.get('chains', {})
    if not chains:
        logger.warning("[DATA QUALITY] Empty chains dict")
        return True

    # Sample-check first expiration for data quality
    first_exp = chain['expirations'][0]
    if first_exp not in chains:
        logger.warning(f"[DATA QUALITY] First expiration {first_exp} not in chains")
        return True

    exp_chain = chains[first_exp]
    calls = exp_chain.get('calls', {})
    puts = exp_chain.get('puts', {})

    if not calls or not puts:
        logger.warning("[DATA QUALITY] Missing calls or puts in first expiration")
        return True

    # Check for NaN or invalid prices in first few strikes
    for strike_dict in [calls, puts]:
        for strike, data in list(strike_dict.items())[:3]:  # Check first 3 strikes
            if not isinstance(data, dict):
                logger.warning(f"[DATA QUALITY] Strike {strike} data is not a dict")
                return True

            for price_field in ['bid', 'ask', 'mid']:
                if price_field in data:
                    price = data[price_field]
                    if price is None:
                        logger.warning(f"[DATA QUALITY] {price_field} is None for strike {strike}")
                        return True
                    if isinstance(price, float) and (math.isnan(price) or price <= 0):
                        logger.warning(f"[DATA QUALITY] {price_field} is NaN or <= 0 for strike {strike}")
                        return True

    return False


def has_sufficient_strikes(chain: Dict, min_strikes: int = 5) -> bool:
    """
    Check if chain has sufficient strikes for iron condor construction

    Args:
        chain: Option chain dictionary
        min_strikes: Minimum number of strikes required (default: 5)

    Returns:
        True if sufficient, False otherwise
    """
    if not chain or 'chains' not in chain:
        return False

    for exp, exp_chain in chain['chains'].items():
        calls = exp_chain.get('calls', {})
        puts = exp_chain.get('puts', {})

        if len(calls) >= min_strikes and len(puts) >= min_strikes:
            return True

    return False


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_chain(raw_chain: Dict, source: str) -> Dict:
    """
    Normalize option chain structure from different sources

    Unified structure:
    {
        'expirations': ['20260220', '20260227', ...],
        'chains': {
            '20260220': {
                'calls': {
                    150.0: {'bid': 2.5, 'ask': 2.6, 'mid': 2.55, 'volume': 100, 'open_interest': 500},
                    ...
                },
                'puts': {
                    150.0: {'bid': 1.8, 'ask': 1.9, 'mid': 1.85, 'volume': 50, 'open_interest': 300},
                    ...
                }
            }
        }
    }

    Args:
        raw_chain: Raw chain data from source
        source: 'tradier' or 'yahoo'

    Returns:
        Normalized chain dictionary
    """
    if source == 'tradier':
        return _normalize_tradier_chain(raw_chain)
    elif source == 'yahoo':
        return _normalize_yahoo_chain(raw_chain)
    else:
        logger.error(f"[NORMALIZE] Unknown source: {source}")
        return {}


def _normalize_tradier_chain(raw_chain: Dict) -> Dict:
    """
    Normalize Tradier option chain response from tradier_options_client.

    Input format from tradier_options_client.get_option_chain():
    {
        'calls': {
            150.0: {'bid': 2.5, 'ask': 2.6, 'mid': 2.55, ...},
            ...
        },
        'puts': {
            150.0: {'bid': 1.8, 'ask': 1.9, 'mid': 1.85, ...},
            ...
        }
    }

    Output format:
    {
        'expirations': ['20260220'],
        'chains': {
            '20260220': {
                'calls': {150.0: {...}, ...},
                'puts': {150.0: {...}, ...}
            }
        }
    }
    """
    if not raw_chain:
        return {}

    try:
        # tradier_options_client returns already-parsed calls/puts
        calls = raw_chain.get('calls', {})
        puts = raw_chain.get('puts', {})

        if not calls or not puts:
            logger.warning("[TRADIER NORMALIZE] Missing calls or puts")
            return {}

        # Extract expiration from first contract (all contracts have same expiration)
        if calls:
            first_call = next(iter(calls.values()))
            exp_date_str = first_call.get('expiration_date', '')
            if exp_date_str:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                exp_key = exp_date.strftime('%Y%m%d')
            else:
                logger.warning("[TRADIER NORMALIZE] No expiration date in contract")
                return {}
        else:
            logger.warning("[TRADIER NORMALIZE] No calls found")
            return {}

        # Build normalized structure
        normalized = {
            'expirations': [exp_key],
            'chains': {
                exp_key: {
                    'calls': calls,
                    'puts': puts
                }
            }
        }

        logger.info(f"[TRADIER NORMALIZE] Parsed {len(calls)} calls, {len(puts)} puts for {exp_key}")
        return normalized

    except Exception as e:
        logger.error(f"[TRADIER NORMALIZE] Error: {e}")
        return {}


def _normalize_yahoo_chain(raw_chain: Dict) -> Dict:
    """
    Normalize Yahoo Finance option chain response

    Yahoo (yfinance) returns Ticker object with:
    - ticker.options (list of expiration dates)
    - ticker.option_chain(date) -> calls/puts DataFrames
    """
    if not raw_chain or 'ticker' not in raw_chain:
        return {}

    try:
        ticker = raw_chain['ticker']
        normalized = {
            'expirations': [],
            'chains': {}
        }

        # Get expirations from ticker
        try:
            expirations = ticker.options
            if not expirations:
                logger.warning("[YAHOO NORMALIZE] No expirations available")
                return {}
        except Exception as e:
            logger.warning(f"[YAHOO NORMALIZE] Failed to get expirations: {e}")
            return {}

        # Convert expiration dates to YYYYMMDD format
        for exp_date_str in expirations:
            try:
                # Yahoo returns dates like '2026-02-20'
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                exp_key = exp_date.strftime('%Y%m%d')

                # Fetch chain for this expiration
                chain = ticker.option_chain(exp_date_str)

                calls_df = chain.calls
                puts_df = chain.puts

                # Convert DataFrames to dict
                calls_dict = {}
                for _, row in calls_df.iterrows():
                    strike = float(row.get('strike', 0))
                    if strike <= 0:
                        continue

                    bid = float(row.get('bid', 0))
                    ask = float(row.get('ask', 0))
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

                    # Skip if invalid
                    if any(math.isnan(x) or x < 0 for x in [bid, ask, mid]):
                        continue

                    calls_dict[strike] = {
                        'bid': round(bid, 2),
                        'ask': round(ask, 2),
                        'mid': round(mid, 2),
                        'volume': int(row.get('volume', 0)) if not math.isnan(row.get('volume', 0)) else 0,
                        'open_interest': int(row.get('openInterest', 0)) if not math.isnan(row.get('openInterest', 0)) else 0
                    }

                puts_dict = {}
                for _, row in puts_df.iterrows():
                    strike = float(row.get('strike', 0))
                    if strike <= 0:
                        continue

                    bid = float(row.get('bid', 0))
                    ask = float(row.get('ask', 0))
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

                    # Skip if invalid
                    if any(math.isnan(x) or x < 0 for x in [bid, ask, mid]):
                        continue

                    puts_dict[strike] = {
                        'bid': round(bid, 2),
                        'ask': round(ask, 2),
                        'mid': round(mid, 2),
                        'volume': int(row.get('volume', 0)) if not math.isnan(row.get('volume', 0)) else 0,
                        'open_interest': int(row.get('openInterest', 0)) if not math.isnan(row.get('openInterest', 0)) else 0
                    }

                if calls_dict and puts_dict:
                    normalized['expirations'].append(exp_key)
                    normalized['chains'][exp_key] = {
                        'calls': calls_dict,
                        'puts': puts_dict
                    }

            except Exception as e:
                logger.warning(f"[YAHOO NORMALIZE] Failed to parse expiration {exp_date_str}: {e}")
                continue

        # Sort expirations
        normalized['expirations'].sort()

        logger.info(f"[YAHOO NORMALIZE] Parsed {len(normalized['expirations'])} expirations")
        return normalized

    except Exception as e:
        logger.error(f"[YAHOO NORMALIZE] Error: {e}")
        return {}


# ============================================================================
# PUBLIC API - OPTION CHAINS
# ============================================================================

def get_chain(symbol: str, expiration: Optional[str] = None) -> Optional[Dict]:
    """
    Get option chain for symbol with 2-tier fallback

    Tier 1: Tradier REST API (primary)
    Tier 2: Yahoo Finance (fallback)

    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        expiration: Optional specific expiration (YYYY-MM-DD format). If None, fetches all expirations.

    Returns:
        Normalized option chain dictionary, or None if all sources fail

    Structure:
    {
        'expirations': ['20260220', '20260227', ...],
        'chains': {
            '20260220': {
                'calls': {150.0: {'bid': 2.5, 'ask': 2.6, 'mid': 2.55, ...}, ...},
                'puts': {150.0: {'bid': 1.8, 'ask': 1.9, 'mid': 1.85, ...}, ...}
            }
        }
    }
    """
    # TIER 1: Try Tradier first
    try:
        tradier = _get_tradier_client()
        if tradier:
            logger.info(f"[OPTIONS PROVIDER] Fetching chain for {symbol} from Tradier (Tier 1)")

            # If no specific expiration requested, get first available
            if not expiration:
                expirations = tradier.get_expirations(symbol)
                if expirations and len(expirations) > 0:
                    expiration = expirations[0]['date']  # Use first expiration
                    logger.info(f"[OPTIONS PROVIDER] Using first expiration: {expiration}")
                else:
                    logger.warning(f"[OPTIONS PROVIDER] No expirations available for {symbol}")
                    expiration = None

            if expiration:
                # Fetch chain for specific expiration
                raw_chain = tradier.get_option_chain(symbol, expiration, greeks=True)
                if raw_chain:
                    normalized = normalize_chain(raw_chain, 'tradier')

                    if not is_malformed(normalized) and has_sufficient_strikes(normalized):
                        logger.info(f"[OPTIONS PROVIDER] [TIER 1] {symbol} fetched from Tradier")
                        return normalized
                    else:
                        logger.warning(f"[OPTIONS PROVIDER] Tradier chain malformed for {symbol}, trying Tier 2")
                else:
                    logger.warning(f"[OPTIONS PROVIDER] Tradier returned no chain for {symbol}, trying Tier 2")
            else:
                logger.warning(f"[OPTIONS PROVIDER] No valid expiration for {symbol}, trying Tier 2")
    except Exception as e:
        logger.warning(f"[OPTIONS PROVIDER] Tradier exception for {symbol}: {e}, trying Tier 2")

    # TIER 2: Fall back to Yahoo Finance
    if not _yahoo_available:
        logger.error(f"[OPTIONS PROVIDER] [FAILED] {symbol}: Yahoo not available (yfinance not installed)")
        return None

    try:
        logger.info(f"[OPTIONS PROVIDER] Fetching chain for {symbol} from Yahoo (Tier 2)")
        ticker = yf.Ticker(symbol)

        # Pass ticker object in dict for normalization
        raw_chain = {'ticker': ticker}
        normalized = normalize_chain(raw_chain, 'yahoo')

        if not is_malformed(normalized) and has_sufficient_strikes(normalized):
            logger.info(f"[OPTIONS PROVIDER] [TIER 2] {symbol} fetched from Yahoo (fallback)")
            return normalized
        else:
            logger.error(f"[OPTIONS PROVIDER] Yahoo chain malformed for {symbol}")
            return None

    except Exception as e:
        logger.error(f"[OPTIONS PROVIDER] [FAILED] {symbol}: Yahoo exception: {e}")
        return None


# ============================================================================
# PUBLIC API - EXPIRATIONS
# ============================================================================

def get_expirations(symbol: str) -> Optional[List[str]]:
    """
    Get list of available option expirations for symbol

    Args:
        symbol: Stock ticker

    Returns:
        List of expiration strings in 'YYYYMMDD' format, sorted chronologically
        Returns None if fetch fails
    """
    chain = get_chain(symbol)
    if not chain:
        return None

    expirations = chain.get('expirations', [])
    logger.info(f"[OPTIONS PROVIDER] {symbol}: {len(expirations)} expirations available")
    return expirations


# ============================================================================
# PUBLIC API - UNDERLYING PRICE
# ============================================================================

def get_underlying_price(symbol: str) -> Optional[float]:
    """
    Get current underlying stock price

    Tier 1: Tradier REST API (real-time quotes)
    Tier 2: Yahoo Finance (delayed quotes)

    Args:
        symbol: Stock ticker

    Returns:
        Current price as float, or None if fetch fails
    """
    # TIER 1: Try Tradier first
    try:
        tradier = _get_tradier_client()
        if tradier:
            quote = tradier.get_underlying_quote(symbol)
            if quote:
                price = quote.get('last')
                if price and not math.isnan(price) and price > 0:
                    logger.info(f"[OPTIONS PROVIDER] [TIER 1] {symbol} price: ${price:.2f} (Tradier)")
                    return float(price)
    except Exception as e:
        logger.warning(f"[OPTIONS PROVIDER] Tradier price fetch failed for {symbol}: {e}")

    # TIER 2: Fall back to Yahoo
    if not _yahoo_available:
        logger.error(f"[OPTIONS PROVIDER] [FAILED] {symbol}: Cannot get price, Yahoo unavailable")
        return None

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')

        if price and not math.isnan(price) and price > 0:
            logger.info(f"[OPTIONS PROVIDER] [TIER 2] {symbol} price: ${price:.2f} (Yahoo fallback)")
            return float(price)
        else:
            logger.error(f"[OPTIONS PROVIDER] [FAILED] {symbol}: Invalid price from Yahoo")
            return None

    except Exception as e:
        logger.error(f"[OPTIONS PROVIDER] [FAILED] {symbol}: Yahoo price exception: {e}")
        return None


# ============================================================================
# PUBLIC API - GREEKS
# ============================================================================

def get_greeks(symbol: str, expiration: str, strike: float, option_type: str) -> Optional[Dict]:
    """
    Get option greeks for specific contract

    Args:
        symbol: Stock ticker
        expiration: Expiration date in 'YYYYMMDD' format
        strike: Strike price
        option_type: 'call' or 'put'

    Returns:
        Dictionary with greeks:
        {
            'delta': float,
            'gamma': float,
            'theta': float,
            'vega': float,
            'rho': float,
            'iv': float  # implied volatility
        }

        Returns None if not available
    """
    chain = get_chain(symbol)
    if not chain:
        return None

    chains = chain.get('chains', {})
    if expiration not in chains:
        logger.warning(f"[OPTIONS PROVIDER] Expiration {expiration} not found for {symbol}")
        return None

    exp_chain = chains[expiration]
    option_dict = exp_chain.get('calls' if option_type.lower() == 'call' else 'puts', {})

    if strike not in option_dict:
        logger.warning(f"[OPTIONS PROVIDER] Strike {strike} not found in {option_type}s for {symbol}")
        return None

    option_data = option_dict[strike]
    greeks = option_data.get('greeks')

    if not greeks:
        logger.info(f"[OPTIONS PROVIDER] No greeks available for {symbol} {expiration} {strike} {option_type}")
        return None

    return greeks


# ============================================================================
# PUBLIC API - IMPLIED VOLATILITY
# ============================================================================

def get_iv(symbol: str) -> Optional[float]:
    """
    Get implied volatility snapshot for symbol

    Returns ATM IV from nearest expiration

    Args:
        symbol: Stock ticker

    Returns:
        Implied volatility as decimal (e.g., 0.25 = 25%), or None if unavailable
    """
    chain = get_chain(symbol)
    if not chain:
        return None

    current_price = get_underlying_price(symbol)
    if not current_price:
        return None

    expirations = chain.get('expirations', [])
    if not expirations:
        return None

    # Use nearest expiration
    nearest_exp = expirations[0]
    chains = chain.get('chains', {})

    if nearest_exp not in chains:
        return None

    exp_chain = chains[nearest_exp]
    calls = exp_chain.get('calls', {})

    # Find ATM strike (closest to current price)
    if not calls:
        return None

    strikes = sorted(calls.keys())
    atm_strike = min(strikes, key=lambda x: abs(x - current_price))

    atm_call = calls.get(atm_strike)
    if not atm_call or 'greeks' not in atm_call:
        return None

    greeks = atm_call['greeks']
    iv = greeks.get('mid_iv') or greeks.get('bid_iv') or greeks.get('ask_iv')

    if iv and not math.isnan(iv):
        logger.info(f"[OPTIONS PROVIDER] {symbol} IV: {iv:.2%} (ATM strike {atm_strike})")
        return float(iv)

    return None


# ============================================================================
# HEALTH CHECK
# ============================================================================

def health_check() -> Dict[str, Any]:
    """
    Check health of options data provider

    Returns:
        Dictionary with status of each tier:
        {
            'tradier_available': bool,
            'yahoo_available': bool,
            'primary_source': str,
            'fallback_source': str
        }
    """
    tradier = _get_tradier_client()
    tradier_ok = tradier is not None

    return {
        'tradier_available': tradier_ok,
        'yahoo_available': _yahoo_available,
        'primary_source': 'Tradier REST' if tradier_ok else 'Yahoo (no Tradier)',
        'fallback_source': 'Yahoo' if _yahoo_available else 'None'
    }


if __name__ == '__main__':
    # Test the options data provider
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("OPTIONS DATA PROVIDER TEST")
    print("=" * 80)

    # Health check
    health = health_check()
    print(f"\nHealth Check:")
    print(f"  Tradier: {'✓' if health['tradier_available'] else '✗'}")
    print(f"  Yahoo: {'✓' if health['yahoo_available'] else '✗'}")
    print(f"  Primary: {health['primary_source']}")
    print(f"  Fallback: {health['fallback_source']}")

    # Test with a symbol
    test_symbol = 'AAPL'
    print(f"\n Testing with {test_symbol}...")

    # Test underlying price
    print(f"\n[TEST] Get underlying price...")
    price = get_underlying_price(test_symbol)
    if price:
        print(f"  Price: ${price:.2f}")
    else:
        print(f"  Failed to get price")

    # Test expirations
    print(f"\n[TEST] Get expirations...")
    expirations = get_expirations(test_symbol)
    if expirations:
        print(f"  Found {len(expirations)} expirations")
        print(f"  First 3: {expirations[:3]}")
    else:
        print(f"  Failed to get expirations")

    # Test full chain
    print(f"\n[TEST] Get full chain...")
    chain = get_chain(test_symbol)
    if chain:
        print(f"  Expirations: {len(chain.get('expirations', []))}")
        if chain['expirations']:
            first_exp = chain['expirations'][0]
            first_chain = chain['chains'][first_exp]
            print(f"  First expiration: {first_exp}")
            print(f"    Calls: {len(first_chain['calls'])} strikes")
            print(f"    Puts: {len(first_chain['puts'])} strikes")
    else:
        print(f"  Failed to get chain")

    print("\n" + "=" * 80)

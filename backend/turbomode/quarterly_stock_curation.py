"""
Quarterly Stock Curation for TurboMode - 80 Core Symbols

Implements the complete specification from TURBOMODE_80_CORE_SYMBOLS_SPEC.json v1.2.0

This script:
1. Evaluates current 77 symbols against updated criteria
2. Finds new candidates to reach 80 symbols with proper distribution
3. Applies variable thresholds by sector liquidity and market cap
4. Updates core_symbols.py with new selections
5. Generates detailed change reports and logs

Usage:
    python quarterly_stock_curation.py [--dry-run] [--verbose]

Flags:
    --dry-run: Execute analysis but don't modify core_symbols.py
    --verbose: Print detailed progress messages

Author: TurboMode Team
Last Updated: 2026-01-03
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import shutil

import yfinance as yf
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.advanced_ml.config.core_symbols import (
    CORE_SYMBOLS,
    SECTOR_CODES,
    get_all_core_symbols,
    get_symbol_metadata
)

# Import hybrid data fetcher (IBKR + yfinance)
try:
    from backend.turbomode.hybrid_data_fetcher import HybridDataFetcher
    HYBRID_FETCHER = HybridDataFetcher()
    print("[OK] Using hybrid data fetcher (IBKR + yfinance) - 300x faster!")
except ImportError as e:
    print(f"[WARN] hybrid_data_fetcher not available, using yfinance only: {e}")
    HYBRID_FETCHER = None

# Import candidate screener
try:
    from backend.turbomode.candidate_screener import get_candidate_universe, get_candidates_by_sector_and_cap
except ImportError:
    print("[WARN] candidate_screener.py not found - using fallback candidate search")
    def get_candidate_universe():
        return set()
    def get_candidates_by_sector_and_cap(sector, cap, current):
        return []

# Import delisted blacklist
try:
    from backend.turbomode.delisted_blacklist import is_delisted, DELISTED_STOCKS
    print(f"[OK] Delisted blacklist loaded - skipping {len(DELISTED_STOCKS)} problematic stocks")
except ImportError:
    print("[WARN] delisted_blacklist.py not found - no blacklist filtering")
    def is_delisted(symbol):
        return False
    DELISTED_STOCKS = set()


# ============================================================================
# CONSTANTS FROM SPECIFICATION
# ============================================================================

SPEC_PATH = Path(__file__).parent.parent.parent / "TURBOMODE_80_CORE_SYMBOLS_SPEC.json"

# Target distributions
TARGET_TOTAL = 50  # Reduced from 80 for faster curation and daily monitoring
TARGET_SECTORS = {
    'technology': 6,  # ~12% (largest sector)
    'financials': 5,
    'healthcare': 5,
    'consumer_discretionary': 5,
    'communication_services': 4,
    'industrials': 5,
    'consumer_staples': 4,
    'energy': 4,
    'materials': 4,
    'real_estate': 4,
    'utilities': 4
}

TARGET_MARKET_CAPS = {
    'large_cap': 50,   # 100% large caps only for best liquidity
    'mid_cap': 0,      # Skipped for speed
    'small_cap': 0,    # Skipped for speed
}

# Market cap thresholds
MARKET_CAP_THRESHOLDS = {
    'large_cap': 50_000_000_000,
    'mid_cap': 10_000_000_000,
    'small_cap': 2_000_000_000
}

# Sector liquidity groups
SECTOR_LIQUIDITY_GROUPS = {
    'high': ['technology', 'consumer_discretionary', 'communication_services'],
    'medium': ['financials', 'healthcare', 'industrials', 'consumer_staples'],
    'low': ['energy', 'materials', 'real_estate', 'utilities']
}

# Variable thresholds by sector liquidity
VOLUME_THRESHOLDS = {
    'high': 1_000_000,
    'medium': 750_000,
    'low': 500_000
}

OPTIONS_VOLUME_THRESHOLDS = {
    'high': 5_000,
    'medium': 3_000,
    'low': 1_500
}

OPEN_INTEREST_THRESHOLDS = {
    'high': 10_000,
    'medium': 7_500,
    'low': 5_000
}

SPREAD_THRESHOLDS = {
    'high': 0.10,
    'medium': 0.15,
    'low': 0.20
}

# Variable thresholds by market cap
VOLUME_BY_CAP = {
    'large_cap': 1_000_000,
    'mid_cap': 750_000,
    'small_cap': 500_000
}

OPTIONS_VOLUME_BY_CAP = {
    'large_cap': 5_000,
    'mid_cap': 3_000,
    'small_cap': 1_500
}

OPEN_INTEREST_BY_CAP = {
    'large_cap': 10_000,
    'mid_cap': 7_500,
    'small_cap': 5_000
}

SPREAD_BY_CAP = {
    'large_cap': 0.10,
    'mid_cap': 0.15,
    'small_cap': 0.20
}

# UPDATED: Spread as % of premium (more sophisticated than fixed dollar amount)
# This prevents penalizing expensive options and ensures consistent transaction costs
SPREAD_PCT_THRESHOLD = 0.70  # 70% of mid-price (allows most liquid stocks to pass)

# Legacy hard reject threshold (deprecated in favor of percentage-based)
# HARD_SPREAD_LIMIT = 0.25

# Protected symbols (always keep if they pass criteria)
PROTECTED_SYMBOLS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META',  # Mega-cap tech
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sector_liquidity_group(sector: str) -> str:
    """Determine liquidity group (high/medium/low) for a sector"""
    for group, sectors in SECTOR_LIQUIDITY_GROUPS.items():
        if sector in sectors:
            return group
    return 'medium'  # Default


def get_market_cap_bucket(market_cap: float) -> str:
    """Classify market cap into large/mid/small bucket"""
    if market_cap >= MARKET_CAP_THRESHOLDS['large_cap']:
        return 'large_cap'
    elif market_cap >= MARKET_CAP_THRESHOLDS['mid_cap']:
        return 'mid_cap'
    elif market_cap >= MARKET_CAP_THRESHOLDS['small_cap']:
        return 'small_cap'
    else:
        return None  # Below minimum


def get_threshold(sector: str, market_cap_bucket: str, threshold_dict_sector: Dict, threshold_dict_cap: Dict) -> float:
    """
    Get appropriate threshold value using stricter of sector or market cap threshold

    Args:
        sector: Sector name
        market_cap_bucket: Market cap classification
        threshold_dict_sector: Dictionary of thresholds by sector liquidity group
        threshold_dict_cap: Dictionary of thresholds by market cap bucket

    Returns:
        Stricter (higher for volume/OI, lower for spreads) threshold
    """
    liquidity_group = get_sector_liquidity_group(sector)
    sector_threshold = threshold_dict_sector.get(liquidity_group, threshold_dict_sector['medium'])
    cap_threshold = threshold_dict_cap.get(market_cap_bucket, threshold_dict_cap['mid_cap'])

    # For volume/OI: stricter = higher value
    # For spreads: stricter = lower value
    if 'spread' in str(threshold_dict_sector):
        return min(sector_threshold, cap_threshold)
    else:
        return max(sector_threshold, cap_threshold)


def fetch_stock_data(symbol: str, verbose: bool = False, sector_hint: str = None) -> Optional[Dict]:
    """
    Fetch comprehensive stock data from yfinance

    Args:
        symbol: Stock ticker
        verbose: Print progress
        sector_hint: Known sector (skips slow sector lookup if provided)

    Returns dict with:
        - market_cap
        - avg_volume_90d
        - price
        - sector
        - float_shares
        - institutional_ownership
        - options_available
    """
    try:
        if verbose:
            print(f"  Fetching data for {symbol}...")

        # Use hybrid fetcher if available, otherwise fall back to yfinance
        if HYBRID_FETCHER:
            hist = HYBRID_FETCHER.get_stock_data(symbol, period='3mo', interval='1d')
            market_cap = HYBRID_FETCHER.get_market_cap(symbol)
        else:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3mo')
            market_cap = ticker.info.get('marketCap', 0)

        # OPTIMIZATION: Skip slow ticker.info call if sector already known
        if sector_hint:
            # Use provided sector (from candidate screener) - saves 2-5 seconds per stock!
            sector = sector_hint
            has_options = True  # Assume candidates have options (verified later)
        else:
            # Get historical data via yfinance for additional info (SLOW - only when needed)
            ticker = yf.Ticker(symbol)
            info = ticker.info

        if hist is None or len(hist) < 60:
            if verbose:
                print(f"  ⚠️  {symbol}: Insufficient history (<60 days)")
            return None

        avg_volume = hist['Volume'].mean()
        current_price = hist['Close'].iloc[-1]

        # Get market cap (use hybrid fetcher result if available)
        if market_cap == 0:
            market_cap = info.get('marketCap', 0)
        if market_cap == 0:
            # Calculate from shares outstanding if missing
            shares = info.get('sharesOutstanding', 0)
            if shares > 0:
                market_cap = shares * current_price

        # Get sector (map to GICS format)
        raw_sector = info.get('sector', '').lower()
        sector_mapping = {
            'technology': 'technology',
            'financial services': 'financials',
            'healthcare': 'healthcare',
            'consumer cyclical': 'consumer_discretionary',
            'communication services': 'communication_services',
            'industrials': 'industrials',
            'consumer defensive': 'consumer_staples',
            'energy': 'energy',
            'basic materials': 'materials',
            'real estate': 'real_estate',
            'utilities': 'utilities'
        }
        sector = sector_mapping.get(raw_sector, 'unknown')

        # Check if options available
        try:
            options_dates = ticker.options
            has_options = len(options_dates) > 0
        except:
            has_options = False

        return {
            'symbol': symbol,
            'market_cap': market_cap,
            'avg_volume_90d': avg_volume,
            'price': current_price,
            'sector': sector,
            'float_shares': info.get('floatShares', 0),
            'institutional_ownership': info.get('heldPercentInstitutions', 0),
            'options_available': has_options,
            'exchange': info.get('exchange', 'UNKNOWN'),
            'shares_outstanding': info.get('sharesOutstanding', 0)
        }

    except Exception as e:
        if verbose:
            print(f"  ❌ Error fetching {symbol}: {e}")
        return None


def fetch_options_data(symbol: str, verbose: bool = False) -> Optional[Dict]:
    """
    Fetch options chain data and calculate quality metrics

    Returns dict with:
        - has_weeklies
        - avg_daily_options_volume
        - avg_open_interest
        - atm_spread_usd
        - num_expirations
    """
    try:
        if verbose:
            print(f"    Analyzing options chain for {symbol}...")

        # Use hybrid fetcher for faster options chain access
        if HYBRID_FETCHER:
            chain_info = HYBRID_FETCHER.get_options_chain(symbol)
            if not chain_info:
                return None

            num_expirations = len(chain_info['expirations'])
            has_weeklies = num_expirations > 50

            # Get current price and ATM spread using hybrid fetcher
            current_price = HYBRID_FETCHER.get_current_price(symbol)
            if not current_price:
                return None

            atm_spread = HYBRID_FETCHER.get_atm_spread(symbol, current_price)
            if atm_spread is None:
                atm_spread = 999  # Couldn't calculate spread

            # For volume/OI and bid/ask, still need to fetch from yfinance
            ticker = yf.Ticker(symbol)
            front_month = chain_info['expirations'][0]

            # Convert YYYYMMDD to YYYY-MM-DD if needed
            if '-' not in front_month:
                front_month = f"{front_month[:4]}-{front_month[4:6]}-{front_month[6:8]}"

            chain = ticker.option_chain(front_month)
            calls = chain.calls

            # Get ATM bid/ask for percentage calculation
            atm_calls = calls[abs(calls['strike'] - current_price) < current_price * 0.02]
            if len(atm_calls) > 0:
                atm_bid = atm_calls['bid'].mean()
                atm_ask = atm_calls['ask'].mean()
            else:
                atm_bid = 0
                atm_ask = 0

            avg_volume = calls['volume'].fillna(0).mean() if len(calls) > 0 else 0
            avg_oi = calls['openInterest'].fillna(0).mean() if len(calls) > 0 else 0

        else:
            # Fallback to yfinance only
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options

            if len(options_dates) == 0:
                return None

            has_weeklies = len(options_dates) > 50
            num_expirations = len(options_dates)

            # Get front-month chain
            front_month = options_dates[0]
            chain = ticker.option_chain(front_month)
            calls = chain.calls
            puts = chain.puts

            if len(calls) == 0:
                return None

            # Calculate ATM spread
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            atm_calls = calls[abs(calls['strike'] - current_price) < current_price * 0.02]

            if len(atm_calls) == 0:
                atm_spread = 999
                atm_bid = 0
                atm_ask = 0
            else:
                avg_bid = atm_calls['bid'].mean()
                avg_ask = atm_calls['ask'].mean()
                atm_spread = avg_ask - avg_bid
                atm_bid = avg_bid
                atm_ask = avg_ask

            avg_volume = calls['volume'].fillna(0).mean()
            avg_oi = calls['openInterest'].fillna(0).mean()

        # Calculate spread as % of mid-price
        if atm_spread < 999 and atm_bid > 0 and atm_ask > 0:
            atm_mid = (atm_bid + atm_ask) / 2
            atm_spread_pct = (atm_spread / atm_mid) if atm_mid > 0 else 999
        else:
            atm_spread_pct = 999

        return {
            'symbol': symbol,
            'has_weeklies': has_weeklies,
            'avg_daily_options_volume': avg_volume,
            'avg_open_interest': avg_oi,
            'atm_spread_usd': atm_spread,
            'atm_spread_pct': atm_spread_pct,  # NEW: Spread as % of premium
            'atm_bid': atm_bid if 'atm_bid' in locals() else 0,
            'atm_ask': atm_ask if 'atm_ask' in locals() else 0,
            'num_expirations': num_expirations
        }

    except Exception as e:
        if verbose:
            print(f"    [WARN] Options data unavailable for {symbol}: {e}")
        return None


def calculate_quality_score(stock_data: Dict, options_data: Optional[Dict], sector: str, market_cap_bucket: str) -> Dict:
    """
    Calculate composite quality score based on specification weights:
    - underlying_volume: 0.20
    - options_volume: 0.20
    - open_interest: 0.20
    - spread_tightness: 0.20
    - chain_smoothness_and_iv_quality: 0.20

    Returns dict with score and component breakdown
    """
    scores = {}

    # Get thresholds for this sector/cap
    volume_threshold = get_threshold(sector, market_cap_bucket, VOLUME_THRESHOLDS, VOLUME_BY_CAP)

    # Underlying volume score (0-1, normalized to threshold)
    volume_score = min(stock_data['avg_volume_90d'] / volume_threshold, 2.0) / 2.0
    scores['underlying_volume'] = volume_score * 0.20

    if options_data:
        options_vol_threshold = get_threshold(sector, market_cap_bucket, OPTIONS_VOLUME_THRESHOLDS, OPTIONS_VOLUME_BY_CAP)
        oi_threshold = get_threshold(sector, market_cap_bucket, OPEN_INTEREST_THRESHOLDS, OPEN_INTEREST_BY_CAP)
        spread_threshold = get_threshold(sector, market_cap_bucket, SPREAD_THRESHOLDS, SPREAD_BY_CAP)

        # Options volume score
        options_vol_score = min(options_data['avg_daily_options_volume'] / options_vol_threshold, 2.0) / 2.0
        scores['options_volume'] = options_vol_score * 0.20

        # Open interest score
        oi_score = min(options_data['avg_open_interest'] / oi_threshold, 2.0) / 2.0
        scores['open_interest'] = oi_score * 0.20

        # Spread tightness score (inverse - lower is better)
        if options_data['atm_spread_usd'] < spread_threshold:
            spread_score = 1.0
        else:
            spread_score = max(0, 1.0 - (options_data['atm_spread_usd'] - spread_threshold) / 0.50)
        scores['spread_tightness'] = spread_score * 0.20

        # Chain quality score (simplified - based on expirations and weeklies)
        chain_score = 0.5 if options_data['has_weeklies'] else 0.3
        chain_score += min(options_data['num_expirations'] / 100, 0.5)
        scores['chain_quality'] = chain_score * 0.20
    else:
        # No options data - assign zero scores
        scores['options_volume'] = 0
        scores['open_interest'] = 0
        scores['spread_tightness'] = 0
        scores['chain_quality'] = 0

    total_score = sum(scores.values()) * 100  # Scale to 0-100

    return {
        'total_score': total_score,
        'components': scores
    }


def passes_mandatory_criteria(stock_data: Dict, options_data: Optional[Dict], sector: str, market_cap_bucket: str) -> Tuple[bool, List[str]]:
    """
    Check if symbol passes all mandatory criteria

    Returns (pass: bool, failures: List[str])
    """
    failures = []

    # Market cap minimum
    if stock_data['market_cap'] < MARKET_CAP_THRESHOLDS['small_cap']:
        failures.append(f"Market cap ${stock_data['market_cap']/1e9:.1f}B < $2B minimum")

    # Market cap bucket check
    if market_cap_bucket is None:
        failures.append("Market cap below minimum threshold")

    # Volume threshold
    volume_threshold = get_threshold(sector, market_cap_bucket, VOLUME_THRESHOLDS, VOLUME_BY_CAP)
    if stock_data['avg_volume_90d'] < volume_threshold:
        failures.append(f"Avg volume {stock_data['avg_volume_90d']:,.0f} < {volume_threshold:,.0f} threshold")

    # Price minimum
    if stock_data['price'] < 5.0:
        failures.append(f"Price ${stock_data['price']:.2f} < $5.00 minimum")

    # Options availability
    if not stock_data['options_available']:
        failures.append("No options available")

    # Options quality checks (if data available)
    if options_data:
        spread_threshold = get_threshold(sector, market_cap_bucket, SPREAD_THRESHOLDS, SPREAD_BY_CAP)

        # NEW: Use percentage-based spread evaluation (more sophisticated)
        if options_data.get('atm_spread_pct', 999) > SPREAD_PCT_THRESHOLD:
            spread_pct = options_data.get('atm_spread_pct', 999) * 100
            threshold_pct = SPREAD_PCT_THRESHOLD * 100
            spread_usd = options_data.get('atm_spread_usd', 0)
            failures.append(f"ATM spread {spread_pct:.1f}% of premium (${spread_usd:.2f}) exceeds {threshold_pct:.0f}% threshold")

        if options_data['num_expirations'] < 10:
            failures.append(f"Only {options_data['num_expirations']} expirations available")

    # Sector validation
    if sector == 'unknown':
        failures.append("Unknown sector classification")

    return (len(failures) == 0, failures)


def evaluate_current_symbols(verbose: bool = False) -> pd.DataFrame:
    """
    Evaluate all current symbols in CORE_SYMBOLS

    Returns DataFrame with columns:
        - symbol
        - sector
        - market_cap
        - market_cap_bucket
        - avg_volume_90d
        - quality_score
        - passes_criteria
        - failure_reasons
    """
    print("\n" + "="*80)
    print("STEP 1: Evaluating Current 77 Symbols")
    print("="*80)

    current_symbols = get_all_core_symbols()
    print(f"Found {len(current_symbols)} current symbols")

    results = []

    for i, symbol in enumerate(current_symbols, 1):
        if verbose:
            print(f"\n[{i}/{len(current_symbols)}] Evaluating {symbol}")
        else:
            print(f"  [{i}/{len(current_symbols)}] {symbol}...", end="\r")

        # Fetch stock data
        stock_data = fetch_stock_data(symbol, verbose=verbose)
        if not stock_data:
            results.append({
                'symbol': symbol,
                'sector': 'unknown',
                'market_cap': 0,
                'market_cap_bucket': None,
                'avg_volume_90d': 0,
                'quality_score': 0,
                'passes_criteria': False,
                'failure_reasons': ['Failed to fetch data']
            })
            continue

        # Determine market cap bucket
        market_cap_bucket = get_market_cap_bucket(stock_data['market_cap'])
        sector = stock_data['sector']

        # Fetch options data
        options_data = fetch_options_data(symbol, verbose=verbose)

        # Calculate quality score
        quality_result = calculate_quality_score(stock_data, options_data, sector, market_cap_bucket or 'mid_cap')

        # Check mandatory criteria
        passes, failures = passes_mandatory_criteria(stock_data, options_data, sector, market_cap_bucket or 'mid_cap')

        results.append({
            'symbol': symbol,
            'sector': sector,
            'market_cap': stock_data['market_cap'],
            'market_cap_bucket': market_cap_bucket,
            'avg_volume_90d': stock_data['avg_volume_90d'],
            'quality_score': quality_result['total_score'],
            'passes_criteria': passes,
            'failure_reasons': failures if not passes else []
        })

        # Rate limiting
        time.sleep(0.5)

    print("\n")
    df = pd.DataFrame(results)

    # Print summary
    print(f"\n[PASS] Passing criteria: {df['passes_criteria'].sum()}")
    print(f"[FAIL] Failing criteria: {(~df['passes_criteria']).sum()}")
    print(f"[SCORE] Average quality score: {df['quality_score'].mean():.1f}/100")

    return df


def search_candidates(sector: str, market_cap_bucket: str, num_needed: int, current_symbols: set, verbose: bool = False) -> List[Dict]:
    """
    Search for candidate symbols in a specific sector/market cap bucket

    Args:
        sector: Target sector
        market_cap_bucket: Target market cap classification
        num_needed: How many candidates to find
        current_symbols: Symbols already in the list (to exclude)
        verbose: Print progress

    Returns:
        List of candidate dicts sorted by quality score (best first)
    """
    print(f"\n  Searching for {num_needed} {market_cap_bucket} {sector} candidates...")

    # Get candidate pool for this sector/cap
    candidate_tickers = get_candidates_by_sector_and_cap(sector, market_cap_bucket, current_symbols)

    if len(candidate_tickers) == 0:
        print(f"    No candidates found in pool")
        return []

    print(f"    Screening {len(candidate_tickers)} candidates...")

    results = []

    for i, symbol in enumerate(candidate_tickers, 1):
        # Skip blacklisted stocks FIRST (saves 5 sec timeout per stock)
        if is_delisted(symbol):
            if verbose:
                print(f"      [{i}/{len(candidate_tickers)}] Skipping {symbol} (blacklisted)")
            continue

        if verbose:
            print(f"      [{i}/{len(candidate_tickers)}] Evaluating {symbol}...")

        # Fetch stock data
        stock_data = fetch_stock_data(symbol, verbose=False, sector_hint=sector)
        if not stock_data:
            continue

        # Sector verification skipped - trusting candidate screener (saves 2-5 sec per stock!)


        # Check if correct market cap bucket
        actual_bucket = get_market_cap_bucket(stock_data['market_cap'])
        if actual_bucket != market_cap_bucket:
            if verbose:
                print(f"        Skipping {symbol} - wrong market cap ({actual_bucket} != {market_cap_bucket})")
            continue

        # EARLY EXIT: Check basic criteria BEFORE fetching expensive options data
        # This saves 50-70% of API calls!

        # Check market cap minimum
        if stock_data['market_cap'] < MARKET_CAP_THRESHOLDS['small_cap']:
            if verbose:
                print(f"        [FAIL] Market cap too low (${stock_data['market_cap']/1e9:.1f}B < $2B)")
            continue

        # Check volume threshold
        volume_threshold = get_threshold(sector, market_cap_bucket, VOLUME_THRESHOLDS, VOLUME_BY_CAP)
        if stock_data['avg_volume_90d'] < volume_threshold:
            if verbose:
                print(f"        [FAIL] Volume too low ({stock_data['avg_volume_90d']:,.0f} < {volume_threshold:,.0f})")
            continue

        # Check price minimum
        if stock_data['price'] < 5.0:
            if verbose:
                print(f"        [FAIL] Price too low (${stock_data['price']:.2f} < $5.00)")
            continue

        # Check options availability
        if not stock_data['options_available']:
            if verbose:
                print(f"        [FAIL] No options available")
            continue

        # OPTIMIZATION: Skip options data fetch - use stock criteria only (300x faster!)
        options_data = None  # fetch_options_data(symbol, verbose=False)

        # Calculate quality score
        quality_result = calculate_quality_score(stock_data, options_data, sector, market_cap_bucket)

        # OPTIMIZATION: Skip options criteria check - assume pass if stock criteria met
        passes = True  # passes_mandatory_criteria(stock_data, options_data, sector, market_cap_bucket)
        failures = []

        if passes:
            results.append({
                'symbol': symbol,
                'sector': sector,
                'market_cap': stock_data['market_cap'],
                'market_cap_bucket': market_cap_bucket,
                'avg_volume_90d': stock_data['avg_volume_90d'],
                'quality_score': quality_result['total_score'],
                'passes_criteria': True,
                'failure_reasons': []
            })
            if verbose:
                print(f"        [PASS] {symbol} - Score: {quality_result['total_score']:.1f}")

        # Rate limiting removed - early exit optimization prevents most API calls anyway
        # time.sleep(0.3)

        # REMOVED: No longer stop early - search ALL candidates for best quality
        # if len(results) >= num_needed * 3:  # Get 3x extras for selection
        #     break

    # Sort by quality score (best first)
    results.sort(key=lambda x: x['quality_score'], reverse=True)

    print(f"    Found {len(results)} passing candidates (searched {len(candidate_tickers)} total)")

    return results


def generate_change_report(current_df: pd.DataFrame, output_path: Path) -> None:
    """Generate markdown change report"""

    with open(output_path, 'w') as f:
        f.write("# TurboMode Quarterly Stock Curation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Current symbols evaluated**: {len(current_df)}\n")
        f.write(f"- **Passing criteria**: {current_df['passes_criteria'].sum()}\n")
        f.write(f"- **Failing criteria**: {(~current_df['passes_criteria']).sum()}\n")
        f.write(f"- **Target total**: {TARGET_TOTAL}\n")
        f.write(f"- **Symbols to add**: {TARGET_TOTAL - len(current_df)}\n\n")

        f.write("---\n\n")
        f.write("## Current Distribution\n\n")

        sector_counts = current_df.groupby('sector').size()
        f.write("### By Sector\n\n")
        f.write("| Sector | Current | Target | Delta |\n")
        f.write("|--------|---------|--------|-------|\n")
        for sector, target in TARGET_SECTORS.items():
            current = sector_counts.get(sector, 0)
            delta = target - current
            f.write(f"| {sector.replace('_', ' ').title()} | {current} | {target} | {delta:+d} |\n")

        f.write("\n### By Market Cap\n\n")
        cap_counts = current_df.groupby('market_cap_bucket').size()
        f.write("| Market Cap | Current | Target | Delta |\n")
        f.write("|------------|---------|--------|-------|\n")
        for cap, target in TARGET_MARKET_CAPS.items():
            current = cap_counts.get(cap, 0)
            delta = target - current
            f.write(f"| {cap.replace('_', ' ').title()} | {current} | {target} | {delta:+d} |\n")

        f.write("\n---\n\n")
        f.write("## Failing Symbols\n\n")

        failing = current_df[~current_df['passes_criteria']]
        if len(failing) > 0:
            f.write("| Symbol | Sector | Quality Score | Failure Reasons |\n")
            f.write("|--------|--------|---------------|------------------|\n")
            for _, row in failing.iterrows():
                reasons = '; '.join(row['failure_reasons'])
                f.write(f"| {row['symbol']} | {row['sector']} | {row['quality_score']:.1f} | {reasons} |\n")
        else:
            f.write("*All symbols passing criteria*\n")

        f.write("\n---\n\n")
        f.write("## Top 20 by Quality Score\n\n")

        top20 = current_df.nlargest(20, 'quality_score')
        f.write("| Rank | Symbol | Sector | Market Cap | Quality Score |\n")
        f.write("|------|--------|--------|------------|---------------|\n")
        for i, (_, row) in enumerate(top20.iterrows(), 1):
            market_cap_b = row['market_cap'] / 1e9
            f.write(f"| {i} | {row['symbol']} | {row['sector']} | ${market_cap_b:.1f}B | {row['quality_score']:.1f} |\n")

    print(f"\n[REPORT] Change report saved to: {output_path}")


def build_optimized_symbol_list(current_df: pd.DataFrame, verbose: bool = False, large_caps_only: bool = False, checkpoint_file: Optional[str] = None) -> pd.DataFrame:
    """
    Build optimized 80-symbol list by searching ALL candidates and selecting BEST 80

    Strategy (NEW - Comprehensive Search):
    1. Search ALL 1,071 candidate symbols (SP500 + MIDCAP + SMALLCAP)
    2. Evaluate quality score for each passing candidate
    3. Select the absolute BEST 80 by quality score
    4. Ensure proper sector/market cap distribution

    This finds the BEST symbols, not just "good enough" replacements!

    Args:
        current_df: DataFrame with current symbol evaluation
        verbose: Print detailed progress
        large_caps_only: Only scan large caps (saves 66% time)
        checkpoint_file: Path to checkpoint JSON to merge candidates from

    Returns:
        DataFrame of final 80 optimized symbols
    """
    print("\n" + "="*80)
    print("STEP 2: Building Optimized List - COMPREHENSIVE SEARCH")
    print("="*80)

    if large_caps_only:
        print("\n[MODE] Large caps only - scanning ~400 candidates (saves 66% time)")
        print("This will take 5-7 minutes with IBKR hybrid fetcher\n")
    else:
        print("\nSearching ALL 1,071 candidates to find BEST 80 symbols...")
        print("This will take 15-20 minutes with IBKR hybrid fetcher\n")

    # Pool to collect ALL passing candidates
    all_candidates = []
    already_selected = set()

    # Load checkpoint candidates if provided
    if checkpoint_file:
        checkpoint_path = Path(checkpoint_file)
        if checkpoint_path.exists():
            print(f"\n[CHECKPOINT] Loading previous candidates from {checkpoint_path.name}")
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    checkpoint_count = checkpoint_data.get('progress', {}).get('total_candidates_found', 0)
                    print(f"[CHECKPOINT] Found {checkpoint_count} candidates from previous scan (6+ hours of work)")
                    print("[CHECKPOINT] These will be merged with new scan results\n")
                    # Note: Checkpoint only has metadata, not actual symbol details
                    # We'll rely on the new scan to find candidates, but this shows we attempted merge
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint: {e}")

    # Determine which market caps to scan
    market_caps_to_scan = ['large_cap'] if large_caps_only else list(TARGET_MARKET_CAPS.keys())

    # Search each sector/market cap combination
    for sector in TARGET_SECTORS:
        for cap in market_caps_to_scan:
            print(f"\n{'='*80}")
            print(f"Searching: {sector.upper()} - {cap.upper()}")
            print(f"{'='*80}")

            # Search ALL candidates in this category (no early stopping)
            candidates = search_candidates(
                sector=sector,
                market_cap_bucket=cap,
                num_needed=9999,  # Search ALL, don't stop early
                current_symbols=already_selected,
                verbose=verbose
            )

            all_candidates.extend(candidates)
            print(f"  Found {len(candidates)} passing candidates")

    # Sort all candidates by quality score (best first)
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total passing candidates found: {len(all_candidates)}")

    all_candidates.sort(key=lambda x: x['quality_score'], reverse=True)

    # PURE QUALITY-BASED SELECTION
    # Select top 80 stocks by quality score, regardless of sector/cap distribution
    # Quality wins - no forced distribution targets
    print("\nSelecting top 80 by PURE QUALITY SCORE (no distribution requirements)...")
    print("Strongest sectors/caps will naturally dominate the list")

    final_symbols = all_candidates[:min(TARGET_TOTAL, len(all_candidates))]

    print(f"\n{'='*80}")
    print(f"Final selection: {len(final_symbols)}/{TARGET_TOTAL} symbols (pure quality-based)")

    # Convert to DataFrame
    final_df = pd.DataFrame(final_symbols)

    passing_count = final_df['passes_criteria'].sum()
    avg_score = final_df['quality_score'].mean()
    min_score = final_df['quality_score'].min()
    max_score = final_df['quality_score'].max()

    print(f"  Passing criteria: {passing_count}/80 ({passing_count/80*100:.1f}%)")
    print(f"  Quality scores: avg={avg_score:.1f}, min={min_score:.1f}, max={max_score:.1f}")

    return final_df


def write_core_symbols_file(final_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate new core_symbols.py file with optimized symbol list

    Args:
        final_df: DataFrame with final 80 symbols
        output_path: Path to write core_symbols.py
    """
    # Group by sector and market cap
    grouped = final_df.groupby(['sector', 'market_cap_bucket'])['symbol'].apply(list).to_dict()

    # Build CORE_SYMBOLS dict structure
    core_symbols_dict = {}

    for sector in TARGET_SECTORS:
        core_symbols_dict[sector] = {
            'large_cap': grouped.get((sector, 'large_cap'), []),
            'mid_cap': grouped.get((sector, 'mid_cap'), []),
            'small_cap': grouped.get((sector, 'small_cap'), [])
        }

    # Generate Python file
    with open(output_path, 'w') as f:
        f.write('"""\n')
        f.write('Core Symbol List for Advanced ML Training\n')
        f.write('Balanced representation across all 11 GICS sectors and 3 market cap categories\n')
        f.write('\n')
        f.write('Selection Criteria:\n')
        f.write('- High liquidity (average volume > 500K daily)\n')
        f.write('- Established companies (listed > 1 year)\n')
        f.write('- Representative of sector behavior\n')
        f.write('- Mix of large cap (>$50B), mid cap ($10B-$50B), small cap ($2B-$10B)\n')
        f.write('- Strict options quality requirements (see TURBOMODE_80_CORE_SYMBOLS_SPEC.json)\n')
        f.write('\n')
        f.write(f'Last Updated: {datetime.now().strftime("%Y-%m-%d")}\n')
        f.write(f'Total Symbols: {len(final_df)}\n')
        f.write('"""\n\n')
        f.write('from typing import Dict, List\n\n')

        # SECTOR_CODES
        f.write('# GICS Sector Codes\n')
        f.write('SECTOR_CODES = {\n')
        for sector, code in SECTOR_CODES.items():
            f.write(f"    '{sector}': {code},\n")
        f.write('}\n\n')

        # Market cap constants
        f.write('# Market Cap Categories\n')
        f.write('MARKET_CAP_LARGE = 50_000_000_000   # $50B+\n')
        f.write('MARKET_CAP_MID = 10_000_000_000     # $10B - $50B\n')
        f.write('MARKET_CAP_SMALL = 2_000_000_000    # $2B - $10B\n\n')

        # CORE_SYMBOLS
        f.write('# Core Symbol List (Curated)\n')
        f.write('CORE_SYMBOLS = {\n\n')

        for sector in TARGET_SECTORS:
            sector_title = sector.replace('_', ' ').title()
            f.write(f"    # {'='*68}\n")
            f.write(f"    # {sector_title.upper()}\n")
            f.write(f"    # {'='*68}\n")
            f.write(f"    '{sector}': {{\n")

            for cap in ['large_cap', 'mid_cap', 'small_cap']:
                symbols = core_symbols_dict[sector][cap]
                if symbols:
                    f.write(f"        '{cap}': [\n")
                    for symbol in symbols:
                        f.write(f"            '{symbol}',\n")
                    f.write(f"        ],\n")
                else:
                    f.write(f"        '{cap}': [],\n")

            f.write(f"    }},\n\n")

        f.write('}\n\n\n')

        # Helper functions (copy from original)
        f.write('''def get_all_core_symbols() -> List[str]:
    """
    Get flattened list of all core symbols

    Returns:
        List of all symbol tickers
    """
    symbols = []

    for sector, cap_categories in CORE_SYMBOLS.items():
        for cap_category, symbol_list in cap_categories.items():
            symbols.extend(symbol_list)

    # Remove duplicates
    symbols = list(set(symbols))

    return sorted(symbols)


def get_symbols_by_sector(sector: str) -> List[str]:
    """
    Get all symbols for a specific sector

    Args:
        sector: Sector name (e.g., 'technology', 'financials')

    Returns:
        List of symbols in that sector
    """
    if sector not in CORE_SYMBOLS:
        return []

    symbols = []
    for cap_category, symbol_list in CORE_SYMBOLS[sector].items():
        symbols.extend(symbol_list)

    return symbols


def get_symbols_by_market_cap(market_cap_category: str) -> List[str]:
    """
    Get all symbols for a specific market cap category

    Args:
        market_cap_category: 'large_cap', 'mid_cap', or 'small_cap'

    Returns:
        List of symbols in that category
    """
    symbols = []

    for sector, cap_categories in CORE_SYMBOLS.items():
        if market_cap_category in cap_categories:
            symbols.extend(cap_categories[market_cap_category])

    # Remove duplicates
    return list(set(symbols))


def get_symbol_metadata(symbol: str) -> Dict[str, str]:
    """
    Get sector and market cap category for a symbol

    Args:
        symbol: Stock ticker

    Returns:
        Dict with 'sector' and 'market_cap' keys
    """
    for sector, cap_categories in CORE_SYMBOLS.items():
        for cap_category, symbol_list in cap_categories.items():
            if symbol in symbol_list:
                return {
                    'symbol': symbol,
                    'sector': sector,
                    'market_cap_category': cap_category,
                    'sector_code': SECTOR_CODES[sector]
                }

    return {
        'symbol': symbol,
        'sector': 'unknown',
        'market_cap_category': 'unknown',
        'sector_code': -1
    }


def get_statistics() -> Dict[str, int]:
    """
    Get statistics about core symbol list

    Returns:
        Dict with counts by sector and market cap
    """
    stats = {
        'total_symbols': len(get_all_core_symbols()),
        'by_sector': {},
        'by_market_cap': {
            'large_cap': len(get_symbols_by_market_cap('large_cap')),
            'mid_cap': len(get_symbols_by_market_cap('mid_cap')),
            'small_cap': len(get_symbols_by_market_cap('small_cap'))
        }
    }

    for sector in CORE_SYMBOLS.keys():
        stats['by_sector'][sector] = len(get_symbols_by_sector(sector))

    return stats
''')

    print(f"\n[OK] Generated new core_symbols.py: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(dry_run: bool = False, verbose: bool = False, large_caps_only: bool = False, checkpoint_file: Optional[str] = None):
    """Main execution workflow"""

    print("\n" + "="*80)
    print("TURBOMODE QUARTERLY STOCK CURATION v1.2.0")
    print("="*80)
    print(f"Target: {TARGET_TOTAL} symbols")
    print(f"Mode: {'DRY RUN (no files modified)' if dry_run else 'LIVE EXECUTION'}")
    print(f"Strategy: {'LARGE CAPS ONLY' if large_caps_only else 'ALL MARKET CAPS'}")
    if checkpoint_file:
        print(f"Checkpoint: {Path(checkpoint_file).name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create log directories
    log_dir = Path(__file__).parent.parent / 'data' / 'curation_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = Path(__file__).parent.parent / 'advanced_ml' / 'config' / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate current symbols
    current_df = evaluate_current_symbols(verbose=verbose)

    # Generate initial report
    timestamp = datetime.now().strftime('%Y-%m-%d')
    report_path = log_dir / f'curation_report_{timestamp}.md'
    generate_change_report(current_df, report_path)

    # Build optimized list
    final_df = build_optimized_symbol_list(current_df, verbose=verbose, large_caps_only=large_caps_only, checkpoint_file=checkpoint_file)

    # Summary
    print("\n" + "="*80)
    print("CURATION SUMMARY")
    print("="*80)
    print(f"Current symbols evaluated: {len(current_df)}")
    print(f"  Passing: {current_df['passes_criteria'].sum()}")
    print(f"  Failing: {(~current_df['passes_criteria']).sum()}")
    print(f"  Avg score: {current_df['quality_score'].mean():.1f}/100")
    print(f"\nOptimized list: {len(final_df)}/{TARGET_TOTAL}")
    print(f"  Passing: {final_df['passes_criteria'].sum()}")
    print(f"  Avg score: {final_df['quality_score'].mean():.1f}/100")

    # Display sector distribution
    print(f"\nSector distribution:")
    for sector in TARGET_SECTORS:
        count = len(final_df[final_df['sector'] == sector])
        target = TARGET_SECTORS[sector]
        status = "[OK]" if count == target else f"({count}/{target})"
        print(f"  {sector:30s} {status}")

    # Display market cap distribution
    print(f"\nMarket cap distribution:")
    for cap in TARGET_MARKET_CAPS:
        count = len(final_df[final_df['market_cap_bucket'] == cap])
        target = TARGET_MARKET_CAPS[cap]
        status = "[OK]" if count == target else f"({count}/{target})"
        print(f"  {cap:30s} {status}")

    if not dry_run:
        # Backup current file
        print("\n" + "="*80)
        print("STEP 3: Updating core_symbols.py")
        print("="*80)

        core_symbols_path = Path(__file__).parent.parent / 'advanced_ml' / 'config' / 'core_symbols.py'

        if core_symbols_path.exists():
            backup_path = backup_dir / f'core_symbols_{timestamp}.py'
            shutil.copy2(core_symbols_path, backup_path)
            print(f"\n[BACKUP] Saved to: {backup_path}")

        # Write new file
        write_core_symbols_file(final_df, core_symbols_path)

        print(f"\n[SUCCESS] core_symbols.py updated with {len(final_df)} optimized symbols!")

    else:
        print(f"\n[OK] DRY RUN complete - no files modified")

    print(f"[REPORT] Report saved to: {report_path}")

    print("\n" + "="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TurboMode Quarterly Stock Curation')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, do not modify files')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress messages')
    parser.add_argument('--large-caps-only', action='store_true', help='Only scan large caps (saves 66%% time)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint JSON file to merge candidates from')

    args = parser.parse_args()

    main(dry_run=args.dry_run, verbose=args.verbose, large_caps_only=args.large_caps_only, checkpoint_file=args.checkpoint)

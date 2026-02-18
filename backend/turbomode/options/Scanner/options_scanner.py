"""
OPTIONS INTELLIGENCE SCANNER

Runs the same enrichment logic as the real-time API endpoint,
but stores results in the database for fast cached retrieval.

This scanner should run every 5-10 minutes during market hours
to keep the cache fresh.
"""
import os
import sys
import json
import sqlite3
from datetime import datetime
from time import time

# Add backend to path for imports
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
STOCKAPP_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, STOCKAPP_DIR)

from backend.tradier_client import get_option_chain as tradier_get_option_chain
from backend.turbomode.options.Engines.expiration_selector import select_expiration
from backend.turbomode.options.Engines.wing_selector import select_wings
from backend.turbomode.options.Engines.condor_pricing import calculate_condor_pnl
from backend.turbomode.options.Engines.analytics_engine import compute_analytics
from backend.turbomode.options.Engines.regime_engine import classify_regime
from backend.turbomode.options.Engines.transition_engine import detect_transition
from backend.turbomode.options.Engines.signal_intelligence import enrich_signal
from backend.turbomode.options.Data.snapshot_loader import load_snapshot
from backend.turbomode.options.Data.snapshot_writer import write_snapshot
from backend.turbomode.options.Data.db_writer import write_enriched_to_db
from backend.turbomode.options.Scanner.scanner_logger import log

# Database paths
OPTIONS_DB_PATH = os.path.join(BACKEND_DIR, 'turbomode', 'options', 'Data', 'options_intel.db')

# Symbol universe path
CORE_230_PATH = os.path.join(BACKEND_DIR, '..', 'config', 'symbols', 'CORE_230.json')


def get_current_price(symbol):
    """
    Fetch current price for a symbol using 2-tier fallback system.

    Priority order:
    1. Tradier (live real-time data - PRIMARY)
    2. Yahoo Finance (fallback for stocks only)

    Args:
        symbol: Stock ticker symbol

    Returns:
        float: Current price, or None if all sources fail
    """
    # Tier 1: Try Tradier first (live real-time - PRIMARY)
    try:
        from backend.tradier_client import get_last_price as tradier_price
        price = tradier_price(symbol)
        if price is not None and price > 0:
            log(f'[OPTIONS_SCANNER] Fetched price for {symbol} from Tradier: ${price:.2f}')
            return float(price)
    except Exception as e:
        log(f'[OPTIONS_SCANNER] Tradier price fetch failed for {symbol}: {e}')

    # Tier 2: Try Yahoo Finance (fallback for stocks only)
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Try multiple price fields in order of preference
        for field in ['regularMarketPrice', 'currentPrice', 'previousClose']:
            if field in info and info[field] is not None:
                price = float(info[field])
                if price > 0:
                    log(f'[OPTIONS_SCANNER] Fetched price for {symbol} from Yahoo (fallback): ${price:.2f}')
                    return price
    except Exception as e:
        log(f'[OPTIONS_SCANNER] Yahoo price fetch failed for {symbol}: {e}')

    # All sources failed
    log(f'[OPTIONS_SCANNER] All price sources failed for {symbol}')
    return None


def get_option_chain(symbol):
    """
    Fetch option chain for a symbol using Tradier ONLY.

    Tradier is the ONLY live provider for options chains.
    Yahoo is NEVER used for options chains (stocks and ingestion only).

    Args:
        symbol: Stock ticker symbol

    Returns:
        dict: Option chain data with structure:
        {
            'expirations': ['2026-02-14', '2026-02-21', ...],
            'chains': {
                '2026-02-14': {
                    'calls': {strike: {data}, ...},
                    'puts': {strike: {data}, ...}
                },
                ...
            }
        }
        or None if Tradier fails
    """
    # Tradier ONLY (no fallback for options chains)
    try:
        chain_data = tradier_get_option_chain(symbol)
        if chain_data and 'expirations' in chain_data and 'chains' in chain_data:
            log(f'[OPTIONS_SCANNER] Fetched chain for {symbol} from Tradier')
            return chain_data
        else:
            log(f'[OPTIONS_SCANNER] Tradier returned empty chain for {symbol}')
            return None
    except Exception as e:
        log(f'[OPTIONS_SCANNER] Tradier chain fetch failed for {symbol}: {e}')
        return None


def load_symbol_universe():
    """
    Load symbol universe from CORE_230.json.

    Returns:
        List of dictionaries with symbol data from CORE_230.json
    """
    try:
        with open(CORE_230_PATH, 'r') as f:
            data = json.load(f)
        log(f'[OPTIONS_SCANNER] Loaded {len(data)} symbols from CORE_230.json')
        return data
    except Exception as e:
        log(f'[OPTIONS_SCANNER] ERROR: Failed to load CORE_230.json: {e}')
        return []


def run_options_scanner():
    """
    Main scanner function that enriches all active signals and writes to cache.

    Returns:
        Dictionary with scan results: {success: bool, count: int, errors: list, runtime: float}
    """
    start_time = time()
    log('[OPTIONS_SCANNER] Starting scan...')

    try:
        # 1. Load symbol universe from CORE_230.json (no stock DB dependency)
        symbol_data = load_symbol_universe()

        if not symbol_data:
            log('[OPTIONS_SCANNER] No symbols loaded from CORE_230.json')
            return {'success': False, 'count': 0, 'errors': ['Failed to load symbols']}

        log(f'[OPTIONS_SCANNER] Loaded {len(symbol_data)} symbols from CORE_230.json')

        # 2. Process each symbol
        enriched_signals = []
        errors = []

        for idx, symbol_info in enumerate(symbol_data):
            try:
                symbol = symbol_info['ticker']
                sector = symbol_info.get('sector', 'Unknown')

                log(f'[OPTIONS_SCANNER] Processing {idx+1}/{len(symbol_data)}: {symbol}')

                # Fetch current price (Tradier primary, Yahoo fallback for stocks only)
                current_price = get_current_price(symbol)

                if current_price is None:
                    log(f'[OPTIONS_SCANNER] Skipping {symbol}: No price available')
                    errors.append(f'{symbol}: No price available')
                    continue

                # Placeholder values (no stock DB dependency)
                base_signal = 'NEUTRAL'  # Not using stock signals
                model_confidence = 0.5  # Neutral confidence
                model_probability = 0.5  # Neutral probability
                age_days = 0  # Fresh data

                # Fetch option chain (Tradier ONLY - no fallback for options)
                chain_data = get_option_chain(symbol)

                if not chain_data or 'expirations' not in chain_data:
                    log(f'[OPTIONS_SCANNER] No option chain for {symbol}')
                    errors.append(f'{symbol}: No option chain')
                    continue

                # Select expiration
                expiration = select_expiration(chain_data['expirations'])

                if not expiration or expiration not in chain_data['chains']:
                    log(f'[OPTIONS_SCANNER] No valid expiration for {symbol}')
                    errors.append(f'{symbol}: No valid expiration')
                    continue

                chain = chain_data['chains'][expiration]
                calls = chain.get('calls', {})
                puts = chain.get('puts', {})

                if not calls or not puts:
                    log(f'[OPTIONS_SCANNER] Incomplete chain for {symbol}')
                    errors.append(f'{symbol}: Incomplete chain')
                    continue

                # Select wings (using current price as stop levels since no signal data)
                wings = select_wings(calls, puts, current_price,
                                   current_price * 1.05,  # 5% above as upper stop
                                   current_price * 0.95)  # 5% below as lower stop

                if not wings:
                    log(f'[OPTIONS_SCANNER] No valid wings for {symbol}')
                    errors.append(f'{symbol}: No valid wings')
                    continue

                short_call_strike, short_put_strike = wings

                # Calculate P&L
                pricing = calculate_condor_pnl(calls, puts, short_call_strike, short_put_strike)
                pnl = pricing['pnl'] if pricing and 'pnl' in pricing else None

                # Compute analytics
                analytics = compute_analytics(
                    symbol=symbol,
                    current_price=current_price,
                    calls=calls,
                    puts=puts,
                    short_call_strike=short_call_strike,
                    short_put_strike=short_put_strike,
                    pnl=pnl,
                    age_days=age_days
                )

                # Classify regime
                regime_state = classify_regime(
                    drift=analytics['drift'],
                    pressure_index=analytics['pressure_index'],
                    volatility_regime=analytics['volatility_regime'],
                    quality_score=analytics['quality_score'],
                    pnl=pnl
                )

                # Get previous snapshot for transition detection
                previous_snapshot = load_snapshot(symbol)

                # Detect transition
                if previous_snapshot:
                    transition_state, transition_strength = detect_transition(
                        current_regime=regime_state,
                        previous_regime=previous_snapshot.get('regime_state'),
                        drift=analytics['drift'],
                        previous_drift=previous_snapshot.get('drift'),
                        quality_score=analytics['quality_score'],
                        previous_quality_score=previous_snapshot.get('quality_score')
                    )
                else:
                    transition_state = "No Transition"
                    transition_strength = 0.0

                # Build regime data dict
                regime_data = {
                    'regime_state': regime_state,
                    'transition_state': transition_state,
                    'transition_strength': transition_strength
                }

                # Enrich signal with intelligence
                enriched = enrich_signal(
                    symbol=symbol,
                    base_signal=base_signal,
                    model_confidence=model_confidence,
                    model_probability=model_probability,
                    analytics=analytics,
                    regime_data=regime_data,
                    age_days=age_days
                )

                # Add symbol data fields (no stock DB dependency)
                enriched['current_price'] = current_price
                enriched['stop_upper'] = current_price * 1.05
                enriched['stop_lower'] = current_price * 0.95
                enriched['prob_buy'] = 0.33
                enriched['prob_sell'] = 0.33
                enriched['prob_hold'] = 0.34
                enriched['created_at'] = datetime.now().isoformat()
                enriched['sector'] = sector

                # Add options-specific fields
                enriched['short_call_strike'] = short_call_strike
                enriched['short_put_strike'] = short_put_strike
                enriched['expiration'] = expiration

                # Add pricing fields
                if pricing:
                    enriched['max_credit'] = pricing.get('max_credit')
                    enriched['max_profit'] = pricing.get('max_profit')
                    enriched['max_loss'] = pricing.get('max_loss')
                    enriched['pnl'] = pricing.get('pnl')

                # Add analytics fields
                enriched['drift'] = analytics.get('drift')
                enriched['pressure_index'] = analytics.get('pressure_index')
                enriched['volatility_regime'] = analytics.get('volatility_regime')
                enriched['quality_score'] = analytics.get('quality_score')

                # Add regime fields
                enriched['regime_state'] = regime_state
                enriched['transition_state'] = transition_state
                enriched['transition_strength'] = transition_strength

                enriched_signals.append(enriched)
                log(f'[OPTIONS_SCANNER] ✓ Enriched {symbol}')

            except Exception as e:
                log(f'[OPTIONS_SCANNER] ERROR processing {symbol}: {e}')
                errors.append(f'{symbol}: {str(e)}')
                continue

        # 4. Write all enriched signals to options_intel.db
        if enriched_signals:
            count = 0
            for enriched in enriched_signals:
                write_snapshot(enriched['symbol'], enriched)
                write_enriched_to_db(enriched)
                count += 1

            end_time = time()
            runtime = end_time - start_time
            avg_time_per_signal = runtime / count if count > 0 else 0

            log(f'[OPTIONS_SCANNER] Scan complete: {count}/{len(symbol_data)} symbols cached')
            log(f'[OPTIONS_SCANNER] Runtime: {runtime:.2f}s (avg {avg_time_per_signal:.2f}s/symbol)')
            log(f'[OPTIONS_SCANNER] Errors: {len(errors)} symbols failed')

            # Error recovery: Log failed symbols for retry
            if errors:
                log(f'[OPTIONS_SCANNER] Failed symbols: {", ".join([e.split(":")[0] for e in errors[:10]])}')

            return {
                'success': True,
                'count': count,
                'total': len(symbol_data),
                'errors': errors,
                'runtime': runtime,
                'avg_time_per_signal': avg_time_per_signal
            }
        else:
            end_time = time()
            runtime = end_time - start_time
            log('[OPTIONS_SCANNER] No signals successfully enriched')
            log(f'[OPTIONS_SCANNER] Runtime: {runtime:.2f}s')
            return {'success': False, 'count': 0, 'errors': errors, 'runtime': runtime}

    except Exception as e:
        end_time = time()
        runtime = end_time - start_time
        log(f'[OPTIONS_SCANNER] FATAL ERROR: {e}')
        log(f'[OPTIONS_SCANNER] Runtime before crash: {runtime:.2f}s')
        return {'success': False, 'count': 0, 'errors': [str(e)], 'runtime': runtime}


if __name__ == '__main__':
    result = run_options_scanner()
    print(f'\nScan Result: {result}')

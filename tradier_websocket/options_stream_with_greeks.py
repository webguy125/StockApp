"""
Tradier Options Streaming with Greeks
Combines websocket streaming (real-time prices) with REST API polling (Greeks)

Architecture:
- Websocket: Real-time bid/ask/trade updates (continuous)
- REST API: Greeks updates every 60 seconds (delta, gamma, theta, vega, rho, IV)
- Local cache: Stores latest Greeks for each contract

ALL TRADIER FIELDS CAPTURED:
✅ symbol                 - Option contract symbol
✅ strike                 - Strike price
✅ expiration             - Expiration date
✅ type                   - Call or Put
✅ bid                    - Bid price (from websocket)
✅ ask                    - Ask price (from websocket)
✅ last                   - Last trade price (from websocket)
✅ bid_size               - Bid size in contracts (from websocket)
✅ ask_size               - Ask size in contracts (from websocket)
✅ volume                 - Trading volume (from websocket)
✅ open_interest          - Open interest (from REST API, for liquidity scoring)
✅ change                 - Dollar change (from REST API, for volatility regime)
✅ percent_change         - Percentage change (from REST API, for volatility regime)
✅ delta                  - Delta (from REST API)
✅ gamma                  - Gamma (from REST API)
✅ theta                  - Theta (from REST API)
✅ vega                   - Vega (from REST API)
✅ rho                    - Rho (from REST API)
✅ implied_volatility     - IV (from REST API)
✅ underlying_price       - Underlying stock price (from websocket feed)
✅ days_to_expiration     - Calculated from expiration date
✅ contract_multiplier    - Standard 100 for equity options
"""
import os
import sys
import json
import asyncio
import websockets
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import time

# Load API key
TRADIER_API_KEY = "pplYfsA91vM8AAFoSmLB4naoaDa5"

if not TRADIER_API_KEY:
    print("[ERROR] TRADIER_API_KEY not set")
    sys.exit(1)

# Tradier API endpoints
TRADIER_API_URL = "https://api.tradier.com"
TRADIER_SESSION_ENDPOINT = "/v1/markets/events/session"
TRADIER_OPTIONS_CHAIN_ENDPOINT = "/v1/markets/options/chains"
TRADIER_WS_URL = "wss://ws.tradier.com/v1/markets/events"

# Configuration
TEST_SYMBOLS = ["AAPL", "SPY", "TSLA"]
NUM_OPTIONS_PER_SYMBOL = 3
GREEKS_UPDATE_INTERVAL = 60  # Update Greeks every 60 seconds

# Global cache for Greeks data
greeks_cache = {}  # {option_symbol: {delta, gamma, theta, vega, rho, iv, open_interest, change, last_updated}}
greeks_lock = threading.Lock()

# Global cache for underlying prices
underlying_prices = {}  # {underlying_symbol: {price, last_updated}}
underlying_lock = threading.Lock()

# Contract multiplier (standard for equity options)
CONTRACT_MULTIPLIER = 100


def calculate_days_to_expiration(expiration_str):
    """
    Calculate days to expiration from expiration date string
    Args:
        expiration_str: Date string in format YYYY-MM-DD
    Returns:
        int: Days to expiration (can be negative if expired)
    """
    try:
        exp_date = datetime.strptime(expiration_str, '%Y-%m-%d')
        today = datetime.now()
        delta = exp_date - today
        return delta.days
    except:
        return None


def get_options_contracts(symbol, num_contracts=3):
    """
    Fetch real options contract symbols for a given underlying symbol
    Returns list of option contract info including Greeks
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching options chain for {symbol}...")

    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }

    try:
        # Step 1: Get available expiration dates
        exp_response = requests.get(
            TRADIER_API_URL + "/v1/markets/options/expirations",
            headers=headers,
            params={"symbol": symbol},
            timeout=10
        )

        if exp_response.status_code != 200:
            print(f"[ERROR] Expiration request failed: {exp_response.status_code}")
            return []

        exp_data = exp_response.json()

        # Extract first available expiration date
        expiration_date = None
        if "expirations" in exp_data and "date" in exp_data["expirations"]:
            dates = exp_data["expirations"]["date"]
            if isinstance(dates, list) and len(dates) > 0:
                expiration_date = dates[0]
            elif isinstance(dates, str):
                expiration_date = dates

        if not expiration_date:
            print(f"[ERROR] No expiration dates found for {symbol}")
            return []

        print(f"  Using expiration: {expiration_date}")

        # Step 2: Get options chain with Greeks
        response = requests.get(
            TRADIER_API_URL + TRADIER_OPTIONS_CHAIN_ENDPOINT,
            headers=headers,
            params={
                "symbol": symbol,
                "expiration": expiration_date,
                "greeks": "true"  # Request Greeks data
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"[ERROR] Options chain request failed: {response.status_code}")
            return []

        data = response.json()

        # Extract option symbols and Greeks
        options_list = []

        if "options" in data and "option" in data["options"]:
            options_data = data["options"]["option"]

            if isinstance(options_data, dict):
                options_data = [options_data]

            # Sort by volume to get most liquid
            sorted_options = sorted(
                options_data,
                key=lambda x: x.get('volume', 0) if x.get('volume') else 0,
                reverse=True
            )

            # Get top N most liquid options
            for opt in sorted_options[:num_contracts]:
                symbol_name = opt.get('symbol', '')
                option_type = opt.get('option_type', 'unknown')
                strike = opt.get('strike', 0)
                volume = opt.get('volume', 0)

                # Extract Greeks
                greeks = opt.get('greeks', {})

                if symbol_name:
                    option_info = {
                        'symbol': symbol_name,
                        'type': option_type,
                        'strike': strike,
                        'volume': volume,
                        'underlying': symbol,
                        'expiration': expiration_date,
                        'greeks': {
                            'delta': greeks.get('delta', None),
                            'gamma': greeks.get('gamma', None),
                            'theta': greeks.get('theta', None),
                            'vega': greeks.get('vega', None),
                            'rho': greeks.get('rho', None),
                            'mid_iv': greeks.get('mid_iv', None),
                            'smv_vol': greeks.get('smv_vol', None),
                        },
                        'open_interest': opt.get('open_interest', 0),
                        'change': opt.get('change', None),
                        'change_percentage': opt.get('change_percentage', None),
                    }
                    options_list.append(option_info)

                    # Initialize Greeks cache with all metadata
                    with greeks_lock:
                        greeks_cache[symbol_name] = {
                            **option_info['greeks'],
                            'open_interest': option_info['open_interest'],
                            'change': option_info['change'],
                            'change_percentage': option_info['change_percentage'],
                            'last_updated': datetime.now()
                        }

                    print(f"  Found: {symbol_name} ({option_type} @ ${strike}, vol: {volume}, OI: {option_info['open_interest']})")
                    if option_info['greeks']['delta'] is not None:
                        print(f"    Greeks: Delta={greeks.get('delta'):.4f}, Gamma={greeks.get('gamma'):.4f}, "
                              f"Theta={greeks.get('theta'):.4f}, Vega={greeks.get('vega'):.4f}, IV={greeks.get('mid_iv', 0):.2%}")
                    if option_info['change'] is not None:
                        print(f"    Change: ${option_info['change']:+.2f} ({option_info['change_percentage']:+.2f}%)")

        return options_list

    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return []


def update_greeks_for_contracts(options_info_list):
    """
    Update Greeks for all options contracts
    Called periodically in a separate thread
    """
    while True:
        time.sleep(GREEKS_UPDATE_INTERVAL)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Updating Greeks...")

        # Group options by underlying symbol and expiration
        by_underlying = {}
        for opt in options_info_list:
            key = (opt['underlying'], opt['expiration'])
            if key not in by_underlying:
                by_underlying[key] = []
            by_underlying[key].append(opt)

        headers = {
            "Authorization": f"Bearer {TRADIER_API_KEY}",
            "Accept": "application/json"
        }

        # Fetch updated Greeks for each underlying/expiration combo
        for (underlying, expiration), opts in by_underlying.items():
            try:
                response = requests.get(
                    TRADIER_API_URL + TRADIER_OPTIONS_CHAIN_ENDPOINT,
                    headers=headers,
                    params={
                        "symbol": underlying,
                        "expiration": expiration,
                        "greeks": "true"
                    },
                    timeout=10
                )

                if response.status_code != 200:
                    print(f"  [WARNING] Failed to update {underlying} Greeks")
                    continue

                data = response.json()

                if "options" in data and "option" in data["options"]:
                    options_data = data["options"]["option"]
                    if isinstance(options_data, dict):
                        options_data = [options_data]

                    # Update cache with Greeks and metadata
                    updates = 0
                    with greeks_lock:
                        for opt_data in options_data:
                            opt_symbol = opt_data.get('symbol', '')
                            if opt_symbol in greeks_cache:
                                greeks = opt_data.get('greeks', {})
                                greeks_cache[opt_symbol].update({
                                    'delta': greeks.get('delta', None),
                                    'gamma': greeks.get('gamma', None),
                                    'theta': greeks.get('theta', None),
                                    'vega': greeks.get('vega', None),
                                    'rho': greeks.get('rho', None),
                                    'mid_iv': greeks.get('mid_iv', None),
                                    'smv_vol': greeks.get('smv_vol', None),
                                    'open_interest': opt_data.get('open_interest', 0),
                                    'change': opt_data.get('change', None),
                                    'change_percentage': opt_data.get('change_percentage', None),
                                    'last_updated': datetime.now()
                                })
                                updates += 1

                    print(f"  Updated {updates} contracts for {underlying}")

            except Exception as e:
                print(f"  [ERROR] Failed to update {underlying}: {e}")


def create_streaming_session():
    """Create a streaming session and return session ID"""
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json"
    }

    try:
        response = requests.post(
            TRADIER_API_URL + TRADIER_SESSION_ENDPOINT,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            print(f"[ERROR] Session creation failed: {response.status_code}")
            return None

        session_data = response.json()

        if "stream" in session_data and "sessionid" in session_data["stream"]:
            return session_data["stream"]["sessionid"]

        return None

    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return None


async def stream_options_with_greeks(session_id, option_symbols, options_info_list):
    """Stream real-time options data and display with cached Greeks"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to websocket...")
    print(f"Session ID: {session_id}")
    print()

    # Build lookup for option metadata (strike, expiration, type)
    option_metadata = {}
    underlying_symbols = set()
    for opt in options_info_list:
        option_metadata[opt['symbol']] = {
            'strike': opt['strike'],
            'expiration': opt['expiration'],
            'type': opt['type'],
            'underlying': opt['underlying'],
            'days_to_expiration': calculate_days_to_expiration(opt['expiration'])
        }
        underlying_symbols.add(opt['underlying'])

    # Convert to list for subscription
    underlying_symbols_list = list(underlying_symbols)

    try:
        async with websockets.connect(TRADIER_WS_URL, ssl=True, compression=None) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket connected!")
            print()

            # Subscribe to both options AND underlying symbols for price feed
            all_symbols = option_symbols + underlying_symbols_list
            subscribe_message = {
                "symbols": all_symbols,
                "sessionid": session_id,
                "filter": ["quote", "trade"],
                "linebreak": True
            }

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Subscribing to {len(option_symbols)} option contracts...")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Subscribing to {len(underlying_symbols_list)} underlying symbols: {', '.join(underlying_symbols_list)}")
            print()

            await websocket.send(json.dumps(subscribe_message))

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Streaming with Greeks...")
            print("=" * 120)
            print()

            message_count = 0
            max_messages = 50  # Show more messages

            try:
                while message_count < max_messages:
                    message = await asyncio.wait_for(websocket.recv(), timeout=120)
                    message_count += 1

                    try:
                        data = json.loads(message)
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                        msg_type = data.get('type', 'unknown')
                        symbol = data.get('symbol', '')

                        # Check if this is an underlying stock or option contract
                        is_underlying = symbol in underlying_symbols_list

                        # Handle underlying price updates
                        if is_underlying:
                            price = None
                            if msg_type == 'quote':
                                bid = data.get('bid')
                                ask = data.get('ask')
                                if bid and ask:
                                    price = (float(bid) + float(ask)) / 2
                            elif msg_type == 'trade':
                                price = data.get('price') or data.get('last')
                                if price:
                                    price = float(price)

                            if price:
                                with underlying_lock:
                                    underlying_prices[symbol] = {
                                        'price': price,
                                        'last_updated': datetime.now()
                                    }
                                print(f"[{timestamp}] UNDERLYING UPDATE - {symbol}: ${price:.2f}")
                                print("-" * 120)
                                continue

                        # Get option metadata
                        metadata = option_metadata.get(symbol, {})

                        # Get cached Greeks for this contract
                        cached_greeks = None
                        with greeks_lock:
                            cached_greeks = greeks_cache.get(symbol, {})

                        # Get underlying price
                        underlying_price = None
                        if metadata:
                            underlying_symbol = metadata.get('underlying')
                            with underlying_lock:
                                underlying_data = underlying_prices.get(underlying_symbol, {})
                                underlying_price = underlying_data.get('price')

                        print(f"[{timestamp}] {msg_type.upper()} - {symbol}")

                        # Display option metadata with expiration info
                        if metadata:
                            dte = metadata.get('days_to_expiration')
                            dte_str = f"({dte}d)" if dte is not None else ""
                            print(f"  Contract: {metadata.get('underlying', 'N/A')} "
                                  f"{metadata.get('expiration', 'N/A')} {dte_str} "
                                  f"${metadata.get('strike', 0):.2f} {metadata.get('type', 'N/A').upper()}")
                            print(f"  Contract Multiplier: {CONTRACT_MULTIPLIER}")

                        # Display underlying price if available
                        if underlying_price is not None:
                            print(f"  Underlying Price: ${underlying_price:.2f}")

                        # Helper function to safely convert to float
                        def to_float(val):
                            if val is None:
                                return None
                            if isinstance(val, str):
                                try:
                                    return float(val)
                                except:
                                    return None
                            return float(val)

                        # Extract all Tradier fields from message
                        bid = to_float(data.get('bid'))
                        ask = to_float(data.get('ask'))
                        last = to_float(data.get('last'))
                        price = to_float(data.get('price'))  # Trade price
                        bid_size = data.get('bidsz') or data.get('bidsize')
                        ask_size = data.get('asksz') or data.get('asksize')
                        volume = data.get('volume') or data.get('cvol')  # cvol = cumulative volume
                        size = data.get('size')  # Trade size
                        open_interest = data.get('open_interest') or data.get('openinterest')
                        change = to_float(data.get('change'))
                        percent_change = to_float(data.get('change_percentage'))

                        # Display price data based on message type
                        if msg_type == 'quote':
                            price_parts = []
                            if bid is not None and ask is not None:
                                spread = ask - bid
                                price_parts.append(f"Bid: ${bid:.2f}")
                                price_parts.append(f"Ask: ${ask:.2f}")
                                price_parts.append(f"Spread: ${spread:.2f}")
                            if last is not None:
                                price_parts.append(f"Last: ${last:.2f}")
                            if price_parts:
                                print(f"  {' | '.join(price_parts)}")

                            size_parts = []
                            if bid_size is not None:
                                size_parts.append(f"Bid Size: {bid_size}")
                            if ask_size is not None:
                                size_parts.append(f"Ask Size: {ask_size}")
                            if size_parts:
                                print(f"  {' | '.join(size_parts)}")

                            if volume is not None:
                                print(f"  Volume: {volume}")
                            if open_interest is not None:
                                print(f"  Open Interest: {open_interest}")

                            if change is not None or percent_change is not None:
                                change_parts = []
                                if change is not None:
                                    change_parts.append(f"Change: ${change:+.2f}")
                                if percent_change is not None:
                                    change_parts.append(f"({percent_change:+.2f}%)")
                                print(f"  {' '.join(change_parts)}")

                        elif msg_type == 'trade':
                            trade_parts = []
                            if price is not None:
                                trade_parts.append(f"Price: ${price:.2f}")
                            elif last is not None:
                                trade_parts.append(f"Price: ${last:.2f}")
                            if size is not None:
                                trade_parts.append(f"Size: {size}")
                            if volume is not None:
                                trade_parts.append(f"Volume: {volume}")
                            if trade_parts:
                                print(f"  {' | '.join(trade_parts)}")

                        # Display Greeks and additional data from cache
                        if cached_greeks:
                            greeks_parts = []

                            # Delta, Gamma, Theta, Vega, Rho
                            if cached_greeks.get('delta') is not None:
                                greeks_parts.append(f"Delta: {cached_greeks['delta']:.4f}")
                            if cached_greeks.get('gamma') is not None:
                                greeks_parts.append(f"Gamma: {cached_greeks['gamma']:.4f}")
                            if cached_greeks.get('theta') is not None:
                                greeks_parts.append(f"Theta: {cached_greeks['theta']:.4f}")
                            if cached_greeks.get('vega') is not None:
                                greeks_parts.append(f"Vega: {cached_greeks['vega']:.4f}")
                            if cached_greeks.get('rho') is not None:
                                greeks_parts.append(f"Rho: {cached_greeks['rho']:.4f}")

                            # Implied Volatility
                            iv_val = cached_greeks.get('mid_iv') or cached_greeks.get('smv_vol')
                            if iv_val is not None:
                                greeks_parts.append(f"IV: {iv_val:.2%}")

                            if greeks_parts:
                                age = (datetime.now() - cached_greeks['last_updated']).seconds
                                print(f"  Greeks: {' | '.join(greeks_parts)}")
                                print(f"  Greeks Age: {age}s")

                            # Open Interest (important for liquidity scoring)
                            if cached_greeks.get('open_interest') is not None and cached_greeks.get('open_interest') > 0:
                                print(f"  Open Interest: {cached_greeks['open_interest']:,}")

                            # Change / Percent Change (for volatility regime detection)
                            change_val = cached_greeks.get('change')
                            pct_change_val = cached_greeks.get('change_percentage')
                            if change_val is not None or pct_change_val is not None:
                                change_parts = []
                                if change_val is not None:
                                    change_parts.append(f"Change: ${change_val:+.2f}")
                                if pct_change_val is not None:
                                    change_parts.append(f"({pct_change_val:+.2f}%)")
                                if change_parts:
                                    print(f"  {' '.join(change_parts)}")

                        print("-" * 120)

                    except json.JSONDecodeError:
                        pass

            except asyncio.TimeoutError:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No messages for 120 seconds")

            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Received {message_count} messages")

    except Exception as e:
        print(f"[ERROR] Websocket error: {e}")
        import traceback
        traceback.print_exc()


def run_test():
    """Main test function"""
    print("=" * 100)
    print("TRADIER OPTIONS STREAMING WITH GREEKS")
    print("=" * 100)
    print(f"Underlying symbols: {', '.join(TEST_SYMBOLS)}")
    print(f"Options per symbol: {NUM_OPTIONS_PER_SYMBOL}")
    print(f"Greeks update interval: {GREEKS_UPDATE_INTERVAL} seconds")
    print()

    # Step 1: Get options contracts with initial Greeks
    print("STEP 1: Fetching Options Contracts with Greeks")
    print("=" * 100)

    all_options = []
    for symbol in TEST_SYMBOLS:
        options = get_options_contracts(symbol, NUM_OPTIONS_PER_SYMBOL)
        all_options.extend(options)
        print()

    if not all_options:
        print("[ERROR] No options contracts found")
        return

    option_symbols = [opt['symbol'] for opt in all_options]

    print(f"[OK] Found {len(option_symbols)} option contracts with Greeks")
    print()

    # Step 2: Start Greeks update thread
    print("STEP 2: Starting Greeks Update Thread")
    print("=" * 100)
    greeks_thread = threading.Thread(
        target=update_greeks_for_contracts,
        args=(all_options,),
        daemon=True
    )
    greeks_thread.start()
    print(f"[OK] Greeks will update every {GREEKS_UPDATE_INTERVAL} seconds")
    print()

    # Step 3: Create streaming session
    print("STEP 3: Creating Streaming Session")
    print("=" * 100)
    session_id = create_streaming_session()

    if not session_id:
        print("[ERROR] Failed to create streaming session")
        return

    print(f"[OK] Session created: {session_id}")
    print()

    # Step 4: Stream with Greeks
    print("STEP 4: Streaming Options Data with Greeks")
    print("=" * 100)

    asyncio.run(stream_options_with_greeks(session_id, option_symbols, all_options))

    print()
    print("=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    run_test()

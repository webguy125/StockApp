"""
Tradier Real Options Websocket Stream Test
Fetches actual options contracts and streams their data including Greeks

Process:
1. Query options chain via REST API to get real option contract symbols
2. Create streaming session via REST API
3. Connect to WebSocket with session ID
4. Subscribe to actual option contracts (not just stock symbols)
5. Receive options data with Greeks (delta, gamma, theta, vega, IV)
"""
import os
import sys
import json
import asyncio
import websockets
import requests
from datetime import datetime

# Load API key (hardcoded for testing)
TRADIER_API_KEY = "pplYfsA91vM8AAFoSmLB4naoaDa5"

if not TRADIER_API_KEY:
    print("[ERROR] TRADIER_API_KEY not set")
    sys.exit(1)

# Tradier API endpoints
TRADIER_API_URL = "https://api.tradier.com"
TRADIER_SESSION_ENDPOINT = "/v1/markets/events/session"
TRADIER_OPTIONS_CHAIN_ENDPOINT = "/v1/markets/options/chains"
TRADIER_WS_URL = "wss://ws.tradier.com/v1/markets/events"

# Test symbols for options (use liquid stocks with active options)
TEST_SYMBOLS = ["AAPL", "SPY", "TSLA"]
NUM_OPTIONS_PER_SYMBOL = 3  # Get 3 ATM options per symbol


def get_options_contracts(symbol, num_contracts=3):
    """
    Fetch real options contract symbols for a given underlying symbol
    Returns list of option contract symbols near ATM (at-the-money)
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
            print(f"[ERROR] Expiration request failed with status code: {exp_response.status_code}")
            print(f"Response: {exp_response.text}")
            return []

        exp_data = exp_response.json()

        # Extract first available expiration date
        expiration_date = None
        if "expirations" in exp_data and "date" in exp_data["expirations"]:
            dates = exp_data["expirations"]["date"]
            if isinstance(dates, list) and len(dates) > 0:
                expiration_date = dates[0]  # Use nearest expiration
            elif isinstance(dates, str):
                expiration_date = dates

        if not expiration_date:
            print(f"[ERROR] No expiration dates found for {symbol}")
            return []

        print(f"  Using expiration: {expiration_date}")

        # Step 2: Get options chain for that expiration
        response = requests.get(
            TRADIER_API_URL + TRADIER_OPTIONS_CHAIN_ENDPOINT,
            headers=headers,
            params={
                "symbol": symbol,
                "expiration": expiration_date,
                "greeks": "true"
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"[ERROR] Options chain request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []

        data = response.json()

        # Extract option symbols
        options_list = []

        if "options" in data and "option" in data["options"]:
            options_data = data["options"]["option"]

            # If single option, convert to list
            if isinstance(options_data, dict):
                options_data = [options_data]

            # Get ATM options (calls and puts near current price)
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

                if symbol_name:
                    options_list.append({
                        'symbol': symbol_name,
                        'type': option_type,
                        'strike': strike,
                        'volume': volume
                    })
                    print(f"  Found: {symbol_name} ({option_type} @ ${strike}, vol: {volume})")

        return options_list

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return []


def create_streaming_session():
    """Create a streaming session and return session ID"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating streaming session...")

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
            print(f"[ERROR] Session creation failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

        session_data = response.json()
        print(f"Session response: {json.dumps(session_data, indent=2)}")
        print()

        # Extract session ID
        if "stream" in session_data and "sessionid" in session_data["stream"]:
            session_id = session_data["stream"]["sessionid"]
            print(f"[OK] Session created: {session_id}")
            return session_id
        else:
            print(f"[ERROR] No session ID in response")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return None


async def stream_options_data(session_id, option_symbols):
    """Stream real-time options data with Greeks"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to websocket...")
    print(f"Endpoint: {TRADIER_WS_URL}")
    print(f"Session ID: {session_id}")
    print()

    try:
        # Connect to websocket
        async with websockets.connect(TRADIER_WS_URL, ssl=True, compression=None) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket connected!")
            print()

            # Subscribe to options updates using session ID
            subscribe_message = {
                "symbols": option_symbols,
                "sessionid": session_id,
                "filter": ["quote", "trade"],  # Real-time quotes and trades
                "linebreak": True
            }

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending subscription request...")
            print(f"Subscribing to {len(option_symbols)} option contracts:")
            for sym in option_symbols:
                print(f"  - {sym}")
            print()

            await websocket.send(json.dumps(subscribe_message))

            # Listen for messages
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening for options data...")
            print("=" * 80)
            print()

            message_count = 0
            timeout_seconds = 60
            max_messages = 30  # Collect more messages for options

            try:
                while message_count < max_messages:
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                    message_count += 1

                    # Parse and display message
                    try:
                        data = json.loads(message)
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                        msg_type = data.get('type', 'unknown')
                        symbol = data.get('symbol', '')

                        print(f"[{timestamp}] Message #{message_count} - {msg_type.upper()}")
                        print(f"  Contract: {symbol}")

                        # Display relevant fields based on message type
                        if msg_type == 'quote':
                            if 'bid' in data and 'ask' in data:
                                bid = float(data['bid']) if isinstance(data['bid'], str) else data['bid']
                                ask = float(data['ask']) if isinstance(data['ask'], str) else data['ask']
                                print(f"  Bid: ${bid:.2f}  Ask: ${ask:.2f}")

                            if 'bidsize' in data:
                                print(f"  Bid Size: {data['bidsize']}")
                            if 'asksize' in data:
                                print(f"  Ask Size: {data['asksize']}")

                        elif msg_type == 'trade':
                            if 'price' in data:
                                price = float(data['price']) if isinstance(data['price'], str) else data['price']
                                print(f"  Price: ${price:.2f}")

                            if 'size' in data:
                                print(f"  Size: {data['size']}")

                            if 'volume' in data or 'cvol' in data:
                                vol = data.get('volume', data.get('cvol', 'N/A'))
                                print(f"  Volume: {vol}")

                        # Greeks (if available)
                        greeks_found = False
                        if 'delta' in data:
                            delta = float(data['delta']) if isinstance(data['delta'], str) else data['delta']
                            print(f"  Delta: {delta:.4f}")
                            greeks_found = True
                        if 'gamma' in data:
                            gamma = float(data['gamma']) if isinstance(data['gamma'], str) else data['gamma']
                            print(f"  Gamma: {gamma:.4f}")
                            greeks_found = True
                        if 'theta' in data:
                            theta = float(data['theta']) if isinstance(data['theta'], str) else data['theta']
                            print(f"  Theta: {theta:.4f}")
                            greeks_found = True
                        if 'vega' in data:
                            vega = float(data['vega']) if isinstance(data['vega'], str) else data['vega']
                            print(f"  Vega: {vega:.4f}")
                            greeks_found = True
                        if 'rho' in data:
                            rho = float(data['rho']) if isinstance(data['rho'], str) else data['rho']
                            print(f"  Rho: {rho:.4f}")
                            greeks_found = True
                        if 'impliedVolatility' in data or 'iv' in data:
                            iv = data.get('impliedVolatility', data.get('iv', 0))
                            iv_val = float(iv) if isinstance(iv, str) else iv
                            print(f"  Implied Volatility: {iv_val:.2%}")
                            greeks_found = True

                        if not greeks_found and msg_type == 'quote':
                            print("  (Greeks not available in this update)")

                        print("-" * 80)

                    except json.JSONDecodeError:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Non-JSON message: {message}")
                        print("-" * 80)

            except asyncio.TimeoutError:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No messages received for {timeout_seconds} seconds")
                print("This might be normal if market is closed or low options activity")

            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Test Summary:")
            print(f"  Total messages: {message_count}")
            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Closing connection...")

    except websockets.exceptions.WebSocketException as e:
        print(f"[ERROR] Websocket error: {e}")
        if hasattr(e, 'status_code'):
            if e.status_code == 401:
                print("Authentication failed - check your API key or session ID")
            elif e.status_code == 403:
                print("Access forbidden - check your API permissions")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def run_test():
    """Main test function"""
    print("=" * 80)
    print("TRADIER REAL OPTIONS WEBSOCKET STREAM TEST")
    print("=" * 80)
    print(f"Underlying symbols: {', '.join(TEST_SYMBOLS)}")
    print(f"Options per symbol: {NUM_OPTIONS_PER_SYMBOL}")
    print()

    # Step 1: Get real options contract symbols
    print("STEP 1: Fetching Options Contracts")
    print("=" * 80)

    all_options = []
    for symbol in TEST_SYMBOLS:
        options = get_options_contracts(symbol, NUM_OPTIONS_PER_SYMBOL)
        all_options.extend(options)
        print()

    if not all_options:
        print("[ERROR] No options contracts found")
        print("This might be due to:")
        print("  - Market is closed")
        print("  - API rate limits")
        print("  - Account doesn't have options data access")
        return

    # Extract just the option symbols for subscription
    option_symbols = [opt['symbol'] for opt in all_options]

    print(f"[OK] Found {len(option_symbols)} option contracts total")
    print()

    # Step 2: Create streaming session
    print("STEP 2: Creating Streaming Session")
    print("=" * 80)
    session_id = create_streaming_session()

    if not session_id:
        print("\n[ERROR] Failed to create streaming session")
        print("Cannot proceed with websocket test")
        return

    print()
    print("STEP 3: Streaming Options Data")
    print("=" * 80)

    # Step 3: Stream options data
    asyncio.run(stream_options_data(session_id, option_symbols))

    print()
    print("=" * 80)
    print("REAL OPTIONS STREAM TEST COMPLETE")
    print("=" * 80)
    print()
    print("Note: Greeks may not appear in every message.")
    print("They are typically included in quote messages during market hours.")


if __name__ == "__main__":
    run_test()

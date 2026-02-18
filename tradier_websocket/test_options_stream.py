"""
Tradier Options Websocket Stream Test
Tests live options chain streaming for specified symbols
Two-step process:
1. Create session via REST API
2. Connect to WebSocket with session ID
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
    print("[ERROR] TRADIER_API_KEY environment variable not set")
    print("\nTo set it (Windows):")
    print("  set TRADIER_API_KEY=your_key_here")
    print("\nTo set it (Linux/Mac):")
    print("  export TRADIER_API_KEY=your_key_here")
    sys.exit(1)

# Tradier API endpoints
TRADIER_API_URL = "https://api.tradier.com"
TRADIER_SESSION_ENDPOINT = "/v1/markets/events/session"
TRADIER_WS_URL = "wss://ws.tradier.com/v1/markets/events"

# Test symbols for options (use liquid stocks with active options)
TEST_SYMBOLS = ["AAPL", "SPY", "TSLA"]

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

async def test_options_stream(session_id):
    """Test Tradier options websocket streaming with session ID"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to websocket...")
    print(f"Endpoint: {TRADIER_WS_URL}")
    print(f"Session ID: {session_id}")
    print()

    try:
        # Connect to websocket (no authentication headers needed, session ID is used instead)
        async with websockets.connect(TRADIER_WS_URL, ssl=True, compression=None) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected successfully!")
            print()

            # Subscribe to options updates using session ID
            # Tradier supports: quote, trade, summary, timesale, tradex (extended)
            subscribe_message = {
                "symbols": TEST_SYMBOLS,
                "sessionid": session_id,
                "filter": ["quote", "trade"],  # Real-time quotes and trades
                "linebreak": True
            }

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending options subscription request...")
            print(f"Request: {json.dumps(subscribe_message, indent=2)}")
            print()

            await websocket.send(json.dumps(subscribe_message))

            # Listen for messages (timeout after 60 seconds for options)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening for options updates...")
            print("=" * 80)
            print()

            message_count = 0
            options_messages = 0
            timeout_seconds = 60
            max_messages = 20  # Collect more messages for options

            try:
                while message_count < max_messages:
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                    message_count += 1

                    # Parse and display message
                    try:
                        data = json.loads(message)
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                        # Check if this is options data (symbol contains options-like format)
                        msg_type = data.get('type', 'unknown')
                        symbol = data.get('symbol', '')

                        # Options symbols typically have format like: AAPL230120C00150000
                        is_option = len(symbol) > 10 or 'C' in symbol[-10:] or 'P' in symbol[-10:]

                        if is_option:
                            options_messages += 1
                            print(f"[{timestamp}] OPTIONS Message #{options_messages}:")
                        else:
                            print(f"[{timestamp}] Message #{message_count} ({msg_type}):")

                        # Pretty print the data
                        if is_option:
                            # Highlight key options fields
                            print(f"  Symbol: {symbol}")
                            print(f"  Type: {msg_type}")

                            if 'bid' in data and 'ask' in data:
                                bid = float(data['bid']) if isinstance(data['bid'], str) else data['bid']
                                ask = float(data['ask']) if isinstance(data['ask'], str) else data['ask']
                                print(f"  Bid: ${bid:.2f}  Ask: ${ask:.2f}")

                            if 'last' in data:
                                last = float(data['last']) if isinstance(data['last'], str) else data['last']
                                print(f"  Last: ${last:.2f}")

                            if 'price' in data:
                                price = float(data['price']) if isinstance(data['price'], str) else data['price']
                                print(f"  Price: ${price:.2f}")

                            if 'volume' in data:
                                print(f"  Volume: {data['volume']}")

                            # Greeks (if available)
                            if 'delta' in data:
                                delta = float(data['delta']) if isinstance(data['delta'], str) else data['delta']
                                print(f"  Delta: {delta:.4f}")
                            if 'gamma' in data:
                                gamma = float(data['gamma']) if isinstance(data['gamma'], str) else data['gamma']
                                print(f"  Gamma: {gamma:.4f}")
                            if 'theta' in data:
                                theta = float(data['theta']) if isinstance(data['theta'], str) else data['theta']
                                print(f"  Theta: {theta:.4f}")
                            if 'vega' in data:
                                vega = float(data['vega']) if isinstance(data['vega'], str) else data['vega']
                                print(f"  Vega: {vega:.4f}")
                            if 'impliedVolatility' in data:
                                iv = float(data['impliedVolatility']) if isinstance(data['impliedVolatility'], str) else data['impliedVolatility']
                                print(f"  IV: {iv:.2%}")
                        else:
                            print(json.dumps(data, indent=2))

                        print("-" * 80)
                    except json.JSONDecodeError:
                        # Some messages might not be JSON (like status messages)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Non-JSON message: {message}")
                        print("-" * 80)

            except asyncio.TimeoutError:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No messages received for {timeout_seconds} seconds")
                print("This might be normal if market is closed or low options activity")

            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Test Summary:")
            print(f"  Total messages: {message_count}")
            print(f"  Options messages: {options_messages}")
            print(f"  Stock messages: {message_count - options_messages}")
            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Closing connection...")

    except websockets.exceptions.WebSocketException as e:
        print(f"[ERROR] Websocket error: {e}")
        # Check for specific status codes
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
    print("TRADIER OPTIONS WEBSOCKET STREAM TEST")
    print("=" * 80)
    print(f"Test symbols: {', '.join(TEST_SYMBOLS)}")
    print()

    # Step 1: Create streaming session
    session_id = create_streaming_session()

    if not session_id:
        print("\n[ERROR] Failed to create streaming session")
        print("Cannot proceed with websocket test")
        return

    print()
    print("=" * 80)

    # Step 2: Connect to websocket with session ID
    asyncio.run(test_options_stream(session_id))

    print()
    print("=" * 80)
    print("OPTIONS STREAM TEST COMPLETE")
    print("=" * 80)
    print()
    print("Note: Tradier may require specific subscription level for options data")
    print("Check your account permissions if no options data is received")

if __name__ == "__main__":
    # Run complete test (session creation + websocket connection)
    run_test()

"""
Tradier Websocket Connection Test
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
    print("  set TRADIER_API_KEY=pplYfsA91vM8AAFoSmLB4naoaDa5")
    print("\nTo set it (Linux/Mac):")
    print("  export TRADIER_API_KEY=pplYfsA91vM8AAFoSmLB4naoaDa5")
    sys.exit(1)

# Tradier API endpoints
TRADIER_API_URL = "https://api.tradier.com"
TRADIER_SESSION_ENDPOINT = "/v1/markets/events/session"
TRADIER_WS_URL = "wss://ws.tradier.com/v1/markets/events"

# Test symbols to subscribe to
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]

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

async def test_tradier_websocket(session_id):
    """Test Tradier websocket connection with session ID"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to websocket...")
    print(f"Endpoint: {TRADIER_WS_URL}")
    print(f"Session ID: {session_id}")
    print()

    try:
        # Connect to websocket (no authentication headers needed, session ID is used instead)
        async with websockets.connect(TRADIER_WS_URL, ssl=True, compression=None) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket connected!")
            print()

            # Subscribe to quote updates using session ID
            subscribe_message = {
                "symbols": TEST_SYMBOLS,
                "sessionid": session_id,
                "filter": ["quote", "trade"],  # quote, trade, summary, timesale
                "linebreak": True
            }

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending subscription request...")
            print(f"Request: {json.dumps(subscribe_message, indent=2)}")
            print()

            await websocket.send(json.dumps(subscribe_message))

            # Listen for messages (timeout after 30 seconds)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening for messages...")
            print("=" * 80)
            print()

            message_count = 0
            timeout_seconds = 30

            try:
                while message_count < 20:  # Receive first 20 messages
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                    message_count += 1

                    # Parse and display message
                    try:
                        data = json.loads(message)
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                        print(f"[{timestamp}] Message #{message_count}:")
                        print(json.dumps(data, indent=2))
                        print("-" * 80)
                    except json.JSONDecodeError:
                        # Some messages might not be JSON (like status messages)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Non-JSON message: {message}")
                        print("-" * 80)

            except asyncio.TimeoutError:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No messages received for {timeout_seconds} seconds")
                print("This might be normal if market is closed or symbols are not actively trading")

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
    print("TRADIER WEBSOCKET CONNECTION TEST")
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
    asyncio.run(test_tradier_websocket(session_id))

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Run complete test (session creation + websocket connection)
    run_test()

"""
Test Tradier WebSocket Integration
Tests the full pipeline: Tradier WS -> Backend -> Socket.IO -> Frontend
"""
import os
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_tradier_ws_client():
    """Test standalone Tradier WebSocket client"""
    print("=" * 80)
    print("TEST 1: Tradier WebSocket Client")
    print("=" * 80)

    try:
        from tradier_websocket_client import TradierWebSocketClient

        # Load API key from environment
        api_key = os.environ.get("TRADIER_API_KEY")
        if not api_key:
            print("[FAIL] TRADIER_API_KEY not set in environment")
            return False

        print(f"[OK] API key loaded: {api_key[:8]}...")

        # Create client
        client = TradierWebSocketClient(api_key)
        print("[OK] Client created")

        # Test data collection
        received_quotes = []

        def on_quote(symbol, data):
            received_quotes.append((symbol, data))
            print(f"  [DATA] {symbol}: ${data.get('last', 0):.2f} (bid: ${data.get('bid', 0):.2f}, ask: ${data.get('ask', 0):.2f})")

        # Subscribe to test symbols
        test_symbols = ["AAPL", "MSFT", "TSLA"]
        for symbol in test_symbols:
            client.subscribe(symbol)
            client.register_callback(symbol, on_quote)
            print(f"[OK] Subscribed to {symbol}")

        # Start streaming
        client.start()
        print("[OK] WebSocket started")

        # Wait for quotes
        print("\nWaiting for quotes (30 seconds)...")
        time.sleep(30)

        # Stop client
        client.stop()
        print("\n[OK] Client stopped")

        # Check results
        print(f"\n[DATA] Received {len(received_quotes)} total quotes")

        # Group by symbol
        by_symbol = {}
        for symbol, data in received_quotes:
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(data)

        for symbol in test_symbols:
            count = len(by_symbol.get(symbol, []))
            status = "[OK]" if count > 0 else "[FAIL]"
            print(f"{status} {symbol}: {count} quotes")

        # Success if we got at least one quote for each symbol
        success = all(symbol in by_symbol and len(by_symbol[symbol]) > 0 for symbol in test_symbols)

        if success:
            print("\n[OK] TEST 1 PASSED: All symbols received quotes")
        else:
            print("\n[FAIL] TEST 1 FAILED: Some symbols did not receive quotes")

        return success

    except Exception as e:
        print(f"\n[FAIL] TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_socket_io_integration():
    """Test Socket.IO integration"""
    print("\n" + "=" * 80)
    print("TEST 2: Socket.IO Integration")
    print("=" * 80)

    try:
        import socketio

        # Create Socket.IO client
        sio = socketio.Client()

        received_updates = []

        @sio.on('ticker_update')
        def on_ticker(data):
            received_updates.append(data)
            print(f"  [DATA] {data.get('symbol')}: ${data.get('price', 0):.2f}")

        @sio.on('connect')
        def on_connect():
            print("[OK] Connected to Flask Socket.IO")

            # Subscribe to test symbols
            test_symbols = ["AAPL", "MSFT"]
            for symbol in test_symbols:
                sio.emit('subscribe_ticker', {'symbol': symbol})
                print(f"[OK] Subscribed to {symbol}")

        # Connect to Flask server
        print("Connecting to Flask server...")
        sio.connect('http://127.0.0.1:5000')

        # Wait for updates
        print("\nWaiting for ticker updates (30 seconds)...")
        time.sleep(30)

        # Disconnect
        sio.disconnect()
        print("\n[OK] Disconnected")

        # Check results
        print(f"\n[DATA] Received {len(received_updates)} ticker updates")

        # Group by symbol
        by_symbol = {}
        for update in received_updates:
            symbol = update.get('symbol')
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(update)

        for symbol, updates in by_symbol.items():
            print(f"  {symbol}: {len(updates)} updates")

        # Success if we got at least one update
        success = len(received_updates) > 0

        if success:
            print("\n[OK] TEST 2 PASSED: Received ticker updates via Socket.IO")
        else:
            print("\n[FAIL] TEST 2 FAILED: No ticker updates received")

        return success

    except Exception as e:
        print(f"\n[FAIL] TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n")
    print("*" * 80)
    print("Tradier WebSocket Integration Test Suite")
    print("*" * 80)
    print("\n")

    # Run tests
    test1_passed = test_tradier_ws_client()
    test2_passed = test_socket_io_integration()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Tradier WS Client): {'[OK] PASSED' if test1_passed else '[FAIL] FAILED'}")
    print(f"Test 2 (Socket.IO Integration): {'[OK] PASSED' if test2_passed else '[FAIL] FAILED'}")

    if test1_passed and test2_passed:
        print("\n[OK] ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        sys.exit(1)

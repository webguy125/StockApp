"""
Simple test to verify Coinbase WebSocket is sending data
"""
import websocket
import json
import time
import threading

received_messages = 0

def on_message(ws, message):
    global received_messages
    received_messages += 1
    data = json.loads(message)
    msg_type = data.get('type', 'unknown')
    print(f"[{received_messages}] Received type={msg_type}: {str(data)[:150]}...")

def on_error(ws, error):
    print(f"ERROR: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed: status={close_status_code} msg={close_msg}")

def on_open(ws):
    print("=" * 60)
    print("WebSocket connection OPENED")
    print("=" * 60)

    # Subscribe to BTC-USD ticker
    subscribe_msg = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["ticker"]
    }

    print(f"Sending subscription: {json.dumps(subscribe_msg)}")
    ws.send(json.dumps(subscribe_msg))
    print("Subscription message sent. Waiting for data...")
    print("=" * 60)

if __name__ == "__main__":
    print("Testing Coinbase WebSocket Feed...")
    print("Endpoint: wss://ws-feed.exchange.coinbase.com")
    print("=" * 60)

    ws = websocket.WebSocketApp(
        "wss://ws-feed.exchange.coinbase.com",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # Run in a thread so we can terminate after 10 seconds
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    print("Waiting for 10 seconds to receive data...")
    time.sleep(10)

    print(f"\nTest complete. Total messages received: {received_messages}")
    ws.close()

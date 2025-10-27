"""
Coinbase WebSocket Manager
Handles real-time price streaming using WebSocket
"""

from flask_socketio import SocketIO, emit
import json
import threading
import websocket
import time

class CoinbaseWebSocketManager:
    def __init__(self, socketio, volume_tracker=None):
        self.socketio = socketio
        self.volume_tracker = volume_tracker
        self.active_streams = {}
        self.base_url = "wss://ws-feed.exchange.coinbase.com"  # Coinbase Advanced Trade WebSocket

    def normalize_symbol(self, symbol):
        """Normalize to Coinbase format (BTC-USD)"""
        symbol = symbol.upper().replace('/', '-')

        # If already has dash, return as-is
        if '-' in symbol:
            return symbol

        # If ends with USD/USDT, add dash
        if symbol.endswith('USD'):
            base = symbol[:-3]
            return f"{base}-USD"
        elif symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}-USD"
        else:
            # Add -USD as default quote currency
            return f"{symbol}-USD"

    def start_ticker_stream(self, symbol):
        """
        Start streaming real-time ticker (price) data

        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'BTC', 'ETH-USD')
        """
        normalized_symbol = self.normalize_symbol(symbol)
        stream_name = f"{normalized_symbol}@ticker"

        if stream_name in self.active_streams:
            self.stop_stream(stream_name)

        print(f">> Starting Coinbase ticker stream: {stream_name}")

        def on_message(ws, message):
            try:
                data = json.loads(message)

                # DEBUG: Log all messages to diagnose
                msg_type = data.get('type', 'unknown')
                print(f">> [CB WS] {stream_name} - Message type: {msg_type}")

                # Handle ticker updates
                if data.get('type') == 'ticker':
                    # Get today's volume from tracker (accumulated from REST + trades)
                    today_volume = 0
                    if self.volume_tracker:
                        today_volume = self.volume_tracker.get_volume(data['product_id'])

                    ticker_data = {
                        'symbol': data['product_id'],
                        'price': float(data.get('price', 0)),
                        'volume_24h': float(data.get('volume_24h', 0)),  # Keep for reference
                        'volume_today': today_volume,  # NEW: Today's actual candle volume
                        'time': data.get('time'),
                        'bid': float(data.get('best_bid', 0)),
                        'ask': float(data.get('best_ask', 0))
                    }

                    # Emit to frontend
                    print(f">> [TICKER] {ticker_data['symbol']} @ ${ticker_data['price']:.2f} | Vol: {today_volume:.2f} BTC")
                    self.socketio.emit('ticker_update', ticker_data)

            except Exception as e:
                print(f">> Error processing ticker message: {e}")

        def on_error(ws, error):
            print(f">> WebSocket error for {stream_name}: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f">> WebSocket closed for {stream_name}")
            if stream_name in self.active_streams:
                del self.active_streams[stream_name]

        def on_open(ws):
            print(f">> WebSocket connected: {stream_name}")

            # Subscribe to ticker channel
            subscribe_message = {
                "type": "subscribe",
                "product_ids": [normalized_symbol],
                "channels": ["ticker"]
            }
            print(f">> [TICKER SUB] Sending: {json.dumps(subscribe_message)}")
            try:
                ws.send(json.dumps(subscribe_message))
                print(f">> [TICKER SUB] Subscription message sent for {stream_name}")
            except Exception as e:
                print(f">> [TICKER SUB ERROR] Failed to send subscription: {e}")

        ws = websocket.WebSocketApp(
            self.base_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Run forever with ping to keep connection alive
        ws_thread = threading.Thread(
            target=lambda: ws.run_forever(ping_interval=20, ping_timeout=10),
            daemon=True
        )
        ws_thread.start()

        self.active_streams[stream_name] = {
            'ws': ws,
            'thread': ws_thread,
            'symbol': normalized_symbol
        }

    def start_matches_stream(self, symbol):
        """
        Start streaming matches (trades) data

        Args:
            symbol: Trading pair
        """
        normalized_symbol = self.normalize_symbol(symbol)
        stream_name = f"{normalized_symbol}@matches"

        if stream_name in self.active_streams:
            self.stop_stream(stream_name)

        print(f">> Starting Coinbase matches stream: {stream_name}")

        def on_message(ws, message):
            try:
                data = json.loads(message)

                # Handle match (trade) updates
                if data.get('type') == 'match':
                    trade_size = float(data.get('size', 0))
                    symbol = data['product_id']

                    # Add trade volume to today's accumulator
                    if self.volume_tracker:
                        self.volume_tracker.add_trade(symbol, trade_size)

                    trade_data = {
                        'symbol': symbol,
                        'price': float(data.get('price', 0)),
                        'size': trade_size,
                        'side': data.get('side'),
                        'time': data.get('time')
                    }

                    # Emit to frontend (for tick charts, etc.)
                    self.socketio.emit('coinbase_trade_update', trade_data)

            except Exception as e:
                print(f">> Error processing matches message: {e}")

        def on_error(ws, error):
            print(f">> WebSocket error for {stream_name}: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f">> WebSocket closed for {stream_name}")
            if stream_name in self.active_streams:
                del self.active_streams[stream_name]

        def on_open(ws):
            print(f">> WebSocket connected: {stream_name}")

            # Subscribe to matches channel
            subscribe_message = {
                "type": "subscribe",
                "product_ids": [normalized_symbol],
                "channels": ["matches"]
            }
            ws.send(json.dumps(subscribe_message))

        ws = websocket.WebSocketApp(
            self.base_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Run forever with ping to keep connection alive
        ws_thread = threading.Thread(
            target=lambda: ws.run_forever(ping_interval=20, ping_timeout=10),
            daemon=True
        )
        ws_thread.start()

        self.active_streams[stream_name] = {
            'ws': ws,
            'thread': ws_thread,
            'symbol': normalized_symbol
        }

    def stop_stream(self, stream_name):
        """Stop a specific stream"""
        if stream_name in self.active_streams:
            print(f">> Stopping stream: {stream_name}")
            try:
                stream_info = self.active_streams[stream_name]
                stream_info['ws'].close()
            except Exception as e:
                print(f">> Error closing stream {stream_name}: {e}")
            finally:
                # Always remove from active streams even if close fails
                if stream_name in self.active_streams:
                    del self.active_streams[stream_name]

    def stop_all_streams(self):
        """Stop all active streams"""
        print(f">> Stopping all Coinbase WebSocket streams")
        for stream_name in list(self.active_streams.keys()):
            self.stop_stream(stream_name)

    def get_active_streams(self):
        """Get list of active streams"""
        return list(self.active_streams.keys())

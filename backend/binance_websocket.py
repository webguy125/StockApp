"""
Binance WebSocket Manager
Handles real-time price streaming using WebSocket
"""

from flask import Flask
from flask_socketio import SocketIO, emit
import json
import threading
import websocket
import time

class BinanceWebSocketManager:
    def __init__(self, socketio):
        self.socketio = socketio
        self.active_streams = {}  # {symbol: websocket_connection}
        self.base_url = "wss://stream.binance.us:9443/ws"  # Use Binance US WebSocket

    def normalize_symbol(self, symbol):
        """Normalize symbol to Binance format (lowercase)"""
        symbol = symbol.upper().replace('-', '').replace('/', '')

        # Handle USD -> USDT conversion first
        if symbol.endswith('USD') and not symbol.endswith('USDT') and not symbol.endswith('BUSD'):
            symbol = symbol[:-3] + 'USDT'

        # Add USDT if no quote currency
        # Only check for trading pair quote currencies (not base currencies like BTC, ETH)
        quote_currencies = ['USDT', 'BUSD']
        has_quote = any(symbol.endswith(q) for q in quote_currencies)

        if not has_quote:
            symbol = f"{symbol}USDT"

        return symbol.lower()

    def start_kline_stream(self, symbol, interval='1m'):
        """
        Start streaming kline (candlestick) data

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'BTC', 'BTC-USD')
            interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
        """
        normalized_symbol = self.normalize_symbol(symbol)
        stream_name = f"{normalized_symbol}@kline_{interval}"

        # Stop existing stream if any
        if stream_name in self.active_streams:
            self.stop_stream(stream_name)

        print(f"🚀 Starting Binance WebSocket stream: {stream_name}")

        # Create WebSocket connection
        ws_url = f"{self.base_url}/{stream_name}"

        def on_message(ws, message):
            try:
                data = json.loads(message)

                if 'k' in data:
                    kline = data['k']

                    # Format kline data
                    candle_data = {
                        'symbol': data['s'],
                        'time': kline['t'],
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'is_closed': kline['x']  # True when candle is complete
                    }

                    # Emit to frontend via Socket.IO
                    self.socketio.emit('binance_kline_update', candle_data)

                    if candle_data['is_closed']:
                        print(f"📊 {stream_name}: New candle closed - O:{candle_data['open']:.2f} H:{candle_data['high']:.2f} L:{candle_data['low']:.2f} C:{candle_data['close']:.2f}")

            except Exception as e:
                print(f"❌ Error processing WebSocket message: {e}")

        def on_error(ws, error):
            print(f"❌ WebSocket error for {stream_name}: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f"🔌 WebSocket closed for {stream_name}")
            if stream_name in self.active_streams:
                del self.active_streams[stream_name]

        def on_open(ws):
            print(f"✅ WebSocket connected: {stream_name}")

        # Create and start WebSocket in a thread
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()

        self.active_streams[stream_name] = {
            'ws': ws,
            'thread': ws_thread,
            'symbol': normalized_symbol,
            'interval': interval
        }

    def start_ticker_stream(self, symbol):
        """
        Start streaming real-time ticker (price) data

        Args:
            symbol: Trading pair
        """
        normalized_symbol = self.normalize_symbol(symbol)
        stream_name = f"{normalized_symbol}@ticker"

        if stream_name in self.active_streams:
            self.stop_stream(stream_name)

        print(f"🚀 Starting Binance ticker stream: {stream_name}")

        ws_url = f"{self.base_url}/{stream_name}"

        def on_message(ws, message):
            try:
                data = json.loads(message)

                ticker_data = {
                    'symbol': data['s'],
                    'price': float(data['c']),
                    'change': float(data['p']),
                    'change_percent': float(data['P']),
                    'high': float(data['h']),
                    'low': float(data['l']),
                    'volume': float(data['v']),
                    'time': data['E']
                }

                # Emit to frontend
                self.socketio.emit('binance_ticker_update', ticker_data)

            except Exception as e:
                print(f"❌ Error processing ticker message: {e}")

        def on_error(ws, error):
            print(f"❌ WebSocket error for {stream_name}: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f"🔌 WebSocket closed for {stream_name}")
            if stream_name in self.active_streams:
                del self.active_streams[stream_name]

        def on_open(ws):
            print(f"✅ WebSocket connected: {stream_name}")

        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()

        self.active_streams[stream_name] = {
            'ws': ws,
            'thread': ws_thread,
            'symbol': normalized_symbol
        }

    def stop_stream(self, stream_name):
        """Stop a specific stream"""
        if stream_name in self.active_streams:
            print(f"🛑 Stopping stream: {stream_name}")
            stream_info = self.active_streams[stream_name]
            stream_info['ws'].close()
            del self.active_streams[stream_name]

    def stop_all_streams(self):
        """Stop all active streams"""
        print("🛑 Stopping all Binance WebSocket streams")
        for stream_name in list(self.active_streams.keys()):
            self.stop_stream(stream_name)

    def get_active_streams(self):
        """Get list of active streams"""
        return list(self.active_streams.keys())

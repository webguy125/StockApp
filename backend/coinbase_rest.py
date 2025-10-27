"""
Coinbase REST API Client for Historical Data
Simple and clean implementation
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

class CoinbaseREST:
    def __init__(self):
        self.base_url = "https://api.exchange.coinbase.com"

    def get_daily_candles(self, symbol="BTC-USD", days=30):
        """
        Get daily candles from Coinbase
        Returns last N days of daily data
        """
        # Coinbase uses granularity in seconds: 86400 = 1 day
        granularity = 86400

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Format for Coinbase API (ISO 8601)
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()

        # Build request
        url = f"{self.base_url}/products/{symbol}/candles"
        params = {
            'start': start_iso,
            'end': end_iso,
            'granularity': granularity
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Coinbase returns: [timestamp, low, high, open, close, volume]
            candles = response.json()

            # Convert to our format
            result = []
            for candle in candles:
                timestamp, low, high, open_price, close_price, volume = candle
                # Use UTC to match Yahoo Finance date handling
                date_str = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
                result.append({
                    'Date': date_str,
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close_price,
                    'Volume': volume
                })

            # Sort by date (Coinbase returns newest first)
            result.sort(key=lambda x: x['Date'])

            # Remove duplicate dates if any
            seen_dates = set()
            unique_result = []
            for candle in result:
                if candle['Date'] not in seen_dates:
                    seen_dates.add(candle['Date'])
                    unique_result.append(candle)

            print(f"[SUCCESS] Fetched {len(unique_result)} daily candles from Coinbase")
            return unique_result

        except Exception as e:
            print(f"[ERROR] Error fetching Coinbase data: {e}")
            return []

    def get_current_day_volume(self, symbol="BTC-USD"):
        """
        Get today's volume from Coinbase (just the current incomplete candle)
        Returns the volume for today in BTC
        """
        # Coinbase uses granularity in seconds: 86400 = 1 day
        granularity = 86400

        # Get today's data (just 1 day)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

        # Format for Coinbase API (ISO 8601)
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()

        # Build request
        url = f"{self.base_url}/products/{symbol}/candles"
        params = {
            'start': start_iso,
            'end': end_iso,
            'granularity': granularity
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Coinbase returns: [timestamp, low, high, open, close, volume]
            candles = response.json()

            if candles and len(candles) > 0:
                # Get the most recent candle (first in array, Coinbase returns newest first)
                latest = candles[0]
                timestamp, low, high, open_price, close_price, volume = latest
                date_str = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')

                print(f"[VOLUME] {symbol} {date_str}: {volume:.2f} BTC")
                return {
                    'symbol': symbol,
                    'date': date_str,
                    'volume': volume,
                    'price': close_price
                }
            else:
                print(f"[VOLUME] No data for {symbol}")
                return {'symbol': symbol, 'volume': 0, 'price': 0}

        except Exception as e:
            print(f"[ERROR] Error fetching current volume for {symbol}: {e}")
            return {'symbol': symbol, 'volume': 0, 'price': 0}
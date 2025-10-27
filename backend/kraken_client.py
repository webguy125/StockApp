"""
Kraken API Client for Crypto Data
- REST API for historical OHLC data
- WebSocket for real-time ticker and trades
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class KrakenClient:
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Kraken pair mapping (BTC-USD -> XXBTZUSD format)
        self.pair_map = {
            'BTC-USD': 'XXBTZUSD',
            'ETH-USD': 'XETHZUSD',
            'SOL-USD': 'SOLUSD',
            'XRP-USD': 'XXRPZUSD',
            'DOGE-USD': 'XDGUSD',
            'BNB-USD': None,  # Not available on Kraken
            'ADA-USD': 'ADAUSD',
            'AVAX-USD': 'AVAXUSD',
            'DOT-USD': 'DOTUSD',
            'MATIC-USD': 'MATICUSD',
            'LINK-USD': 'LINKUSD',
            'UNI-USD': 'UNIUSD',
            'AAVE-USD': 'AAVEUSD',
        }

    def _rate_limit(self):
        """Ensure we don't spam the API"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint, params=None):
        """Make API request with rate limiting"""
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('error') and len(data['error']) > 0:
                raise Exception(f"Kraken API Error: {data['error']}")

            return data.get('result', {})
        except Exception as e:
            print(f">> Kraken API Error: {e}")
            raise

    def normalize_symbol(self, symbol):
        """
        Convert standard symbol format to Kraken format
        Examples:
            'BTC-USD' -> 'XXBTZUSD'
            'BTC' -> 'XXBTZUSD'
            'ETH-USD' -> 'XETHZUSD'
        """
        symbol = symbol.upper().replace('/', '-')

        # If already has dash, look it up
        if '-' in symbol:
            kraken_pair = self.pair_map.get(symbol)
            if kraken_pair:
                return kraken_pair
            # Try to construct it
            base, quote = symbol.split('-')
            if base == 'BTC':
                base = 'XBT'  # Kraken uses XBT for Bitcoin
            return f'X{base}Z{quote}'

        # Add -USD if no quote currency
        return self.normalize_symbol(f"{symbol}-USD")

    def get_ohlc(self, symbol, interval=1, since=None):
        """
        Get OHLC data from Kraken

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            interval: Timeframe in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Return data since given timestamp (Unix timestamp)

        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        kraken_pair = self.normalize_symbol(symbol)

        params = {
            'pair': kraken_pair,
            'interval': interval
        }

        if since:
            params['since'] = since

        print(f">> Fetching Kraken OHLC: {kraken_pair} interval={interval}min")

        data = self._make_request('/OHLC', params)

        # Kraken returns data under the pair name
        if kraken_pair not in data:
            # Try alternative pair names
            for key in data.keys():
                if key != 'last':
                    kraken_pair = key
                    break

        if kraken_pair not in data:
            raise Exception(f"No OHLC data found for {symbol}")

        ohlc_data = data[kraken_pair]

        # Kraken returns: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])

        # Convert to correct types
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['Open'] = df['open'].astype(float)
        df['High'] = df['high'].astype(float)
        df['Low'] = df['low'].astype(float)
        df['Close'] = df['close'].astype(float)
        df['Volume'] = df['volume'].astype(float)

        # Keep only needed columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # print(f">> Fetched {len(df)} candles for {symbol}")  # Commented out - too verbose
        # if not df.empty:
        #     first_candle = df.iloc[0]['Date']
        #     last_candle = df.iloc[-1]['Date']
        #     print(f">> Candle range: {first_candle} to {last_candle}")  # Commented out - too verbose

        return df

    def aggregate_candles(self, df, group_size, base_interval_minutes):
        """
        Aggregate candles by combining multiple candles into one, aligned to proper time boundaries

        Args:
            df: DataFrame with OHLC data
            group_size: Number of candles to combine (e.g., 2 for 2h from 1h data)
            base_interval_minutes: The base interval in minutes (e.g., 60 for 1h candles)

        Returns:
            DataFrame with aggregated candles aligned to proper time boundaries
        """
        if df.empty or group_size <= 1:
            return df

        # Calculate the target interval in minutes
        target_interval_minutes = base_interval_minutes * group_size

        # Group candles by aligned time boundaries
        # For 2h candles (120 min), align to 00:00, 02:00, 04:00, etc.
        # For 12h candles (720 min), align to 00:00, 12:00
        aggregated = []

        for idx, row in df.iterrows():
            timestamp = row['Date']

            # Round down to the nearest target interval
            # Convert to timestamp, divide by interval seconds, floor, multiply back
            interval_seconds = target_interval_minutes * 60
            ts = timestamp.timestamp() if hasattr(timestamp, 'timestamp') else pd.Timestamp(timestamp).timestamp()
            aligned_ts = (ts // interval_seconds) * interval_seconds
            aligned_time = pd.Timestamp(aligned_ts, unit='s', tz='UTC')

            # Find or create aggregated candle for this time boundary
            if not aggregated or aggregated[-1]['Date'] != aligned_time:
                # Start a new aggregated candle
                aggregated.append({
                    'Date': aligned_time,
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Close': row['Close'],
                    'Volume': row['Volume']
                })
            else:
                # Update existing aggregated candle
                agg = aggregated[-1]
                agg['High'] = max(agg['High'], row['High'])
                agg['Low'] = min(agg['Low'], row['Low'])
                agg['Close'] = row['Close']  # Last close in the period
                agg['Volume'] += row['Volume']

        result = pd.DataFrame(aggregated)
        # print(f">> Aggregated {len(df)} candles into {len(result)} candles (group_size={group_size}, interval={target_interval_minutes}min)")  # Commented out - too verbose

        return result

    def get_data_by_period(self, symbol, interval='5m', period='1d'):
        """
        Get data for a specific period (matches yfinance interface)

        Args:
            symbol: Trading pair
            interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y'

        Returns:
            DataFrame with candlestick data
        """
        # Define intervals that need aggregation: {requested_interval: (base_interval, group_size)}
        aggregation_map = {
            '15s': (1, 0),     # Fetch 1m, split into 4 (approximation)
            '45s': (1, 0),     # Fetch 1m, split into ~1.3 (approximation)
            '2m': (1, 2),      # Fetch 1m, combine 2
            '3m': (1, 3),      # Fetch 1m, combine 3
            '10m': (5, 2),     # Fetch 5m, combine 2
            '45m': (15, 3),    # Fetch 15m, combine 3
            '2h': (60, 2),     # Fetch 1h, combine 2
            '3h': (60, 3),     # Fetch 1h, combine 3
            '12h': (240, 3),   # Fetch 4h, combine 3
        }

        # Map intervals to Kraken interval (in minutes)
        interval_map = {
            '1s': 1,      # Kraken doesn't have seconds, use 1min as fallback
            '5s': 1,
            '10s': 1,
            '15s': 1,     # Will use 1m data
            '30s': 1,
            '45s': 1,     # Will use 1m data
            '1m': 1,
            '2m': 1,      # Will be aggregated later
            '3m': 1,      # Will be aggregated later
            '5m': 5,
            '10m': 5,     # Will be aggregated later
            '15m': 15,
            '30m': 30,
            '45m': 15,    # Will be aggregated later
            '1h': 60,
            '2h': 60,     # Will be aggregated later
            '3h': 60,     # Will be aggregated later
            '4h': 240,
            '12h': 240,   # Will be aggregated later
            '1d': 1440,
            '3d': 1440,   # Use daily candles
            '5d': 1440,
            '1wk': 10080,
            '2wk': 10080
        }

        kraken_interval = interval_map.get(interval, 5)
        needs_aggregation = interval in aggregation_map

        # Calculate date range based on period
        # EXTENDED: Like TradingView, fetch maximum historical data for backtesting
        # The period label is just a UI hint, but we load as much data as practical
        end_time = datetime.now()

        # Strategy: Load extensive historical data based on interval granularity
        # - Intraday (1m-1h): Limited by API constraints, but still generous
        # - Multi-hour (2h-12h): Several months to years
        # - Daily+: Go back to 2014 or earlier for maximum backtesting capability

        # Handle YTD (Year-To-Date) specially
        if period == 'ytd':
            start_time = datetime(end_time.year, 1, 1)  # January 1st of current year
        else:
            # Extended period mapping for maximum historical data
            # Adjust based on interval to balance data volume and usefulness
            if interval in ['1m', '3m', '5m']:
                # Very granular: limit to avoid excessive data (but still generous)
                period_map = {
                    '1d': timedelta(days=7),      # 7 days of minute data
                    '5d': timedelta(days=30),     # 1 month
                    '1mo': timedelta(days=90),    # 3 months
                    '3mo': timedelta(days=180),   # 6 months
                    '6mo': timedelta(days=365),   # 1 year
                    '1y': timedelta(days=365*2),  # 2 years
                    '5y': timedelta(days=365*3),  # 3 years
                }
            elif interval in ['15m', '30m', '45m', '1h']:
                # Hourly-ish: medium range
                period_map = {
                    '1d': timedelta(days=30),     # 1 month
                    '5d': timedelta(days=90),     # 3 months
                    '1mo': timedelta(days=180),   # 6 months
                    '3mo': timedelta(days=365),   # 1 year
                    '6mo': timedelta(days=365*2), # 2 years
                    '1y': timedelta(days=365*5),  # 5 years
                    '5y': timedelta(days=365*10), # 10 years
                }
            elif interval in ['2h', '4h', '12h']:
                # Multi-hour: long range
                period_map = {
                    '1d': timedelta(days=180),    # 6 months
                    '5d': timedelta(days=365),    # 1 year
                    '1mo': timedelta(days=365*2), # 2 years
                    '3mo': timedelta(days=365*5), # 5 years
                    '6mo': timedelta(days=365*10),# 10 years
                    '1y': timedelta(days=365*11), # 11 years (back to ~2014)
                    '5y': timedelta(days=365*11), # 11 years
                }
            else:
                # Daily and above: maximum historical data (like TradingView)
                period_map = {
                    '1d': timedelta(days=365*11), # 11 years back to ~2014
                    '5d': timedelta(days=365*11),
                    '1mo': timedelta(days=365*11),
                    '3mo': timedelta(days=365*11),
                    '6mo': timedelta(days=365*11),
                    '1y': timedelta(days=365*11),
                    '5y': timedelta(days=365*11),
                }

            period_delta = period_map.get(period, timedelta(days=365*11))
            start_time = end_time - period_delta

            print(f">> EXTENDED RANGE: Fetching from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({period_delta.days} days)")

        # Kraken returns max 720 datapoints, so we may need multiple requests
        # Calculate how many candles we need
        total_seconds = (end_time - start_time).total_seconds()
        total_minutes = total_seconds / 60
        num_candles = int(total_minutes / kraken_interval)

        # print(f">> DEBUG: period={period}, interval={kraken_interval}min, num_candles={num_candles}")  # Commented out - too verbose

        if num_candles <= 720:
            # Single request
            since = int(start_time.timestamp())
            df = self.get_ohlc(symbol, kraken_interval, since=since)

            # Apply aggregation if needed
            if needs_aggregation:
                base_interval, group_size = aggregation_map[interval]
                df = self.aggregate_candles(df, group_size, base_interval)

            return df
        else:
            # Multiple requests needed
            # Strategy: Fetch in chunks, working forward from start_time
            # print(f">> Period {period} requires {num_candles} candles, fetching in chunks...")  # Commented out - too verbose

            all_data = []
            current_since = int(start_time.timestamp())
            chunk_count = 0
            max_chunks = min((num_candles // 720) + 2, 100)  # Estimate chunks needed

            while chunk_count < max_chunks:
                df_chunk = self.get_ohlc(symbol, kraken_interval, since=current_since)

                if df_chunk.empty:
                    print(f">> No more data available at chunk {chunk_count + 1}")
                    break

                # Check if we got any new data (not duplicates)
                if all_data:
                    last_existing_date = all_data[-1].iloc[-1]['Date']
                    new_data = df_chunk[df_chunk['Date'] > last_existing_date]
                    if not new_data.empty:
                        all_data.append(new_data)
                    else:
                        # No new data, we've reached the limit of available historical data
                        print(f">> Reached end of available historical data at chunk {chunk_count + 1}")
                        break
                else:
                    all_data.append(df_chunk)

                chunk_count += 1

                # Get the last timestamp for next iteration
                last_timestamp = df_chunk.iloc[-1]['Date']

                # Check if we've reached or passed the end time
                if last_timestamp >= end_time:
                    print(f">> Reached present time at chunk {chunk_count}")
                    break

                # Progress indicator
                if chunk_count % 5 == 0:
                    oldest = all_data[0].iloc[0]['Date']
                    newest = all_data[-1].iloc[-1]['Date']
                    print(f">> Progress: {chunk_count} chunks, data from {oldest} to {newest}")

                # Update since to get next chunk (use timestamp + small increment to avoid overlap)
                # For Kraken, we need to go to the next candle period
                current_since = int(last_timestamp.timestamp()) + (kraken_interval * 60)

            if not all_data:
                raise Exception(f"No data fetched for {symbol}")

            print(f">> Fetched {chunk_count} total chunks")

            # Combine all chunks
            result = pd.concat(all_data, ignore_index=True)

            # Debug: Check for duplicates before removing
            duplicates_before = result.duplicated(subset=['Date']).sum()
            if duplicates_before > 0:
                print(f">> WARNING: Found {duplicates_before} duplicate timestamps before deduplication")
                duplicate_dates = result[result.duplicated(subset=['Date'], keep=False)]['Date'].unique()
                for dup_date in duplicate_dates:
                    dup_rows = result[result['Date'] == dup_date]
                    print(f"   Duplicate at {dup_date}:")
                    for idx, row in dup_rows.iterrows():
                        print(f"     O:{row['Open']} H:{row['High']} L:{row['Low']} C:{row['Close']}")

            result = result.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
            result = result.reset_index(drop=True)

            # print(f">> Combined {len(result)} candles from {len(all_data)} chunks (removed {duplicates_before} duplicates)")  # Commented out - too verbose

            # Apply aggregation if needed
            if needs_aggregation:
                base_interval, group_size = aggregation_map[interval]
                result = self.aggregate_candles(result, group_size, base_interval)

            return result

    def get_ticker(self, symbol):
        """Get current ticker data"""
        kraken_pair = self.normalize_symbol(symbol)

        data = self._make_request('/Ticker', params={'pair': kraken_pair})

        # Find the pair data
        for key, value in data.items():
            if key.upper().replace('/', '').replace('-', '') == kraken_pair.upper().replace('/', '').replace('-', ''):
                return {
                    'symbol': symbol,
                    'price': float(value['c'][0]),  # Last trade closed
                    'volume': float(value['v'][1]),  # Volume today
                    'high': float(value['h'][1]),  # High today
                    'low': float(value['l'][1]),  # Low today
                }

        raise Exception(f"No ticker data found for {symbol}")

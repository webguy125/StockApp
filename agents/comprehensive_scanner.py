"""
Comprehensive Scanner Agent - S&P 500 + Top 100 Cryptos
Integrates with Agent Self-Learning Loop

Purpose: Comprehensive market scanner for both stocks and cryptocurrencies
Data Sources:
  - Polygon API for S&P 500 stocks (real-time & historical)
  - CoinGecko API for top 100 cryptocurrencies
Output: C:\StockApp\agents\repository\scanner_output.json

Features:
- Pulls fresh data from Polygon + CoinGecko
- Calculates comprehensive technical indicators
- Filters by volume, volatility, liquidity
- Outputs ready-to-use data for heat maps
- Integrates with existing agent learning loop
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

# Imports
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import emoji codec
try:
    from language.emoji_codec import EmojiCodec
except:
    EmojiCodec = None


class ComprehensiveScanner:
    """
    Comprehensive scanner for S&P 500 stocks and top 100 cryptocurrencies.
    Replaces existing scanners with robust, multi-source data collection.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec() if EmojiCodec else None
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.repository_path.mkdir(parents=True, exist_ok=True)

        # API Setup
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.coingecko = CoinGeckoAPI()

        # S&P 500 symbols cache (will fetch dynamically)
        self.sp500_symbols = []

        # Filtering thresholds
        self.thresholds = {
            'min_volume_usd': {
                'stock_default': 50_000_000,     # $50M for stocks
                'crypto_default': 10_000_000,    # $10M for crypto
                'large_cap': 500_000_000,        # $500M for large caps
            },
            'min_volatility_pct': {
                'stock': 0.5,    # 0.5% min volatility for stocks
                'crypto': 1.0    # 1.0% min volatility for crypto
            },
            'min_price': 1.0,              # Minimum price $1
            'max_price': 100000,           # Maximum price (filter outliers)
            'min_market_cap': 100_000_000  # $100M minimum market cap
        }

        print(f"🔧 Comprehensive Scanner initialized")
        print(f"   Polygon API: {'✅ Configured' if self.polygon_api_key else '❌ Not configured'}")
        print(f"   CoinGecko API: ✅ Configured")

    def get_sp500_symbols(self) -> List[str]:
        """
        Get S&P 500 symbols from Wikipedia or fallback to predefined list
        """
        try:
            # Try to fetch from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df['Symbol'].tolist()

            # Clean symbols (remove dots, etc.)
            symbols = [s.replace('.', '-') for s in symbols]

            print(f"✅ Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            return symbols[:100]  # Limit to top 100 for performance

        except Exception as e:
            print(f"⚠️  Failed to fetch S&P 500 list: {e}")
            print("   Using fallback list of major stocks")

            # Fallback to major stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX',
                'LLY', 'ABBV', 'MRK', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD',
                'CSCO', 'ABT', 'ACN', 'DHR', 'VZ', 'ADBE', 'NKE', 'CRM', 'TXN',
                'NEE', 'CMCSA', 'PM', 'DIS', 'LIN', 'ORCL', 'WFC', 'NFLX', 'AMD',
                'INTC', 'BA', 'CAT', 'GE', 'F', 'GM', 'PYPL', 'QCOM', 'IBM', 'UBER'
            ]

    def get_top_cryptos(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get top cryptocurrencies by market cap from CoinGecko
        """
        try:
            print(f"📡 Fetching top {limit} cryptocurrencies from CoinGecko...")

            # Get top cryptos by market cap
            cryptos = self.coingecko.get_coins_markets(
                vs_currency='usd',
                order='market_cap_desc',
                per_page=limit,
                page=1,
                sparkline=False,
                price_change_percentage='1h,24h,7d'
            )

            print(f"✅ Fetched {len(cryptos)} cryptocurrencies")
            return cryptos

        except Exception as e:
            print(f"❌ Error fetching cryptos from CoinGecko: {e}")
            return []

    def fetch_stock_data_polygon(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock data from Polygon API
        """
        if not self.polygon_api_key:
            return None

        try:
            from polygon import RESTClient

            client = RESTClient(self.polygon_api_key)

            # Get latest quote
            quote = client.get_last_quote(symbol)

            # Get previous close for calculations
            prev_close_response = client.get_previous_close(symbol)
            prev_close = prev_close_response[0] if prev_close_response else None

            # Get aggregates for volume
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

            aggs = client.get_aggs(
                symbol,
                1,
                'day',
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            current_price = quote.ask_price if hasattr(quote, 'ask_price') else 0
            volume = aggs[0].volume if aggs and len(aggs) > 0 else 0
            prev_close_price = prev_close.close if prev_close and hasattr(prev_close, 'close') else current_price

            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume_24h': volume,
                'price_change_24h': current_price - prev_close_price if prev_close_price else 0,
                'price_change_pct_24h': ((current_price - prev_close_price) / prev_close_price * 100) if prev_close_price else 0,
                'market_type': 'stock',
                'source': 'polygon'
            }

        except Exception as e:
            print(f"   ⚠️  Polygon API error for {symbol}: {e}")
            return None

    def fetch_stock_data_yfinance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock data from Yahoo Finance (fallback)
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='5d')

            if hist.empty:
                return None

            current_price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
            volume = info.get('regularMarketVolume', 0)
            prev_close = info.get('previousClose', current_price)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume_24h': volume,
                'price_change_24h': current_price - prev_close,
                'price_change_pct_24h': ((current_price - prev_close) / prev_close * 100) if prev_close else 0,
                'market_cap': info.get('marketCap', 0),
                'market_type': 'stock',
                'source': 'yfinance'
            }

        except Exception as e:
            print(f"   ⚠️  YFinance error for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, symbol: str, market_type: str) -> Dict[str, Any]:
        """
        Calculate technical indicators using historical data
        """
        try:
            import yfinance as yf
            from ta.momentum import RSIIndicator
            from ta.trend import MACD, SMAIndicator, EMAIndicator
            from ta.volatility import AverageTrueRange, BollingerBands

            # Fetch historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo', interval='1d')

            if hist.empty or len(hist) < 14:
                return {}

            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']

            indicators = {}

            # RSI
            try:
                rsi = RSIIndicator(close, window=14)
                indicators['rsi'] = float(rsi.rsi().iloc[-1])
            except:
                indicators['rsi'] = 50.0

            # MACD
            try:
                macd = MACD(close)
                indicators['macd'] = float(macd.macd().iloc[-1])
                indicators['macd_signal'] = float(macd.macd_signal().iloc[-1])
                indicators['macd_histogram'] = float(macd.macd_diff().iloc[-1])
            except:
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_histogram'] = 0.0

            # Moving Averages
            try:
                sma_20 = SMAIndicator(close, window=20)
                sma_50 = SMAIndicator(close, window=50) if len(close) >= 50 else None
                ema_12 = EMAIndicator(close, window=12)
                ema_26 = EMAIndicator(close, window=26)

                indicators['sma_20'] = float(sma_20.sma_indicator().iloc[-1])
                indicators['sma_50'] = float(sma_50.sma_indicator().iloc[-1]) if sma_50 else indicators['sma_20']
                indicators['ema_12'] = float(ema_12.ema_indicator().iloc[-1])
                indicators['ema_26'] = float(ema_26.ema_indicator().iloc[-1])
            except:
                current_price = float(close.iloc[-1])
                indicators['sma_20'] = current_price
                indicators['sma_50'] = current_price
                indicators['ema_12'] = current_price
                indicators['ema_26'] = current_price

            # ATR (volatility)
            try:
                atr = AverageTrueRange(high, low, close, window=14)
                indicators['atr'] = float(atr.average_true_range().iloc[-1])
            except:
                indicators['atr'] = 0.0

            # Bollinger Bands
            try:
                bb = BollingerBands(close, window=20, window_dev=2)
                indicators['bb_upper'] = float(bb.bollinger_hband().iloc[-1])
                indicators['bb_middle'] = float(bb.bollinger_mavg().iloc[-1])
                indicators['bb_lower'] = float(bb.bollinger_lband().iloc[-1])
            except:
                current_price = float(close.iloc[-1])
                indicators['bb_upper'] = current_price * 1.02
                indicators['bb_middle'] = current_price
                indicators['bb_lower'] = current_price * 0.98

            # Volume metrics
            try:
                avg_volume = float(volume.mean())
                current_volume = float(volume.iloc[-1])
                indicators['avg_volume'] = avg_volume
                indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            except:
                indicators['avg_volume'] = 0.0
                indicators['volume_ratio'] = 1.0

            return indicators

        except Exception as e:
            print(f"   ⚠️  Indicator calculation error for {symbol}: {e}")
            return {}

    def calculate_volatility(self, symbol: str) -> float:
        """
        Calculate 24h volatility percentage
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d', interval='1h')

            if hist.empty:
                return 0.0

            high = hist['High'].max()
            low = hist['Low'].min()
            avg = (high + low) / 2

            volatility = ((high - low) / avg * 100) if avg > 0 else 0.0
            return float(volatility)

        except:
            return 0.0

    def scan_stocks(self) -> List[Dict[str, Any]]:
        """
        Scan S&P 500 stocks
        """
        print("\n📊 Scanning S&P 500 Stocks...")

        # Get S&P 500 symbols
        self.sp500_symbols = self.get_sp500_symbols()

        results = []
        scanned = 0
        passed = 0

        for symbol in self.sp500_symbols:
            scanned += 1
            print(f"   [{scanned}/{len(self.sp500_symbols)}] Scanning {symbol}...", end=' ')

            # Try Polygon first, fallback to yfinance
            data = self.fetch_stock_data_polygon(symbol)
            if not data:
                data = self.fetch_stock_data_yfinance(symbol)

            if not data:
                print("❌ No data")
                continue

            # Calculate volatility
            volatility = self.calculate_volatility(symbol)
            data['volatility_24h'] = volatility

            # Apply filters
            price = data.get('current_price', 0)
            volume_usd = data.get('volume_24h', 0) * price
            min_volatility = self.thresholds['min_volatility_pct']['stock']

            # Filter criteria
            if price < self.thresholds['min_price']:
                print(f"❌ Price too low (${price:.2f})")
                continue

            if volume_usd < self.thresholds['min_volume_usd']['stock_default']:
                print(f"❌ Volume too low (${volume_usd/1e6:.1f}M)")
                continue

            if volatility < min_volatility:
                print(f"❌ Low volatility ({volatility:.2f}%)")
                continue

            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(symbol, 'stock')
            data['indicators'] = indicators

            passed += 1
            print(f"✅ Passed (V:{volatility:.1f}%, Vol:${volume_usd/1e6:.0f}M)")

            results.append(data)

            # Rate limiting
            time.sleep(0.1)

        print(f"\n✅ Stock scan complete: {passed}/{scanned} passed filters")
        return results

    def scan_cryptos(self) -> List[Dict[str, Any]]:
        """
        Scan top 100 cryptocurrencies
        """
        print("\n💰 Scanning Top 100 Cryptocurrencies...")

        cryptos = self.get_top_cryptos(100)
        results = []
        passed = 0

        for idx, crypto in enumerate(cryptos, 1):
            symbol = crypto['symbol'].upper()
            print(f"   [{idx}/{len(cryptos)}] Scanning {symbol}...", end=' ')

            try:
                # Extract data from CoinGecko response
                data = {
                    'symbol': f"{symbol}-USD",
                    'name': crypto['name'],
                    'current_price': crypto['current_price'],
                    'market_cap': crypto['market_cap'],
                    'volume_24h': crypto['total_volume'],
                    'price_change_24h': crypto.get('price_change_24h', 0),
                    'price_change_pct_24h': crypto.get('price_change_percentage_24h', 0),
                    'price_change_pct_7d': crypto.get('price_change_percentage_7d_in_currency', 0),
                    'volatility_24h': abs(crypto.get('price_change_percentage_24h', 0)),
                    'market_type': 'crypto',
                    'source': 'coingecko'
                }

                # Apply filters
                price = data['current_price']
                volume_usd = data['volume_24h']
                volatility = data['volatility_24h']
                market_cap = data['market_cap']

                min_volatility = self.thresholds['min_volatility_pct']['crypto']

                if volume_usd < self.thresholds['min_volume_usd']['crypto_default']:
                    print(f"❌ Low volume (${volume_usd/1e6:.1f}M)")
                    continue

                if market_cap < self.thresholds['min_market_cap']:
                    print(f"❌ Low market cap (${market_cap/1e6:.1f}M)")
                    continue

                if volatility < min_volatility:
                    print(f"❌ Low volatility ({volatility:.2f}%)")
                    continue

                # Calculate technical indicators
                indicators = self.calculate_technical_indicators(f"{symbol}-USD", 'crypto')
                data['indicators'] = indicators

                passed += 1
                print(f"✅ Passed (V:{volatility:.1f}%, Vol:${volume_usd/1e6:.0f}M)")

                results.append(data)

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

            # Rate limiting
            time.sleep(0.05)

        print(f"\n✅ Crypto scan complete: {passed}/{len(cryptos)} passed filters")
        return results

    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """
        Run full comprehensive scan of both stocks and cryptos
        """
        print("=" * 80)
        print("🚀 COMPREHENSIVE SCANNER - S&P 500 + Top 100 Cryptos")
        print("=" * 80)

        start_time = datetime.now()

        # Scan stocks
        stocks = self.scan_stocks()

        # Scan cryptos
        cryptos = self.scan_cryptos()

        # Combine results
        all_candidates = stocks + cryptos

        # Sort by a composite score (volatility * volume)
        for candidate in all_candidates:
            volume_score = candidate.get('volume_24h', 0) / 1e9  # Normalize to billions
            volatility_score = candidate.get('volatility_24h', 0)
            candidate['scan_score'] = volume_score * volatility_score

        all_candidates.sort(key=lambda x: x.get('scan_score', 0), reverse=True)

        # Prepare output
        scan_output = {
            'timestamp': datetime.now().isoformat(),
            'scan_duration_seconds': (datetime.now() - start_time).total_seconds(),
            'total_scanned': len(self.sp500_symbols) + 100,
            'total_passed': len(all_candidates),
            'stocks_passed': len(stocks),
            'cryptos_passed': len(cryptos),
            'candidates': all_candidates,
            'thresholds': self.thresholds
        }

        # Save to repository
        output_file = self.repository_path / "scanner_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scan_output, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print(f"✅ SCAN COMPLETE")
        print(f"   Duration: {scan_output['scan_duration_seconds']:.1f}s")
        print(f"   Total Scanned: {scan_output['total_scanned']}")
        print(f"   Total Passed: {scan_output['total_passed']}")
        print(f"     - Stocks: {scan_output['stocks_passed']}")
        print(f"     - Cryptos: {scan_output['cryptos_passed']}")
        print(f"   Output: {output_file}")
        print("=" * 80)

        return scan_output

    def integrate_with_learning_loop(self, scan_results: Dict[str, Any]):
        """
        Integration hook for the agent self-learning loop
        This function is called after scanning to feed data into the learning system
        """
        print("\n🔗 Integrating with Agent Self-Learning Loop...")

        try:
            # The fusion agent will automatically pick up scanner_output.json
            # We can optionally trigger it here
            print("   ✅ Scanner output saved for fusion agent")
            print("   💡 Run fusion_agent.py to process these signals")

            # Optionally trigger fusion agent automatically
            # import subprocess
            # subprocess.run([sys.executable, "fusion_agent.py"], cwd=Path(__file__).parent)

        except Exception as e:
            print(f"   ⚠️  Integration warning: {e}")


def main():
    """
    Main execution function
    """
    scanner = ComprehensiveScanner()
    scan_results = scanner.run_comprehensive_scan()
    scanner.integrate_with_learning_loop(scan_results)


if __name__ == "__main__":
    main()

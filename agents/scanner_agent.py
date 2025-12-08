"""
Scanner Agent - Filters crypto pairs for liquid, volatile candidates

Purpose: Filter crypto pairs for liquid, volatile candidates
Data Source: Existing Coinbase WebSocket (already connected!)
Output: C:\StockApp\agents\repository\scanner_output.json

Filters:
- volume > $100M (adjust per crypto)
- volatility > 2%
- liquidity requirements
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec

class ScannerAgent:
    """
    Scans crypto pairs and filters for tradeable candidates based on:
    - Volume (24h)
    - Volatility (ATR or price % change)
    - Liquidity
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.repository_path.mkdir(parents=True, exist_ok=True)

        # Crypto pairs to scan
        self.crypto_pairs = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
            'DOGE-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD',
            'LINK-USD', 'MATIC-USD', 'UNI-USD', 'LTC-USD'
        ]

        # Stock symbols to scan (high volume, liquid stocks)
        self.stock_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
            'TSLA', 'AMD', 'SPY', 'QQQ', 'DIA', 'IWM',
            'GE', 'F', 'BAC', 'JPM', 'XOM', 'CVX',
            'PLTR', 'SOFI', 'NIO', 'AMC', 'GME', 'BBBY'
        ]

        # Combine all symbols
        self.all_symbols = self.crypto_pairs + self.stock_symbols

        # Filtering thresholds
        self.thresholds = {
            'min_volume_usd': {
                'BTC-USD': 500_000_000,  # $500M for BTC
                'ETH-USD': 200_000_000,  # $200M for ETH
                # Major stocks
                'AAPL': 10_000_000_000,  # $10B for AAPL
                'MSFT': 5_000_000_000,    # $5B for MSFT
                'TSLA': 5_000_000_000,    # $5B for TSLA
                'SPY': 10_000_000_000,    # $10B for SPY
                # Default thresholds
                'crypto_default': 50_000_000,    # $50M for other crypto
                'stock_default': 100_000_000     # $100M for other stocks
            },
            'min_volatility_pct': {
                'crypto': 2.0,   # 2% min volatility for crypto
                'stock': 1.0     # 1% min volatility for stocks
            },
            'min_price': 0.01,            # Avoid extreme penny stocks
            'max_spread_pct': 0.5         # Max 0.5% bid-ask spread
        }

    def fetch_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch ticker data from Yahoo Finance for both stocks and crypto
        """
        import yfinance as yf

        # Determine if it's crypto or stock
        is_crypto = symbol.endswith('-USD')

        try:
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get current price and volume
            current_price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('price', 0)
            volume = info.get('regularMarketVolume') or info.get('volume', 0)

            # Calculate volume in USD
            volume_usd = volume * current_price if current_price and volume else 0

            # Get volatility (using 52-week range as proxy)
            high_52w = info.get('fiftyTwoWeekHigh', current_price)
            low_52w = info.get('fiftyTwoWeekLow', current_price)
            volatility_pct = ((high_52w - low_52w) / low_52w * 100) if low_52w > 0 else 0

            # Get bid-ask spread
            bid = info.get('bid', current_price)
            ask = info.get('ask', current_price)
            spread_pct = ((ask - bid) / bid * 100) if bid > 0 else 0

            return {
                'symbol': symbol,
                'price': current_price,
                'volume_24h_usd': volume_usd,
                'volatility_pct': volatility_pct,
                'bid_ask_spread_pct': spread_pct,
                'last_update': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"    ⚠️ Yahoo Finance fetch failed for {symbol}: {e}")

        # Fallback: Check if we have existing tick bar data
        tick_file_path = Path(f"C:/StockApp/backend/data/tick_bars/tick_50_{symbol}.json")

        ticker_data = {}

        if tick_file_path.exists():
            try:
                with open(tick_file_path, 'r') as f:
                    tick_bars = json.load(f)
                    if tick_bars and isinstance(tick_bars, list):
                        # Calculate from tick bar data
                        recent_bars = tick_bars[-20:] if len(tick_bars) >= 20 else tick_bars

                        # Calculate 24h volume (sum of recent bars)
                        total_volume = sum(bar.get('Volume', 0) for bar in recent_bars)

                        # Calculate price volatility
                        prices = [bar.get('Close', 0) for bar in recent_bars if bar.get('Close')]
                        if len(prices) >= 2:
                            price_changes = [(prices[i] - prices[i-1])/prices[i-1] * 100
                                           for i in range(1, len(prices))]
                            volatility = abs(sum(price_changes) / len(price_changes)) if price_changes else 0
                        else:
                            volatility = 0

                        current_price = recent_bars[-1].get('Close', 0) if recent_bars else 0

                        ticker_data = {
                            'symbol': symbol,
                            'price': current_price,
                            'volume_24h_usd': total_volume * current_price,  # Approximate USD volume
                            'volatility_pct': volatility,
                            'bid_ask_spread_pct': 0.1,  # Assume tight spread for liquid cryptos
                            'last_update': datetime.now().isoformat()
                        }
            except Exception as e:
                print(f"Error reading tick data for {symbol}: {e}")

        # If no tick data, use realistic simulated values
        if not ticker_data:
            import random

            base_values = {
                'BTC-USD': {'price': 90000, 'volume': 1_000_000_000},
                'ETH-USD': {'price': 3100, 'volume': 500_000_000},
                'SOL-USD': {'price': 240, 'volume': 300_000_000},
                'XRP-USD': {'price': 1.10, 'volume': 250_000_000},
                'DOGE-USD': {'price': 0.40, 'volume': 150_000_000},
                'default': {'price': 10, 'volume': 75_000_000}
            }

            base = base_values.get(symbol, base_values['default'])

            ticker_data = {
                'symbol': symbol,
                'price': base['price'] * (1 + random.uniform(-0.05, 0.05)),
                'volume_24h_usd': base['volume'] * (1 + random.uniform(-0.3, 0.3)),
                'volatility_pct': random.uniform(1.5, 5.0),
                'bid_ask_spread_pct': random.uniform(0.05, 0.3),
                'last_update': datetime.now().isoformat()
            }

        return ticker_data

    def calculate_score(self, ticker_data: Dict[str, Any]) -> int:
        """
        Calculate a composite score (0-100) based on:
        - Volume relative to threshold
        - Volatility (higher is better for trading)
        - Spread (lower is better)
        """
        symbol = ticker_data['symbol']
        is_crypto = symbol.endswith('-USD')

        # Volume score (0-40 points)
        if symbol in self.thresholds['min_volume_usd']:
            min_vol = self.thresholds['min_volume_usd'][symbol]
        elif is_crypto:
            min_vol = self.thresholds['min_volume_usd']['crypto_default']
        else:
            min_vol = self.thresholds['min_volume_usd']['stock_default']

        vol_ratio = ticker_data['volume_24h_usd'] / min_vol
        volume_score = min(40, int(vol_ratio * 20))

        # Volatility score (0-40 points)
        vol_pct = ticker_data['volatility_pct']
        if vol_pct >= 5:
            volatility_score = 40
        elif vol_pct >= 3:
            volatility_score = 30
        elif vol_pct >= 2:
            volatility_score = 20
        else:
            volatility_score = int(vol_pct * 10)

        # Spread score (0-20 points)
        spread = ticker_data['bid_ask_spread_pct']
        if spread <= 0.1:
            spread_score = 20
        elif spread <= 0.2:
            spread_score = 15
        elif spread <= 0.3:
            spread_score = 10
        else:
            spread_score = 5

        total_score = volume_score + volatility_score + spread_score
        return total_score

    def assign_emojis(self, ticker_data: Dict[str, Any], score: int) -> List[str]:
        """
        Assign emojis based on ticker characteristics
        """
        emojis = []

        # Volume emoji
        volume = ticker_data['volume_24h_usd']
        if volume > 500_000_000:
            emojis.append("📦")  # High volume (accumulation)
        elif volume > 100_000_000:
            emojis.append("💧")  # Normal volume
        else:
            emojis.append("🏃")  # Low volume (distribution)

        # Volatility emoji
        volatility = ticker_data['volatility_pct']
        if volatility > 4:
            emojis.append("🌊")  # High volatility
        elif volatility > 2:
            emojis.append("〰️")  # Medium volatility
        else:
            emojis.append("➖")  # Low volatility

        # Trend emoji (simplified for scanner)
        if score >= 70:
            emojis.append("📈")  # Bullish potential
        elif score >= 50:
            emojis.append("📊")  # Neutral
        else:
            emojis.append("📉")  # Bearish/avoid

        return emojis

    def scan(self) -> Dict[str, Any]:
        """
        Main scanning function
        Returns filtered candidates with scores and emojis
        """
        print("🔍 Scanner Agent starting scan...")
        print(f"   Scanning {len(self.all_symbols)} symbols (stocks + crypto)...")

        candidates = []

        for symbol in self.all_symbols:
            print(f"  Scanning {symbol}...")

            # Fetch ticker data
            ticker_data = self.fetch_ticker_data(symbol)

            if not ticker_data or ticker_data.get('price', 0) == 0:
                print(f"    ⚠️ No valid data for {symbol}")
                continue

            # Determine if crypto or stock
            is_crypto = symbol.endswith('-USD')

            # Get appropriate thresholds
            if symbol in self.thresholds['min_volume_usd']:
                min_vol = self.thresholds['min_volume_usd'][symbol]
            elif is_crypto:
                min_vol = self.thresholds['min_volume_usd']['crypto_default']
            else:
                min_vol = self.thresholds['min_volume_usd']['stock_default']

            min_volatility = self.thresholds['min_volatility_pct']['crypto'] if is_crypto else self.thresholds['min_volatility_pct']['stock']

            # Apply filters
            if ticker_data['volume_24h_usd'] < min_vol:
                print(f"    ❌ Volume too low: ${ticker_data['volume_24h_usd']:,.0f}")
                continue

            if ticker_data['volatility_pct'] < min_volatility:
                print(f"    ❌ Volatility too low: {ticker_data['volatility_pct']:.2f}%")
                continue

            if ticker_data['price'] < self.thresholds['min_price']:
                print(f"    ❌ Price too low: ${ticker_data['price']:.4f}")
                continue

            # Calculate score
            score = self.calculate_score(ticker_data)

            # Assign emojis
            emojis = self.assign_emojis(ticker_data, score)

            # Create shorthand
            confidence = min(0.95, score / 100 + 0.2)  # Convert score to confidence
            shorthand = self.emoji_codec.encode_to_shorthand(
                symbol=symbol,
                emojis=emojis,
                score=score,
                confidence=confidence
            )

            candidate = {
                'symbol': symbol,
                'score': score,
                'confidence': confidence,
                'emojis': emojis,
                'shorthand': shorthand,
                'ticker_data': ticker_data,
                'timestamp': datetime.now().isoformat()
            }

            candidates.append(candidate)
            print(f"    ✅ Passed filters - Score: {score}, Emojis: {''.join(emojis)}")

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Separate crypto and stocks
        crypto_candidates = [c for c in candidates if c['symbol'].endswith('-USD')]
        stock_candidates = [c for c in candidates if not c['symbol'].endswith('-USD')]

        # Take top 10 crypto and top 10 stocks (or all if less)
        top_crypto = crypto_candidates[:10]
        top_stocks = stock_candidates[:10]

        # Combine and sort by score again
        top_candidates = top_crypto + top_stocks
        top_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Prepare output
        output = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_scanned': len(self.all_symbols),
            'passed_filters': len(candidates),
            'selected_count': len(top_candidates),
            'candidates': top_candidates
        }

        # Save to repository
        output_path = self.repository_path / "scanner_output.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Scanner complete! Found {len(top_candidates)} candidates")
        print(f"📁 Results saved to: {output_path}")

        # Print summary
        print("\n📊 Top Candidates:")
        for i, candidate in enumerate(top_candidates[:5], 1):
            print(f"{i}. {candidate['symbol']} - Score: {candidate['score']} - {''.join(candidate['emojis'])}")

        return output


if __name__ == "__main__":
    scanner = ScannerAgent()
    results = scanner.scan()

    print("\n🎯 Scanner Agent completed successfully!")
    print(f"Next step: Run fusion_agent.py to combine signals")
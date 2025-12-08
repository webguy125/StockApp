"""
Indicators Agent - Calculates technical indicators for analysis

Purpose: Calculate RSI, MACD, MAs, ATR from existing price data
Data Source: Yahoo Finance via yfinance
Output: C:\StockApp\agents\repository\indicators_agent\<symbol>.json

Indicators:
- RSI (14-period)
- MACD (12, 26, 9)
- Moving Averages (20, 50, 200)
- ATR (14-period)
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class IndicatorsAgent:
    """
    Calculates technical indicators and generates trading signals.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository/indicators_agent")
        self.repository_path.mkdir(parents=True, exist_ok=True)

        # Indicator parameters
        self.params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ma_short': 20,
            'ma_medium': 50,
            'ma_long': 200,
            'atr_period': 14
        }

        # Signal thresholds
        self.thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'atr_high_volatility': 0.02  # 2% of price
        }

    def fetch_price_data(self, symbol: str, period: str = '3mo') -> Optional[pd.DataFrame]:
        """Fetch price data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                print(f"  ⚠️ No price data available for {symbol}")
                return None

            return data
        except Exception as e:
            print(f"  ❌ Error fetching data for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series,
                      fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()

        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        })

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def analyze_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze all indicators and generate signals"""
        close = data['Close']

        # Calculate indicators
        rsi = self.calculate_rsi(close, self.params['rsi_period'])
        macd_df = self.calculate_macd(close,
                                     self.params['macd_fast'],
                                     self.params['macd_slow'],
                                     self.params['macd_signal'])
        atr = self.calculate_atr(data, self.params['atr_period'])

        # Moving averages
        ma_20 = close.rolling(window=self.params['ma_short']).mean()
        ma_50 = close.rolling(window=self.params['ma_medium']).mean()
        ma_200 = close.rolling(window=self.params['ma_long']).mean() if len(close) >= 200 else None

        # Get latest values
        latest = {
            'price': close.iloc[-1],
            'rsi': rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None,
            'macd': macd_df['macd'].iloc[-1] if not macd_df.empty else None,
            'macd_signal': macd_df['signal'].iloc[-1] if not macd_df.empty else None,
            'macd_histogram': macd_df['histogram'].iloc[-1] if not macd_df.empty else None,
            'ma_20': ma_20.iloc[-1] if not ma_20.empty and not pd.isna(ma_20.iloc[-1]) else None,
            'ma_50': ma_50.iloc[-1] if not ma_50.empty and not pd.isna(ma_50.iloc[-1]) else None,
            'ma_200': ma_200.iloc[-1] if ma_200 is not None and not ma_200.empty and not pd.isna(ma_200.iloc[-1]) else None,
            'atr': atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else None,
            'atr_pct': (atr.iloc[-1] / close.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else None
        }

        # Generate signals
        signals = {
            'rsi_signal': self._get_rsi_signal(latest['rsi']),
            'macd_signal': self._get_macd_signal(latest['macd'], latest['macd_signal']),
            'ma_signal': self._get_ma_signal(latest['price'], latest['ma_20'], latest['ma_50'], latest['ma_200']),
            'trend': self._get_trend(close),
            'volatility': 'high' if latest['atr_pct'] and latest['atr_pct'] > self.thresholds['atr_high_volatility'] else 'normal'
        }

        return {
            'indicators': latest,
            'signals': signals
        }

    def _get_rsi_signal(self, rsi: Optional[float]) -> str:
        """Interpret RSI value"""
        if rsi is None:
            return 'unknown'
        elif rsi < self.thresholds['rsi_oversold']:
            return 'oversold'
        elif rsi > self.thresholds['rsi_overbought']:
            return 'overbought'
        else:
            return 'neutral'

    def _get_macd_signal(self, macd: Optional[float], signal: Optional[float]) -> str:
        """Interpret MACD crossover"""
        if macd is None or signal is None:
            return 'unknown'
        elif macd > signal:
            return 'bullish'
        else:
            return 'bearish'

    def _get_ma_signal(self, price: float, ma20: Optional[float],
                      ma50: Optional[float], ma200: Optional[float]) -> str:
        """Interpret moving average positions"""
        if ma20 is None or ma50 is None:
            return 'unknown'

        # Check for golden/death cross
        if ma200 is not None:
            if ma50 > ma200 and ma20 > ma50:
                return 'strong_bullish'  # Golden cross
            elif ma50 < ma200 and ma20 < ma50:
                return 'strong_bearish'  # Death cross

        # Check price position relative to MAs
        if price > ma20 and price > ma50:
            return 'bullish'
        elif price < ma20 and price < ma50:
            return 'bearish'
        else:
            return 'neutral'

    def _get_trend(self, prices: pd.Series, lookback: int = 20) -> str:
        """Determine price trend over recent period"""
        if len(prices) < lookback:
            return 'unknown'

        recent = prices.iloc[-lookback:]
        start = recent.iloc[0]
        end = recent.iloc[-1]
        change_pct = ((end - start) / start) * 100

        if change_pct > 5:
            return 'strong_uptrend'
        elif change_pct > 1:
            return 'uptrend'
        elif change_pct < -5:
            return 'strong_downtrend'
        elif change_pct < -1:
            return 'downtrend'
        else:
            return 'sideways'

    def calculate_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate composite score from indicators"""
        score = 50  # Base score
        signals = analysis['signals']
        indicators = analysis['indicators']

        # RSI contribution (0-20 points)
        if signals['rsi_signal'] == 'oversold':
            score += 15  # Potential bounce
        elif signals['rsi_signal'] == 'overbought':
            score -= 10  # Potential reversal

        # MACD contribution (0-20 points)
        if signals['macd_signal'] == 'bullish':
            score += 15
        elif signals['macd_signal'] == 'bearish':
            score -= 15

        # MA contribution (0-25 points)
        if signals['ma_signal'] == 'strong_bullish':
            score += 25
        elif signals['ma_signal'] == 'bullish':
            score += 15
        elif signals['ma_signal'] == 'bearish':
            score -= 15
        elif signals['ma_signal'] == 'strong_bearish':
            score -= 25

        # Trend contribution (0-15 points)
        if signals['trend'] == 'strong_uptrend':
            score += 15
        elif signals['trend'] == 'uptrend':
            score += 10
        elif signals['trend'] == 'downtrend':
            score -= 10
        elif signals['trend'] == 'strong_downtrend':
            score -= 15

        # Volatility adjustment
        if signals['volatility'] == 'high':
            score = int(score * 0.9)  # Reduce confidence in high volatility

        return max(0, min(100, score))

    def assign_emojis(self, analysis: Dict[str, Any]) -> List[str]:
        """Assign emojis based on indicator signals"""
        emojis = []
        signals = analysis['signals']

        # Trend emoji
        if signals['trend'] in ['strong_uptrend', 'uptrend']:
            emojis.append("📈")
        elif signals['trend'] in ['strong_downtrend', 'downtrend']:
            emojis.append("📉")
        else:
            emojis.append("➖")

        # RSI emoji
        if signals['rsi_signal'] == 'oversold':
            emojis.append("🔵")  # Blue = oversold (buy signal)
        elif signals['rsi_signal'] == 'overbought':
            emojis.append("🔴")  # Red = overbought (sell signal)
        else:
            emojis.append("⚪")  # White = neutral

        # MACD emoji
        if signals['macd_signal'] == 'bullish':
            emojis.append("✅")  # Green check = bullish crossover
        elif signals['macd_signal'] == 'bearish':
            emojis.append("❌")  # Red X = bearish crossover
        else:
            emojis.append("⏸️")  # Pause = no clear signal

        # MA emoji
        if signals['ma_signal'] == 'strong_bullish':
            emojis.append("🌟")  # Golden cross
        elif signals['ma_signal'] == 'strong_bearish':
            emojis.append("💀")  # Death cross
        elif signals['ma_signal'] == 'bullish':
            emojis.append("☀️")  # Above MAs
        elif signals['ma_signal'] == 'bearish':
            emojis.append("🌧️")  # Below MAs
        else:
            emojis.append("☁️")  # Mixed signals

        return emojis

    def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol's indicators"""
        print(f"  Analyzing indicators for {symbol}...")

        # Fetch price data
        data = self.fetch_price_data(symbol)

        if data is None or data.empty:
            print(f"    ❌ No data available for {symbol}")
            return None

        # Perform analysis
        analysis = self.analyze_indicators(data)

        # Calculate score and assign emojis
        score = self.calculate_score(analysis)
        emojis = self.assign_emojis(analysis)
        confidence = min(0.95, score / 100 + 0.3)

        # Create shorthand
        shorthand = self.emoji_codec.encode_to_shorthand(
            symbol=symbol,
            emojis=emojis,
            score=score,
            confidence=confidence
        )

        # Prepare output
        output = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'agent': 'indicators_agent',
            'signals': {
                'emoji': ''.join(emojis),
                'score': score,
                'confidence': confidence,
                'metadata': analysis
            },
            'shorthand': shorthand
        }

        # Print summary
        indicators = analysis['indicators']
        signals = analysis['signals']

        print(f"    ✅ Score: {score}, Emojis: {''.join(emojis)}")
        if indicators['rsi'] is not None:
            print(f"       RSI: {indicators['rsi']:.2f} ({signals['rsi_signal']})")
        if indicators['macd'] is not None:
            print(f"       MACD: {signals['macd_signal']}")
        print(f"       Trend: {signals['trend']}")
        print(f"       MA Signal: {signals['ma_signal']}")

        return output

    def analyze_candidates(self, scanner_output_path: str = None) -> Dict[str, Any]:
        """Analyze all candidates from scanner output"""
        # Load scanner output
        if not scanner_output_path:
            scanner_output_path = "C:/StockApp/agents/repository/scanner_output.json"

        try:
            with open(scanner_output_path, 'r', encoding='utf-8') as f:
                scanner_data = json.load(f)
        except Exception as e:
            print(f"❌ Could not load scanner output: {e}")
            return {'error': str(e)}

        candidates = scanner_data.get('candidates', [])

        if not candidates:
            print("❌ No candidates found in scanner output")
            return {'error': 'No candidates'}

        print(f"🔍 Indicators Agent analyzing {len(candidates)} candidates...")

        results = []
        for candidate in candidates:  # Process all candidates
            symbol = candidate['symbol']
            analysis = self.analyze_symbol(symbol)

            if analysis:
                # Save individual symbol analysis
                symbol_file = self.repository_path / f"{symbol}.json"
                with open(symbol_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)

                results.append(analysis)

        # Save aggregate results
        aggregate_output = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'indicators_agent',
            'symbols_analyzed': len(results),
            'analyses': results
        }

        aggregate_file = self.repository_path / "indicators_analysis.json"
        with open(aggregate_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate_output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Indicators Agent complete! Analyzed {len(results)} symbols")
        print(f"📁 Results saved to: {self.repository_path}")

        # Print summary
        print("\n📊 Indicators Analysis Summary:")
        for i, result in enumerate(results, 1):
            symbol = result['symbol']
            score = result['signals']['score']
            emojis = result['signals']['emoji']
            print(f"{i}. {symbol} - Score: {score} - {emojis}")

        return aggregate_output


if __name__ == "__main__":
    agent = IndicatorsAgent()
    results = agent.analyze_candidates()

    print("\n🎯 Indicators Agent completed successfully!")
    print(f"Next step: Run volume_agent.py for Ord/Weis analysis")
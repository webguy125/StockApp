"""
Volume Agent - Analyzes volume patterns using ORD/Weis Wave methodology

Purpose: Analyze volume for accumulation/distribution patterns
Data Source: Yahoo Finance via yfinance
Output: C:\StockApp\agents\repository\volume_agent\<symbol>.json

Analysis:
- ORD Volume (Order Flow)
- Weis Wave Volume
- Accumulation/Distribution
- Volume Profile
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class VolumeAgent:
    """
    Analyzes volume patterns using ORD Volume and Weis Wave methodology.
    Detects accumulation/distribution and volume imbalances.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository/volume_agent")
        self.repository_path.mkdir(parents=True, exist_ok=True)

        # Analysis parameters
        self.params = {
            'lookback': 20,  # Bars to analyze
            'volume_spike_threshold': 1.5,  # 150% of average
            'weis_sensitivity': 0.001,  # 0.1% price change for new wave
        }

    def fetch_price_data(self, symbol: str, period: str = '1mo') -> Optional[pd.DataFrame]:
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

    def detect_swing_points(self, data: pd.DataFrame, lookback: int = 2) -> Dict[str, List]:
        """Detect swing highs and lows for Weis Wave analysis"""
        highs = []
        lows = []

        for i in range(lookback, len(data) - lookback):
            # Check for swing high
            is_high = True
            for j in range(1, lookback + 1):
                if data['High'].iloc[i] <= data['High'].iloc[i - j] or \
                   data['High'].iloc[i] <= data['High'].iloc[i + j]:
                    is_high = False
                    break

            if is_high:
                highs.append(i)

            # Check for swing low
            is_low = True
            for j in range(1, lookback + 1):
                if data['Low'].iloc[i] >= data['Low'].iloc[i - j] or \
                   data['Low'].iloc[i] >= data['Low'].iloc[i + j]:
                    is_low = False
                    break

            if is_low:
                lows.append(i)

        return {'highs': highs, 'lows': lows}

    def calculate_weis_waves(self, data: pd.DataFrame) -> List[Dict]:
        """Calculate Weis Wave volume aggregations"""
        waves = []
        current_wave = None
        threshold = self.params['weis_sensitivity']

        for i in range(1, len(data)):
            price_change = (data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i-1]

            if current_wave is None:
                # Start first wave
                current_wave = {
                    'start_idx': i-1,
                    'direction': 'up' if price_change > 0 else 'down',
                    'volume': data['Volume'].iloc[i-1] + data['Volume'].iloc[i],
                    'price_change': price_change
                }
            else:
                # Check if we need to start a new wave
                if (current_wave['direction'] == 'up' and price_change < -threshold) or \
                   (current_wave['direction'] == 'down' and price_change > threshold):
                    # End current wave, start new one
                    current_wave['end_idx'] = i-1
                    current_wave['bars'] = current_wave['end_idx'] - current_wave['start_idx'] + 1
                    waves.append(current_wave)

                    current_wave = {
                        'start_idx': i,
                        'direction': 'up' if price_change > 0 else 'down',
                        'volume': data['Volume'].iloc[i],
                        'price_change': price_change
                    }
                else:
                    # Continue current wave
                    current_wave['volume'] += data['Volume'].iloc[i]
                    current_wave['price_change'] += price_change

        # Add final wave
        if current_wave:
            current_wave['end_idx'] = len(data) - 1
            current_wave['bars'] = current_wave['end_idx'] - current_wave['start_idx'] + 1
            waves.append(current_wave)

        return waves

    def calculate_ord_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ORD Volume (Order Flow) metrics"""
        # Calculate price and volume changes
        price_changes = data['Close'].pct_change()
        volume_data = data['Volume'].values

        # Separate buy and sell volume (simplified)
        buy_volume = []
        sell_volume = []

        for i in range(1, len(data)):
            if price_changes.iloc[i] > 0:
                # Up bar - mostly buying
                buy_volume.append(volume_data[i] * 0.7)
                sell_volume.append(volume_data[i] * 0.3)
            elif price_changes.iloc[i] < 0:
                # Down bar - mostly selling
                buy_volume.append(volume_data[i] * 0.3)
                sell_volume.append(volume_data[i] * 0.7)
            else:
                # No change - split evenly
                buy_volume.append(volume_data[i] * 0.5)
                sell_volume.append(volume_data[i] * 0.5)

        # Calculate cumulative delta
        cumulative_delta = np.cumsum(np.array(buy_volume) - np.array(sell_volume))

        # Calculate recent metrics
        recent_buy = sum(buy_volume[-self.params['lookback']:]) if buy_volume else 0
        recent_sell = sum(sell_volume[-self.params['lookback']:]) if sell_volume else 0
        recent_delta = recent_buy - recent_sell

        return {
            'recent_buy_volume': recent_buy,
            'recent_sell_volume': recent_sell,
            'recent_delta': recent_delta,
            'delta_trend': 'bullish' if recent_delta > 0 else 'bearish',
            'buy_sell_ratio': recent_buy / recent_sell if recent_sell > 0 else float('inf'),
            'cumulative_delta': cumulative_delta[-1] if len(cumulative_delta) > 0 else 0
        }

    def analyze_accumulation_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze accumulation/distribution patterns"""
        # Calculate Money Flow Multiplier
        mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mfm = mfm.fillna(0)  # Handle division by zero

        # Calculate Money Flow Volume
        mfv = mfm * data['Volume']

        # Calculate Accumulation/Distribution Line
        ad_line = mfv.cumsum()

        # Analyze recent trend
        if len(ad_line) >= self.params['lookback']:
            recent_ad = ad_line.iloc[-self.params['lookback']:]
            ad_slope = (recent_ad.iloc[-1] - recent_ad.iloc[0]) / self.params['lookback']

            if ad_slope > 0:
                ad_signal = 'accumulation'
            elif ad_slope < 0:
                ad_signal = 'distribution'
            else:
                ad_signal = 'neutral'
        else:
            ad_slope = 0
            ad_signal = 'unknown'

        # Check for divergences
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-self.params['lookback']]) / data['Close'].iloc[-self.params['lookback']]
        ad_change = ad_slope * self.params['lookback'] / abs(ad_line.iloc[-1]) if ad_line.iloc[-1] != 0 else 0

        divergence = None
        if price_change > 0 and ad_change < 0:
            divergence = 'bearish_divergence'  # Price up, AD down
        elif price_change < 0 and ad_change > 0:
            divergence = 'bullish_divergence'  # Price down, AD up

        return {
            'signal': ad_signal,
            'slope': ad_slope,
            'current_value': ad_line.iloc[-1] if len(ad_line) > 0 else 0,
            'divergence': divergence
        }

    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume distribution across price levels"""
        # Create price bins
        price_range = data['High'].max() - data['Low'].min()
        num_bins = 20
        bin_size = price_range / num_bins

        volume_profile = {}
        for i in range(num_bins):
            lower = data['Low'].min() + i * bin_size
            upper = lower + bin_size

            # Find bars in this price range
            mask = ((data['High'] >= lower) & (data['Low'] <= upper))
            volume_in_range = data.loc[mask, 'Volume'].sum()

            volume_profile[f"{lower:.2f}-{upper:.2f}"] = volume_in_range

        # Find Point of Control (highest volume price)
        poc_range = max(volume_profile, key=volume_profile.get)
        poc_volume = volume_profile[poc_range]
        total_volume = sum(volume_profile.values())

        # Determine if current price is near POC
        current_price = data['Close'].iloc[-1]
        poc_lower = float(poc_range.split('-')[0])
        poc_upper = float(poc_range.split('-')[1])
        near_poc = poc_lower <= current_price <= poc_upper

        return {
            'poc_range': poc_range,
            'poc_volume_pct': (poc_volume / total_volume * 100) if total_volume > 0 else 0,
            'near_poc': near_poc,
            'current_price': current_price
        }

    def calculate_score(self, ord_volume: Dict, ad_analysis: Dict,
                       weis_waves: List[Dict], volume_profile: Dict) -> int:
        """Calculate composite score from volume analysis"""
        score = 50  # Base score

        # ORD Volume contribution (0-25 points)
        if ord_volume['delta_trend'] == 'bullish':
            if ord_volume['buy_sell_ratio'] > 2:
                score += 25
            elif ord_volume['buy_sell_ratio'] > 1.5:
                score += 15
            else:
                score += 10
        else:
            if ord_volume['buy_sell_ratio'] < 0.5:
                score -= 25
            elif ord_volume['buy_sell_ratio'] < 0.67:
                score -= 15
            else:
                score -= 10

        # A/D contribution (0-20 points)
        if ad_analysis['signal'] == 'accumulation':
            score += 15
            if ad_analysis['divergence'] == 'bullish_divergence':
                score += 10  # Bonus for bullish divergence
        elif ad_analysis['signal'] == 'distribution':
            score -= 15
            if ad_analysis['divergence'] == 'bearish_divergence':
                score -= 10  # Penalty for bearish divergence

        # Weis Wave contribution (0-15 points)
        if weis_waves and len(weis_waves) >= 2:
            last_wave = weis_waves[-1]
            prev_wave = weis_waves[-2]

            # Check if volume is increasing in trend direction
            if last_wave['direction'] == 'up' and last_wave['volume'] > prev_wave['volume']:
                score += 15
            elif last_wave['direction'] == 'down' and last_wave['volume'] > prev_wave['volume']:
                score -= 15

        # Volume Profile contribution (0-10 points)
        if volume_profile['near_poc']:
            score += 5  # Near high volume node (support/resistance)

        return max(0, min(100, score))

    def assign_emojis(self, ord_volume: Dict, ad_analysis: Dict,
                     weis_waves: List[Dict], volume_profile: Dict) -> List[str]:
        """Assign emojis based on volume analysis"""
        emojis = []

        # ORD Volume emoji
        if ord_volume['buy_sell_ratio'] > 1.5:
            emojis.append("📦")  # High buy volume (accumulation)
        elif ord_volume['buy_sell_ratio'] < 0.67:
            emojis.append("📤")  # High sell volume (distribution)
        else:
            emojis.append("⚖️")  # Balanced volume

        # A/D emoji
        if ad_analysis['signal'] == 'accumulation':
            emojis.append("💰")  # Accumulation
        elif ad_analysis['signal'] == 'distribution':
            emojis.append("💸")  # Distribution
        else:
            emojis.append("〰️")  # Neutral

        # Divergence emoji
        if ad_analysis['divergence'] == 'bullish_divergence':
            emojis.append("🔄⬆️")  # Bullish divergence
        elif ad_analysis['divergence'] == 'bearish_divergence':
            emojis.append("🔄⬇️")  # Bearish divergence

        # Weis Wave emoji
        if weis_waves and len(weis_waves) >= 2:
            last_wave = weis_waves[-1]
            if last_wave['direction'] == 'up':
                emojis.append("🌊⬆️")  # Up wave
            else:
                emojis.append("🌊⬇️")  # Down wave

        # Volume Profile emoji
        if volume_profile['near_poc']:
            emojis.append("🎯")  # Near point of control

        return emojis

    def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol's volume patterns"""
        print(f"  Analyzing volume for {symbol}...")

        # Fetch price data
        data = self.fetch_price_data(symbol)

        if data is None or data.empty:
            print(f"    ❌ No data available for {symbol}")
            return None

        # Perform various analyses
        ord_volume = self.calculate_ord_volume(data)
        ad_analysis = self.analyze_accumulation_distribution(data)
        weis_waves = self.calculate_weis_waves(data)
        volume_profile = self.analyze_volume_profile(data)

        # Calculate score and assign emojis
        score = self.calculate_score(ord_volume, ad_analysis, weis_waves, volume_profile)
        emojis = self.assign_emojis(ord_volume, ad_analysis, weis_waves, volume_profile)
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
            'agent': 'volume_agent',
            'signals': {
                'emoji': ''.join(emojis),
                'score': score,
                'confidence': confidence,
                'metadata': {
                    'ord_volume': ord_volume,
                    'accumulation_distribution': ad_analysis,
                    'weis_waves': {
                        'total_waves': len(weis_waves),
                        'last_wave': weis_waves[-1] if weis_waves else None
                    },
                    'volume_profile': volume_profile
                }
            },
            'shorthand': shorthand
        }

        # Print summary
        print(f"    ✅ Score: {score}, Emojis: {''.join(emojis)}")
        print(f"       ORD: {ord_volume['delta_trend']} (B/S ratio: {ord_volume['buy_sell_ratio']:.2f})")
        print(f"       A/D: {ad_analysis['signal']}")
        if ad_analysis['divergence']:
            print(f"       ⚠️ {ad_analysis['divergence']}")
        if weis_waves:
            print(f"       Weis: {weis_waves[-1]['direction']} wave")

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

        print(f"🔍 Volume Agent analyzing {len(candidates)} candidates...")

        results = []
        for candidate in candidates:  # Process all candidates
            symbol = candidate['symbol']
            analysis = self.analyze_symbol(symbol)

            if analysis:
                # Save individual symbol analysis
                symbol_file = self.repository_path / f"{symbol}.json"
                with open(symbol_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

                results.append(analysis)

        # Save aggregate results
        aggregate_output = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'volume_agent',
            'symbols_analyzed': len(results),
            'analyses': results
        }

        aggregate_file = self.repository_path / "volume_analysis.json"
        with open(aggregate_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate_output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print(f"\n✅ Volume Agent complete! Analyzed {len(results)} symbols")
        print(f"📁 Results saved to: {self.repository_path}")

        # Print summary
        print("\n📊 Volume Analysis Summary:")
        for i, result in enumerate(results, 1):
            symbol = result['symbol']
            score = result['signals']['score']
            emojis = result['signals']['emoji']
            print(f"{i}. {symbol} - Score: {score} - {emojis}")

        return aggregate_output


if __name__ == "__main__":
    agent = VolumeAgent()
    results = agent.analyze_candidates()

    print("\n🎯 Volume Agent completed successfully!")
    print(f"Next step: Run fusion_agent.py to combine all signals")
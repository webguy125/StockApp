"""
Tick Agent - Analyzes tick bar patterns for momentum and trend signals

Purpose: Analyze existing tick bar data for patterns
Data Source: C:\StockApp\backend\data\tick_bars\
Output: C:\StockApp\agents\repository\tick_agent\<symbol>.json

Analysis:
- 50-tick bars for short-term patterns
- 100-tick bars for trend confirmation
- Volume imbalances
- Momentum shifts
"""

import json
import os
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class TickAgent:
    """
    Analyzes tick bar data for momentum and trend patterns.
    Uses 50-tick and 100-tick bars for different timeframes.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository/tick_agent")
        self.repository_path.mkdir(parents=True, exist_ok=True)

        self.tick_data_path = Path("C:/StockApp/backend/data/tick_bars")

        # Analysis thresholds
        self.thresholds = {
            'trend_bars': 5,  # Number of bars to determine trend
            'volume_spike': 2.0,  # Volume spike multiplier
            'momentum_threshold': 0.001,  # 0.1% price change for momentum
        }

    def load_tick_bars(self, symbol: str, tick_size: int = 50) -> Optional[List[Dict]]:
        """Load tick bar data for a symbol"""
        file_path = self.tick_data_path / f"tick_{tick_size}_{symbol}.json"

        if not file_path.exists():
            print(f"  ⚠️ No tick data found for {symbol} ({tick_size}-tick)")
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else None
        except Exception as e:
            print(f"  ❌ Error loading tick data for {symbol}: {e}")
            return None

    def analyze_trend(self, bars: List[Dict]) -> Dict[str, Any]:
        """Analyze trend from recent bars"""
        if len(bars) < self.thresholds['trend_bars']:
            return {'direction': 'neutral', 'strength': 0}

        recent_bars = bars[-self.thresholds['trend_bars']:]

        # Calculate price changes
        price_changes = []
        for i in range(1, len(recent_bars)):
            change = (recent_bars[i]['Close'] - recent_bars[i-1]['Close']) / recent_bars[i-1]['Close']
            price_changes.append(change)

        # Determine trend direction and strength
        avg_change = sum(price_changes) / len(price_changes) if price_changes else 0

        if avg_change > self.thresholds['momentum_threshold']:
            direction = 'bullish'
            strength = min(100, int(abs(avg_change) * 10000))  # Scale to 0-100
        elif avg_change < -self.thresholds['momentum_threshold']:
            direction = 'bearish'
            strength = min(100, int(abs(avg_change) * 10000))
        else:
            direction = 'neutral'
            strength = int(abs(avg_change) * 5000)

        # Check for consistency
        bullish_bars = sum(1 for change in price_changes if change > 0)
        bearish_bars = sum(1 for change in price_changes if change < 0)

        consistency = max(bullish_bars, bearish_bars) / len(price_changes) if price_changes else 0.5

        return {
            'direction': direction,
            'strength': strength,
            'consistency': consistency,
            'avg_change_pct': avg_change * 100
        }

    def analyze_volume(self, bars: List[Dict]) -> Dict[str, Any]:
        """Analyze volume patterns"""
        if len(bars) < 10:
            return {'pattern': 'insufficient_data', 'intensity': 0}

        recent_bars = bars[-20:] if len(bars) >= 20 else bars
        volumes = [bar['Volume'] for bar in recent_bars]

        if not volumes:
            return {'pattern': 'no_volume', 'intensity': 0}

        avg_volume = statistics.mean(volumes)
        std_volume = statistics.stdev(volumes) if len(volumes) > 1 else 0
        last_volume = volumes[-1]

        # Check for volume spike
        if std_volume > 0:
            z_score = (last_volume - avg_volume) / std_volume
        else:
            z_score = 0

        # Determine volume pattern
        if z_score > 2:
            pattern = 'spike_high'
            intensity = min(100, int(z_score * 20))
        elif z_score < -2:
            pattern = 'spike_low'
            intensity = min(100, int(abs(z_score) * 20))
        elif last_volume > avg_volume * 1.5:
            pattern = 'increasing'
            intensity = 60
        elif last_volume < avg_volume * 0.5:
            pattern = 'decreasing'
            intensity = 40
        else:
            pattern = 'normal'
            intensity = 50

        return {
            'pattern': pattern,
            'intensity': intensity,
            'z_score': z_score,
            'last_vs_avg': last_volume / avg_volume if avg_volume > 0 else 1
        }

    def analyze_momentum(self, bars_50: List[Dict], bars_100: List[Dict]) -> Dict[str, Any]:
        """Analyze momentum using multiple tick bar sizes"""
        momentum = {}

        # Short-term momentum (50-tick)
        if bars_50 and len(bars_50) >= 3:
            recent = bars_50[-3:]
            price_change = (recent[-1]['Close'] - recent[0]['Close']) / recent[0]['Close']

            if price_change > 0.002:  # 0.2%
                momentum['short_term'] = 'strong_bullish'
            elif price_change > 0.001:
                momentum['short_term'] = 'bullish'
            elif price_change < -0.002:
                momentum['short_term'] = 'strong_bearish'
            elif price_change < -0.001:
                momentum['short_term'] = 'bearish'
            else:
                momentum['short_term'] = 'neutral'

            momentum['short_term_change'] = price_change * 100
        else:
            momentum['short_term'] = 'unknown'
            momentum['short_term_change'] = 0

        # Medium-term momentum (100-tick)
        if bars_100 and len(bars_100) >= 3:
            recent = bars_100[-3:]
            price_change = (recent[-1]['Close'] - recent[0]['Close']) / recent[0]['Close']

            if price_change > 0.003:  # 0.3%
                momentum['medium_term'] = 'strong_bullish'
            elif price_change > 0.0015:
                momentum['medium_term'] = 'bullish'
            elif price_change < -0.003:
                momentum['medium_term'] = 'strong_bearish'
            elif price_change < -0.0015:
                momentum['medium_term'] = 'bearish'
            else:
                momentum['medium_term'] = 'neutral'

            momentum['medium_term_change'] = price_change * 100
        else:
            momentum['medium_term'] = 'unknown'
            momentum['medium_term_change'] = 0

        return momentum

    def calculate_score(self, trend: Dict, volume: Dict, momentum: Dict) -> int:
        """Calculate composite score from analysis"""
        score = 50  # Base score

        # Trend contribution (0-30 points)
        if trend['direction'] == 'bullish':
            score += min(30, trend['strength'] * 0.3)
        elif trend['direction'] == 'bearish':
            score -= min(20, trend['strength'] * 0.2)

        # Volume contribution (0-20 points)
        if volume['pattern'] == 'spike_high':
            score += min(20, volume['intensity'] * 0.2)
        elif volume['pattern'] == 'increasing':
            score += 10
        elif volume['pattern'] == 'spike_low':
            score -= 10

        # Momentum contribution (0-20 points)
        if momentum['short_term'] == 'strong_bullish':
            score += 15
        elif momentum['short_term'] == 'bullish':
            score += 10
        elif momentum['short_term'] == 'strong_bearish':
            score -= 15
        elif momentum['short_term'] == 'bearish':
            score -= 10

        if momentum['medium_term'] == 'strong_bullish':
            score += 10
        elif momentum['medium_term'] == 'bullish':
            score += 5
        elif momentum['medium_term'] == 'strong_bearish':
            score -= 10
        elif momentum['medium_term'] == 'bearish':
            score -= 5

        # Ensure score stays in 0-100 range
        return max(0, min(100, int(score)))

    def assign_emojis(self, trend: Dict, volume: Dict, momentum: Dict) -> List[str]:
        """Assign emojis based on tick analysis"""
        emojis = []

        # Trend emoji
        if trend['direction'] == 'bullish':
            if trend['strength'] > 70:
                emojis.append("🚀")  # Strong bullish
            else:
                emojis.append("📈")  # Bullish
        elif trend['direction'] == 'bearish':
            if trend['strength'] > 70:
                emojis.append("💀")  # Strong bearish
            else:
                emojis.append("📉")  # Bearish
        else:
            emojis.append("➖")  # Neutral

        # Volume emoji
        if volume['pattern'] == 'spike_high':
            emojis.append("📦")  # High volume (accumulation)
        elif volume['pattern'] == 'increasing':
            emojis.append("💧")  # Increasing volume
        elif volume['pattern'] == 'spike_low':
            emojis.append("🏜️")  # Low volume (desert)
        else:
            emojis.append("〰️")  # Normal volume

        # Momentum emoji
        if momentum['short_term'] in ['strong_bullish', 'bullish'] and \
           momentum['medium_term'] in ['strong_bullish', 'bullish']:
            emojis.append("⚡")  # Strong momentum
        elif momentum['short_term'] in ['strong_bearish', 'bearish'] and \
             momentum['medium_term'] in ['strong_bearish', 'bearish']:
            emojis.append("🔻")  # Negative momentum
        else:
            emojis.append("🔄")  # Mixed/rotating momentum

        return emojis

    def analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol's tick data"""
        print(f"  Analyzing tick data for {symbol}...")

        # Load tick bars
        bars_50 = self.load_tick_bars(symbol, 50)
        bars_100 = self.load_tick_bars(symbol, 100)

        if not bars_50 and not bars_100:
            print(f"    ❌ No tick data available for {symbol}")
            return None

        # Use whichever data is available
        primary_bars = bars_50 if bars_50 else bars_100

        # Perform analysis
        trend = self.analyze_trend(primary_bars)
        volume = self.analyze_volume(primary_bars)
        momentum = self.analyze_momentum(bars_50, bars_100)

        # Calculate score and assign emojis
        score = self.calculate_score(trend, volume, momentum)
        emojis = self.assign_emojis(trend, volume, momentum)
        confidence = min(0.95, score / 100 + 0.3)  # Convert score to confidence

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
            'agent': 'tick_agent',
            'signals': {
                'emoji': ''.join(emojis),
                'score': score,
                'confidence': confidence,
                'metadata': {
                    'trend': trend,
                    'volume': volume,
                    'momentum': momentum,
                    'tick_bars_analyzed': {
                        '50_tick': len(bars_50) if bars_50 else 0,
                        '100_tick': len(bars_100) if bars_100 else 0
                    }
                }
            },
            'shorthand': shorthand
        }

        print(f"    ✅ Score: {score}, Emojis: {''.join(emojis)}")
        print(f"       Trend: {trend['direction']} ({trend['strength']})")
        print(f"       Volume: {volume['pattern']} ({volume['intensity']})")
        print(f"       Momentum: Short={momentum['short_term']}, Medium={momentum['medium_term']}")

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

        print(f"🔍 Tick Agent analyzing {len(candidates)} candidates...")

        results = []
        for candidate in candidates:
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
            'agent': 'tick_agent',
            'symbols_analyzed': len(results),
            'analyses': results
        }

        aggregate_file = self.repository_path / "tick_analysis.json"
        with open(aggregate_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate_output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Tick Agent complete! Analyzed {len(results)} symbols")
        print(f"📁 Results saved to: {self.repository_path}")

        # Print summary
        print("\n📊 Tick Analysis Summary:")
        for i, result in enumerate(results[:5], 1):
            symbol = result['symbol']
            score = result['signals']['score']
            emojis = result['signals']['emoji']
            print(f"{i}. {symbol} - Score: {score} - {emojis}")

        return aggregate_output


if __name__ == "__main__":
    agent = TickAgent()
    results = agent.analyze_candidates()

    print("\n🎯 Tick Agent completed successfully!")
    print(f"Next step: Run indicators_agent.py for technical analysis")
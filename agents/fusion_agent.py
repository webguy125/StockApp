"""
Fusion Agent - Combines signals from all agents with weighted scoring

Purpose: Aggregate and weight signals from all technical agents
Input: Individual agent analyses from repository
Output: C:\StockApp\agents\repository\fusion_output.json

Weighting:
- Technical (Indicators): 40%
- Volume (ORD/Weis): 25%
- Tick Analysis: 20%
- Fundamentals: 10% (placeholder for now)
- News: 5% (placeholder for now)
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


class FusionAgent:
    """
    Fuses signals from all agents into a unified trading signal.
    Applies weighted scoring based on agent reliability and signal strength.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.fusion_path = self.repository_path / "fusion"
        self.fusion_path.mkdir(parents=True, exist_ok=True)

        # Weights for different agents
        self.weights = {
            'indicators_agent': 0.40,  # 40% - Technical indicators
            'volume_agent': 0.25,      # 25% - Volume analysis
            'tick_agent': 0.20,        # 20% - Tick patterns
            'fundamentals_agent': 0.10,  # 10% - Fundamentals (placeholder)
            'news_agent': 0.05         # 5% - News sentiment (placeholder)
        }

        # Minimum confidence thresholds
        self.min_confidence = {
            'individual': 0.3,  # Min confidence for individual agent
            'combined': 0.5     # Min combined confidence to recommend
        }

    def load_agent_data(self, agent_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Load analysis data for a specific agent and symbol"""
        agent_file = self.repository_path / agent_name / f"{symbol}.json"

        if not agent_file.exists():
            # Try aggregate file
            aggregate_file = self.repository_path / agent_name / f"{agent_name.replace('_agent', '_analysis')}.json"
            if aggregate_file.exists():
                try:
                    with open(aggregate_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Find symbol in analyses
                        for analysis in data.get('analyses', []):
                            if analysis.get('symbol') == symbol:
                                return analysis
                except Exception as e:
                    print(f"    ⚠️ Error loading {agent_name} data: {e}")
            return None

        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    ⚠️ Error loading {agent_name} data for {symbol}: {e}")
            return None

    def collect_all_signals(self, symbol: str) -> Dict[str, Any]:
        """Collect signals from all agents for a symbol"""
        signals = {}

        # Try to load from each agent
        for agent_name in ['indicators_agent', 'volume_agent', 'tick_agent']:
            data = self.load_agent_data(agent_name, symbol)
            if data:
                signals[agent_name] = data
                print(f"    ✅ Loaded {agent_name} signals")
            else:
                print(f"    ⚠️ No {agent_name} data for {symbol}")

        # Placeholder for future agents
        # signals['fundamentals_agent'] = None
        # signals['news_agent'] = None

        return signals

    def calculate_weighted_score(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted score from all agent signals"""
        total_score = 0
        total_weight = 0
        total_confidence = 0
        contributing_agents = []

        for agent_name, weight in self.weights.items():
            if agent_name in signals and signals[agent_name]:
                agent_data = signals[agent_name]
                agent_signals = agent_data.get('signals', {})

                score = agent_signals.get('score', 50)
                confidence = agent_signals.get('confidence', 0.5)

                # Only include if confidence meets threshold
                if confidence >= self.min_confidence['individual']:
                    weighted_score = score * weight
                    total_score += weighted_score
                    total_weight += weight
                    total_confidence += confidence * weight
                    contributing_agents.append({
                        'agent': agent_name,
                        'score': score,
                        'confidence': confidence,
                        'weight': weight,
                        'contribution': weighted_score
                    })

        # Normalize scores
        if total_weight > 0:
            final_score = total_score / total_weight
            final_confidence = total_confidence / total_weight
        else:
            final_score = 50  # Neutral
            final_confidence = 0.3  # Low confidence

        return {
            'total_score': final_score,
            'confidence': final_confidence,
            'total_weight_used': total_weight,
            'contributing_agents': contributing_agents
        }

    def determine_recommendation(self, score: float, confidence: float) -> str:
        """Determine trading recommendation based on score and confidence"""
        # First check extreme scores - these should override confidence
        if score <= 20:
            return 'strong_sell'  # Very bearish, warn regardless of confidence
        elif score >= 80:
            return 'strong_buy'  # Very bullish, signal regardless of confidence

        # For moderate scores, check confidence
        if confidence < self.min_confidence['combined']:
            return 'hold'  # Not confident enough for moderate signals

        # Normal thresholds with adequate confidence
        if score >= 65:
            return 'buy'
        elif score <= 35:
            return 'sell'
        else:
            return 'hold'

    def aggregate_emojis(self, signals: Dict[str, Any]) -> List[str]:
        """Aggregate and prioritize emojis from all agents"""
        all_emojis = []
        emoji_counts = {}

        # Collect all emojis
        for agent_name, agent_data in signals.items():
            if agent_data:
                agent_signals = agent_data.get('signals', {})
                emoji_str = agent_signals.get('emoji', '')

                # Convert string to list of emojis
                if emoji_str:
                    for emoji in emoji_str:
                        if emoji and emoji != ' ':
                            all_emojis.append(emoji)
                            emoji_counts[emoji] = emoji_counts.get(emoji, 0) + 1

        # Select most frequent/relevant emojis
        final_emojis = []

        # Add trend emoji (most common directional)
        trend_emojis = ['📈', '📉', '➖', '🚀', '💀']
        for emoji in trend_emojis:
            if emoji in emoji_counts and len(final_emojis) < 1:
                final_emojis.append(emoji)
                break

        # Add signal strength emoji
        signal_emojis = ['✅', '❌', '⚠️', '🔥', '❄️']
        for emoji in signal_emojis:
            if emoji in emoji_counts and len(final_emojis) < 2:
                final_emojis.append(emoji)

        # Add volume/momentum emoji
        volume_emojis = ['📦', '💰', '💸', '⚖️', '🌊']
        for emoji in volume_emojis:
            if emoji in emoji_counts and len(final_emojis) < 3:
                final_emojis.append(emoji)

        # Fill with most common if needed
        if len(final_emojis) < 3:
            sorted_emojis = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)
            for emoji, count in sorted_emojis:
                if emoji not in final_emojis and len(final_emojis) < 4:
                    final_emojis.append(emoji)

        return final_emojis if final_emojis else ['🤔']  # Uncertain if no emojis

    def fuse_symbol(self, symbol: str) -> Dict[str, Any]:
        """Fuse all signals for a single symbol"""
        print(f"\n  🔄 Fusing signals for {symbol}...")

        # Collect signals from all agents
        signals = self.collect_all_signals(symbol)

        if not signals:
            print(f"    ❌ No signals available for {symbol}")
            return None

        # Calculate weighted score
        scoring = self.calculate_weighted_score(signals)

        # Determine recommendation
        recommendation = self.determine_recommendation(
            scoring['total_score'],
            scoring['confidence']
        )

        # Aggregate emojis
        emojis = self.aggregate_emojis(signals)

        # Create shorthand
        shorthand = self.emoji_codec.encode_to_shorthand(
            symbol=symbol,
            emojis=emojis,
            score=int(scoring['total_score']),
            confidence=scoring['confidence']
        )

        # Prepare fusion output
        fusion_output = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'agent': 'fusion_agent',
            'total_score': scoring['total_score'],
            'confidence': scoring['confidence'],
            'recommendation': recommendation,
            'emojis': emojis,
            'emoji_string': ''.join(emojis),
            'shorthand': shorthand,
            'weighted_scores': {
                agent['agent']: {
                    'score': agent['score'],
                    'confidence': agent['confidence'],
                    'weight': agent['weight'],
                    'contribution': agent['contribution']
                }
                for agent in scoring['contributing_agents']
            },
            'metadata': {
                'total_weight_used': scoring['total_weight_used'],
                'agents_contributing': len(scoring['contributing_agents']),
                'signals_collected': len(signals)
            }
        }

        # Print summary
        print(f"    ✅ Fusion Score: {scoring['total_score']:.1f}")
        print(f"       Confidence: {scoring['confidence']:.2f}")
        print(f"       Recommendation: {recommendation.upper()}")
        print(f"       Emojis: {''.join(emojis)}")
        print(f"       Contributing Agents: {len(scoring['contributing_agents'])}")

        return fusion_output

    def fuse_all_candidates(self) -> Dict[str, Any]:
        """Fuse signals for all candidates from scanner"""
        # Load scanner output
        scanner_file = self.repository_path / "scanner_output.json"

        if not scanner_file.exists():
            print("❌ No scanner output found. Run scanner_agent.py first.")
            return {'error': 'No scanner output'}

        try:
            with open(scanner_file, 'r', encoding='utf-8') as f:
                scanner_data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading scanner output: {e}")
            return {'error': str(e)}

        candidates = scanner_data.get('candidates', [])

        if not candidates:
            print("❌ No candidates in scanner output")
            return {'error': 'No candidates'}

        print(f"🔀 Fusion Agent processing {len(candidates)} candidates...")

        fusion_results = []
        for candidate in candidates:
            symbol = candidate['symbol']
            fusion = self.fuse_symbol(symbol)

            if fusion:
                # Save individual fusion result
                symbol_file = self.fusion_path / f"{symbol}.json"
                with open(symbol_file, 'w', encoding='utf-8') as f:
                    json.dump(fusion, f, indent=2, ensure_ascii=False)

                fusion_results.append(fusion)

        # Sort by score and confidence
        fusion_results.sort(key=lambda x: (x['total_score'] * x['confidence']), reverse=True)

        # Categorize by recommendation
        recommendations = {
            'strong_buy': [],
            'buy': [],
            'hold': [],
            'sell': [],
            'strong_sell': []
        }

        for result in fusion_results:
            rec = result['recommendation']
            recommendations[rec].append(result)

        # Prepare aggregate output
        aggregate_output = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'fusion_agent',
            'total_symbols': len(fusion_results),
            'recommendations': {
                rec: len(symbols) for rec, symbols in recommendations.items()
            },
            'top_opportunities': fusion_results[:5],  # Top 5 by score
            'all_fusions': fusion_results,
            'by_recommendation': recommendations
        }

        # Save aggregate output
        aggregate_file = self.repository_path / "fusion_output.json"
        with open(aggregate_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate_output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Fusion complete! Processed {len(fusion_results)} symbols")
        print(f"📁 Results saved to: {aggregate_file}")

        # Print summary
        print("\n📊 Fusion Summary:")
        print(f"   Strong Buy: {len(recommendations['strong_buy'])} symbols")
        print(f"   Buy: {len(recommendations['buy'])} symbols")
        print(f"   Hold: {len(recommendations['hold'])} symbols")
        print(f"   Sell: {len(recommendations['sell'])} symbols")
        print(f"   Strong Sell: {len(recommendations['strong_sell'])} symbols")

        if recommendations['strong_buy']:
            print("\n🔥 Top Strong Buy Signals:")
            for i, result in enumerate(recommendations['strong_buy'][:3], 1):
                print(f"   {i}. {result['symbol']} - Score: {result['total_score']:.1f} - {result['emoji_string']}")

        if recommendations['buy']:
            print("\n✅ Top Buy Signals:")
            for i, result in enumerate(recommendations['buy'][:3], 1):
                print(f"   {i}. {result['symbol']} - Score: {result['total_score']:.1f} - {result['emoji_string']}")

        return aggregate_output


if __name__ == "__main__":
    fusion = FusionAgent()
    results = fusion.fuse_all_candidates()

    if 'error' not in results:
        print("\n🎯 Fusion Agent completed successfully!")
        print("Next step: Run supreme_leader.py for position governance")
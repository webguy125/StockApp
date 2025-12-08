"""
Archivist Agent - Stores features and outcomes for learning

Purpose: Archive trade outcomes with their original signals for learning
Input: Evaluation results and original signal features
Output: Structured archive for training data

Archive Structure:
- Trade features (technical indicators, volume, patterns)
- Entry/exit conditions
- Outcome (win/loss/scratch)
- Quality score
- Market context
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class ArchivistAgent:
    """
    Archives trade outcomes with their features for the learning loop.
    Creates structured datasets for model training.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.archive_path = self.repository_path / "archive"
        self.archive_path.mkdir(parents=True, exist_ok=True)

        # Archive categories
        self.categories = {
            'wins': self.archive_path / 'wins',
            'losses': self.archive_path / 'losses',
            'quality_trades': self.archive_path / 'quality_trades',
            'learning_cases': self.archive_path / 'learning_cases'
        }

        # Create category directories
        for category_path in self.categories.values():
            category_path.mkdir(exist_ok=True)

        # Archive metadata
        self.archive_meta = self.load_archive_metadata()

    def load_archive_metadata(self) -> Dict[str, Any]:
        """Load or create archive metadata"""
        meta_file = self.archive_path / "archive_metadata.json"

        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        # Initialize metadata
        return {
            'created': datetime.now().isoformat(),
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'quality_wins': 0,
            'quality_losses': 0,
            'last_updated': datetime.now().isoformat()
        }

    def save_archive_metadata(self):
        """Save archive metadata"""
        meta_file = self.archive_path / "archive_metadata.json"
        self.archive_meta['last_updated'] = datetime.now().isoformat()

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.archive_meta, f, indent=2, ensure_ascii=False)

    def load_evaluation_report(self) -> Optional[Dict[str, Any]]:
        """Load the latest evaluation report"""
        report_file = self.repository_path / "evaluation_report.json"

        if not report_file.exists():
            print("❌ No evaluation report found. Run evaluator_agent.py first.")
            return None

        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading evaluation report: {e}")
            return None

    def load_signal_features(self, symbol: str, timestamp: str) -> Dict[str, Any]:
        """Load original signal features for a trade"""
        features = {}

        # Load from different agent outputs
        agents = ['indicators_agent', 'volume_agent', 'tick_agent', 'fusion']

        for agent in agents:
            agent_path = self.repository_path / agent / f"{symbol}.json"
            if agent_path.exists():
                try:
                    with open(agent_path, 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)

                        # Extract relevant features
                        if agent == 'indicators_agent':
                            features['indicators'] = agent_data.get('indicators', {})
                            features['signals'] = agent_data.get('signals', {})
                        elif agent == 'volume_agent':
                            features['volume_analysis'] = agent_data.get('volume_signals', {})
                        elif agent == 'tick_agent':
                            features['tick_patterns'] = agent_data.get('tick_patterns', {})
                        elif agent == 'fusion':
                            features['fusion_score'] = agent_data.get('total_score', 50)
                            features['fusion_confidence'] = agent_data.get('confidence', 0.5)
                except:
                    pass

        return features

    def extract_market_context(self, symbol: str, timestamp: str) -> Dict[str, Any]:
        """Extract market context at time of trade"""
        context = {
            'symbol': symbol,
            'timestamp': timestamp,
            'day_of_week': datetime.fromisoformat(timestamp).strftime('%A') if timestamp else None,
            'hour_of_day': datetime.fromisoformat(timestamp).hour if timestamp else None
        }

        # Add market type
        context['market_type'] = 'crypto' if symbol.endswith('-USD') else 'stock'

        # Check for major symbols
        major_cryptos = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        major_stocks = ['AAPL', 'MSFT', 'TSLA', 'SPY', 'QQQ']

        if symbol in major_cryptos or symbol in major_stocks:
            context['liquidity'] = 'high'
        else:
            context['liquidity'] = 'medium'

        return context

    def create_archive_entry(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured archive entry from evaluation"""
        metrics = evaluation['metrics']
        symbol = metrics['symbol']

        # Generate unique ID for this trade
        trade_id = hashlib.md5(
            f"{symbol}_{evaluation.get('entry_timestamp', '')}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Load original features
        features = self.load_signal_features(
            symbol,
            evaluation.get('entry_timestamp', '')
        )

        # Extract market context
        context = self.extract_market_context(
            symbol,
            evaluation.get('entry_timestamp', '')
        )

        # Create archive entry
        archive_entry = {
            'trade_id': trade_id,
            'symbol': symbol,
            'verdict': evaluation['verdict'],
            'verdict_emoji': evaluation['verdict_emoji'],
            'quality_score': evaluation['quality_score'],

            # Performance metrics
            'metrics': {
                'return_pct': metrics['return_pct'],
                'return_usd': metrics['return_usd'],
                'hold_time': metrics['hold_time'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'max_gain_pct': metrics['max_gain_pct'],
                'risk_reward_realized': metrics['risk_reward_realized']
            },

            # Entry/Exit details
            'trade_details': {
                'entry_price': metrics['entry_price'],
                'exit_price': metrics['exit_price'],
                'entry_time': evaluation.get('entry_timestamp'),
                'exit_time': evaluation.get('exit_timestamp'),
                'position_size': metrics['position_size']
            },

            # Original signals and features
            'features': features,

            # Market context
            'market_context': context,

            # Archive metadata
            'archived_at': datetime.now().isoformat(),
            'archive_version': '1.0'
        }

        return archive_entry

    def categorize_trade(self, archive_entry: Dict[str, Any]) -> str:
        """Determine which category to archive trade in"""
        verdict = archive_entry['verdict']
        quality_score = archive_entry['quality_score']

        if verdict == 'quality_win' or (verdict == 'win' and quality_score >= 80):
            return 'quality_trades'
        elif verdict in ['win', 'quality_win']:
            return 'wins'
        elif verdict in ['loss', 'quality_loss']:
            return 'losses'
        else:
            return 'learning_cases'

    def save_archive_entry(self, archive_entry: Dict[str, Any], category: str):
        """Save archive entry to appropriate category"""
        # Save to category folder
        category_path = self.categories[category]
        trade_id = archive_entry['trade_id']
        symbol = archive_entry['symbol']

        # Create filename with symbol and trade ID
        filename = f"{symbol}_{trade_id}.json"
        filepath = category_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(archive_entry, f, indent=2, ensure_ascii=False)

        print(f"       📁 Archived to {category}/{filename}")

    def create_training_dataset(self) -> Dict[str, Any]:
        """Create a training dataset from archived trades"""
        training_data = {
            'features': [],
            'labels': [],
            'metadata': []
        }

        # Load all archived trades
        for category, path in self.categories.items():
            for file in path.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        entry = json.load(f)

                        # Extract features for training
                        feature_vector = {
                            'fusion_score': entry['features'].get('fusion_score', 50),
                            'fusion_confidence': entry['features'].get('fusion_confidence', 0.5),
                            'rsi': entry['features'].get('indicators', {}).get('rsi', 50),
                            'macd_signal': entry['features'].get('indicators', {}).get('macd_histogram', 0),
                            'volume_signal': entry['features'].get('volume_analysis', {}).get('ord_signal', 0)
                        }

                        # Label (1 for win, 0 for loss)
                        label = 1 if entry['verdict'] in ['win', 'quality_win'] else 0

                        training_data['features'].append(feature_vector)
                        training_data['labels'].append(label)
                        training_data['metadata'].append({
                            'trade_id': entry['trade_id'],
                            'symbol': entry['symbol'],
                            'quality_score': entry['quality_score']
                        })
                except:
                    pass

        # Save training dataset
        if training_data['features']:
            dataset_file = self.archive_path / "training_dataset.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)

            print(f"   📊 Training dataset created with {len(training_data['features'])} samples")

        return training_data

    def archive_all_evaluations(self) -> Dict[str, Any]:
        """Archive all evaluated trades"""
        print("📚 Archivist Agent archiving trade outcomes...")

        # Load evaluation report
        evaluation_report = self.load_evaluation_report()
        if not evaluation_report:
            return {'error': 'No evaluation report'}

        evaluations = evaluation_report.get('evaluations', [])

        if not evaluations:
            print("   No evaluations to archive")
            return {'error': 'No evaluations'}

        print(f"   Archiving {len(evaluations)} trade evaluations...")

        archived_count = {
            'wins': 0,
            'losses': 0,
            'quality_trades': 0,
            'learning_cases': 0
        }

        for evaluation in evaluations:
            symbol = evaluation['metrics']['symbol']
            print(f"  📝 Archiving {symbol}...")

            # Create archive entry
            archive_entry = self.create_archive_entry(evaluation)

            # Determine category
            category = self.categorize_trade(archive_entry)
            archived_count[category] += 1

            # Save to archive
            self.save_archive_entry(archive_entry, category)

            # Update metadata
            self.archive_meta['total_trades'] += 1
            verdict = evaluation['verdict']
            if verdict in ['win', 'quality_win']:
                self.archive_meta['wins'] += 1
                if verdict == 'quality_win':
                    self.archive_meta['quality_wins'] += 1
            elif verdict in ['loss', 'quality_loss']:
                self.archive_meta['losses'] += 1
                if verdict == 'quality_loss':
                    self.archive_meta['quality_losses'] += 1

        # Save updated metadata
        self.save_archive_metadata()

        # Create training dataset
        training_dataset = self.create_training_dataset()

        # Calculate statistics
        total_archived = sum(archived_count.values())
        win_rate = (self.archive_meta['wins'] / self.archive_meta['total_trades'] * 100) if self.archive_meta['total_trades'] > 0 else 0

        # Prepare archive report
        archive_report = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'archivist_agent',
            'archived_this_session': total_archived,
            'archive_breakdown': archived_count,
            'total_in_archive': self.archive_meta['total_trades'],
            'archive_statistics': {
                'total_trades': self.archive_meta['total_trades'],
                'wins': self.archive_meta['wins'],
                'losses': self.archive_meta['losses'],
                'quality_wins': self.archive_meta['quality_wins'],
                'quality_losses': self.archive_meta['quality_losses'],
                'win_rate_pct': win_rate,
                'training_samples': len(training_dataset.get('features', []))
            }
        }

        # Save archive report
        report_file = self.repository_path / "archive_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(archive_report, f, indent=2, ensure_ascii=False)

        print(f"\n📚 Archiving Complete:")
        print(f"   Archived: {total_archived} trades")
        print(f"   Categories:")
        print(f"     Quality Trades: {archived_count['quality_trades']}")
        print(f"     Wins: {archived_count['wins']}")
        print(f"     Losses: {archived_count['losses']}")
        print(f"     Learning Cases: {archived_count['learning_cases']}")
        print(f"   Total in Archive: {self.archive_meta['total_trades']}")
        print(f"   Historical Win Rate: {win_rate:.1f}%")

        print(f"\n📁 Report saved to: {report_file}")

        return archive_report


if __name__ == "__main__":
    archivist = ArchivistAgent()
    results = archivist.archive_all_evaluations()

    if 'error' not in results:
        print("\n📚 Archivist Agent ready to preserve trading knowledge!")
        print("Next step: Run criteria_auditor.py to verify signal integrity")
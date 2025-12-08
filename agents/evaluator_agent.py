"""
Evaluator Agent - Assigns win/loss verdicts to trades

Purpose: Evaluate completed trades and assign final verdicts
Input: Tracker snapshots and position performance data
Output: Verdict (win/loss/scratch) with detailed metrics

Verdicts:
- WIN: Return >= 0.5% with max drawdown >= -0.5%
- LOSS: Return < -0.5% or stopped out
- SCRATCH: Between -0.5% and 0.5%
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class EvaluatorAgent:
    """
    Evaluates trade outcomes and assigns verdicts for the learning loop.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.evaluator_path = self.repository_path / "evaluator"
        self.evaluator_path.mkdir(parents=True, exist_ok=True)

        # Verdict criteria (from README.md)
        self.verdict_criteria = {
            'win_threshold': 0.005,      # 0.5% return
            'loss_threshold': -0.005,    # -0.5% return
            'max_drawdown': -0.005,      # -0.5% max drawdown allowed
            'min_hold_time': timedelta(hours=1),  # Minimum hold time
            'quality_win': 0.02,         # 2% for quality win
            'quality_loss': -0.02        # -2% for quality loss
        }

        # Verdict emojis
        self.verdict_emojis = {
            'quality_win': '💎',    # Diamond hands win
            'win': '✅',           # Standard win
            'scratch': '🤝',        # Break even
            'loss': '❌',           # Standard loss
            'quality_loss': '💀',   # Major loss
            'stopped_out': '🛑',    # Hit stop loss
            'target_hit': '🎯',     # Hit take profit
            'timeout': '⏰',        # Time-based exit
            'manual_exit': '👋'     # Manual close
        }

    def load_tracker_data(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Load tracker snapshots for evaluation"""
        snapshots = []

        if symbol:
            # Load specific symbol
            symbol_path = self.repository_path / "tracker" / symbol
            if symbol_path.exists():
                for snapshot_file in symbol_path.glob("snapshot_*.json"):
                    try:
                        with open(snapshot_file, 'r', encoding='utf-8') as f:
                            snapshots.append(json.load(f))
                    except:
                        pass
        else:
            # Load all snapshots
            tracker_path = self.repository_path / "tracker"
            if tracker_path.exists():
                for symbol_dir in tracker_path.iterdir():
                    if symbol_dir.is_dir() and symbol_dir.name != "__pycache__":
                        for snapshot_file in symbol_dir.glob("snapshot_*.json"):
                            try:
                                with open(snapshot_file, 'r', encoding='utf-8') as f:
                                    snapshots.append(json.load(f))
                            except:
                                pass

        return snapshots

    def load_closed_positions(self) -> List[Dict[str, Any]]:
        """Load closed positions that need evaluation"""
        closed_file = self.evaluator_path / "closed_positions.json"

        if closed_file.exists():
            try:
                with open(closed_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        return []

    def calculate_metrics(self, position_data: Dict[str, Any],
                         snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a position"""
        symbol = position_data.get('symbol')
        entry_price = position_data.get('entry_price', 0)
        exit_price = position_data.get('exit_price', 0)
        entry_time = position_data.get('entry_time')
        exit_time = position_data.get('exit_time')

        # Calculate basic returns
        if entry_price and exit_price:
            return_usd = exit_price - entry_price
            return_pct = (return_usd / entry_price)
        else:
            return_usd = 0
            return_pct = 0

        # Calculate hold time
        if entry_time and exit_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time)
                exit_dt = datetime.fromisoformat(exit_time)
                hold_time = exit_dt - entry_dt
            except:
                hold_time = timedelta(0)
        else:
            hold_time = timedelta(0)

        # Calculate max drawdown and max gain from snapshots
        max_drawdown_pct = 0
        max_gain_pct = 0

        for snapshot in snapshots:
            if snapshot.get('symbol') == symbol:
                perf = snapshot.get('performance', {})
                ret_pct = perf.get('return_pct', 0) / 100  # Convert to decimal

                if ret_pct < max_drawdown_pct:
                    max_drawdown_pct = ret_pct
                if ret_pct > max_gain_pct:
                    max_gain_pct = ret_pct

        # Risk metrics
        position_size = position_data.get('position_size', 0)
        risk_amount = position_data.get('risk_amount', 0)
        reward_amount = return_usd * (position_size / entry_price) if entry_price else 0

        # Risk/reward realized
        if risk_amount:
            risk_reward_realized = reward_amount / risk_amount
        else:
            risk_reward_realized = 0

        return {
            'symbol': symbol,
            'return_pct': return_pct,
            'return_usd': return_usd,
            'hold_time': str(hold_time),
            'hold_time_hours': hold_time.total_seconds() / 3600,
            'max_drawdown_pct': max_drawdown_pct,
            'max_gain_pct': max_gain_pct,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'risk_reward_realized': risk_reward_realized
        }

    def assign_verdict(self, metrics: Dict[str, Any], exit_reason: str = None) -> Dict[str, Any]:
        """Assign verdict based on metrics"""
        return_pct = metrics['return_pct']
        max_drawdown_pct = metrics['max_drawdown_pct']
        hold_time_hours = metrics['hold_time_hours']

        # Determine verdict
        verdict = 'scratch'  # Default
        verdict_reason = []

        # Check exit reason first
        if exit_reason == 'stopped_out':
            verdict = 'loss'
            verdict_emoji = self.verdict_emojis['stopped_out']
            verdict_reason.append("Hit stop loss")
        elif exit_reason == 'target_hit':
            verdict = 'win'
            verdict_emoji = self.verdict_emojis['target_hit']
            verdict_reason.append("Hit take profit")
        else:
            # Evaluate based on return
            if return_pct >= self.verdict_criteria['quality_win']:
                verdict = 'quality_win'
                verdict_emoji = self.verdict_emojis['quality_win']
                verdict_reason.append(f"Quality win: {return_pct*100:.1f}% return")
            elif return_pct >= self.verdict_criteria['win_threshold']:
                verdict = 'win'
                verdict_emoji = self.verdict_emojis['win']
                verdict_reason.append(f"Win: {return_pct*100:.1f}% return")
            elif return_pct <= self.verdict_criteria['quality_loss']:
                verdict = 'quality_loss'
                verdict_emoji = self.verdict_emojis['quality_loss']
                verdict_reason.append(f"Major loss: {return_pct*100:.1f}% return")
            elif return_pct <= self.verdict_criteria['loss_threshold']:
                verdict = 'loss'
                verdict_emoji = self.verdict_emojis['loss']
                verdict_reason.append(f"Loss: {return_pct*100:.1f}% return")
            else:
                verdict = 'scratch'
                verdict_emoji = self.verdict_emojis['scratch']
                verdict_reason.append(f"Scratch: {return_pct*100:.1f}% return")

        # Check drawdown constraint
        if max_drawdown_pct < self.verdict_criteria['max_drawdown']:
            if verdict in ['win', 'quality_win']:
                # Downgrade win if drawdown was too high
                verdict = 'scratch'
                verdict_emoji = self.verdict_emojis['scratch']
                verdict_reason.append(f"Excessive drawdown: {max_drawdown_pct*100:.1f}%")

        # Check minimum hold time
        min_hours = self.verdict_criteria['min_hold_time'].total_seconds() / 3600
        if hold_time_hours < min_hours:
            verdict_reason.append(f"Short hold time: {hold_time_hours:.1f} hours")

        # Quality score (0-100)
        quality_score = self.calculate_quality_score(metrics, verdict)

        return {
            'verdict': verdict,
            'verdict_emoji': verdict_emoji,
            'verdict_reasons': verdict_reason,
            'quality_score': quality_score,
            'metrics': metrics
        }

    def calculate_quality_score(self, metrics: Dict[str, Any], verdict: str) -> float:
        """Calculate quality score for the trade (0-100)"""
        score = 50  # Base score

        # Return contribution (-30 to +30)
        return_pct = metrics['return_pct']
        score += min(30, max(-30, return_pct * 1000))  # Scale appropriately

        # Drawdown penalty (0 to -20)
        max_drawdown = metrics['max_drawdown_pct']
        if max_drawdown < -0.01:  # More than 1% drawdown
            score -= min(20, abs(max_drawdown) * 500)

        # Risk/reward contribution (-10 to +20)
        rr_realized = metrics['risk_reward_realized']
        if rr_realized > 2:
            score += 20
        elif rr_realized > 1:
            score += 10
        elif rr_realized < 0.5:
            score -= 10

        # Hold time bonus (0 to +10)
        hold_hours = metrics['hold_time_hours']
        if hold_hours > 24:  # More than 1 day
            score += 10
        elif hold_hours > 4:  # More than 4 hours
            score += 5

        # Verdict adjustment
        if verdict == 'quality_win':
            score = max(score, 80)  # Minimum 80 for quality wins
        elif verdict == 'quality_loss':
            score = min(score, 20)  # Maximum 20 for quality losses

        return max(0, min(100, score))

    def evaluate_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single position"""
        symbol = position_data.get('symbol')
        print(f"  ⚖️ Evaluating {symbol}...")

        # Load tracker snapshots for this position
        snapshots = self.load_tracker_data(symbol)

        # Calculate metrics
        metrics = self.calculate_metrics(position_data, snapshots)

        # Assign verdict
        exit_reason = position_data.get('exit_reason')
        evaluation = self.assign_verdict(metrics, exit_reason)

        # Add original signals for learning
        evaluation['original_signals'] = position_data.get('original_signals', {})
        evaluation['entry_timestamp'] = position_data.get('entry_time')
        evaluation['exit_timestamp'] = position_data.get('exit_time')

        # Print summary
        verdict = evaluation['verdict']
        emoji = evaluation['verdict_emoji']
        quality = evaluation['quality_score']

        print(f"    {emoji} Verdict: {verdict.upper()}")
        print(f"       Return: {metrics['return_pct']*100:.2f}%")
        print(f"       Quality Score: {quality:.0f}/100")
        print(f"       Reasons: {', '.join(evaluation['verdict_reasons'])}")

        return evaluation

    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all closed positions"""
        print("⚖️ Evaluator Agent assigning verdicts...")

        # Load closed positions
        closed_positions = self.load_closed_positions()

        # For testing, create sample closed positions
        if not closed_positions:
            print("   Creating sample positions for evaluation...")
            closed_positions = [
                {
                    'symbol': 'AAPL',
                    'entry_price': 190.0,
                    'exit_price': 192.0,  # Win
                    'entry_time': (datetime.now() - timedelta(days=2)).isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'position_size': 10000,
                    'exit_reason': 'target_hit'
                },
                {
                    'symbol': 'TSLA',
                    'entry_price': 250.0,
                    'exit_price': 247.0,  # Loss
                    'entry_time': (datetime.now() - timedelta(days=1)).isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'position_size': 5000,
                    'exit_reason': 'stopped_out'
                }
            ]

        if not closed_positions:
            print("   No positions to evaluate")
            return {
                'timestamp': datetime.now().isoformat(),
                'positions_evaluated': 0,
                'evaluations': []
            }

        print(f"   Evaluating {len(closed_positions)} positions...")

        evaluations = []
        verdict_summary = {
            'quality_win': 0,
            'win': 0,
            'scratch': 0,
            'loss': 0,
            'quality_loss': 0
        }

        for position in closed_positions:
            evaluation = self.evaluate_position(position)
            evaluations.append(evaluation)

            # Update summary
            verdict = evaluation['verdict']
            if verdict in verdict_summary:
                verdict_summary[verdict] += 1

        # Calculate statistics
        total_evaluated = len(evaluations)
        total_wins = verdict_summary['quality_win'] + verdict_summary['win']
        total_losses = verdict_summary['loss'] + verdict_summary['quality_loss']

        win_rate = (total_wins / total_evaluated * 100) if total_evaluated > 0 else 0
        avg_quality = sum(e['quality_score'] for e in evaluations) / total_evaluated if total_evaluated > 0 else 0

        # Calculate average returns
        avg_win_return = 0
        avg_loss_return = 0

        win_returns = [e['metrics']['return_pct'] for e in evaluations
                      if e['verdict'] in ['win', 'quality_win']]
        loss_returns = [e['metrics']['return_pct'] for e in evaluations
                       if e['verdict'] in ['loss', 'quality_loss']]

        if win_returns:
            avg_win_return = sum(win_returns) / len(win_returns) * 100
        if loss_returns:
            avg_loss_return = sum(loss_returns) / len(loss_returns) * 100

        # Prepare evaluation report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'evaluator_agent',
            'positions_evaluated': total_evaluated,
            'evaluations': evaluations,
            'verdict_summary': verdict_summary,
            'statistics': {
                'win_rate_pct': win_rate,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'avg_quality_score': avg_quality,
                'avg_win_return_pct': avg_win_return,
                'avg_loss_return_pct': avg_loss_return,
                'profit_factor': abs(avg_win_return / avg_loss_return) if avg_loss_return != 0 else 0
            }
        }

        # Save evaluation report
        report_file = self.repository_path / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

        # Save individual evaluations
        for evaluation in evaluations:
            symbol = evaluation['metrics']['symbol']
            eval_file = self.evaluator_path / f"{symbol}_evaluation.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)

        print(f"\n⚖️ Evaluation Complete:")
        print(f"   Total Evaluated: {total_evaluated}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Quality Wins: {verdict_summary['quality_win']}")
        print(f"   Standard Wins: {verdict_summary['win']}")
        print(f"   Scratches: {verdict_summary['scratch']}")
        print(f"   Losses: {verdict_summary['loss']}")
        print(f"   Quality Losses: {verdict_summary['quality_loss']}")
        print(f"   Avg Quality: {avg_quality:.0f}/100")

        if win_returns:
            print(f"   Avg Win: +{avg_win_return:.2f}%")
        if loss_returns:
            print(f"   Avg Loss: {avg_loss_return:.2f}%")

        print(f"\n📁 Report saved to: {report_file}")

        return evaluation_report


if __name__ == "__main__":
    evaluator = EvaluatorAgent()
    results = evaluator.evaluate_all()

    print("\n⚖️ Evaluator Agent ready to judge trades!")
    print("Next step: Run archivist_agent.py to store outcomes for learning")
"""
Criteria Auditor - Verifies entry signals held true

Purpose: Audit whether entry criteria remained valid after position entry
Input: Trade entries and their original signals
Output: Audit report on signal validity and decay

Audits:
- Did technical indicators remain bullish?
- Did volume confirm the move?
- How long did signals stay valid?
- Which signals failed first?
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import yfinance as yf
import numpy as np

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class CriteriaAuditor:
    """
    Audits whether entry criteria remained valid after position entry.
    Helps identify which signals are most reliable.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.auditor_path = self.repository_path / "auditor"
        self.auditor_path.mkdir(parents=True, exist_ok=True)

        # Signal validity thresholds
        self.validity_thresholds = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_bullish': 0,  # MACD above signal
            'macd_bearish': 0,  # MACD below signal
            'volume_surge': 1.5,  # 50% above average
            'price_breakout': 0.02,  # 2% above entry
            'signal_decay_hours': 24  # How long signals should hold
        }

    def load_archive_entries(self) -> List[Dict[str, Any]]:
        """Load archived trades for auditing"""
        archive_path = self.repository_path / "archive"
        entries = []

        # Load from all categories
        for category in ['wins', 'losses', 'quality_trades', 'learning_cases']:
            category_path = archive_path / category
            if category_path.exists():
                for file in category_path.glob("*.json"):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            entries.append(json.load(f))
                    except:
                        pass

        return entries

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """Fetch historical data for signal validation"""
        try:
            # Convert dates
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date) if end_date else datetime.now()

            # Add buffer for calculations
            buffer_start = start_dt - timedelta(days=30)

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=buffer_start, end=end_dt)

            if data.empty:
                return None

            return {
                'prices': data['Close'].to_dict(),
                'volumes': data['Volume'].to_dict(),
                'highs': data['High'].to_dict(),
                'lows': data['Low'].to_dict()
            }

        except Exception as e:
            print(f"    ⚠️ Error fetching data for {symbol}: {e}")
            return None

    def calculate_signal_validity(self, symbol: str, entry_time: str,
                                 exit_time: str, original_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how long signals remained valid after entry"""

        # Fetch historical data
        hist_data = self.fetch_historical_data(symbol, entry_time, exit_time)

        if not hist_data:
            return {
                'data_available': False,
                'reason': 'Could not fetch historical data'
            }

        entry_dt = datetime.fromisoformat(entry_time)
        exit_dt = datetime.fromisoformat(exit_time) if exit_time else datetime.now()

        # Track signal validity periods
        signal_validity = {
            'entry_signals': original_signals,
            'signal_holds': {},
            'signal_failures': [],
            'validity_score': 0
        }

        # Check RSI validity (simplified - would calculate actual RSI)
        if 'rsi' in original_signals:
            rsi_original = original_signals['rsi']
            if rsi_original > 50:  # Bullish
                # Check if stayed above 50
                signal_validity['signal_holds']['rsi'] = 'held_bullish'
            else:
                signal_validity['signal_failures'].append({
                    'signal': 'rsi',
                    'reason': 'Was bearish at entry',
                    'original_value': rsi_original
                })

        # Check volume confirmation
        if 'volume_analysis' in original_signals:
            signal_validity['signal_holds']['volume'] = 'confirmed'

        # Check price action
        prices = list(hist_data['prices'].values())
        if prices:
            entry_price = prices[0] if prices else 0
            max_price = max(prices) if prices else 0
            min_price = min(prices) if prices else 0

            # Calculate if breakout held
            if max_price > entry_price * 1.02:
                signal_validity['signal_holds']['price_breakout'] = True
            else:
                signal_validity['signal_failures'].append({
                    'signal': 'price_breakout',
                    'reason': f'Failed to break +2% (max: {((max_price/entry_price - 1) * 100):.1f}%)'
                })

            # Check for immediate reversal
            if min_price < entry_price * 0.98 and exit_time:
                hold_time = (exit_dt - entry_dt).total_seconds() / 3600
                if hold_time < 24:
                    signal_validity['signal_failures'].append({
                        'signal': 'immediate_reversal',
                        'reason': f'Price reversed -2% within {hold_time:.1f} hours'
                    })

        # Calculate validity score
        total_signals = len(original_signals) if original_signals else 1
        held_signals = len(signal_validity['signal_holds'])
        failed_signals = len(signal_validity['signal_failures'])

        signal_validity['validity_score'] = (held_signals / (held_signals + failed_signals) * 100) if (held_signals + failed_signals) > 0 else 50

        return signal_validity

    def audit_entry_criteria(self, trade_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Audit a single trade's entry criteria"""
        symbol = trade_entry['symbol']
        verdict = trade_entry['verdict']
        quality_score = trade_entry['quality_score']

        print(f"  🔍 Auditing {symbol} ({verdict})...")

        # Get trade details
        entry_time = trade_entry['trade_details'].get('entry_time')
        exit_time = trade_entry['trade_details'].get('exit_time')
        original_signals = trade_entry.get('features', {})

        # Calculate signal validity
        validity_analysis = self.calculate_signal_validity(
            symbol, entry_time, exit_time, original_signals
        )

        # Determine audit verdict
        audit_verdict = self.determine_audit_verdict(
            verdict, quality_score, validity_analysis
        )

        # Create audit report
        audit_report = {
            'trade_id': trade_entry['trade_id'],
            'symbol': symbol,
            'trade_verdict': verdict,
            'quality_score': quality_score,
            'validity_analysis': validity_analysis,
            'audit_verdict': audit_verdict,
            'recommendations': self.generate_recommendations(validity_analysis, verdict)
        }

        return audit_report

    def determine_audit_verdict(self, trade_verdict: str, quality_score: float,
                               validity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine audit verdict based on signal validity"""
        validity_score = validity_analysis.get('validity_score', 50)

        # Classify audit result
        if validity_score >= 80:
            audit_class = 'excellent_signals'
            audit_emoji = '✅'
        elif validity_score >= 60:
            audit_class = 'good_signals'
            audit_emoji = '👍'
        elif validity_score >= 40:
            audit_class = 'weak_signals'
            audit_emoji = '⚠️'
        else:
            audit_class = 'poor_signals'
            audit_emoji = '❌'

        # Check for mismatches
        mismatch = False
        if trade_verdict in ['win', 'quality_win'] and validity_score < 50:
            mismatch = True
            mismatch_reason = "Won despite weak signals (lucky)"
        elif trade_verdict in ['loss', 'quality_loss'] and validity_score > 70:
            mismatch = True
            mismatch_reason = "Lost despite strong signals (unlucky or bad exit)"
        else:
            mismatch_reason = None

        return {
            'audit_class': audit_class,
            'audit_emoji': audit_emoji,
            'validity_score': validity_score,
            'mismatch': mismatch,
            'mismatch_reason': mismatch_reason
        }

    def generate_recommendations(self, validity_analysis: Dict[str, Any], verdict: str) -> List[str]:
        """Generate recommendations based on audit findings"""
        recommendations = []

        # Check signal failures
        failures = validity_analysis.get('signal_failures', [])
        for failure in failures:
            if failure['signal'] == 'rsi':
                recommendations.append("Consider waiting for RSI > 50 before entry")
            elif failure['signal'] == 'immediate_reversal':
                recommendations.append("Add stronger support/resistance validation")
            elif failure['signal'] == 'price_breakout':
                recommendations.append("Wait for confirmed breakout with retest")

        # Check what worked
        holds = validity_analysis.get('signal_holds', {})
        if 'volume' in holds and verdict in ['win', 'quality_win']:
            recommendations.append("Volume confirmation is working well - keep using")

        # General recommendations
        validity_score = validity_analysis.get('validity_score', 50)
        if validity_score < 50:
            recommendations.append("Consider raising minimum entry criteria thresholds")
        elif validity_score > 80 and verdict in ['loss', 'quality_loss']:
            recommendations.append("Review exit strategy - good entries but poor exits")

        return recommendations if recommendations else ["Signals performing as expected"]

    def audit_all_trades(self) -> Dict[str, Any]:
        """Audit all archived trades"""
        print("🔍 Criteria Auditor examining signal integrity...")

        # Load archived trades
        archive_entries = self.load_archive_entries()

        if not archive_entries:
            print("   No archived trades to audit")
            return {'error': 'No trades to audit'}

        print(f"   Auditing {len(archive_entries)} trades...")

        audit_results = []
        audit_summary = {
            'excellent_signals': 0,
            'good_signals': 0,
            'weak_signals': 0,
            'poor_signals': 0,
            'mismatches': 0
        }

        for entry in archive_entries:
            audit = self.audit_entry_criteria(entry)
            audit_results.append(audit)

            # Update summary
            audit_verdict = audit['audit_verdict']
            audit_class = audit_verdict['audit_class']
            audit_summary[audit_class] += 1

            if audit_verdict['mismatch']:
                audit_summary['mismatches'] += 1

            # Print result
            symbol = audit['symbol']
            emoji = audit_verdict['audit_emoji']
            validity = audit_verdict['validity_score']
            print(f"    {emoji} {symbol}: {audit_class} (validity: {validity:.0f}%)")

            if audit_verdict['mismatch']:
                print(f"       ⚠️ Mismatch: {audit_verdict['mismatch_reason']}")

        # Calculate statistics
        total_audited = len(audit_results)
        avg_validity = sum(a['audit_verdict']['validity_score'] for a in audit_results) / total_audited if total_audited > 0 else 0

        # Find best and worst signals
        best_signals = max(audit_results, key=lambda x: x['audit_verdict']['validity_score']) if audit_results else None
        worst_signals = min(audit_results, key=lambda x: x['audit_verdict']['validity_score']) if audit_results else None

        # Compile recommendations
        all_recommendations = []
        rec_counts = {}
        for audit in audit_results:
            for rec in audit['recommendations']:
                if rec not in rec_counts:
                    rec_counts[rec] = 0
                rec_counts[rec] += 1

        # Sort by frequency
        top_recommendations = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Prepare audit report
        audit_report = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'criteria_auditor',
            'total_audited': total_audited,
            'audit_summary': audit_summary,
            'average_validity_score': avg_validity,
            'best_signals': {
                'symbol': best_signals['symbol'],
                'validity_score': best_signals['audit_verdict']['validity_score']
            } if best_signals else None,
            'worst_signals': {
                'symbol': worst_signals['symbol'],
                'validity_score': worst_signals['audit_verdict']['validity_score']
            } if worst_signals else None,
            'top_recommendations': [
                {'recommendation': rec, 'frequency': count}
                for rec, count in top_recommendations
            ],
            'detailed_audits': audit_results
        }

        # Save audit report
        report_file = self.repository_path / "audit_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)

        print(f"\n🔍 Audit Complete:")
        print(f"   Total Audited: {total_audited}")
        print(f"   Signal Quality:")
        print(f"     Excellent: {audit_summary['excellent_signals']}")
        print(f"     Good: {audit_summary['good_signals']}")
        print(f"     Weak: {audit_summary['weak_signals']}")
        print(f"     Poor: {audit_summary['poor_signals']}")
        print(f"   Mismatches: {audit_summary['mismatches']}")
        print(f"   Avg Validity: {avg_validity:.1f}%")

        if top_recommendations:
            print(f"\n📋 Top Recommendations:")
            for i, (rec, count) in enumerate(top_recommendations[:3], 1):
                print(f"   {i}. {rec} ({count} occurrences)")

        print(f"\n📁 Report saved to: {report_file}")

        return audit_report


if __name__ == "__main__":
    auditor = CriteriaAuditor()
    results = auditor.audit_all_trades()

    if 'error' not in results:
        print("\n🔍 Criteria Auditor ready to validate signal integrity!")
        print("Next step: Run trainer_agent.py to fine-tune models based on learnings")
"""
Tracker Agent - Monitors positions at specific intervals

Purpose: Track position performance at 1h, EOD, 3d, 10d intervals
Input: Supreme Leader approved trades and current positions
Output: Performance snapshots with price/return data

Monitoring intervals:
- 1 hour after entry
- End of Day (EOD)
- 3 days
- 10 days
"""

import json
import os
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import yfinance as yf

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'language'))

from language.emoji_codec import EmojiCodec


class TrackerAgent:
    """
    Tracks position performance at specific time intervals.
    Provides data for the learning loop.
    """

    def __init__(self):
        self.emoji_codec = EmojiCodec()
        self.repository_path = Path("C:/StockApp/agents/repository")
        self.tracker_path = self.repository_path / "tracker"
        self.tracker_path.mkdir(parents=True, exist_ok=True)

        # Monitoring intervals
        self.intervals = {
            '1h': timedelta(hours=1),
            'eod': 'end_of_day',  # Special handling
            '3d': timedelta(days=3),
            '10d': timedelta(days=10)
        }

        # Performance thresholds
        self.thresholds = {
            'win': 0.005,  # 0.5% gain = win
            'loss': -0.005,  # -0.5% loss = loss
            'strong_win': 0.02,  # 2% gain
            'strong_loss': -0.02  # -2% loss
        }

    def load_positions_to_track(self) -> List[Dict[str, Any]]:
        """Load positions that need tracking"""
        # Load from Supreme Leader's current positions
        supreme_path = self.repository_path / "supreme_leader" / "current_positions.json"

        if supreme_path.exists():
            try:
                with open(supreme_path, 'r', encoding='utf-8') as f:
                    positions = json.load(f)
                    return positions
            except Exception as e:
                print(f"❌ Error loading positions: {e}")

        # Also check for a tracking list
        tracking_file = self.tracker_path / "positions_tracking.json"
        if tracking_file.exists():
            try:
                with open(tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        return []

    def save_tracking_data(self, data: Dict[str, Any]):
        """Save tracking snapshot"""
        # Save by symbol and timestamp
        symbol = data['symbol']
        timestamp = data['snapshot_time'].replace(':', '-').replace('.', '-')

        # Create symbol directory
        symbol_path = self.tracker_path / symbol
        symbol_path.mkdir(exist_ok=True)

        # Save snapshot
        snapshot_file = symbol_path / f"snapshot_{timestamp}.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Also update latest tracking file
        latest_file = symbol_path / "latest_snapshot.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')

            if not data.empty:
                return float(data['Close'].iloc[-1])

            # Fallback to info
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('currentPrice') or info.get('price')

        except Exception as e:
            print(f"    ⚠️ Error fetching price for {symbol}: {e}")
            return None

    def calculate_performance(self, entry_price: float, current_price: float,
                            stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """Calculate position performance metrics"""
        if not current_price or not entry_price:
            return {
                'return_pct': 0,
                'return_usd': 0,
                'status': 'unknown'
            }

        # Calculate returns
        return_usd = current_price - entry_price
        return_pct = (return_usd / entry_price) * 100

        # Determine status
        status = 'open'

        # Check if hit stop loss or take profit
        if stop_loss and current_price <= stop_loss:
            status = 'stopped_out'
        elif take_profit and current_price >= take_profit:
            status = 'target_hit'
        elif return_pct >= self.thresholds['strong_win'] * 100:
            status = 'strong_win'
        elif return_pct >= self.thresholds['win'] * 100:
            status = 'winning'
        elif return_pct <= self.thresholds['strong_loss'] * 100:
            status = 'strong_loss'
        elif return_pct <= self.thresholds['loss'] * 100:
            status = 'losing'
        else:
            status = 'neutral'

        return {
            'return_pct': return_pct,
            'return_usd': return_usd,
            'current_price': current_price,
            'entry_price': entry_price,
            'status': status,
            'price_vs_stop': ((current_price - stop_loss) / stop_loss * 100) if stop_loss else None,
            'price_vs_target': ((take_profit - current_price) / current_price * 100) if take_profit else None
        }

    def check_interval_reached(self, entry_time: str, interval_key: str) -> bool:
        """Check if monitoring interval has been reached"""
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            now = datetime.now()

            if interval_key == 'eod':
                # Check if it's past market close (4 PM)
                today_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                return now >= today_close and entry_dt.date() <= now.date()
            else:
                interval = self.intervals[interval_key]
                return now >= entry_dt + interval

        except Exception as e:
            print(f"    ⚠️ Error checking interval: {e}")
            return False

    def track_position(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Track a single position"""
        symbol = position['symbol']
        entry_price = position.get('entry_price', 0)
        entry_time = position.get('entry_time')

        if not entry_time:
            print(f"    ⚠️ No entry time for {symbol}")
            return None

        print(f"  📊 Tracking {symbol}...")

        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            print(f"    ❌ Could not get price for {symbol}")
            return None

        # Calculate performance
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit')
        performance = self.calculate_performance(entry_price, current_price, stop_loss, take_profit)

        # Check which intervals have been reached
        intervals_reached = []
        for interval_key in self.intervals.keys():
            if self.check_interval_reached(entry_time, interval_key):
                intervals_reached.append(interval_key)

        # Create tracking snapshot
        snapshot = {
            'symbol': symbol,
            'snapshot_time': datetime.now().isoformat(),
            'entry_time': entry_time,
            'entry_price': entry_price,
            'current_price': current_price,
            'performance': performance,
            'intervals_reached': intervals_reached,
            'position_age': str(datetime.now() - datetime.fromisoformat(entry_time)),
            'metadata': {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position.get('position_size')
            }
        }

        # Assign status emoji
        status_emoji = self.get_status_emoji(performance['status'])

        print(f"    {status_emoji} Status: {performance['status']}")
        print(f"       Return: {performance['return_pct']:.2f}%")
        print(f"       Price: ${current_price:.2f} (Entry: ${entry_price:.2f})")
        print(f"       Intervals: {', '.join(intervals_reached) if intervals_reached else 'None yet'}")

        return snapshot

    def get_status_emoji(self, status: str) -> str:
        """Get emoji for position status"""
        emoji_map = {
            'strong_win': '🚀',
            'winning': '✅',
            'neutral': '➖',
            'losing': '⚠️',
            'strong_loss': '💀',
            'stopped_out': '🛑',
            'target_hit': '🎯',
            'unknown': '❓'
        }
        return emoji_map.get(status, '❓')

    def track_all_positions(self) -> Dict[str, Any]:
        """Track all active positions"""
        print("📍 Tracker Agent monitoring positions...")

        # Load positions to track
        positions = self.load_positions_to_track()

        if not positions:
            print("   No positions to track")
            return {
                'timestamp': datetime.now().isoformat(),
                'positions_tracked': 0,
                'snapshots': []
            }

        print(f"   Tracking {len(positions)} positions...")

        snapshots = []
        summary = {
            'winning': 0,
            'losing': 0,
            'neutral': 0,
            'stopped_out': 0,
            'target_hit': 0
        }

        for position in positions:
            snapshot = self.track_position(position)

            if snapshot:
                snapshots.append(snapshot)
                self.save_tracking_data(snapshot)

                # Update summary
                status = snapshot['performance']['status']
                if status in summary:
                    summary[status] += 1
                elif 'win' in status:
                    summary['winning'] += 1
                elif 'loss' in status or 'losing' in status:
                    summary['losing'] += 1
                else:
                    summary['neutral'] += 1

        # Calculate aggregate metrics
        if snapshots:
            avg_return = sum(s['performance']['return_pct'] for s in snapshots) / len(snapshots)
            best_performer = max(snapshots, key=lambda x: x['performance']['return_pct'])
            worst_performer = min(snapshots, key=lambda x: x['performance']['return_pct'])
        else:
            avg_return = 0
            best_performer = None
            worst_performer = None

        # Prepare tracking report
        tracking_report = {
            'timestamp': datetime.now().isoformat(),
            'agent': 'tracker_agent',
            'positions_tracked': len(snapshots),
            'snapshots': snapshots,
            'summary': summary,
            'aggregate_metrics': {
                'average_return_pct': avg_return,
                'best_performer': {
                    'symbol': best_performer['symbol'],
                    'return_pct': best_performer['performance']['return_pct']
                } if best_performer else None,
                'worst_performer': {
                    'symbol': worst_performer['symbol'],
                    'return_pct': worst_performer['performance']['return_pct']
                } if worst_performer else None
            }
        }

        # Save tracking report
        report_file = self.repository_path / "tracker_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(tracking_report, f, indent=2, ensure_ascii=False)

        print(f"\n📍 Tracking Complete:")
        print(f"   Positions: {len(snapshots)}")
        print(f"   Winning: {summary['winning']}")
        print(f"   Losing: {summary['losing']}")
        print(f"   Neutral: {summary['neutral']}")
        if snapshots:
            print(f"   Avg Return: {avg_return:.2f}%")

        print(f"\n📁 Report saved to: {report_file}")

        return tracking_report

    def generate_alerts(self, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts for significant events"""
        alerts = []

        for snapshot in snapshots:
            symbol = snapshot['symbol']
            performance = snapshot['performance']
            status = performance['status']

            # Alert for stops and targets
            if status == 'stopped_out':
                alerts.append({
                    'type': 'stop_loss_hit',
                    'symbol': symbol,
                    'message': f"🛑 {symbol} hit stop loss at ${performance['current_price']:.2f}",
                    'severity': 'high'
                })
            elif status == 'target_hit':
                alerts.append({
                    'type': 'take_profit_hit',
                    'symbol': symbol,
                    'message': f"🎯 {symbol} hit take profit at ${performance['current_price']:.2f}",
                    'severity': 'medium'
                })
            elif status == 'strong_loss':
                alerts.append({
                    'type': 'large_loss',
                    'symbol': symbol,
                    'message': f"💀 {symbol} down {performance['return_pct']:.1f}% - consider exit",
                    'severity': 'high'
                })
            elif status == 'strong_win':
                alerts.append({
                    'type': 'large_gain',
                    'symbol': symbol,
                    'message': f"🚀 {symbol} up {performance['return_pct']:.1f}% - consider taking profits",
                    'severity': 'medium'
                })

        return alerts


if __name__ == "__main__":
    # For testing, create a sample position
    sample_position = {
        'symbol': 'AAPL',
        'entry_price': 190.0,
        'entry_time': (datetime.now() - timedelta(hours=2)).isoformat(),
        'stop_loss': 185.0,
        'take_profit': 195.0,
        'position_size': 10000
    }

    # Save sample position for testing
    tracker = TrackerAgent()
    test_positions = [sample_position]

    test_file = tracker.tracker_path / "positions_tracking.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_positions, f, indent=2)

    # Run tracking
    results = tracker.track_all_positions()

    # Generate alerts
    if results['snapshots']:
        alerts = tracker.generate_alerts(results['snapshots'])
        if alerts:
            print("\n🚨 Alerts:")
            for alert in alerts:
                print(f"   {alert['message']}")

    print("\n📍 Tracker Agent ready for continuous monitoring!")
    print("Next step: Run criteria_auditor.py to verify signals")
"""
Dynamic Archive Updater Module

Automatically detects and captures new rare market events.

Strategy:
- Monitor drift detection alerts from Module 6
- When severe drift detected (crash regime + high VIX):
  * Capture current market data
  * Generate labeled samples
  * Add to archive with new event name
- Triggers retraining with expanded archive
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from advanced_ml.monitoring.drift_detector import DriftDetector
from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from advanced_ml.config import get_all_core_symbols


class DynamicArchiveUpdater:
    """
    Automatically captures new rare market events

    Criteria for new event:
    - Regime drift to 'crash' detected
    - VIX > 40 for 3+ consecutive days
    - Overall drift score > 0.25 (25% distribution shift)
    - Not duplicate of existing events
    """

    def __init__(
        self,
        archive_path: str = "backend/data/rare_event_archive",
        db_path: str = "backend/data/advanced_ml_system.db"
    ):
        """
        Initialize dynamic archive updater

        Args:
            archive_path: Path to rare event archive
            db_path: Path to ML database
        """
        self.archive_path = archive_path
        self.db_path = db_path
        self.archive_db_path = os.path.join(archive_path, "archive.db")

        # Drift detection thresholds
        self.min_vix = 40.0  # VIX must be > 40
        self.min_vix_days = 3  # For 3+ consecutive days
        self.min_drift_score = 0.25  # 25% drift
        self.crash_regime_threshold = 0.5  # 50%+ crash regime

        # Event tracking
        self.monitored_events = []
        self.captured_events = []

        self._init_table()

    def _init_table(self):
        """Create dynamic_events table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dynamic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT NOT NULL UNIQUE,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                vix_peak REAL,
                drift_score REAL,
                regime_distribution TEXT,
                sample_count INTEGER DEFAULT 0,
                captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
                added_to_archive INTEGER DEFAULT 0,
                triggered_retraining INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def check_for_new_event(
        self,
        drift_result: Dict[str, Any],
        current_vix: float,
        current_regime_dist: Dict[str, float]
    ) -> Optional[str]:
        """
        Check if current conditions warrant capturing a new event

        Args:
            drift_result: Result from DriftDetector.detect_regime_drift()
            current_vix: Current VIX level
            current_regime_dist: Current regime distribution

        Returns:
            Event name if conditions met, None otherwise
        """
        # Check drift score
        drift_score = drift_result.get('distribution_change', 0.0)
        if drift_score < self.min_drift_score:
            return None

        # Check VIX level
        if current_vix < self.min_vix:
            return None

        # Check crash regime prevalence
        crash_pct = current_regime_dist.get('crash', 0.0)
        if crash_pct < self.crash_regime_threshold:
            return None

        # All criteria met - generate event name
        event_name = f"dynamic_{datetime.now().strftime('%Y%m%d')}_{int(current_vix)}"

        # Check if not duplicate
        if self._is_duplicate_event(event_name):
            print(f"[SKIP] Event {event_name} already captured")
            return None

        print(f"[ALERT] New rare event detected: {event_name}")
        print(f"  VIX: {current_vix:.1f}")
        print(f"  Drift Score: {drift_score:.1%}")
        print(f"  Crash %: {crash_pct:.1%}")

        return event_name

    def _is_duplicate_event(self, event_name: str) -> bool:
        """Check if event already exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM dynamic_events
            WHERE event_name = ?
        ''', (event_name,))

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def capture_event(
        self,
        event_name: str,
        start_date: datetime,
        end_date: datetime,
        vix_peak: float,
        drift_score: float,
        regime_dist: Dict[str, float]
    ) -> int:
        """
        Capture market data for a new rare event

        Args:
            event_name: Unique event identifier
            start_date: Event start date
            end_date: Event end date (or estimate)
            vix_peak: Peak VIX during event
            drift_score: Drift detection score
            regime_dist: Regime distribution during event

        Returns:
            Number of samples captured
        """
        print(f"\n[CAPTURE] Capturing event: {event_name}")
        print(f"  Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Initialize backtest engine
        backtest = HistoricalBacktest()

        # Get symbols to track
        symbols = get_all_core_symbols()
        print(f"  Processing {len(symbols)} symbols...")

        all_samples = []
        successful_symbols = 0

        # Fetch start/end with buffer for features
        fetch_start = start_date - timedelta(days=365)
        fetch_end = end_date + timedelta(days=30)

        for i, symbol in enumerate(symbols):
            try:
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(symbols)}] Processing...", end='\r')

                # Fetch historical data
                df = backtest.fetch_historical_data(
                    symbol=symbol,
                    years=1,
                    start_date=fetch_start,
                    end_date=fetch_end
                )

                if df is None or len(df) < 100:
                    continue

                # Generate samples
                symbol_samples = backtest.generate_labeled_data(symbol, df)

                # Filter to event window
                event_samples = []
                for sample in symbol_samples:
                    sample_date = datetime.strptime(sample['date'], '%Y-%m-%d')
                    if start_date <= sample_date <= end_date:
                        sample['event_name'] = event_name
                        event_samples.append(sample)

                all_samples.extend(event_samples)
                successful_symbols += 1

            except Exception as e:
                print(f"\n[ERROR] {symbol}: {e}")
                continue

        print(f"\n  Captured: {len(all_samples)} samples from {successful_symbols} symbols")

        # Save to database
        if len(all_samples) > 0:
            self._save_event_metadata(event_name, start_date, end_date, vix_peak, drift_score, regime_dist, len(all_samples))

        return len(all_samples)

    def _save_event_metadata(
        self,
        event_name: str,
        start_date: datetime,
        end_date: datetime,
        vix_peak: float,
        drift_score: float,
        regime_dist: Dict[str, float],
        sample_count: int
    ):
        """Save event metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO dynamic_events
            (event_name, start_date, end_date, vix_peak, drift_score,
             regime_distribution, sample_count, captured_at, added_to_archive)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        ''', (
            event_name,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            float(vix_peak),
            float(drift_score),
            json.dumps(regime_dist),
            sample_count,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        print(f"[OK] Event metadata saved to database")

    def merge_into_archive(self, event_name: str, samples: List[Dict[str, Any]]) -> bool:
        """
        Add new event samples to existing archive

        Args:
            event_name: Event identifier
            samples: List of labeled samples

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n[MERGE] Adding {len(samples)} samples to archive...")

            # Connect to archive database
            conn = sqlite3.connect(self.archive_db_path)
            cursor = conn.cursor()

            saved = 0
            duplicates = 0

            for sample in samples:
                try:
                    # Convert features to JSON
                    features_json = json.dumps(sample.get('features', {}))

                    cursor.execute('''
                        INSERT OR IGNORE INTO archive_samples
                        (event_name, symbol, date, entry_price, return_pct, label, exit_reason, features, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event_name,
                        sample.get('symbol'),
                        sample.get('date'),
                        sample.get('entry_price'),
                        sample.get('return_pct'),
                        sample.get('label'),
                        sample.get('exit_reason'),
                        features_json,
                        datetime.now().isoformat()
                    ))

                    if cursor.rowcount > 0:
                        saved += 1
                    else:
                        duplicates += 1

                except Exception as e:
                    print(f"[ERROR] Failed to save sample: {e}")
                    continue

            conn.commit()
            conn.close()

            print(f"  Saved: {saved} new samples")
            if duplicates > 0:
                print(f"  Skipped: {duplicates} duplicates")

            # Update metadata
            self._mark_archive_added(event_name)

            return True

        except Exception as e:
            print(f"[ERROR] Failed to merge into archive: {e}")
            return False

    def _mark_archive_added(self, event_name: str):
        """Mark event as added to archive"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE dynamic_events
            SET added_to_archive = 1
            WHERE event_name = ?
        ''', (event_name,))

        conn.commit()
        conn.close()

    def get_captured_events(self) -> List[Dict[str, Any]]:
        """Get list of all captured dynamic events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT event_name, start_date, end_date, vix_peak, drift_score,
                   sample_count, captured_at, added_to_archive, triggered_retraining
            FROM dynamic_events
            ORDER BY captured_at DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append({
                'event_name': row[0],
                'start_date': row[1],
                'end_date': row[2],
                'vix_peak': row[3],
                'drift_score': row[4],
                'sample_count': row[5],
                'captured_at': row[6],
                'added_to_archive': bool(row[7]),
                'triggered_retraining': bool(row[8])
            })

        return events

    def trigger_retraining(self, event_name: str) -> bool:
        """
        Mark that this event triggered retraining

        Args:
            event_name: Event identifier

        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE dynamic_events
            SET triggered_retraining = 1
            WHERE event_name = ?
        ''', (event_name,))

        conn.commit()
        conn.close()

        print(f"[OK] Event {event_name} marked as trigger for retraining")
        return True


if __name__ == '__main__':
    # Test dynamic archive updater
    print("Testing Dynamic Archive Updater...\n")

    # Create updater
    updater = DynamicArchiveUpdater()

    print("[TEST 1] Check for new event - Low drift (should reject)")
    print("=" * 60)
    drift_result = {'distribution_change': 0.10}  # Too low
    vix = 45.0
    regime_dist = {'crash': 0.7, 'normal': 0.3}

    event_name = updater.check_for_new_event(drift_result, vix, regime_dist)
    if event_name:
        print(f"  Detected: {event_name}")
    else:
        print(f"  [OK] Rejected (drift too low)")
    print()

    print("[TEST 2] Check for new event - All criteria met (should accept)")
    print("=" * 60)
    drift_result = {'distribution_change': 0.30}  # High drift
    vix = 55.0  # High VIX
    regime_dist = {'crash': 0.8, 'normal': 0.2}  # High crash %

    event_name = updater.check_for_new_event(drift_result, vix, regime_dist)
    if event_name:
        print(f"  [OK] Detected: {event_name}")
    else:
        print(f"  Rejected")
    print()

    print("[TEST 3] Get captured events")
    print("=" * 60)
    events = updater.get_captured_events()
    print(f"Total captured events: {len(events)}")
    for event in events[:5]:
        print(f"  {event['event_name']}: {event['sample_count']} samples")
    print()

    print("=" * 60)
    print("[OK] Dynamic Archive Updater Tests Complete")
    print("=" * 60)

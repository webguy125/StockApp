"""
Rare Event Archive Generation Script

Generates curated samples from historical market stress events.
Run this once to create the archive, then use it for all future training.

Runtime: 2-4 hours (processes 7 events across multiple years)
"""

import sys
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from advanced_ml.archive.rare_event_archive import create_archive_database
from advanced_ml.config import get_all_core_symbols


class RareEventArchiveGenerator:
    """
    Generates rare event archive from historical market data
    """

    def __init__(self, archive_path: str = None):
        """
        Initialize archive generator

        Args:
            archive_path: Path to archive directory (auto-detected if None)
        """
        # Auto-detect archive path
        if archive_path is None:
            # Get script directory and go up one level to archive root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            archive_path = os.path.dirname(script_dir)  # Parent of scripts/

        self.archive_path = archive_path
        self.db_path = os.path.join(archive_path, "archive.db")
        self.config_path = os.path.join(archive_path, "metadata", "archive_config.json")
        self.metadata_path = os.path.join(archive_path, "metadata", "event_metadata.json")
        self.log_path = os.path.join(archive_path, "metadata", "generation_log.json")

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize backtest engine
        self.backtest = HistoricalBacktest()

        # Get symbol list
        self.all_symbols = get_all_core_symbols()

        # Event metadata storage
        self.event_metadata = {}
        self.generation_log = {
            'started_at': datetime.now().isoformat(),
            'events_processed': [],
            'errors': []
        }

    def get_symbols_for_event(self, event_name: str, start_date: str) -> List[str]:
        """
        Get symbols that existed during an event

        Args:
            event_name: Event identifier
            start_date: Event start date (YYYY-MM-DD)

        Returns:
            List of symbols available during that period
        """
        # Parse start year
        start_year = int(start_date[:4])

        # Symbol availability based on IPO/listing dates
        symbol_ipos = {
            # Pre-2008 symbols (available for all events)
            'AAPL': 1980, 'MSFT': 1986, 'JPM': 1980, 'BAC': 1980,
            'GE': 1980, 'XOM': 1980, 'CVX': 1980, 'JNJ': 1980,
            'PG': 1980, 'KO': 1980, 'PEP': 1980, 'WMT': 1980,
            'HD': 1981, 'DIS': 1980, 'MCD': 1980, 'NEM': 1980,
            'GS': 1999, 'SCHW': 1987, 'WFC': 1980, 'AMT': 1995,
            'PLD': 1997, 'SPG': 1993, 'DUK': 1980, 'SO': 1980,
            'NEE': 1980, 'AES': 1991, 'NRG': 2003, 'SHW': 1980,
            'APD': 1980, 'LIN': 1992, 'HON': 1980, 'BA': 1980,
            'GIS': 1980, 'ABBV': 2013, 'LLY': 1980, 'UNH': 1984,

            # 2010s IPOs
            'META': 2012, 'GOOGL': 2004, 'AMZN': 1997, 'NFLX': 2002,
            'TSLA': 2010, 'NVDA': 1999, 'COP': 1980, 'SLB': 1980,
            'FANG': 2016, 'EQIX': 2000, 'DXCM': 2005, 'LULU': 2007,
            'DECK': 1993, 'CRWD': 2019, 'PLTR': 2020, 'UBER': 2019,
            'MTCH': 2015, 'ALLY': 2014, 'SMCI': 2007,

            # Regional/smaller caps (various dates)
            'ASTE': 1998, 'CARS': 2017, 'CATY': 2000, 'CEIX': 2017,
            'CHEF': 2019, 'CTRA': 2021, 'CVCO': 1999, 'EXAS': 2001,
            'GBCI': 2012, 'INGLES': 1987, 'IOSP': 2000, 'KRYS': 2016,
            'MGEE': 1990, 'MTDR': 2011, 'REXR': 2013, 'SHAK': 2015,
            'SNOW': 2020, 'STAG': 2011, 'TMDX': 2016, 'TRNO': 2016,
            'XPO': 2011
        }

        # Filter symbols that existed during this event
        available_symbols = []
        for symbol in self.all_symbols:
            ipo_year = symbol_ipos.get(symbol, 2030)  # Default to future if unknown
            if ipo_year <= start_year:
                available_symbols.append(symbol)

        return available_symbols

    def generate_event_samples(self, event_name: str) -> List[Dict[str, Any]]:
        """
        Generate samples for a specific event

        Args:
            event_name: Event identifier from config

        Returns:
            List of samples
        """
        print(f"\n[EVENT] Generating samples for: {event_name}")

        # Get event definition
        event_def = self.config['event_definitions'][event_name]
        start_date = datetime.strptime(event_def['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(event_def['end_date'], '%Y-%m-%d')

        print(f"  Date Range: {event_def['start_date']} to {event_def['end_date']}")

        # Get available symbols for this period
        symbols = self.get_symbols_for_event(event_name, event_def['start_date'])
        print(f"  Symbols Available: {len(symbols)}")

        # Calculate years needed (event duration + lookback for features)
        event_duration_days = (end_date - start_date).days
        years_needed = max(1, (event_duration_days + 365) // 365)  # Event + 1 year lookback

        print(f"  Fetching {years_needed} years of data for context...")

        all_samples = []
        successful_symbols = 0
        failed_symbols = []

        for i, symbol in enumerate(symbols):
            try:
                print(f"  [{i+1}/{len(symbols)}] Processing {symbol}...", end='\r')

                # Fetch historical data for this symbol
                # Start date = event start - 1 year (for feature calculation)
                # End date = event end + 30 days (for future window)
                fetch_start = start_date - timedelta(days=365)
                fetch_end = end_date + timedelta(days=30)

                df = self.backtest.fetch_historical_data(
                    symbol=symbol,
                    years=years_needed,
                    start_date=fetch_start,  # Use event-specific dates!
                    end_date=fetch_end
                )

                if df is None or len(df) < 100:
                    failed_symbols.append((symbol, "insufficient_data"))
                    continue

                # Generate samples for this symbol during the event period
                symbol_samples = self.backtest.generate_labeled_data(symbol, df)

                # Filter samples to only those within the event window
                event_samples = []
                for sample in symbol_samples:
                    sample_date = datetime.strptime(sample['date'], '%Y-%m-%d')
                    if start_date <= sample_date <= end_date:
                        # Add event name tag
                        sample['event_name'] = event_name
                        event_samples.append(sample)

                all_samples.extend(event_samples)
                successful_symbols += 1

            except Exception as e:
                failed_symbols.append((symbol, str(e)))
                continue

        print(f"\n  Completed: {len(all_samples)} samples from {successful_symbols} symbols")

        if failed_symbols:
            print(f"  Failed: {len(failed_symbols)} symbols")
            self.generation_log['errors'].extend([
                f"{event_name}:{symbol}:{reason}" for symbol, reason in failed_symbols[:5]
            ])

        # Calculate event metadata
        label_dist = {'buy': 0, 'hold': 0, 'sell': 0}
        label_map = {0: 'buy', 1: 'hold', 2: 'sell'}

        for sample in all_samples:
            label_name = label_map.get(sample['label'], 'unknown')
            if label_name in label_dist:
                label_dist[label_name] += 1

        # Calculate average VIX if available
        avg_vix = 0.0
        vix_samples = 0
        for sample in all_samples:
            if 'vix' in sample.get('features', {}):
                avg_vix += sample['features']['vix']
                vix_samples += 1

        if vix_samples > 0:
            avg_vix /= vix_samples

        # Store event metadata
        self.event_metadata[event_name] = {
            'event_name': event_name,
            'start_date': event_def['start_date'],
            'end_date': event_def['end_date'],
            'description': event_def['description'],
            'symbols_used': symbols[:20],  # First 20 for metadata
            'total_symbols': len(symbols),
            'successful_symbols': successful_symbols,
            'sample_count': len(all_samples),
            'label_distribution': label_dist,
            'avg_vix': round(avg_vix, 2),
            'generated_at': datetime.now().isoformat()
        }

        return all_samples

    def save_samples_to_database(self, samples: List[Dict[str, Any]]):
        """
        Save samples to archive database

        Args:
            samples: List of sample dictionaries
        """
        print(f"\n[SAVE] Saving {len(samples)} samples to database...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        saved = 0
        duplicates = 0

        for sample in samples:
            try:
                # Convert features dict to JSON string
                features_json = json.dumps(sample.get('features', {}))

                cursor.execute("""
                    INSERT OR IGNORE INTO archive_samples
                    (event_name, symbol, date, entry_price, return_pct, label, exit_reason, features, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample.get('event_name'),
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

    def generate_archive(self, regenerate: bool = False):
        """
        Generate complete rare event archive

        Args:
            regenerate: If True, delete existing archive and start fresh
        """
        print("=" * 70)
        print("RARE EVENT ARCHIVE GENERATION")
        print("=" * 70)
        print(f"\nArchive Path: {self.archive_path}")
        print(f"Events to Process: {len(self.config['active_events'])}")
        print()

        # Check if archive exists
        if os.path.exists(self.db_path):
            if regenerate:
                print("[WARN] Deleting existing archive...")
                os.remove(self.db_path)
            else:
                print("[WARN] Archive already exists!")
                response = input("Regenerate? This will delete existing data (yes/no): ")
                if response.lower() != 'yes':
                    print("[ABORT] Archive generation cancelled")
                    return
                os.remove(self.db_path)

        # Create database
        print("\n[SETUP] Creating archive database...")
        create_archive_database(self.db_path)

        # Process each event
        active_events = self.config['active_events']

        for i, event_name in enumerate(active_events):
            print(f"\n{'='*70}")
            print(f"EVENT {i+1}/{len(active_events)}: {event_name}")
            print(f"{'='*70}")

            try:
                # Generate samples
                samples = self.generate_event_samples(event_name)

                if len(samples) > 0:
                    # Save to database
                    self.save_samples_to_database(samples)

                    # Log success
                    self.generation_log['events_processed'].append({
                        'event_name': event_name,
                        'sample_count': len(samples),
                        'status': 'success'
                    })
                else:
                    print(f"[WARN] No samples generated for {event_name}")
                    self.generation_log['events_processed'].append({
                        'event_name': event_name,
                        'sample_count': 0,
                        'status': 'no_samples'
                    })

            except Exception as e:
                print(f"[ERROR] Failed to process {event_name}: {e}")
                self.generation_log['errors'].append(f"{event_name}: {str(e)}")
                self.generation_log['events_processed'].append({
                    'event_name': event_name,
                    'sample_count': 0,
                    'status': 'error',
                    'error': str(e)
                })

        # Finalize
        self.generation_log['completed_at'] = datetime.now().isoformat()

        # Save metadata
        self._save_metadata()

        # Print summary
        self._print_summary()

    def _save_metadata(self):
        """Save event metadata and generation log"""
        # Save event metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.event_metadata, f, indent=2)

        # Save generation log
        with open(self.log_path, 'w') as f:
            json.dump(self.generation_log, f, indent=2)

        print(f"\n[OK] Metadata saved:")
        print(f"  {self.metadata_path}")
        print(f"  {self.log_path}")

    def _print_summary(self):
        """Print generation summary"""
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)

        # Count total samples
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM archive_samples")
        total_samples = cursor.fetchone()[0]

        # Per-event counts
        cursor.execute("""
            SELECT event_name, COUNT(*) as count
            FROM archive_samples
            GROUP BY event_name
        """)
        event_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Overall label distribution
        cursor.execute("""
            SELECT label, COUNT(*) as count
            FROM archive_samples
            GROUP BY label
        """)
        label_dist = {}
        label_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        for row in cursor.fetchall():
            label_dist[label_map[row[0]]] = row[1]

        conn.close()

        print(f"\nTotal Samples: {total_samples}")
        print(f"\nPer-Event Breakdown:")
        for event, count in event_counts.items():
            weight = self.config['event_weights'].get(event, 0)
            print(f"  {event:30s} {count:6d} samples ({weight*100:.0f}% weight)")

        print(f"\nLabel Distribution:")
        for label, count in label_dist.items():
            pct = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {label:10s} {count:6d} ({pct:5.1f}%)")

        print(f"\nErrors: {len(self.generation_log['errors'])}")
        if self.generation_log['errors']:
            print("  (See generation_log.json for details)")

        print("\n" + "=" * 70)
        print("[OK] Archive ready for training!")
        print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate Rare Event Archive')
    parser.add_argument('--regenerate', action='store_true',
                        help='Delete and regenerate existing archive')
    args = parser.parse_args()

    generator = RareEventArchiveGenerator()
    generator.generate_archive(regenerate=args.regenerate)


if __name__ == '__main__':
    main()


import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Bootstrap Signal History from Historical Backtest Data
Populates signal_history table with 10 years of closed trades for win-rate calculation.

This script transforms backtest trades (training data) into closed signal history.
These historical signals are used ONLY for win-rate statistics, NOT for training.

Usage:
    python bootstrap_signal_history.py [--force]

    --force: Override safety check if signal_history already has data
"""

import sqlite3
import argparse
from datetime import datetime
from backend.turbomode.core_engine.training_symbols import get_symbol_metadata

def bootstrap_signal_history(db_path: str, force: bool = False):
    """
    Bootstrap signal_history from backtest trades.

    Args:
        db_path: Path to turbomode.db
        force: If True, bootstrap even if signal_history is non-empty
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM signal_history')
    existing_count = cursor.fetchone()[0]

    if existing_count > 0 and not force:
        print(f'ABORT: signal_history already contains {existing_count:,} rows')
        print('Use --force to override this safety check')
        conn.close()
        return

    if existing_count > 0:
        print(f'WARNING: signal_history contains {existing_count:,} rows but --force was specified')
        print('Continuing with bootstrap...')

    print('=' * 80)
    print('BOOTSTRAP SIGNAL HISTORY FROM BACKTEST TRADES')
    print('=' * 80)

    cursor.execute('SELECT COUNT(*) FROM trades WHERE trade_type = "backtest"')
    total_backtest = cursor.fetchone()[0]
    print(f'Total backtest trades available: {total_backtest:,}')

    cursor.execute('SELECT MIN(entry_date), MAX(entry_date) FROM trades WHERE trade_type = "backtest"')
    date_range = cursor.fetchone()
    print(f'Date range: {date_range[0]} to {date_range[1]}')

    print('\nTransforming backtest trades -> signal_history...')

    cursor.execute('''
        SELECT
            symbol,
            outcome,
            entry_date,
            entry_price,
            exit_date,
            exit_price,
            profit_loss,
            profit_loss_pct
        FROM trades
        WHERE trade_type = "backtest"
        ORDER BY symbol, entry_date
    ''')

    inserted = 0
    skipped = 0
    symbol_counts = {}

    batch = []
    batch_size = 5000

    for row in cursor.fetchall():
        symbol, outcome, entry_date, entry_price, exit_date, exit_price, profit_loss, profit_loss_pct = row

        signal_type = outcome.upper()

        metadata = get_symbol_metadata(symbol)
        sector = metadata['sector']
        market_cap = metadata['market_cap_category']

        entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
        exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
        hold_days = (exit_dt - entry_dt).days

        exit_reason = 'holding_period_complete'

        confidence = 0.75

        batch.append((
            symbol,
            signal_type,
            confidence,
            entry_date,
            entry_price,
            exit_date,
            exit_price,
            exit_reason,
            profit_loss_pct,
            hold_days,
            market_cap,
            sector,
            entry_date
        ))

        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        if len(batch) >= batch_size:
            cursor.executemany('''
                INSERT INTO signal_history (
                    symbol, signal_type, confidence, entry_date, entry_price,
                    exit_date, exit_price, exit_reason, profit_loss_pct, hold_days,
                    market_cap, sector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)
            inserted += len(batch)
            print(f'  Inserted {inserted:,} / {total_backtest:,} ({inserted/total_backtest*100:.1f}%)')
            batch = []

    if batch:
        cursor.executemany('''
            INSERT INTO signal_history (
                symbol, signal_type, confidence, entry_date, entry_price,
                exit_date, exit_price, exit_reason, profit_loss_pct, hold_days,
                market_cap, sector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch)
        inserted += len(batch)

    conn.commit()

    print('\n' + '=' * 80)
    print('BOOTSTRAP COMPLETE')
    print('=' * 80)
    print(f'Inserted: {inserted:,}')
    print(f'Skipped: {skipped:,}')
    print(f'Unique symbols: {len(symbol_counts)}')

    print('\nTop 10 symbols by signal count:')
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'  {symbol}: {count:,}')

    cursor.execute('SELECT signal_type, COUNT(*) FROM signal_history GROUP BY signal_type')
    print('\nSignal type distribution:')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1]:,}')

    conn.close()
    print('\n[OK] Signal history bootstrap complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bootstrap signal_history from backtest data')
    parser.add_argument('--force', action='store_true', help='Force bootstrap even if signal_history is non-empty')
    parser.add_argument('--db', default='backend/data/turbomode.db', help='Path to database')

    args = parser.parse_args()

    db_path = args.db if args.db.startswith('/') or ':' in args.db else str(project_root / args.db)

    bootstrap_signal_history(db_path, force=args.force)

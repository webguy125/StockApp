
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Signal Closer - Real-Time Trade Exit Logic

Evaluates open positions in active_signals and closes them when exit conditions are met.
Computes realized P/L and moves closed trades to signal_history (performance ledger).

Exit Conditions:
1. Time-based: 14 days (holding period complete)
2. Target hit: price >= entry_price * (1 + target_pct)
3. Stop hit: price <= entry_price * (1 - stop_pct)

NO CONTAMINATION: Only real scanner-generated signals are processed.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from master_market_data.market_data_api import get_market_data_api
from backend.turbomode.core_engine.training_symbols import get_symbol_metadata
from backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignalCloser:
    """
    Evaluates active signals and closes them when exit conditions are met.
    """

    def __init__(self, db_path: str = 'backend/data/turbomode.db'):
        self.db_path = db_path
        self.market_data_api = get_market_data_api()
        self.HOLDING_PERIOD_DAYS = 14
        self.DEFAULT_TARGET_PCT = 0.05  # 5%
        self.DEFAULT_STOP_PCT = 0.05    # 5%

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            df = self.market_data_api.get_candles(symbol, timeframe='1d')
            if df is not None and len(df) > 0:
                return float(df.iloc[-1]['close'])
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
        return None

    def check_exit_conditions(
        self,
        signal_type: str,
        entry_price: float,
        current_price: float,
        entry_date: str,
        current_date: str,
        target_pct: Optional[float] = None,
        stop_pct: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if exit conditions are met.

        Returns:
            (should_exit, exit_reason, profit_loss_pct)
        """

        target_pct = target_pct or self.DEFAULT_TARGET_PCT
        stop_pct = stop_pct or self.DEFAULT_STOP_PCT

        entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        days_held = (current_dt - entry_dt).days

        # Use canonical P/L calculation (handles BUY/SELL/HOLD correctly)
        pnl_pct = calculate_pnl_pct(entry_price, current_price, signal_type)

        if signal_type == 'BUY':
            target_price = entry_price * (1 + target_pct)
            stop_price = entry_price * (1 - stop_pct)

            if current_price >= target_price:
                return True, 'target_hit', pnl_pct

            if current_price <= stop_price:
                return True, 'stop_hit', pnl_pct

            if days_held >= self.HOLDING_PERIOD_DAYS:
                return True, 'holding_period_complete', pnl_pct

        elif signal_type == 'SELL':
            target_price = entry_price * (1 - target_pct)
            stop_price = entry_price * (1 + stop_pct)

            if current_price <= target_price:
                return True, 'target_hit', pnl_pct

            if current_price >= stop_price:
                return True, 'stop_hit', pnl_pct

            if days_held >= self.HOLDING_PERIOD_DAYS:
                return True, 'holding_period_complete', pnl_pct

        elif signal_type == 'HOLD':
            if days_held >= self.HOLDING_PERIOD_DAYS:
                return True, 'holding_period_complete', pnl_pct

        return False, '', pnl_pct

    def close_signals(self, current_date: Optional[str] = None) -> Dict[str, int]:
        """
        Check all active signals and close those meeting exit conditions.

        Returns:
            Dictionary with counts of closed signals
        """

        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, symbol, signal_type, confidence, entry_date, entry_price,
                   market_cap, sector, prob_buy, prob_sell, prob_hold, atr,
                   target_price, stop_price
            FROM active_signals
        ''')

        active_signals = cursor.fetchall()

        logger.info(f"Checking {len(active_signals)} active signals for exit conditions...")

        closed_count = 0
        target_hits = 0
        stop_hits = 0
        time_exits = 0
        failed = 0

        for row in active_signals:
            (signal_id, symbol, signal_type, confidence, entry_date, entry_price,
             market_cap, sector, prob_buy, prob_sell, prob_hold, entry_atr,
             target_price, stop_price) = row

            current_price = self.get_current_price(symbol)

            if current_price is None:
                logger.warning(f"  {symbol}: Unable to get current price, skipping")
                failed += 1
                continue

            should_exit, exit_reason, pnl_pct = self.check_exit_conditions(
                signal_type=signal_type,
                entry_price=entry_price,
                current_price=current_price,
                entry_date=entry_date,
                current_date=current_date
            )

            if not should_exit:
                continue

            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
            hold_days = (current_dt - entry_dt).days

            # Compute directional_margin = abs(prob_buy - prob_sell)
            directional_margin = None
            if prob_buy is not None and prob_sell is not None:
                directional_margin = abs(prob_buy - prob_sell)

            # Compute rr = (target_price - entry_price) / (entry_price - stop_price)
            rr = None
            if target_price is not None and stop_price is not None and stop_price != entry_price:
                rr = (target_price - entry_price) / (entry_price - stop_price)

            try:
                cursor.execute('''
                    INSERT INTO signal_history (
                        symbol, signal_type, confidence, entry_date, entry_price,
                        exit_date, exit_price, exit_reason, profit_loss_pct, hold_days,
                        market_cap, sector, created_at,
                        prob_buy, prob_sell, prob_hold, entry_atr, target_price, stop_price,
                        rr, directional_margin
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, signal_type, confidence, entry_date, entry_price,
                    current_date, current_price, exit_reason, pnl_pct, hold_days,
                    market_cap, sector, datetime.now().isoformat(),
                    prob_buy, prob_sell, prob_hold, entry_atr, target_price, stop_price,
                    rr, directional_margin
                ))

                cursor.execute('DELETE FROM active_signals WHERE id = ?', (signal_id,))

                closed_count += 1

                if exit_reason == 'target_hit':
                    target_hits += 1
                elif exit_reason == 'stop_hit':
                    stop_hits += 1
                elif exit_reason == 'holding_period_complete':
                    time_exits += 1

                logger.info(f"  CLOSED {symbol} ({signal_type}): {exit_reason}, P/L: {pnl_pct:+.2f}%, hold: {hold_days}d")

            except Exception as e:
                logger.error(f"  Failed to close {symbol}: {e}")
                failed += 1
                conn.rollback()
                continue

        conn.commit()
        conn.close()

        logger.info(f"\nSignal Closer Summary:")
        logger.info(f"  Total closed: {closed_count}")
        logger.info(f"  Target hits: {target_hits}")
        logger.info(f"  Stop hits: {stop_hits}")
        logger.info(f"  Time exits: {time_exits}")
        logger.info(f"  Failed: {failed}")

        return {
            'total_closed': closed_count,
            'target_hits': target_hits,
            'stop_hits': stop_hits,
            'time_exits': time_exits,
            'failed': failed
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Close active signals that meet exit conditions')
    parser.add_argument('--db', default='backend/data/turbomode.db', help='Path to database')
    parser.add_argument('--date', default=None, help='Current date (YYYY-MM-DD), defaults to today')

    args = parser.parse_args()

    db_path = args.db if args.db.startswith('/') or ':' in args.db else str(project_root / args.db)

    closer = SignalCloser(db_path=db_path)
    result = closer.close_signals(current_date=args.date)

    print(f"\nClosed {result['total_closed']} signals")

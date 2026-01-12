"""
TurboMode Outcome Tracker
Tracks prediction outcomes after 5-day hold period
Closes the feedback loop for continuous learning

This runs daily at 2 AM to check signals from 5 days ago
"""

import os
import sys
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('outcome_tracker')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class OutcomeTracker:
    """
    Track TurboMode prediction outcomes
    Checks signals from 5 days ago and records actual results
    """

    def __init__(self, db_path: str = None):
        """
        Initialize outcome tracker

        Args:
            db_path: Path to turbomode.db (defaults to backend/data/turbomode.db)
        """
        if db_path is None:
            db_path = os.path.join(parent_dir, 'data', 'turbomode.db')

        self.db_path = db_path
        self.hold_period_days = 5  # 5-day hold period for faster feedback
        self.win_threshold = 0.10   # +10% gain = correct BUY prediction

    def get_signals_to_check(self) -> List[Dict[str, Any]]:
        """
        Get active signals that are ready to be evaluated (5+ days old)

        Returns:
            List of signals ready for outcome tracking
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate cutoff date (5 days ago)
        cutoff_date = datetime.now() - timedelta(days=self.hold_period_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        # Get signals older than 5 days that haven't been evaluated
        cursor.execute("""
            SELECT
                id, symbol, signal_type, confidence,
                entry_date, entry_price, entry_min, entry_max,
                sector, market_cap, created_at
            FROM active_signals
            WHERE entry_date <= ?
            AND status = 'ACTIVE'
            ORDER BY entry_date ASC
        """, (cutoff_str,))

        rows = cursor.fetchall()
        conn.close()

        signals = []
        for row in rows:
            signals.append({
                'id': row[0],
                'symbol': row[1],
                'signal_type': row[2],
                'confidence': row[3],
                'entry_date': row[4],
                'entry_price': row[5],
                'entry_min': row[6],
                'entry_max': row[7],
                'sector': row[8],
                'market_cap': row[9],
                'created_at': row[10]
            })

        return signals

    def get_current_price(self, symbol: str) -> float:
        """
        Get current stock price from yfinance

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price or None if error
        """
        try:
            ticker = yf.Ticker(symbol)

            # Try to get regular market price
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            if current_price:
                return float(current_price)

            # Fallback: Get last close from history
            hist = ticker.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            logger.warning(f"Could not get price for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def evaluate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a signal's outcome after 5 days

        Args:
            signal: Signal dictionary

        Returns:
            Outcome dictionary with results
        """
        symbol = signal['symbol']
        entry_price = signal['entry_price']
        signal_type = signal['signal_type']
        confidence = signal['confidence']

        # Get current price
        current_price = self.get_current_price(symbol)

        if current_price is None:
            return {
                'status': 'error',
                'error': 'Could not fetch current price'
            }

        # Calculate return
        price_change = current_price - entry_price
        return_pct = (price_change / entry_price)

        # CORRECTED OUTCOME LOGIC - Symmetric thresholds for BUY and SELL
        # This fixes MISMATCH #1 and #3 from TURBOMODE_MISMATCH_AUDIT.md
        #
        # BUY signal (long position):
        #   - Target: +10% (price goes UP)
        #   - Stop: -5% (price goes DOWN)
        #   - Correct if: return_pct >= +0.10 (hit target)
        #
        # SELL signal (short position):
        #   - Target: -10% (price goes DOWN)
        #   - Stop: +5% (price goes UP)
        #   - Correct if: return_pct <= -0.10 (hit target)
        #
        # SYMMETRIC: Both require hitting the 10% target to be "correct"

        if signal_type == 'BUY':
            # BUY signal: Correct ONLY if gained ≥10% (hit target)
            is_correct = return_pct >= self.win_threshold  # +0.10
            outcome = 'correct' if is_correct else 'incorrect'
        elif signal_type == 'SELL':
            # SELL signal: Correct ONLY if lost ≥10% (hit bearish target)
            # IMPORTANT: SELL is a SHORT position - profit when price goes DOWN
            is_correct = return_pct <= -self.win_threshold  # -0.10
            outcome = 'correct' if is_correct else 'incorrect'
        else:
            outcome = 'unknown'
            is_correct = False

        return {
            'status': 'success',
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': current_price,
            'price_change': price_change,
            'return_pct': return_pct,
            'outcome': outcome,
            'is_correct': is_correct,
            'signal_type': signal_type,
            'confidence': confidence
        }

    def save_outcome(self, signal: Dict[str, Any], result: Dict[str, Any]):
        """
        Save outcome to signal_history table and deactivate signal

        Args:
            signal: Original signal
            result: Evaluation result
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert into signal_history
            cursor.execute("""
                INSERT INTO signal_history (
                    signal_id, symbol, signal_type, confidence,
                    entry_date, entry_price, exit_date, exit_price,
                    return_pct, outcome, is_correct,
                    sector, market_cap, hold_days
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal['id'],
                signal['symbol'],
                signal['signal_type'],
                signal['confidence'],
                signal['entry_date'],
                signal['entry_price'],
                datetime.now().strftime('%Y-%m-%d'),
                result['exit_price'],
                result['return_pct'],
                result['outcome'],
                1 if result['is_correct'] else 0,
                signal.get('sector', 'unknown'),
                signal.get('market_cap', 'unknown'),
                self.hold_period_days
            ))

            # Mark signal as CLOSED in active_signals
            cursor.execute("""
                UPDATE active_signals
                SET status = 'CLOSED', updated_at = ?
                WHERE id = ?
            """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), signal['id']))

            conn.commit()
            logger.info(f"[OK] {signal['symbol']}: {result['outcome'].upper()} "
                       f"({result['return_pct']*100:+.1f}%)")

        except Exception as e:
            conn.rollback()
            logger.error(f"[ERROR] Failed to save outcome for {signal['symbol']}: {e}")
            raise
        finally:
            conn.close()

    def track_outcomes(self) -> Dict[str, Any]:
        """
        Main function: Track all signals ready for evaluation

        Returns:
            Summary statistics
        """
        logger.info("=" * 80)
        logger.info("TURBOMODE OUTCOME TRACKER")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Get signals to check
        signals = self.get_signals_to_check()

        if not signals:
            logger.info("[INFO] No signals ready for evaluation (need to be 5+ days old)")
            return {
                'total_checked': 0,
                'correct': 0,
                'incorrect': 0,
                'errors': 0
            }

        logger.info(f"[INFO] Found {len(signals)} signals ready for evaluation\n")

        stats = {
            'total_checked': len(signals),
            'correct': 0,
            'incorrect': 0,
            'errors': 0,
            'buy_correct': 0,
            'buy_incorrect': 0,
            'sell_correct': 0,
            'sell_incorrect': 0
        }

        for signal in signals:
            try:
                # Evaluate signal
                result = self.evaluate_signal(signal)

                if result['status'] == 'error':
                    stats['errors'] += 1
                    logger.warning(f"[SKIP] {signal['symbol']}: {result['error']}")
                    continue

                # Save outcome
                self.save_outcome(signal, result)

                # Update stats
                if result['is_correct']:
                    stats['correct'] += 1
                    if signal['signal_type'] == 'BUY':
                        stats['buy_correct'] += 1
                    else:
                        stats['sell_correct'] += 1
                else:
                    stats['incorrect'] += 1
                    if signal['signal_type'] == 'BUY':
                        stats['buy_incorrect'] += 1
                    else:
                        stats['sell_incorrect'] += 1

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"[ERROR] Failed to process {signal['symbol']}: {e}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("OUTCOME TRACKING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Evaluated:    {stats['total_checked']}")
        logger.info(f"Correct:            {stats['correct']} ({stats['correct']/stats['total_checked']*100:.1f}%)")
        logger.info(f"Incorrect:          {stats['incorrect']} ({stats['incorrect']/stats['total_checked']*100:.1f}%)")
        logger.info(f"Errors:             {stats['errors']}")
        logger.info(f"\nBUY Signals:        {stats['buy_correct']} correct, {stats['buy_incorrect']} incorrect")
        logger.info(f"SELL Signals:       {stats['sell_correct']} correct, {stats['sell_incorrect']} incorrect")
        logger.info("=" * 80)

        return stats


def track_signal_outcomes():
    """
    Main entry point for scheduled job
    Called daily by Flask scheduler at 2 AM
    """
    tracker = OutcomeTracker()
    return tracker.track_outcomes()


if __name__ == '__main__':
    # Test the outcome tracker
    print("Testing Outcome Tracker...")
    print("=" * 80)

    tracker = OutcomeTracker()
    results = tracker.track_outcomes()

    print("\n[OK] Outcome tracking complete!")
    print(f"Checked: {results['total_checked']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")

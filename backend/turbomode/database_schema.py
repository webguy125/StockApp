"""
TurboMode Database Schema
Stores overnight S&P 500 ML predictions and signal lifecycle data
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os


class TurboModeDB:
    """
    Database manager for TurboMode signals

    Tables:
    - active_signals: Current open positions (max 14 days)
    - signal_history: Closed positions (hit target/stop or expired)
    - sector_stats: Daily sector aggregations for Sectors Overview
    """

    def __init__(self, db_path: str = "backend/data/turbomode.db"):
        """
        Initialize TurboMode database

        Args:
            db_path: Path to SQLite database file
        """
        # Convert relative path to absolute path
        if not os.path.isabs(db_path):
            # Get the project root (2 levels up from this file's directory)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            db_path = os.path.join(project_root, db_path)

        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Active signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,  -- 'BUY' or 'SELL'
                confidence REAL NOT NULL,   -- Model confidence (0.0 - 1.0)

                -- Entry data (FIXED - never changes unless signal flips)
                entry_date TEXT NOT NULL,   -- ISO format: YYYY-MM-DD
                entry_price REAL NOT NULL,
                entry_min REAL,
                entry_max REAL,
                signal_timestamp TEXT NOT NULL,  -- When signal was created

                -- Current data (UPDATED each scan)
                current_price REAL NOT NULL,

                -- Targets (based on entry_price)
                target_price REAL NOT NULL,  -- +12% for BUY, -12% for SELL
                stop_price REAL NOT NULL,    -- -7% for BUY, +7% for SELL

                -- Classifications
                market_cap TEXT NOT NULL,    -- 'large_cap', 'mid_cap', 'small_cap'
                sector TEXT NOT NULL,        -- GICS sector name

                -- Lifecycle (UPDATED each scan)
                age_days INTEGER DEFAULT 0,  -- Days since signal_timestamp
                status TEXT DEFAULT 'ACTIVE', -- 'ACTIVE', 'TARGET_HIT', 'STOP_HIT', 'EXPIRED'

                -- Metadata
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                UNIQUE(symbol)  -- Only one signal per symbol (allows flipping BUY<->SELL)
            )
        """)

        # Signal history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,

                -- Entry data
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,

                -- Exit data
                exit_date TEXT NOT NULL,
                exit_price REAL NOT NULL,
                exit_reason TEXT NOT NULL,  -- 'TARGET_HIT', 'STOP_HIT', 'EXPIRED'

                -- Performance
                profit_loss_pct REAL NOT NULL,  -- Actual P&L percentage
                hold_days INTEGER NOT NULL,

                -- Classifications
                market_cap TEXT NOT NULL,
                sector TEXT NOT NULL,

                -- Metadata
                created_at TEXT NOT NULL
            )
        """)

        # Sector statistics table (daily aggregations)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sector_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                sector TEXT NOT NULL,

                -- Signal counts
                total_buy_signals INTEGER DEFAULT 0,
                total_sell_signals INTEGER DEFAULT 0,

                -- Confidence metrics
                avg_buy_confidence REAL DEFAULT 0.0,
                avg_sell_confidence REAL DEFAULT 0.0,

                -- Performance (from recent history)
                win_rate_30d REAL DEFAULT 0.0,  -- Last 30 days
                avg_profit_30d REAL DEFAULT 0.0,

                -- Sentiment
                sentiment TEXT DEFAULT 'NEUTRAL',  -- 'BULLISH', 'BEARISH', 'NEUTRAL'

                created_at TEXT NOT NULL,

                UNIQUE(date, sector)  -- One record per sector per day
            )
        """)

        # Indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_signals_symbol ON active_signals(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_signals_market_cap ON active_signals(market_cap)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_signals_sector ON active_signals(sector)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_signals_signal_type ON active_signals(signal_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_signals_age ON active_signals(age_days)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_symbol ON signal_history(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_sector ON signal_history(sector)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_exit_date ON signal_history(exit_date)")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_stats_date ON sector_stats(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_stats_sector ON sector_stats(sector)")

        conn.commit()
        conn.close()

        print(f"[OK] TurboMode database initialized: {self.db_path}")

    # =========================================================================
    # ACTIVE SIGNALS
    # =========================================================================

    def add_or_update_signal(self, signal: Dict[str, Any], current_price: float) -> str:
        """
        Add new signal OR update existing signal with proper flipping logic

        SIGNAL LIFECYCLE RULES:
        1. If no existing signal: CREATE new signal
        2. If existing signal with SAME type (BUY->BUY): UPDATE current_price and confidence only
        3. If existing signal with DIFFERENT type (BUY->SELL): FLIP signal (reset entry_price, timestamp)

        Args:
            signal: Dictionary with keys:
                - symbol: str
                - signal_type: 'BUY' or 'SELL'
                - confidence: float (0.0 - 1.0)
                - entry_date: str (YYYY-MM-DD)
                - entry_price: float (price at signal generation)
                - target_price: float
                - stop_price: float
                - market_cap: 'large_cap', 'mid_cap', or 'small_cap'
                - sector: str (GICS sector name)
            current_price: Current market price for this symbol

        Returns:
            'CREATED', 'UPDATED', or 'FLIPPED'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        symbol = signal['symbol']
        new_signal_type = signal['signal_type']

        # Check if signal already exists
        cursor.execute("""
            SELECT signal_type, signal_timestamp FROM active_signals
            WHERE symbol = ? AND status = 'ACTIVE'
        """, (symbol,))

        existing = cursor.fetchone()

        # Default entry range to ±2% if not provided
        entry_min = signal.get('entry_min', signal['entry_price'] * 0.98)
        entry_max = signal.get('entry_max', signal['entry_price'] * 1.02)

        if not existing:
            # CREATE: No existing signal
            cursor.execute("""
                INSERT INTO active_signals
                (symbol, signal_type, confidence, entry_date, entry_price, entry_min, entry_max,
                 signal_timestamp, current_price, target_price, stop_price, market_cap, sector,
                 age_days, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                new_signal_type,
                signal['confidence'],
                signal['entry_date'],
                signal['entry_price'],
                entry_min,
                entry_max,
                now,  # signal_timestamp
                current_price,
                signal['target_price'],
                signal['stop_price'],
                signal['market_cap'],
                signal['sector'],
                0,  # age_days
                'ACTIVE',
                now,  # created_at
                now   # updated_at
            ))

            conn.commit()
            conn.close()
            return 'CREATED'

        else:
            existing_signal_type = existing[0]

            if existing_signal_type == new_signal_type:
                # UPDATE: Same signal type, just update current_price and confidence
                cursor.execute("""
                    UPDATE active_signals
                    SET confidence = ?,
                        current_price = ?,
                        updated_at = ?
                    WHERE symbol = ? AND status = 'ACTIVE'
                """, (signal['confidence'], current_price, now, symbol))

                conn.commit()
                conn.close()
                return 'UPDATED'

            else:
                # FLIP: Signal changed direction (BUY->SELL or SELL->BUY)
                # Reset entry_price, signal_timestamp, age_days
                cursor.execute("""
                    UPDATE active_signals
                    SET signal_type = ?,
                        confidence = ?,
                        entry_date = ?,
                        entry_price = ?,
                        entry_min = ?,
                        entry_max = ?,
                        signal_timestamp = ?,
                        current_price = ?,
                        target_price = ?,
                        stop_price = ?,
                        age_days = 0,
                        updated_at = ?
                    WHERE symbol = ? AND status = 'ACTIVE'
                """, (
                    new_signal_type,
                    signal['confidence'],
                    signal['entry_date'],
                    signal['entry_price'],
                    entry_min,
                    entry_max,
                    now,  # Reset signal_timestamp
                    current_price,
                    signal['target_price'],
                    signal['stop_price'],
                    now,  # updated_at
                    symbol
                ))

                conn.commit()
                conn.close()
                return 'FLIPPED'

    def add_signal(self, signal: Dict[str, Any]) -> bool:
        """
        DEPRECATED: Use add_or_update_signal() instead
        Kept for backward compatibility with existing code

        Returns:
            True if successful
        """
        current_price = signal.get('entry_price')  # Fallback to entry_price
        result = self.add_or_update_signal(signal, current_price)
        return result in ['CREATED', 'UPDATED', 'FLIPPED']

    def get_active_signals(self, market_cap: Optional[str] = None,
                          signal_type: Optional[str] = None,
                          limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get active signals, optionally filtered by market cap and signal type

        Args:
            market_cap: Filter by 'large_cap', 'mid_cap', or 'small_cap' (None = all)
            signal_type: Filter by 'BUY' or 'SELL' (None = all)
            limit: Maximum number of results (default 20)

        Returns:
            List of signal dictionaries, sorted by confidence DESC
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM active_signals WHERE status = 'ACTIVE'"
        params = []

        if market_cap:
            query += " AND market_cap = ?"
            params.append(market_cap)

        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)

        query += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_current_price(self, symbol: str, current_price: float) -> bool:
        """
        Update current_price for an existing signal (called during scans)

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE active_signals
            SET current_price = ?,
                updated_at = ?
            WHERE symbol = ? AND status = 'ACTIVE'
        """, (current_price, datetime.now().isoformat(), symbol))

        conn.commit()
        conn.close()
        return True

    def get_active_symbols(self) -> List[str]:
        """
        Get list of symbols with active signals
        Used to exclude from overnight scanning

        Returns:
            List of symbol strings
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT symbol FROM active_signals WHERE status = 'ACTIVE'")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def calculate_effective_confidence(self, confidence: float, age_days: int) -> float:
        """
        Calculate time-decayed effective confidence

        Formula: Effective = Original × (1 - (age / 14) × 0.3)
        - Day 0:  100% of original confidence
        - Day 7:  85% of original confidence
        - Day 14: 70% of original confidence

        Args:
            confidence: Original ML confidence (0.0 - 1.0)
            age_days: Age in days (0-14)

        Returns:
            Effective confidence with time decay applied
        """
        decay_factor = 1.0 - (age_days / 14.0) * 0.3
        return confidence * max(decay_factor, 0.7)  # Minimum 70% of original

    def get_weakest_signal(self, market_cap: str, signal_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the signal with lowest effective confidence for a market cap + type
        Used for replacement logic

        Args:
            market_cap: 'large_cap', 'mid_cap', or 'small_cap'
            signal_type: 'BUY' or 'SELL'

        Returns:
            Signal dict with lowest effective confidence, or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM active_signals
            WHERE market_cap = ? AND signal_type = ? AND status = 'ACTIVE'
        """, (market_cap, signal_type))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Calculate effective confidence for each signal
        signals_with_effective = []
        for row in rows:
            signal = dict(row)
            effective_conf = self.calculate_effective_confidence(
                signal['confidence'],
                signal['age_days']
            )
            signal['effective_confidence'] = effective_conf
            signals_with_effective.append(signal)

        # Return signal with lowest effective confidence
        weakest = min(signals_with_effective, key=lambda x: x['effective_confidence'])
        return weakest

    def count_active_signals(self, market_cap: str, signal_type: str) -> int:
        """
        Count active signals for a market cap + type combination

        Args:
            market_cap: 'large_cap', 'mid_cap', or 'small_cap'
            signal_type: 'BUY' or 'SELL'

        Returns:
            Count of active signals
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM active_signals
            WHERE market_cap = ? AND signal_type = ? AND status = 'ACTIVE'
        """, (market_cap, signal_type))

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def replace_signal(self, old_symbol: str, old_signal_type: str, new_signal: Dict[str, Any]) -> bool:
        """
        Replace an old signal with a new higher-confidence signal
        Moves old signal to history with REPLACED status

        Args:
            old_symbol: Symbol to replace
            old_signal_type: 'BUY' or 'SELL'
            new_signal: New signal dictionary

        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get old signal details
            cursor.execute("""
                SELECT * FROM active_signals
                WHERE symbol = ? AND signal_type = ? AND status = 'ACTIVE'
            """, (old_symbol, old_signal_type))

            old_row = cursor.fetchone()
            if not old_row:
                conn.close()
                return False

            # Move old signal to history
            exit_date = datetime.now().strftime('%Y-%m-%d')
            exit_price = old_row[5]  # entry_price (no actual exit, just replaced)
            hold_days = old_row[11]  # age_days

            cursor.execute("""
                INSERT INTO signal_history
                (symbol, signal_type, confidence, entry_date, entry_price,
                 exit_date, exit_price, exit_reason, profit_loss_pct, hold_days,
                 market_cap, sector, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                old_symbol,
                old_signal_type,
                old_row[3],  # confidence
                old_row[4],  # entry_date
                old_row[5],  # entry_price
                exit_date,
                exit_price,
                'REPLACED',  # Replaced by higher confidence signal
                0.0,  # No P&L (didn't actually trade)
                hold_days,
                old_row[9],  # market_cap
                old_row[10],  # sector
                datetime.now().isoformat()
            ))

            # Delete old signal from active
            cursor.execute("""
                DELETE FROM active_signals
                WHERE symbol = ? AND signal_type = ?
            """, (old_symbol, old_signal_type))

            # Add new signal
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO active_signals
                (symbol, signal_type, confidence, entry_date, entry_price,
                 target_price, stop_price, market_cap, sector, age_days,
                 status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_signal['symbol'],
                new_signal['signal_type'],
                new_signal['confidence'],
                new_signal['entry_date'],
                new_signal['entry_price'],
                new_signal['target_price'],
                new_signal['stop_price'],
                new_signal['market_cap'],
                new_signal['sector'],
                0,  # age_days
                'ACTIVE',
                now,
                now
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"[ERROR] Failed to replace signal: {e}")
            conn.rollback()
            conn.close()
            return False

    def update_signal_age(self):
        """
        Update age_days for all active signals based on signal_timestamp
        Called daily by scanner before generating new signals
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update age_days from signal_timestamp (not entry_date)
        cursor.execute("""
            UPDATE active_signals
            SET age_days = CAST((julianday('now') - julianday(signal_timestamp)) AS INTEGER),
                updated_at = ?
            WHERE status = 'ACTIVE'
        """, (datetime.now().isoformat(),))

        conn.commit()

        # Expire signals older than 14 days
        cursor.execute("""
            UPDATE active_signals
            SET status = 'EXPIRED',
                updated_at = ?
            WHERE status = 'ACTIVE' AND age_days >= 14
        """, (datetime.now().isoformat(),))

        expired_count = cursor.rowcount
        conn.commit()
        conn.close()

        if expired_count > 0:
            print(f"[INFO] Expired {expired_count} signals (14+ days old)")

        return expired_count

    def close_signal(self, symbol: str, signal_type: str,
                    exit_price: float, exit_reason: str):
        """
        Close an active signal and move to history

        Args:
            symbol: Stock symbol
            signal_type: 'BUY' or 'SELL'
            exit_price: Actual exit price
            exit_reason: 'TARGET_HIT', 'STOP_HIT', or 'EXPIRED'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get active signal
        cursor.execute("""
            SELECT * FROM active_signals
            WHERE symbol = ? AND signal_type = ? AND status = 'ACTIVE'
        """, (symbol, signal_type))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        # Calculate P&L
        entry_price = row[5]  # entry_price column
        if signal_type == 'BUY':
            profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SELL
            profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100

        # Add to history
        exit_date = datetime.now().strftime('%Y-%m-%d')
        hold_days = row[11]  # age_days column

        cursor.execute("""
            INSERT INTO signal_history
            (symbol, signal_type, confidence, entry_date, entry_price,
             exit_date, exit_price, exit_reason, profit_loss_pct, hold_days,
             market_cap, sector, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            signal_type,
            row[3],  # confidence
            row[4],  # entry_date
            entry_price,
            exit_date,
            exit_price,
            exit_reason,
            profit_loss_pct,
            hold_days,
            row[9],  # market_cap
            row[10],  # sector
            datetime.now().isoformat()
        ))

        # Update active signal status
        cursor.execute("""
            UPDATE active_signals
            SET status = ?, updated_at = ?
            WHERE symbol = ? AND signal_type = ?
        """, (exit_reason, datetime.now().isoformat(), symbol, signal_type))

        conn.commit()
        conn.close()

        return True

    # =========================================================================
    # SECTOR STATISTICS
    # =========================================================================

    def update_sector_stats(self, date: str, sector: str, stats: Dict[str, Any]):
        """
        Update sector statistics for a given date

        Args:
            date: YYYY-MM-DD
            sector: GICS sector name
            stats: Dictionary with keys:
                - total_buy_signals: int
                - total_sell_signals: int
                - avg_buy_confidence: float
                - avg_sell_confidence: float
                - win_rate_30d: float (optional)
                - avg_profit_30d: float (optional)
                - sentiment: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO sector_stats
            (date, sector, total_buy_signals, total_sell_signals,
             avg_buy_confidence, avg_sell_confidence, win_rate_30d,
             avg_profit_30d, sentiment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            sector,
            stats.get('total_buy_signals', 0),
            stats.get('total_sell_signals', 0),
            stats.get('avg_buy_confidence', 0.0),
            stats.get('avg_sell_confidence', 0.0),
            stats.get('win_rate_30d', 0.0),
            stats.get('avg_profit_30d', 0.0),
            stats.get('sentiment', 'NEUTRAL'),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def get_sector_stats(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get sector statistics for a specific date (or latest)

        Args:
            date: YYYY-MM-DD (None = latest date in database)

        Returns:
            List of sector stat dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if date is None:
            # Get latest date
            cursor.execute("SELECT MAX(date) FROM sector_stats")
            date = cursor.fetchone()[0]

        if date is None:
            conn.close()
            return []

        cursor.execute("""
            SELECT * FROM sector_stats
            WHERE date = ?
            ORDER BY sector
        """, (date,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall database statistics

        Returns:
            Dictionary with counts and metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM active_signals WHERE status = 'ACTIVE'")
        active_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM signal_history")
        history_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(profit_loss_pct) FROM signal_history WHERE exit_reason = 'TARGET_HIT'")
        avg_win = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT AVG(profit_loss_pct) FROM signal_history WHERE exit_reason = 'STOP_HIT'")
        avg_loss = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT COUNT(*) FROM signal_history WHERE exit_reason = 'TARGET_HIT'")
        targets_hit = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM signal_history WHERE exit_reason = 'STOP_HIT'")
        stops_hit = cursor.fetchone()[0]

        total_closed = targets_hit + stops_hit
        win_rate = (targets_hit / total_closed * 100) if total_closed > 0 else 0.0

        conn.close()

        return {
            'active_signals': active_count,
            'closed_signals': history_count,
            'targets_hit': targets_hit,
            'stops_hit': stops_hit,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss
        }

    def clear_all_data(self):
        """
        Clear all data from database (for testing)
        WARNING: This deletes all signals and history!
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM active_signals")
        cursor.execute("DELETE FROM signal_history")
        cursor.execute("DELETE FROM sector_stats")

        conn.commit()
        conn.close()

        print("[WARNING] All TurboMode data cleared!")


if __name__ == '__main__':
    # Test database initialization
    print("Testing TurboMode Database Schema...")
    print("=" * 60)

    db = TurboModeDB(db_path="backend/data/turbomode_test.db")

    # Test adding a signal
    print("\n[TEST] Adding BUY signal...")
    signal = {
        'symbol': 'AAPL',
        'signal_type': 'BUY',
        'confidence': 0.92,
        'entry_date': '2025-12-26',
        'entry_price': 195.50,
        'target_price': 215.05,  # +10%
        'stop_price': 185.73,    # -5%
        'market_cap': 'large_cap',
        'sector': 'Information Technology'
    }

    success = db.add_signal(signal)
    print(f"  Result: {'SUCCESS' if success else 'FAILED (duplicate?)'}")

    # Test retrieving signals
    print("\n[TEST] Getting active BUY signals for large_cap...")
    signals = db.get_active_signals(market_cap='large_cap', signal_type='BUY', limit=10)
    print(f"  Found {len(signals)} signals")
    for sig in signals:
        print(f"    {sig['symbol']}: {sig['confidence']:.2%} confidence")

    # Test active symbols
    print("\n[TEST] Getting active symbols...")
    active_symbols = db.get_active_symbols()
    print(f"  Active symbols: {active_symbols}")

    # Test sector stats
    print("\n[TEST] Updating sector stats...")
    stats = {
        'total_buy_signals': 15,
        'total_sell_signals': 8,
        'avg_buy_confidence': 0.88,
        'avg_sell_confidence': 0.85,
        'sentiment': 'BULLISH'
    }
    db.update_sector_stats('2025-12-26', 'Information Technology', stats)
    print("  Stats updated")

    # Test getting stats
    print("\n[TEST] Getting database stats...")
    db_stats = db.get_stats()
    print(f"  Active signals: {db_stats['active_signals']}")
    print(f"  Closed signals: {db_stats['closed_signals']}")
    print(f"  Win rate: {db_stats['win_rate']:.2f}%")

    print("\n[OK] Database schema test complete!")

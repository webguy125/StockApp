"""
Trade Tracker
SQLite database for tracking signals, trades, and performance
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


class TradeTracker:
    """
    Tracks all signals generated and trades executed
    Stores data for learning engine
    """

    def __init__(self, db_path: str = "backend/data/trading_system.db"):
        self.db_path = db_path
        self._ensure_db_exists()
        self._init_database()

    def _ensure_db_exists(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                position_size REAL,
                outcome TEXT DEFAULT 'open',
                profit_loss REAL,
                profit_loss_pct REAL,
                exit_reason TEXT,
                analysis_snapshot TEXT,
                features TEXT,
                predictions TEXT,
                model_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Add model_name column to existing trades table (migration)
        try:
            cursor.execute('ALTER TABLE trades ADD COLUMN model_name TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                direction TEXT NOT NULL,
                analyzers TEXT,
                model_predictions TEXT,
                rank INTEGER,
                was_traded INTEGER DEFAULT 0,
                trade_id TEXT,
                model_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        ''')

        # Add model_name column to existing signals table (migration)
        try:
            cursor.execute('ALTER TABLE signals ADD COLUMN model_name TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Analyzer performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyzer_performance (
                analyzer_name TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                trades_count INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_profit REAL DEFAULT 0.0,
                total_profit REAL DEFAULT 0.0,
                current_weight REAL DEFAULT 1.0,
                trend TEXT DEFAULT 'stable',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (analyzer_name, period_start, period_end)
            )
        ''')

        # Patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                features TEXT,
                success_rate REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                trade_count INTEGER DEFAULT 0,
                discovered_date TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')

        conn.commit()
        conn.close()

    def save_signal(self, symbol: str, score: float, confidence: float, direction: str,
                   analyzers: Dict[str, Any], model_predictions: Dict[str, Any] = None,
                   rank: int = None, model_name: str = None) -> str:
        """Save a generated signal"""
        signal_id = str(uuid.uuid4())
        date = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO signals (id, date, symbol, score, confidence, direction, analyzers, model_predictions, rank, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_id,
            date,
            symbol,
            score,
            confidence,
            direction,
            json.dumps(analyzers),
            json.dumps(model_predictions) if model_predictions else None,
            rank,
            model_name
        ))

        conn.commit()
        conn.close()

        return signal_id

    def save_trade(self, symbol: str, entry_price: float, position_size: float = 1.0,
                  analysis_snapshot: Dict[str, Any] = None, features: List[float] = None,
                  predictions: Dict[str, Any] = None, model_name: str = None) -> str:
        """Record a new trade"""
        trade_id = str(uuid.uuid4())
        entry_date = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (id, symbol, entry_date, entry_price, position_size, analysis_snapshot, features, predictions, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id,
            symbol,
            entry_date,
            entry_price,
            position_size,
            json.dumps(analysis_snapshot) if analysis_snapshot else None,
            json.dumps(features) if features else None,
            json.dumps(predictions) if predictions else None,
            model_name
        ))

        conn.commit()
        conn.close()

        return trade_id

    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str = 'manual'):
        """Close an open trade and calculate P&L"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get trade details
        cursor.execute('SELECT entry_price, position_size FROM trades WHERE id = ?', (trade_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            raise ValueError(f"Trade {trade_id} not found")

        entry_price, position_size = row

        # Calculate P&L
        profit_loss = (exit_price - entry_price) * position_size
        profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100

        # Determine outcome
        outcome = 'win' if profit_loss > 0 else 'loss'

        # Update trade
        cursor.execute('''
            UPDATE trades
            SET exit_date = ?, exit_price = ?, profit_loss = ?, profit_loss_pct = ?, outcome = ?, exit_reason = ?
            WHERE id = ?
        ''', (
            datetime.now().isoformat(),
            exit_price,
            profit_loss,
            profit_loss_pct,
            outcome,
            exit_reason,
            trade_id
        ))

        conn.commit()
        conn.close()

    def get_signals(self, limit: int = 100, symbol: str = None) -> List[Dict[str, Any]]:
        """Fetch recent signals"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if symbol:
            cursor.execute('''
                SELECT * FROM signals
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT * FROM signals
                ORDER BY date DESC
                LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_trades(self, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch trades"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status:
            cursor.execute('''
                SELECT * FROM trades
                WHERE outcome = ?
                ORDER BY entry_date DESC
                LIMIT ?
            ''', (status, limit))
        else:
            cursor.execute('''
                SELECT * FROM trades
                ORDER BY entry_date DESC
                LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_performance_stats(self, model_name: str = None) -> Dict[str, Any]:
        """Calculate overall performance statistics (optionally filtered by model)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build WHERE clause
        where_clause = 'WHERE outcome != "open"'
        where_params = []

        if model_name:
            where_clause += ' AND model_name = ?'
            where_params.append(model_name)

        # Total trades
        cursor.execute(f'SELECT COUNT(*) FROM trades {where_clause}', where_params)
        total_trades = cursor.fetchone()[0]

        # Win/loss counts
        win_where = where_clause + ' AND outcome = "win"'
        cursor.execute(f'SELECT COUNT(*) FROM trades {win_where}'.replace(where_clause, where_clause), where_params)
        wins = cursor.fetchone()[0]

        loss_where = where_clause + ' AND outcome = "loss"'
        cursor.execute(f'SELECT COUNT(*) FROM trades {loss_where}'.replace(where_clause, where_clause), where_params)
        losses = cursor.fetchone()[0]

        # P&L stats
        cursor.execute(f'SELECT AVG(profit_loss), SUM(profit_loss) FROM trades {where_clause}', where_params)
        avg_pl, total_pl = cursor.fetchone()

        # Gross profit/loss
        cursor.execute(f'SELECT SUM(profit_loss) FROM trades {win_where}'.replace(where_clause, where_clause), where_params)
        gross_profit = cursor.fetchone()[0] or 0.0

        cursor.execute(f'SELECT SUM(profit_loss) FROM trades {loss_where}'.replace(where_clause, where_clause), where_params)
        gross_loss = abs(cursor.fetchone()[0] or 0.0)

        conn.close()

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_profit_loss': avg_pl or 0.0,
            'total_profit_loss': total_pl or 0.0,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor
        }

    def update_analyzer_performance(self, analyzer_name: str, win_rate: float, avg_profit: float, trades_count: int):
        """Update analyzer performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        period_start = (datetime.now().replace(day=1)).isoformat()
        period_end = datetime.now().isoformat()

        # Calculate weight based on win rate
        new_weight = max(0.1, min(3.0, win_rate * 2.0))

        cursor.execute('''
            INSERT OR REPLACE INTO analyzer_performance
            (analyzer_name, period_start, period_end, trades_count, win_rate, avg_profit, current_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (analyzer_name, period_start, period_end, trades_count, win_rate, avg_profit, new_weight))

        conn.commit()
        conn.close()

    def __repr__(self):
        stats = self.get_performance_stats()
        return f"<TradeTracker trades={stats['total_trades']} win_rate={stats['win_rate']:.1f}%>"

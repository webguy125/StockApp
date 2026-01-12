"""
Sector/Symbol Performance Tracking Module

Tracks model performance by:
1. GICS Sector (Energy, Technology, Healthcare, etc.)
2. Individual Symbols
3. Sector + Regime combinations

Helps identify model strengths and weaknesses across different market sectors.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class SectorTracker:
    """
    Tracks ML model performance by GICS sector and individual symbols

    GICS (Global Industry Classification Standard) provides standardized
    sector classification for stocks.
    """

    # GICS Sector mappings (11 sectors)
    SECTOR_MAP = {
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'FANG', 'NRG', 'AES'],
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'SMCI', 'SNOW'],
        'Healthcare': ['JNJ', 'UNH', 'ABBV', 'LLY', 'DXCM'],
        'Financials': ['JPM', 'BAC', 'GS', 'WFC', 'SCHW', 'ALLY'],
        'Consumer_Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NFLX', 'LULU', 'DECK', 'SHAK', 'CARS'],
        'Consumer_Staples': ['WMT', 'PG', 'KO', 'PEP', 'GIS', 'CHEF'],
        'Industrials': ['BA', 'HON', 'GE', 'APD', 'LIN', 'SHW', 'XPO'],
        'Real_Estate': ['AMT', 'PLD', 'SPG', 'EQIX', 'STAG', 'REXR'],
        'Utilities': ['DUK', 'SO', 'NEE'],
        'Materials': ['NEM', 'SHW', 'LIN', 'CVX'],  # Some overlap (diversified)
        'Communication_Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'MTCH']
    }

    def __init__(self, db_path: str = "backend/data/advanced_ml_system.db"):
        """
        Initialize sector tracker

        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self._init_table()

        # Build reverse lookup: symbol -> sector
        self.symbol_to_sector = {}
        for sector, symbols in self.SECTOR_MAP.items():
            for symbol in symbols:
                # If symbol in multiple sectors, use first one
                if symbol not in self.symbol_to_sector:
                    self.symbol_to_sector[symbol] = sector

    def _init_table(self):
        """Create sector_performance table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sector_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                sector TEXT NOT NULL,
                regime TEXT NOT NULL,
                predicted_label INTEGER NOT NULL,
                actual_label INTEGER,
                confidence REAL NOT NULL,
                correct INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Indexes for efficient querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sector_performance
            ON sector_performance(sector, regime)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_performance
            ON sector_performance(symbol, regime)
        ''')

        conn.commit()
        conn.close()

    def get_sector(self, symbol: str) -> str:
        """
        Get GICS sector for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Sector name or 'Unknown'
        """
        return self.symbol_to_sector.get(symbol, 'Unknown')

    def track_prediction(
        self,
        symbol: str,
        regime: str,
        prediction: int,
        actual: int = None,
        confidence: float = 0.0
    ):
        """
        Record a prediction for sector tracking

        Args:
            symbol: Stock symbol
            regime: Market regime
            prediction: Predicted label (0=buy, 1=hold, 2=sell)
            actual: Actual label (None if not known yet)
            confidence: Prediction confidence [0-1]
        """
        sector = self.get_sector(symbol)

        # Determine if prediction was correct
        if actual is not None:
            correct = 1 if prediction == actual else 0
        else:
            correct = None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO sector_performance
            (symbol, sector, regime, predicted_label, actual_label, confidence, correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            sector,
            regime,
            int(prediction),
            int(actual) if actual is not None else None,
            float(confidence),
            correct
        ))

        conn.commit()
        conn.close()

    def track_batch_predictions(
        self,
        symbols: List[str],
        regimes: List[str],
        predictions: List[int],
        actuals: List[int] = None,
        confidences: List[float] = None
    ):
        """
        Record multiple predictions in batch

        Args:
            symbols: List of stock symbols
            regimes: List of market regimes
            predictions: List of predicted labels
            actuals: List of actual labels (optional)
            confidences: List of confidences (optional)
        """
        n = len(symbols)

        if actuals is None:
            actuals = [None] * n
        if confidences is None:
            confidences = [0.0] * n

        for i in range(n):
            self.track_prediction(
                symbols[i],
                regimes[i],
                predictions[i],
                actuals[i] if actuals[i] is not None else None,
                confidences[i]
            )

    def get_sector_performance(
        self,
        sector: str = None,
        regime: str = None
    ) -> Dict[str, Any]:
        """
        Get accuracy/metrics by sector

        Args:
            sector: Specific sector (None = all sectors)
            regime: Specific regime (None = all regimes)

        Returns:
            Dict with performance by sector:
            {
                'Energy': {'accuracy': 0.72, 'total': 1500, 'by_regime': {...}},
                'Technology': {'accuracy': 0.68, 'total': 2000, ...},
                ...
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        if sector and regime:
            # Specific sector + regime
            cursor.execute('''
                SELECT sector, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE sector = ? AND regime = ? AND correct IS NOT NULL
                GROUP BY sector, regime
            ''', (sector, regime))
        elif sector:
            # Specific sector, all regimes
            cursor.execute('''
                SELECT sector, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE sector = ? AND correct IS NOT NULL
                GROUP BY sector, regime
            ''', (sector,))
        elif regime:
            # All sectors, specific regime
            cursor.execute('''
                SELECT sector, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE regime = ? AND correct IS NOT NULL
                GROUP BY sector, regime
            ''', (regime,))
        else:
            # All sectors, all regimes
            cursor.execute('''
                SELECT sector, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE correct IS NOT NULL
                GROUP BY sector, regime
            ''')

        rows = cursor.fetchall()
        conn.close()

        # Organize results
        sector_stats = {}

        for row in rows:
            sec = row[0]
            reg = row[1]
            total = row[2]
            correct = row[3] if row[3] else 0
            avg_conf = row[4] if row[4] else 0.0

            accuracy = correct / total if total > 0 else 0.0

            if sec not in sector_stats:
                sector_stats[sec] = {
                    'accuracy': 0.0,
                    'total': 0,
                    'correct': 0,
                    'avg_confidence': 0.0,
                    'by_regime': {}
                }

            # Update regime-specific stats
            sector_stats[sec]['by_regime'][reg] = {
                'accuracy': accuracy,
                'total': total,
                'correct': correct,
                'avg_confidence': avg_conf
            }

            # Update overall stats
            sector_stats[sec]['total'] += total
            sector_stats[sec]['correct'] += correct

        # Calculate overall accuracy per sector
        for sec in sector_stats:
            total = sector_stats[sec]['total']
            correct = sector_stats[sec]['correct']
            sector_stats[sec]['accuracy'] = correct / total if total > 0 else 0.0

            # Calculate weighted average confidence
            total_conf = sum(
                stats['avg_confidence'] * stats['total']
                for stats in sector_stats[sec]['by_regime'].values()
            )
            sector_stats[sec]['avg_confidence'] = total_conf / total if total > 0 else 0.0

        return sector_stats

    def get_symbol_performance(
        self,
        symbol: str = None,
        regime: str = None
    ) -> Dict[str, Any]:
        """
        Get accuracy for specific symbols

        Args:
            symbol: Specific symbol (None = all symbols)
            regime: Specific regime (None = all regimes)

        Returns:
            Dict with performance by symbol:
            {
                'AAPL': {'accuracy': 0.70, 'total': 500, 'best_regime': 'normal', 'sector': 'Technology'},
                'TSLA': {'accuracy': 0.65, 'total': 450, 'best_regime': 'high_volatility', ...},
                ...
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        if symbol and regime:
            cursor.execute('''
                SELECT symbol, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE symbol = ? AND regime = ? AND correct IS NOT NULL
                GROUP BY symbol, regime
            ''', (symbol, regime))
        elif symbol:
            cursor.execute('''
                SELECT symbol, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE symbol = ? AND correct IS NOT NULL
                GROUP BY symbol, regime
            ''', (symbol,))
        elif regime:
            cursor.execute('''
                SELECT symbol, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE regime = ? AND correct IS NOT NULL
                GROUP BY symbol, regime
            ''', (regime,))
        else:
            cursor.execute('''
                SELECT symbol, regime,
                       COUNT(*) as total,
                       SUM(correct) as correct_count,
                       AVG(confidence) as avg_conf
                FROM sector_performance
                WHERE correct IS NOT NULL
                GROUP BY symbol, regime
            ''')

        rows = cursor.fetchall()
        conn.close()

        # Organize results
        symbol_stats = {}

        for row in rows:
            sym = row[0]
            reg = row[1]
            total = row[2]
            correct = row[3] if row[3] else 0
            avg_conf = row[4] if row[4] else 0.0

            accuracy = correct / total if total > 0 else 0.0

            if sym not in symbol_stats:
                symbol_stats[sym] = {
                    'accuracy': 0.0,
                    'total': 0,
                    'correct': 0,
                    'sector': self.get_sector(sym),
                    'by_regime': {},
                    'best_regime': ''
                }

            # Update regime-specific stats
            symbol_stats[sym]['by_regime'][reg] = {
                'accuracy': accuracy,
                'total': total,
                'correct': correct,
                'avg_confidence': avg_conf
            }

            # Update overall stats
            symbol_stats[sym]['total'] += total
            symbol_stats[sym]['correct'] += correct

        # Calculate overall accuracy and best regime per symbol
        for sym in symbol_stats:
            total = symbol_stats[sym]['total']
            correct = symbol_stats[sym]['correct']
            symbol_stats[sym]['accuracy'] = correct / total if total > 0 else 0.0

            # Find best regime (highest accuracy)
            best_regime = ''
            best_accuracy = 0.0
            for reg, stats in symbol_stats[sym]['by_regime'].items():
                if stats['accuracy'] > best_accuracy:
                    best_accuracy = stats['accuracy']
                    best_regime = reg

            symbol_stats[sym]['best_regime'] = best_regime

        return symbol_stats

    def get_weakest_sectors(self, n: int = 3) -> List[Tuple[str, float]]:
        """
        Identify sectors with lowest accuracy

        Args:
            n: Number of sectors to return

        Returns:
            List of (sector, accuracy) tuples sorted by worst accuracy
        """
        sector_perf = self.get_sector_performance()

        # Sort by accuracy (ascending)
        sectors_sorted = sorted(
            [(sector, stats['accuracy']) for sector, stats in sector_perf.items()],
            key=lambda x: x[1]
        )

        return sectors_sorted[:n]

    def get_best_symbols(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Identify symbols with highest accuracy

        Args:
            n: Number of symbols to return

        Returns:
            List of (symbol, accuracy) tuples sorted by best accuracy
        """
        symbol_perf = self.get_symbol_performance()

        # Sort by accuracy (descending)
        symbols_sorted = sorted(
            [(symbol, stats['accuracy']) for symbol, stats in symbol_perf.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return symbols_sorted[:n]


if __name__ == '__main__':
    # Test sector tracker
    print("Testing Sector Tracker...\n")

    # Create tracker
    tracker = SectorTracker()

    print("[TEST 1] Symbol-to-Sector Mapping")
    print("=" * 50)
    test_symbols = ['AAPL', 'XOM', 'JPM', 'AMZN', 'UNH']
    for symbol in test_symbols:
        sector = tracker.get_sector(symbol)
        print(f"  {symbol:6s} -> {sector}")
    print()

    print("[TEST 2] Tracking Predictions")
    print("=" * 50)

    # Simulate predictions
    import numpy as np
    np.random.seed(42)

    # Tech stocks - good accuracy
    for _ in range(100):
        symbol = np.random.choice(['AAPL', 'MSFT', 'GOOGL'])
        prediction = np.random.choice([0, 1, 2])
        actual = prediction if np.random.rand() > 0.3 else np.random.choice([0, 1, 2])
        tracker.track_prediction(symbol, 'normal', prediction, actual, 0.8)

    # Energy stocks - poor accuracy
    for _ in range(100):
        symbol = np.random.choice(['XOM', 'CVX', 'COP'])
        prediction = np.random.choice([0, 1, 2])
        actual = prediction if np.random.rand() > 0.6 else np.random.choice([0, 1, 2])
        tracker.track_prediction(symbol, 'crash', prediction, actual, 0.7)

    print("Tracked 200 predictions")
    print()

    print("[TEST 3] Sector Performance")
    print("=" * 50)
    sector_perf = tracker.get_sector_performance()
    for sector, stats in sorted(sector_perf.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {sector:25s} {stats['accuracy']:5.1%}  ({stats['total']} predictions)")
    print()

    print("[TEST 4] Symbol Performance")
    print("=" * 50)
    symbol_perf = tracker.get_symbol_performance()
    for symbol, stats in sorted(symbol_perf.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:10]:
        print(f"  {symbol:6s} {stats['accuracy']:5.1%}  Best: {stats['best_regime']:15s}  ({stats['sector']})")
    print()

    print("[TEST 5] Weakest Sectors")
    print("=" * 50)
    weakest = tracker.get_weakest_sectors(n=3)
    for sector, accuracy in weakest:
        print(f"  {sector:25s} {accuracy:5.1%}")
    print()

    print("=" * 50)
    print("[OK] Sector Tracker Tests Complete")
    print("=" * 50)

"""
Adaptive Stock Ranker with Feedback Loop
Dynamically identifies top 10 most predictable stocks
Tracks rolling performance and adapts to regime changes
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Import canonical symbol normalizer (strict mode)
from backend.turbomode.core_engine.symbol_normalizer import validate_and_normalize, is_canonical


class AdaptiveStockRanker:
    """
    Self-adaptive universe selector that evolves with market conditions.

    Features:
    - Rolling win rate tracking (30/60/90 day windows)
    - Regime change detection
    - Dynamic scoring with recency bias
    - Performance persistence scoring
    - Automated monthly rotation
    """

    def __init__(self, db_path=None):
        # Use absolute path to avoid path resolution issues when called from Flask
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "data", "turbomode.db")

        self.db_path = db_path

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rankings_file = os.path.join(base_dir, "data", "stock_rankings.json")
        self.history_file = os.path.join(base_dir, "data", "ranking_history.json")

    def calculate_rolling_performance(self, df, symbol, window_days):
        """Calculate win rate over a rolling window"""
        # Enforce canonical symbol format (strict mode)
        if not is_canonical(symbol):
            raise ValueError(f"Non-canonical symbol rejected in TurboMode: '{symbol}'. Use canonical format (e.g., BRK-B, BTC-USD)")

        symbol_df = df[df['symbol'] == symbol].copy()

        if len(symbol_df) == 0:
            return {
                'win_rate': 0.0,
                'total_signals': 0,
                'wins': 0,
                'losses': 0
            }

        # Convert entry_date to datetime
        symbol_df['entry_date'] = pd.to_datetime(symbol_df['entry_date'])

        # Get cutoff date for rolling window
        max_date = symbol_df['entry_date'].max()
        cutoff_date = max_date - timedelta(days=window_days)

        # Filter to rolling window
        window_df = symbol_df[symbol_df['entry_date'] >= cutoff_date]

        if len(window_df) == 0:
            return {
                'win_rate': 0.0,
                'total_signals': 0,
                'wins': 0,
                'losses': 0
            }

        # Calculate wins/losses based on profit_loss_pct
        # For BUY signals: win if profit_loss_pct > 0
        # For SELL signals: win if profit_loss_pct < 0
        wins = 0
        losses = 0

        for _, row in window_df.iterrows():
            pnl = row['profit_loss_pct']
            outcome = row['outcome']

            if outcome == 'buy':
                if pnl >= 10.0:  # Hit the +10% target
                    wins += 1
                else:
                    losses += 1
            elif outcome == 'sell':
                if pnl <= -10.0:  # Hit the -10% target
                    wins += 1
                else:
                    losses += 1

        total = wins + losses
        win_rate = wins / total if total > 0 else 0.0

        return {
            'win_rate': win_rate,
            'total_signals': total,
            'wins': wins,
            'losses': losses
        }

    def calculate_composite_score(self, symbol_stats):
        """
        Calculate composite score with recency bias.

        Score = (30d_wr * 0.5) + (60d_wr * 0.3) + (90d_wr * 0.2) +
                (signal_frequency * 0.1) + (persistence_bonus)
        """
        # Win rate components (weighted by recency)
        wr_30d = symbol_stats['win_rate_30d']
        wr_60d = symbol_stats['win_rate_60d']
        wr_90d = symbol_stats['win_rate_90d']

        # Signal frequency (normalized to 0-1 scale, assuming max 100 signals/year)
        signals_per_year = symbol_stats['signals_per_year']
        freq_score = min(signals_per_year / 100.0, 1.0)

        # Persistence bonus: reward stocks that maintain high win rates
        persistence = 0.0
        if wr_30d >= 0.60 and wr_60d >= 0.60:
            persistence += 0.1
        if wr_60d >= 0.60 and wr_90d >= 0.60:
            persistence += 0.1

        # Composite score
        score = (wr_30d * 0.5 +
                 wr_60d * 0.3 +
                 wr_90d * 0.2 +
                 freq_score * 0.1 +
                 persistence)

        return score

    def detect_regime_change(self, symbol_stats):
        """
        Detect if a stock has undergone a regime change.

        Regime change = significant divergence between recent and historical performance
        """
        wr_30d = symbol_stats['win_rate_30d']
        wr_90d = symbol_stats['win_rate_90d']

        # Regime change threshold: 20% divergence
        if abs(wr_30d - wr_90d) > 0.20:
            if wr_30d > wr_90d:
                return 'improving'
            else:
                return 'deteriorating'

        return 'stable'

    def analyze_all_stocks(self):
        """Run comprehensive analysis on all stocks"""
        print("\n" + "=" * 80)
        print("ADAPTIVE STOCK RANKING ANALYSIS")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Load data
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT
            symbol,
            outcome,
            profit_loss_pct,
            entry_date,
            exit_date
        FROM trades
        WHERE trade_type = 'backtest'
        AND outcome IN ('buy', 'sell')
        AND symbol IS NOT NULL
        ORDER BY symbol, entry_date
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            print("[ERROR] No backtest data found")
            return None

        print(f"Total signals: {len(df):,}")
        print(f"Unique symbols: {df['symbol'].nunique()}")
        print()

        # Analyze each symbol
        all_stats = []

        for symbol in df['symbol'].unique():
            # Calculate rolling performance
            perf_30d = self.calculate_rolling_performance(df, symbol, 30)
            perf_60d = self.calculate_rolling_performance(df, symbol, 60)
            perf_90d = self.calculate_rolling_performance(df, symbol, 90)

            # Calculate overall stats
            symbol_df = df[df['symbol'] == symbol]
            total_signals = len(symbol_df)
            signals_per_year = total_signals / 7  # 7 years of data

            buy_signals = len(symbol_df[symbol_df['outcome'] == 'buy'])
            sell_signals = len(symbol_df[symbol_df['outcome'] == 'sell'])

            stats = {
                'symbol': symbol,
                'total_signals': total_signals,
                'signals_per_year': signals_per_year,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'buy_ratio': buy_signals / total_signals if total_signals > 0 else 0,
                'win_rate_30d': perf_30d['win_rate'],
                'signals_30d': perf_30d['total_signals'],
                'wins_30d': perf_30d['wins'],
                'win_rate_60d': perf_60d['win_rate'],
                'signals_60d': perf_60d['total_signals'],
                'win_rate_90d': perf_90d['win_rate'],
                'signals_90d': perf_90d['total_signals']
            }

            # Calculate composite score
            stats['composite_score'] = self.calculate_composite_score(stats)

            # Detect regime
            stats['regime'] = self.detect_regime_change(stats)

            all_stats.append(stats)

        # Sort by composite score
        stats_df = pd.DataFrame(all_stats)
        stats_df = stats_df.sort_values('composite_score', ascending=False)

        return stats_df

    def get_top_10(self, stats_df):
        """Extract top 10 stocks"""
        top_10 = stats_df.head(10)

        print("=" * 80)
        print("TOP 10 MOST PREDICTABLE STOCKS (Adaptive Ranking)")
        print("=" * 80)
        print()
        print(f"{'Rank':<5} {'Symbol':<8} {'Score':<8} {'WR_30d':<8} {'WR_60d':<8} {'WR_90d':<8} {'Sig/Yr':<8} {'Regime':<12}")
        print("-" * 80)

        for idx, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{idx:<5} {row['symbol']:<8} "
                  f"{row['composite_score']:<8.3f} "
                  f"{row['win_rate_30d']*100:<7.1f}% "
                  f"{row['win_rate_60d']*100:<7.1f}% "
                  f"{row['win_rate_90d']*100:<7.1f}% "
                  f"{row['signals_per_year']:<8.1f} "
                  f"{row['regime']:<12}")

        return top_10

    def save_rankings(self, stats_df):
        """Save current rankings to JSON"""
        rankings_data = {
            'timestamp': datetime.now().isoformat(),
            'top_10': stats_df.head(10).to_dict('records'),
            'all_stocks': stats_df.to_dict('records')
        }

        os.makedirs(os.path.dirname(self.rankings_file), exist_ok=True)

        with open(self.rankings_file, 'w') as f:
            json.dump(rankings_data, f, indent=2)

        print(f"\n[OK] Rankings saved to {self.rankings_file}")

        # Append to history
        self.append_to_history(rankings_data)

    def append_to_history(self, rankings_data):
        """Maintain historical record of rankings"""
        history = []

        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        # Add current rankings
        history.append({
            'timestamp': rankings_data['timestamp'],
            'top_10_symbols': [s['symbol'] for s in rankings_data['top_10']]
        })

        # Keep last 24 months of history
        history = history[-24:]

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"[OK] History updated ({len(history)} entries)")

    def load_current_rankings(self):
        """Load most recent rankings"""
        if not os.path.exists(self.rankings_file):
            return None

        with open(self.rankings_file, 'r') as f:
            return json.load(f)

    def run_analysis(self):
        """Execute full analysis pipeline"""
        print("\n" + "=" * 80)
        print("STARTING ADAPTIVE STOCK RANKING ANALYSIS")
        print("=" * 80)

        # Analyze all stocks
        stats_df = self.analyze_all_stocks()

        if stats_df is None:
            return None

        # Get top 10
        top_10 = self.get_top_10(stats_df)

        # Save results
        self.save_rankings(stats_df)

        # Print summary
        print()
        print("=" * 80)
        print("REGIME CHANGE SUMMARY")
        print("=" * 80)

        improving = stats_df[stats_df['regime'] == 'improving']
        deteriorating = stats_df[stats_df['regime'] == 'deteriorating']

        print(f"\nImproving stocks: {len(improving)}")
        if len(improving) > 0:
            print("  " + ", ".join(improving.head(5)['symbol'].tolist()))

        print(f"\nDeteriorating stocks: {len(deteriorating)}")
        if len(deteriorating) > 0:
            print("  " + ", ".join(deteriorating.head(5)['symbol'].tolist()))

        print()
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return {
            'top_10': top_10.to_dict('records'),
            'all_stats': stats_df.to_dict('records'),
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    ranker = AdaptiveStockRanker()
    results = ranker.run_analysis()

    if results:
        print("\n[SUCCESS] Adaptive stock ranking complete!")
        print(f"[INFO] Top 10 symbols: {[s['symbol'] for s in results['top_10']]}")

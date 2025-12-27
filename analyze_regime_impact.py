"""
Analyze Step 11 Results by Market Regime
Shows how current model performs in different market conditions
"""

import sys
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def detect_market_regime(date_str):
    """Detect market regime for a given date"""

    try:
        date = pd.to_datetime(date_str)

        # Get SPY data (market proxy)
        spy = yf.Ticker("SPY")
        end_date = date + timedelta(days=1)
        start_date = date - timedelta(days=250)  # ~1 year for MA200

        spy_data = spy.history(start=start_date, end=end_date)

        if len(spy_data) < 200:
            return None

        # Calculate indicators
        current_price = spy_data['Close'].iloc[-1]
        ma50 = spy_data['Close'].rolling(50).mean().iloc[-1]
        ma200 = spy_data['Close'].rolling(200).mean().iloc[-1]

        # Get VIX for volatility
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(start=date - timedelta(days=30), end=end_date)

        if len(vix_data) == 0:
            vix_current = 20  # Default
        else:
            vix_current = vix_data['Close'].iloc[-1]

        # Determine regime
        regime = {
            'date': date_str,
            'spy_price': current_price,
            'ma50': ma50,
            'ma200': ma200,
            'vix': vix_current
        }

        # Trend regime
        if current_price > ma200 and ma50 > ma200:
            regime['trend'] = 'bull'
        elif current_price < ma200 and ma50 < ma200:
            regime['trend'] = 'bear'
        else:
            regime['trend'] = 'choppy'

        # Volatility regime
        if vix_current < 15:
            regime['volatility'] = 'low'
        elif vix_current < 25:
            regime['volatility'] = 'normal'
        elif vix_current < 35:
            regime['volatility'] = 'high'
        else:
            regime['volatility'] = 'extreme'

        return regime

    except Exception as e:
        print(f"[WARNING] Could not detect regime for {date_str}: {e}")
        return None


def analyze_by_regime():
    """Analyze Step 11 results by market regime"""

    print("=" * 70)
    print("REGIME IMPACT ANALYSIS - Step 11 Results")
    print("=" * 70)
    print()

    # Load database
    db_path = 'backend/data/advanced_ml_system.db'
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)

    # Get all samples with their predictions
    query = """
    SELECT
        symbol,
        entry_date,
        exit_date,
        entry_price,
        exit_price,
        return_pct,
        label
    FROM trades
    ORDER BY entry_date
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        print("[ERROR] No data found in database")
        return

    print(f"[OK] Loaded {len(df)} trades from database")
    print()

    # Detect regime for each trade
    print("[PROGRESS] Detecting market regime for each trade...")
    print("(This may take a few minutes...)")
    print()

    regimes = []
    for idx, row in df.iterrows():
        regime = detect_market_regime(row['entry_date'])
        regimes.append(regime)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} trades...")

    df['regime'] = regimes

    # Remove rows where regime detection failed
    df_valid = df[df['regime'].notna()].copy()

    # Extract regime components
    df_valid['trend_regime'] = df_valid['regime'].apply(lambda x: x['trend'] if x else None)
    df_valid['vol_regime'] = df_valid['regime'].apply(lambda x: x['volatility'] if x else None)

    print()
    print("=" * 70)
    print("RESULTS BY MARKET REGIME")
    print("=" * 70)
    print()

    # Overall stats
    print("Overall Performance:")
    print(f"  Total Trades: {len(df_valid)}")
    print(f"  Buy (profitable): {len(df_valid[df_valid['label'] == 0])} ({len(df_valid[df_valid['label'] == 0])/len(df_valid)*100:.1f}%)")
    print(f"  Hold (neutral): {len(df_valid[df_valid['label'] == 1])} ({len(df_valid[df_valid['label'] == 1])/len(df_valid)*100:.1f}%)")
    print(f"  Sell (loss): {len(df_valid[df_valid['label'] == 2])} ({len(df_valid[df_valid['label'] == 2])/len(df_valid)*100:.1f}%)")
    print()

    # By trend regime
    print("-" * 70)
    print("PERFORMANCE BY TREND REGIME:")
    print("-" * 70)

    for regime in ['bull', 'bear', 'choppy']:
        regime_df = df_valid[df_valid['trend_regime'] == regime]
        if len(regime_df) == 0:
            continue

        n_samples = len(regime_df)
        n_buy = len(regime_df[regime_df['label'] == 0])
        n_hold = len(regime_df[regime_df['label'] == 1])
        n_sell = len(regime_df[regime_df['label'] == 2])

        avg_return = regime_df['return_pct'].mean()
        avg_win = regime_df[regime_df['label'] == 0]['return_pct'].mean() if n_buy > 0 else 0
        avg_loss = regime_df[regime_df['label'] == 2]['return_pct'].mean() if n_sell > 0 else 0

        print(f"\n{regime.upper()} Market:")
        print(f"  Samples: {n_samples} ({n_samples/len(df_valid)*100:.1f}%)")
        print(f"  Buy signals: {n_buy} ({n_buy/n_samples*100:.1f}%)")
        print(f"  Hold signals: {n_hold} ({n_hold/n_samples*100:.1f}%)")
        print(f"  Sell signals: {n_sell} ({n_sell/n_samples*100:.1f}%)")
        print(f"  Average return: {avg_return:.2f}%")
        print(f"  Average win: {avg_win:.2f}%")
        print(f"  Average loss: {avg_loss:.2f}%")

    # By volatility regime
    print()
    print("-" * 70)
    print("PERFORMANCE BY VOLATILITY REGIME:")
    print("-" * 70)

    for regime in ['low', 'normal', 'high', 'extreme']:
        regime_df = df_valid[df_valid['vol_regime'] == regime]
        if len(regime_df) == 0:
            continue

        n_samples = len(regime_df)
        n_buy = len(regime_df[regime_df['label'] == 0])
        n_hold = len(regime_df[regime_df['label'] == 1])
        n_sell = len(regime_df[regime_df['label'] == 2])

        avg_return = regime_df['return_pct'].mean()
        avg_win = regime_df[regime_df['label'] == 0]['return_pct'].mean() if n_buy > 0 else 0
        avg_loss = regime_df[regime_df['label'] == 2]['return_pct'].mean() if n_sell > 0 else 0

        print(f"\n{regime.upper()} Volatility (VIX):")
        print(f"  Samples: {n_samples} ({n_samples/len(df_valid)*100:.1f}%)")
        print(f"  Buy signals: {n_buy} ({n_buy/n_samples*100:.1f}%)")
        print(f"  Hold signals: {n_hold} ({n_hold/n_samples*100:.1f}%)")
        print(f"  Sell signals: {n_sell} ({n_sell/n_samples*100:.1f}%)")
        print(f"  Average return: {avg_return:.2f}%")
        print(f"  Average win: {avg_win:.2f}%")
        print(f"  Average loss: {avg_loss:.2f}%")

    # Combined regime analysis
    print()
    print("-" * 70)
    print("COMBINED REGIME ANALYSIS:")
    print("-" * 70)

    regime_combos = df_valid.groupby(['trend_regime', 'vol_regime']).agg({
        'return_pct': ['count', 'mean'],
        'label': lambda x: (x == 0).sum()  # Count of Buy signals
    }).round(2)

    print()
    print("Samples and Returns by Regime Combination:")
    for (trend, vol), row in regime_combos.iterrows():
        count = row[('return_pct', 'count')]
        avg_ret = row[('return_pct', 'mean')]
        buy_count = row[('label', '<lambda>')]

        print(f"  {trend:6s} + {vol:7s} vol: {count:5.0f} samples, Avg Return: {avg_ret:6.2f}%, Buys: {buy_count:4.0f}")

    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print()

    # Compare best vs worst regime
    bull_normal = df_valid[(df_valid['trend_regime'] == 'bull') & (df_valid['vol_regime'] == 'normal')]
    bear_high = df_valid[(df_valid['trend_regime'] == 'bear') & (df_valid['vol_regime'] == 'high')]

    if len(bull_normal) > 0 and len(bear_high) > 0:
        best_return = bull_normal['return_pct'].mean()
        worst_return = bear_high['return_pct'].mean()

        print(f"Best Regime (Bull + Normal Vol): {best_return:.2f}% avg return")
        print(f"Worst Regime (Bear + High Vol): {worst_return:.2f}% avg return")
        print(f"Difference: {best_return - worst_return:.2f}% ← HUGE GAP!")
        print()
        print("This shows why regime-aware models are critical!")

    print()
    print("=" * 70)

    # Save results
    summary = {
        'total_samples': len(df_valid),
        'by_trend': {},
        'by_volatility': {},
        'date_range': {
            'start': df_valid['entry_date'].min(),
            'end': df_valid['entry_date'].max()
        }
    }

    for regime in ['bull', 'bear', 'choppy']:
        regime_df = df_valid[df_valid['trend_regime'] == regime]
        if len(regime_df) > 0:
            summary['by_trend'][regime] = {
                'samples': len(regime_df),
                'pct': len(regime_df) / len(df_valid) * 100,
                'avg_return': regime_df['return_pct'].mean(),
                'buy_signals': len(regime_df[regime_df['label'] == 0])
            }

    for regime in ['low', 'normal', 'high', 'extreme']:
        regime_df = df_valid[df_valid['vol_regime'] == regime]
        if len(regime_df) > 0:
            summary['by_volatility'][regime] = {
                'samples': len(regime_df),
                'pct': len(regime_df) / len(df_valid) * 100,
                'avg_return': regime_df['return_pct'].mean(),
                'buy_signals': len(regime_df[regime_df['label'] == 0])
            }

    with open('regime_analysis_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("[OK] Results saved to: regime_analysis_results.json")
    print()


if __name__ == '__main__':
    analyze_by_regime()

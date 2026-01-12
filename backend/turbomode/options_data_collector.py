"""
Options Historical Data Collector
Extracts TurboMode signals from last 6 months and simulates options outcomes
for ML training
"""

import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks import analytical as greeks

print("Starting options data collection...")
print("="*80)

# Paths
TURBOMODE_DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'turbomode.db')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_training')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Risk-free rate
RISK_FREE_RATE = 0.04

def get_turbomode_signals():
    """Extract signals from last 6 months"""
    conn = sqlite3.connect(TURBOMODE_DB)

    # Get signals from July-Dec 2025 (6 months)
    query = """
        SELECT symbol, entry_date, signal_type, confidence, entry_price, target_price, stop_price
        FROM active_signals
        WHERE entry_date >= '2025-07-01' AND entry_date <= '2025-12-31'
        ORDER BY entry_date
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"[OK] Extracted {len(df)} TurboMode signals from July-Dec 2025")
    return df

def calculate_hv(ticker, days=30):
    """Calculate historical volatility"""
    try:
        hist = ticker.history(period=f"{days+10}d")
        if len(hist) < 2:
            return 0.30

        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = log_returns.std() * np.sqrt(252)
        return float(volatility)
    except:
        return 0.30

def simulate_option_outcome(symbol, entry_date_str, signal_type, confidence, entry_price, target_price):
    """Simulate 14-day option outcome using Black-Scholes"""

    try:
        entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d')

        # Fetch historical data (entry + 14 days)
        ticker = yf.Ticker(symbol)

        start_date = (entry_date - timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = (entry_date + timedelta(days=20)).strftime('%Y-%m-%d')

        hist = ticker.history(start=start_date, end=end_date, interval='1d')

        if hist.empty or entry_date_str not in hist.index.strftime('%Y-%m-%d').values:
            return None

        # Get stock price at entry
        entry_idx = hist.index[hist.index.strftime('%Y-%m-%d') == entry_date_str][0]
        stock_price_entry = float(hist.loc[entry_idx, 'Close'])

        # Calculate HV
        hv = calculate_hv(ticker, days=30)

        # Determine option type
        option_type = 'CALL' if signal_type == 'BUY' else 'PUT'

        # Calculate target strike (slightly ITM)
        expected_move_pct = abs(target_price - entry_price) / entry_price

        if option_type == 'CALL':
            strike = stock_price_entry * 0.98  # 2% ITM
        else:
            strike = stock_price_entry * 1.02  # 2% ITM

        # DTE: Use 35 days (middle of 30-45 range)
        dte = 35
        T_entry = dte / 365.0

        # Estimate IV (use HV as proxy, slightly higher)
        iv = min(hv * 1.2, 0.80)  # IV typically higher than HV

        # Calculate entry premium using Black-Scholes
        flag = 'c' if option_type == 'CALL' else 'p'
        entry_premium = bs(flag, stock_price_entry, strike, T_entry, RISK_FREE_RATE, iv)

        # Calculate Greeks at entry
        delta_entry = greeks.delta(flag, stock_price_entry, strike, T_entry, RISK_FREE_RATE, iv)
        gamma_entry = greeks.gamma(flag, stock_price_entry, strike, T_entry, RISK_FREE_RATE, iv)
        theta_entry = greeks.theta(flag, stock_price_entry, strike, T_entry, RISK_FREE_RATE, iv) / 365
        vega_entry = greeks.vega(flag, stock_price_entry, strike, T_entry, RISK_FREE_RATE, iv) / 100
        rho_entry = greeks.rho(flag, stock_price_entry, strike, T_entry, RISK_FREE_RATE, iv) / 100

        # Simulate next 14 days
        max_premium = entry_premium
        max_day = 0

        for day in range(1, 15):
            future_date = entry_date + timedelta(days=day)

            if future_date > datetime.now():
                break  # Can't simulate future

            future_date_str = future_date.strftime('%Y-%m-%d')

            if future_date_str not in hist.index.strftime('%Y-%m-%d').values:
                continue

            future_idx = hist.index[hist.index.strftime('%Y-%m-%d') == future_date_str][0]
            stock_price_future = float(hist.loc[future_idx, 'Close'])

            # Calculate remaining time
            T_remaining = max((dte - day) / 365.0, 0.001)

            # Adjust IV slightly (mean reversion)
            iv_adjusted = iv * 0.95 + hv * 0.05  # Slight mean reversion to HV

            # Re-price option
            premium_future = bs(flag, stock_price_future, strike, T_remaining, RISK_FREE_RATE, iv_adjusted)

            if premium_future > max_premium:
                max_premium = premium_future
                max_day = day

        # Calculate final metrics
        max_profit_pct = (max_premium - entry_premium) / entry_premium if entry_premium > 0 else 0
        hit_target = 1 if max_profit_pct >= 0.10 else 0

        # Return data dictionary
        return {
            'symbol': symbol,
            'entry_date': entry_date_str,
            'signal_type': signal_type,
            'confidence': confidence,
            'entry_price': entry_price,
            'target_price': target_price,
            'stock_price_entry': stock_price_entry,
            'historical_vol_30d': hv,
            'option_type': option_type,
            'strike': strike,
            'dte': dte,
            'entry_premium': entry_premium,
            'entry_iv': iv,
            'delta': delta_entry,
            'gamma': gamma_entry,
            'theta': theta_entry,
            'vega': vega_entry,
            'rho': rho_entry,
            'max_premium_14d': max_premium,
            'max_profit_pct': max_profit_pct * 100,
            'days_to_target': max_day if hit_target else None,
            'option_success': hit_target,
            'expected_move_pct': expected_move_pct * 100
        }

    except Exception as e:
        print(f"[ERROR] {symbol} @ {entry_date_str}: {e}")
        return None

def main():
    # Step 1: Get TurboMode signals
    signals_df = get_turbomode_signals()

    if len(signals_df) == 0:
        print("[ERROR] No signals found in database")
        return

    print(f"\n[INFO] Processing {len(signals_df)} signals...")
    print("This may take 1-2 hours depending on API rate limits\n")

    # Step 2: Simulate each signal
    results = []

    for idx, row in signals_df.iterrows():
        if idx % 50 == 0:
            print(f"Progress: {idx}/{len(signals_df)} ({idx/len(signals_df)*100:.1f}%)")

        result = simulate_option_outcome(
            row['symbol'],
            row['entry_date'],
            row['signal_type'],
            row['confidence'],
            row['entry_price'],
            row['target_price']
        )

        if result:
            results.append(result)

    # Step 3: Convert to DataFrame
    results_df = pd.DataFrame(results)

    print(f"\n[OK] Successfully simulated {len(results_df)} options")
    print(f"[OK] Success rate: {results_df['option_success'].mean()*100:.1f}%")

    # Step 4: Save to parquet
    output_file = os.path.join(OUTPUT_DIR, 'labeled_options_training_data.parquet')
    results_df.to_parquet(output_file, index=False)

    print(f"[OK] Saved to: {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("DATA COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total examples: {len(results_df)}")
    print(f"Successful (+10%): {results_df['option_success'].sum()} ({results_df['option_success'].mean()*100:.1f}%)")
    print(f"Failed (<10%): {(1-results_df['option_success']).sum()} ({(1-results_df['option_success'].mean())*100:.1f}%)")
    print(f"Avg profit (winners): {results_df[results_df['option_success']==1]['max_profit_pct'].mean():.1f}%")
    print(f"Avg profit (losers): {results_df[results_df['option_success']==0]['max_profit_pct'].mean():.1f}%")
    print(f"\nFeatures available: {list(results_df.columns)}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

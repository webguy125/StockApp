"""
Track Options Outcomes - Daily Script
Run this daily at 4:30 PM ET to track outcome of all logged options predictions
"""

import sqlite3
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from py_vollib.black_scholes import black_scholes as bs
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_logs', 'options_predictions.db')

def calculate_option_price_bs(S, K, T, r, sigma, option_type):
    """Calculate option price using Black-Scholes"""
    try:
        if T <= 0:
            # Option expired
            if option_type == 'CALL':
                return max(0, S - K)
            else:
                return max(0, K - S)

        flag = 'c' if option_type == 'CALL' else 'p'
        price = bs(flag, S, K, T, r, sigma)
        return price
    except:
        return 0

def track_options_outcomes():
    """Main tracking function"""

    print(f"\n{'='*80}")
    print(f"OPTIONS OUTCOME TRACKING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find all predictions that need tracking
    cursor.execute("""
        SELECT prediction_id, symbol, created_at, option_type, strike, expiration_date,
               entry_premium, entry_iv, stock_price_entry, dte
        FROM options_predictions_log
        WHERE tracking_complete = 0
    """)

    predictions = cursor.fetchall()

    if not predictions:
        print("[INFO] No predictions to track")
        conn.close()
        return

    print(f"[INFO] Found {len(predictions)} predictions to track\n")

    updated_count = 0
    completed_count = 0

    for pred in predictions:
        (pred_id, symbol, created_at_str, option_type, strike, expiration_date_str,
         entry_premium, entry_iv, stock_price_entry, dte_original) = pred

        try:
            created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
            expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d')

            # Calculate days since creation
            days_since_entry = (datetime.now() - created_at).days

            if days_since_entry > 14:
                days_to_track = 14
            else:
                days_to_track = days_since_entry

            # Skip if less than 1 day
            if days_since_entry < 1:
                continue

            # Fetch current stock price
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')

            if hist.empty:
                print(f"[SKIP] {symbol} - No price data available")
                continue

            current_price = float(hist['Close'].iloc[-1])

            # Calculate remaining DTE
            days_to_expiration = (expiration_date - datetime.now()).days
            T = max(days_to_expiration / 365.0, 0.001)

            # Estimate current option price using Black-Scholes
            current_premium = calculate_option_price_bs(
                S=current_price,
                K=strike,
                T=T,
                r=0.04,
                sigma=entry_iv,
                option_type=option_type
            )

            # Calculate profit %
            if entry_premium > 0:
                profit_pct = (current_premium - entry_premium) / entry_premium
            else:
                profit_pct = 0

            # Get existing max premium from database
            cursor.execute("SELECT max_premium_14d FROM options_predictions_log WHERE prediction_id = ?", (pred_id,))
            existing_max = cursor.fetchone()[0]

            if existing_max is None or current_premium > existing_max:
                max_premium = current_premium
                max_premium_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                max_profit_pct = profit_pct
            else:
                max_premium = existing_max
                cursor.execute("SELECT max_premium_date, max_profit_pct FROM options_predictions_log WHERE prediction_id = ?", (pred_id,))
                row = cursor.fetchone()
                max_premium_date = row[0]
                max_profit_pct = row[1] if row[1] is not None else profit_pct

            # Check if hit +10% target
            hit_target = (max_profit_pct >= 0.10)

            # Determine if tracking is complete (14 days passed)
            tracking_complete = (days_since_entry >= 14)

            # Update database
            cursor.execute("""
                UPDATE options_predictions_log
                SET max_premium_14d = ?,
                    max_premium_date = ?,
                    final_premium_14d = ?,
                    hit_10pct_target = ?,
                    max_profit_pct = ?,
                    tracking_complete = ?,
                    outcome_checked_at = ?
                WHERE prediction_id = ?
            """, (
                max_premium,
                max_premium_date,
                current_premium if tracking_complete else None,
                1 if hit_target else 0,
                max_profit_pct * 100,  # Store as percentage
                1 if tracking_complete else 0,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                pred_id
            ))

            # Calculate days to target if hit
            days_to_target = None
            if hit_target and max_premium_date:
                max_date = datetime.strptime(max_premium_date, '%Y-%m-%d %H:%M:%S')
                days_to_target = (max_date - created_at).days

                cursor.execute("""
                    UPDATE options_predictions_log
                    SET days_to_target = ?
                    WHERE prediction_id = ?
                """, (days_to_target, pred_id))

            updated_count += 1

            if tracking_complete:
                completed_count += 1
                status_icon = "[OK]" if hit_target else "[MISS]"
                print(f"{status_icon} {symbol} {option_type} ${strike}")
                print(f"     Entry: ${entry_premium:.2f} → Max: ${max_premium:.2f} ({max_profit_pct*100:+.1f}%)")
                if hit_target and days_to_target:
                    print(f"     Hit target in {days_to_target} days")
                print()
            else:
                print(f"[TRACKING] {symbol} {option_type} ${strike} - Day {days_since_entry}/14")
                print(f"     Current: ${current_premium:.2f} ({profit_pct*100:+.1f}%), Max: ${max_premium:.2f} ({max_profit_pct*100:+.1f}%)")
                print()

        except Exception as e:
            print(f"[ERROR] Failed to track {symbol}: {e}")
            continue

    conn.commit()

    # Generate summary statistics
    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN hit_10pct_target = 1 THEN 1 ELSE 0 END) as hits,
               AVG(CASE WHEN hit_10pct_target = 1 THEN max_profit_pct ELSE NULL END) as avg_win_pct,
               AVG(CASE WHEN hit_10pct_target = 0 THEN max_profit_pct ELSE NULL END) as avg_loss_pct,
               AVG(CASE WHEN hit_10pct_target = 1 THEN days_to_target ELSE NULL END) as avg_days_to_target
        FROM options_predictions_log
        WHERE tracking_complete = 1
    """)

    stats = cursor.fetchone()

    conn.close()

    print(f"\n{'='*80}")
    print(f"TRACKING SUMMARY")
    print(f"{'='*80}")
    print(f"Updated: {updated_count} predictions")
    print(f"Completed: {completed_count} predictions (14 days reached)")

    if stats[0] and stats[0] > 0:
        total_completed = stats[0]
        hits = stats[1] if stats[1] else 0
        hit_rate = (hits / total_completed * 100) if total_completed > 0 else 0
        avg_win = stats[2] if stats[2] else 0
        avg_loss = stats[3] if stats[3] else 0
        avg_days = stats[4] if stats[4] else 0

        print(f"\nALL-TIME PERFORMANCE:")
        print(f"  Total Completed: {total_completed}")
        print(f"  Hit Rate: {hit_rate:.1f}% ({hits}/{total_completed})")
        print(f"  Avg Win: +{avg_win:.1f}%")
        print(f"  Avg Loss/Miss: {avg_loss:+.1f}%")
        if avg_days > 0:
            print(f"  Avg Days to Target: {avg_days:.1f} days")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    track_options_outcomes()

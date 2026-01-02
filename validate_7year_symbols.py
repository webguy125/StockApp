"""
Validate which symbols have 7+ years of historical data available
"""

import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from advanced_ml.config.core_symbols import get_all_core_symbols

# Get all symbols
symbols = get_all_core_symbols()

print("=" * 80)
print("VALIDATING 7-YEAR DATA AVAILABILITY FOR ALL SYMBOLS")
print("=" * 80)
print(f"Total symbols to check: {len(symbols)}")
print(f"Required history: 7 years (back to {(datetime.now() - timedelta(days=7*365)).strftime('%Y-%m-%d')})")
print("=" * 80)

# Track results
valid_symbols = []
invalid_symbols = []
failed_symbols = []

# Calculate date 7 years ago
end_date = datetime.now()
start_date = end_date - timedelta(days=7 * 365 + 30)  # Add buffer

print(f"\nChecking data availability from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...\n")

for i, symbol in enumerate(sorted(symbols), 1):
    print(f"[{i}/{len(symbols)}] {symbol:6s} ... ", end="", flush=True)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')

        if len(df) == 0:
            print("NO DATA")
            failed_symbols.append((symbol, "No data returned"))
            continue

        # Check if we have at least 6.5 years of data (allowing for some gaps)
        first_date = df.index[0]
        # Convert timezone-aware timestamp to naive for comparison
        first_date_naive = first_date.replace(tzinfo=None) if hasattr(first_date, 'tzinfo') else first_date
        years_available = (end_date - first_date_naive).days / 365.25

        if years_available >= 6.5:
            print(f"OK {years_available:.1f} years ({len(df):,} days)")
            valid_symbols.append((symbol, years_available, len(df)))
        else:
            print(f"SHORT: Only {years_available:.1f} years ({len(df):,} days)")
            invalid_symbols.append((symbol, years_available, len(df), first_date.strftime('%Y-%m-%d')))

    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")
        failed_symbols.append((symbol, str(e)))

# Print summary
print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print(f"\n[OK] VALID SYMBOLS (6.5+ years): {len(valid_symbols)}")
if len(valid_symbols) > 0:
    for symbol, years, days in valid_symbols:
        print(f"   {symbol:6s} - {years:.1f} years ({days:,} days)")

print(f"\n[WARNING] INVALID SYMBOLS (< 6.5 years): {len(invalid_symbols)}")
if len(invalid_symbols) > 0:
    for symbol, years, days, first_date in invalid_symbols:
        print(f"   {symbol:6s} - {years:.1f} years ({days:,} days) - Data starts: {first_date}")

print(f"\n[ERROR] FAILED SYMBOLS (errors): {len(failed_symbols)}")
if len(failed_symbols) > 0:
    for symbol, error in failed_symbols:
        print(f"   {symbol:6s} - {error[:60]}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total symbols: {len(symbols)}")
print(f"Valid (6.5+ years): {len(valid_symbols)} ({len(valid_symbols)/len(symbols)*100:.1f}%)")
print(f"Invalid (< 6.5 years): {len(invalid_symbols)} ({len(invalid_symbols)/len(symbols)*100:.1f}%)")
print(f"Failed (errors): {len(failed_symbols)} ({len(failed_symbols)/len(symbols)*100:.1f}%)")

if len(invalid_symbols) > 0 or len(failed_symbols) > 0:
    print("\n" + "=" * 80)
    print("[WARNING] Some symbols don't have 7 years of data!")
    print("=" * 80)
    print("Recommendations:")
    print("1. Remove symbols with insufficient data from core_symbols.py")
    print("2. Or adjust lookback period to match shortest available history")
    print("\nSymbols to potentially remove:")
    for symbol, *_ in invalid_symbols + failed_symbols:
        print(f"   - {symbol}")
else:
    print("\n" + "=" * 80)
    print("[OK] ALL SYMBOLS HAVE SUFFICIENT DATA FOR 7-YEAR BACKTEST!")
    print("=" * 80)
    print("You can proceed with 7-year data generation.")

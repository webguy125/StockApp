"""
Validate Health of All 80 Curated Stocks
Checks for:
- Data availability (not delisted)
- Liquidity (average volume > 500K)
- Active trading (recent data available)
- Valid price data (no NaN/Inf)
"""

import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_ml.config.core_symbols import CORE_SYMBOLS, get_all_core_symbols

print("=" * 80)
print("VALIDATE ALL 80 CURATED STOCKS")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get all symbols
all_symbols = get_all_core_symbols()
print(f"Total symbols to validate: {len(all_symbols)}\n")

# Validation criteria
MIN_VOLUME = 500_000  # 500K daily volume
MIN_DAYS_DATA = 30  # At least 30 days of recent data
LOOKBACK_DAYS = 60  # Check last 60 days

# Results tracking
healthy_stocks = []
failed_stocks = []
low_volume_stocks = []
insufficient_data_stocks = []

print("=" * 80)
print("VALIDATING STOCKS...")
print("=" * 80)

for i, symbol in enumerate(all_symbols, 1):
    print(f"\n[{i}/{len(all_symbols)}] Validating {symbol}...")

    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)

        # Get historical data
        hist = ticker.history(start=start_date, end=end_date, interval='1d')

        if hist.empty:
            print(f"  [X] FAILED: No data available (possibly delisted)")
            failed_stocks.append({
                'symbol': symbol,
                'reason': 'No data available (possibly delisted)',
                'sector': None,
                'market_cap': None
            })
            continue

        # Check data sufficiency
        if len(hist) < MIN_DAYS_DATA:
            print(f"  [!] WARNING: Only {len(hist)} days of data (need {MIN_DAYS_DATA})")
            insufficient_data_stocks.append({
                'symbol': symbol,
                'days': len(hist),
                'sector': None,
                'market_cap': None
            })
            # Still check other criteria

        # Check for valid price data
        if hist['Close'].isnull().any() or hist['Volume'].isnull().any():
            print(f"  [X] FAILED: Missing price/volume data")
            failed_stocks.append({
                'symbol': symbol,
                'reason': 'Missing price/volume data',
                'sector': None,
                'market_cap': None
            })
            continue

        # Calculate average volume
        avg_volume = hist['Volume'].mean()

        if avg_volume < MIN_VOLUME:
            print(f"  [!] WARNING: Low volume ({avg_volume:,.0f} < {MIN_VOLUME:,})")
            low_volume_stocks.append({
                'symbol': symbol,
                'avg_volume': avg_volume,
                'sector': None,
                'market_cap': None
            })
            # Still mark as potentially usable

        # Get latest price
        latest_price = hist['Close'].iloc[-1]
        latest_date = hist.index[-1]

        # Check if data is recent (within last 7 days)
        latest_date_naive = latest_date.replace(tzinfo=None) if hasattr(latest_date, 'tzinfo') else latest_date
        days_old = (datetime.now() - latest_date_naive).days
        if days_old > 7:
            print(f"  [!] WARNING: Data is {days_old} days old (stale)")

        # Success
        print(f"  [OK] HEALTHY: ${latest_price:.2f}, Volume: {avg_volume:,.0f}, {len(hist)} days data")
        healthy_stocks.append({
            'symbol': symbol,
            'price': latest_price,
            'avg_volume': avg_volume,
            'days_data': len(hist),
            'sector': None,  # Will populate from CORE_SYMBOLS
            'market_cap': None
        })

    except Exception as e:
        print(f"  [X] FAILED: {e}")
        failed_stocks.append({
            'symbol': symbol,
            'reason': str(e),
            'sector': None,
            'market_cap': None
        })

# Populate sector and market_cap info
def find_symbol_info(symbol):
    """Find sector and market_cap for a symbol"""
    for sector, caps in CORE_SYMBOLS.items():
        for market_cap, symbols in caps.items():
            if symbol in symbols:
                return sector, market_cap
    return None, None

# Update all stocks with sector/market_cap info
for stock_list in [healthy_stocks, failed_stocks, low_volume_stocks, insufficient_data_stocks]:
    for stock in stock_list:
        sector, market_cap = find_symbol_info(stock['symbol'])
        stock['sector'] = sector
        stock['market_cap'] = market_cap

# Print summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(f"\n[OK] Healthy stocks: {len(healthy_stocks)}")
print(f"[X] Failed stocks: {len(failed_stocks)}")
print(f"[!] Low volume warnings: {len(low_volume_stocks)}")
print(f"[!] Insufficient data warnings: {len(insufficient_data_stocks)}")

if failed_stocks:
    print("\n" + "=" * 80)
    print("FAILED STOCKS (NEED REPLACEMENT)")
    print("=" * 80)
    for stock in failed_stocks:
        print(f"\n  Symbol: {stock['symbol']}")
        print(f"  Sector: {stock['sector']}")
        print(f"  Market Cap: {stock['market_cap']}")
        print(f"  Reason: {stock['reason']}")

if low_volume_stocks:
    print("\n" + "=" * 80)
    print("LOW VOLUME WARNINGS (MAY NEED REPLACEMENT)")
    print("=" * 80)
    for stock in low_volume_stocks:
        print(f"\n  Symbol: {stock['symbol']}")
        print(f"  Sector: {stock['sector']}")
        print(f"  Market Cap: {stock['market_cap']}")
        print(f"  Avg Volume: {stock['avg_volume']:,.0f}")

if insufficient_data_stocks:
    print("\n" + "=" * 80)
    print("INSUFFICIENT DATA WARNINGS")
    print("=" * 80)
    for stock in insufficient_data_stocks:
        print(f"\n  Symbol: {stock['symbol']}")
        print(f"  Sector: {stock['sector']}")
        print(f"  Market Cap: {stock['market_cap']}")
        print(f"  Days of data: {stock['days']}")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if len(failed_stocks) > 0:
    print(f"\n[!] {len(failed_stocks)} stocks MUST be replaced")
    print("These stocks are delisted or have no data available.")

    # Group by sector and market cap for replacement suggestions
    print("\nGrouped by sector/market_cap for easy replacement:")
    failed_by_category = {}
    for stock in failed_stocks:
        key = f"{stock['sector']} - {stock['market_cap']}"
        if key not in failed_by_category:
            failed_by_category[key] = []
        failed_by_category[key].append(stock['symbol'])

    for category, symbols in failed_by_category.items():
        print(f"  {category}: {', '.join(symbols)}")

if len(low_volume_stocks) > 5:
    print(f"\n[!] {len(low_volume_stocks)} stocks have low volume")
    print("Consider replacing if accuracy is still low after retraining.")

if len(healthy_stocks) >= 75:
    print(f"\n[OK] {len(healthy_stocks)}/80 stocks are healthy ({len(healthy_stocks)/80*100:.1f}%)")
    print("System can proceed with data generation after fixing failed stocks.")
else:
    print(f"\n[X] Only {len(healthy_stocks)}/80 stocks are healthy ({len(healthy_stocks)/80*100:.1f}%)")
    print("Significant cleanup needed before data generation.")

print("\n" + "=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Save results to JSON for reference
import json

results = {
    'validation_date': datetime.now().isoformat(),
    'total_symbols': len(all_symbols),
    'healthy_count': len(healthy_stocks),
    'failed_count': len(failed_stocks),
    'low_volume_count': len(low_volume_stocks),
    'insufficient_data_count': len(insufficient_data_stocks),
    'healthy_stocks': healthy_stocks,
    'failed_stocks': failed_stocks,
    'low_volume_stocks': low_volume_stocks,
    'insufficient_data_stocks': insufficient_data_stocks,
    'validation_criteria': {
        'min_volume': MIN_VOLUME,
        'min_days_data': MIN_DAYS_DATA,
        'lookback_days': LOOKBACK_DAYS
    }
}

output_file = 'stock_validation_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n[SAVED] Results saved to {output_file}")

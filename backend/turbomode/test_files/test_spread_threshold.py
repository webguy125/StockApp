"""
Quick test to verify spread threshold with 3 stocks
Tests: AAPL, NVDA, TSLA
"""

import sys
sys.path.insert(0, 'C:\\StockApp\\backend')

from quarterly_stock_curation import (
    fetch_stock_data,
    fetch_options_data,
    passes_mandatory_criteria,
    calculate_quality_score,
    SPREAD_PCT_THRESHOLD
)

print("="*80)
print("SPREAD THRESHOLD TEST - 3 Stock Sample")
print("="*80)
print(f"Current threshold: {SPREAD_PCT_THRESHOLD*100:.0f}% of mid-price\n")

test_symbols = ['AAPL', 'NVDA', 'TSLA']

for symbol in test_symbols:
    print(f"\n{'='*80}")
    print(f"Testing: {symbol}")
    print(f"{'='*80}")

    # Fetch data
    stock_data = fetch_stock_data(symbol, verbose=False)
    if not stock_data:
        print(f"[FAIL] Could not fetch stock data")
        continue

    options_data = fetch_options_data(symbol, verbose=False)
    if not options_data:
        print(f"[FAIL] Could not fetch options data")
        continue

    # Check criteria
    sector = stock_data['sector']
    market_cap_bucket = stock_data.get('market_cap_bucket', 'large_cap')

    passes, failures = passes_mandatory_criteria(stock_data, options_data, sector, market_cap_bucket)
    quality = calculate_quality_score(stock_data, options_data, sector, market_cap_bucket)

    # Display results
    print(f"\nStock Data:")
    print(f"  Price: ${stock_data['price']:.2f}")
    print(f"  Volume: {stock_data['avg_volume_90d']:,.0f}")
    print(f"  Market Cap: ${stock_data['market_cap']/1e9:.1f}B")
    print(f"  Sector: {sector}")

    print(f"\nOptions Data:")
    print(f"  ATM Spread (USD): ${options_data['atm_spread_usd']:.2f}")
    print(f"  ATM Bid: ${options_data['atm_bid']:.2f}")
    print(f"  ATM Ask: ${options_data['atm_ask']:.2f}")
    print(f"  ATM Mid: ${(options_data['atm_bid'] + options_data['atm_ask'])/2:.2f}")
    print(f"  Spread %: {options_data['atm_spread_pct']*100:.1f}%")
    print(f"  Has Weeklies: {options_data['has_weeklies']}")
    print(f"  Expirations: {options_data['num_expirations']}")

    print(f"\nQuality Score: {quality['total_score']:.1f}/100")

    print(f"\nResult: {'[PASS]' if passes else '[FAIL]'}")
    if not passes:
        print(f"Failures:")
        for failure in failures:
            print(f"  - {failure}")

    # Check specifically spread threshold
    spread_pct = options_data['atm_spread_pct']
    if spread_pct <= SPREAD_PCT_THRESHOLD:
        print(f"\n  ✓ Spread {spread_pct*100:.1f}% <= {SPREAD_PCT_THRESHOLD*100:.0f}% threshold [PASS]")
    else:
        print(f"\n  ✗ Spread {spread_pct*100:.1f}% > {SPREAD_PCT_THRESHOLD*100:.0f}% threshold [FAIL]")

print("\n" + "="*80)
print("Test complete!")
print("="*80)

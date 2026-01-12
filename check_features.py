import sys
sys.path.insert(0, 'backend')
import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

cursor.execute('SELECT entry_features_json, symbol FROM trades WHERE trade_type="backtest" LIMIT 1')
row = cursor.fetchone()
features = json.loads(row[0])
symbol = row[1]

print(f"\nSymbol: {symbol}")
print(f"Total features: {len(features)}")
print(f"\nChecking for fundamental features:")

fund_features = [
    'beta', 'short_percent_of_float', 'short_ratio',
    'analyst_target_price', 'profit_margin', 'debt_to_equity',
    'price_to_book', 'price_to_sales', 'return_on_equity',
    'current_ratio', 'revenue_growth', 'forward_pe'
]

found_fundamentals = []
for k in fund_features:
    if k in features:
        found_fundamentals.append(k)
        print(f"  ✓ {k}: {features[k]}")
    else:
        print(f"  ✗ {k}: MISSING")

print(f"\nFundamentals found: {len(found_fundamentals)}/12")

# Check metadata
print(f"\nMetadata features:")
metadata = ['sector_code', 'market_cap_tier', 'symbol_hash']
for k in metadata:
    if k in features:
        print(f"  ✓ {k}: {features[k]}")
    else:
        print(f"  ✗ {k}: MISSING")

conn.close()

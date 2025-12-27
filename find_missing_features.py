import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

# Get a sample with 207 features
cursor.execute('SELECT entry_features_json FROM trades WHERE symbol="AAPL" LIMIT 1')
full_features = json.loads(cursor.fetchone()[0])

# Get a sample with 184 features
cursor.execute('SELECT entry_features_json FROM trades WHERE symbol="BA" LIMIT 1')
partial_features = json.loads(cursor.fetchone()[0])

print(f"Full sample has: {len(full_features)} features")
print(f"Partial sample has: {len(partial_features)} features")
print(f"Difference: {len(full_features) - len(partial_features)} features")

missing = set(full_features.keys()) - set(partial_features.keys())
print(f"\nMissing features ({len(missing)}):")
for feature in sorted(missing):
    print(f"  - {feature}")

# Check which symbols have the 184-feature issue
cursor.execute('SELECT symbol, COUNT(*) FROM trades GROUP BY symbol')
symbols = cursor.fetchall()

print(f"\nChecking feature counts by symbol:")
for symbol, count in symbols:
    cursor.execute('SELECT entry_features_json FROM trades WHERE symbol=? LIMIT 1', (symbol,))
    features = json.loads(cursor.fetchone()[0])
    feature_count = len(features)
    if feature_count != 207:
        print(f"  {symbol}: {feature_count} features (MISSING {207-feature_count})")

conn.close()

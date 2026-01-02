import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()
cursor.execute('SELECT entry_features_json FROM trades WHERE trade_type="backtest" LIMIT 1')
row = cursor.fetchone()
features = json.loads(row[0])

print(f'Total features in DB: {len(features)}')
print(f'Feature count value: {features.get("feature_count", "N/A")}')
print(f'\nAll feature names:')
for k in sorted(features.keys()):
    if k not in ['feature_count', 'last_price', 'last_volume', 'timestamp', 'symbol', 'error']:
        print(f'  {k}')

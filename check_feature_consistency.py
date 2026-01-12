import sys
sys.path.insert(0, 'backend')
import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

cursor.execute('SELECT entry_features_json FROM trades WHERE trade_type="backtest"')
rows = cursor.fetchall()

feature_counts = {}
for row in rows:
    features = json.loads(row[0])
    count = len(features)
    feature_counts[count] = feature_counts.get(count, 0) + 1

print(f'Total samples: {len(rows)}')
print(f'\nFeature count distribution:')
for count in sorted(feature_counts.keys()):
    print(f'  {count} features: {feature_counts[count]} samples')

conn.close()

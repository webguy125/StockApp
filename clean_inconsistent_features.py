import sys
sys.path.insert(0, 'backend')
import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

# Find and delete samples with inconsistent feature counts
cursor.execute('SELECT ROWID, entry_features_json FROM trades WHERE trade_type="backtest"')
rows = cursor.fetchall()

rows_to_delete = []
for rowid, features_json in rows:
    features = json.loads(features_json)
    if len(features) != 193:  # Keep only 193-feature samples
        rows_to_delete.append(rowid)

if rows_to_delete:
    print(f'Found {len(rows_to_delete)} samples with inconsistent feature counts')
    print(f'Deleting ROWIDs: {rows_to_delete}')
    
    placeholders = ','.join(['?' for _ in rows_to_delete])
    cursor.execute(f'DELETE FROM trades WHERE ROWID IN ({placeholders})', rows_to_delete)
    conn.commit()
    print(f'Deleted {len(rows_to_delete)} samples')
else:
    print('All samples have consistent feature counts!')

# Verify
cursor.execute('SELECT COUNT(*) FROM trades WHERE trade_type="backtest"')
remaining = cursor.fetchone()[0]
print(f'\nRemaining samples: {remaining}')

conn.close()

import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

cursor.execute('SELECT features FROM trades WHERE trade_type="backtest" LIMIT 1')
row = cursor.fetchone()

if row and row[0]:
    try:
        features = json.loads(row[0])
    except:
        import pickle
        features = pickle.loads(row[0])

    print(f'Feature count: {len(features)}')
    print(f'Has sector_code: {"sector_code" in features}')
    print(f'Has market_cap_tier: {"market_cap_tier" in features}')
    print(f'Has symbol_hash: {"symbol_hash" in features}')

    if 'sector_code' in features:
        print(f'\nMetadata values:')
        print(f'  sector_code: {features["sector_code"]}')
        print(f'  market_cap_tier: {features["market_cap_tier"]}')
        print(f'  symbol_hash: {features["symbol_hash"]}')

conn.close()

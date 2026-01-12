import sys
sys.path.insert(0, 'backend')
import sqlite3
import json

conn = sqlite3.connect('backend/data/advanced_ml_system.db')
cursor = conn.cursor()

cursor.execute('SELECT entry_features_json, symbol FROM trades WHERE trade_type="backtest" ORDER BY ROWID DESC LIMIT 1')
row = cursor.fetchone()
features = json.loads(row[0])
symbol = row[1]

fund_features = ['beta', 'short_percent_of_float', 'short_ratio', 'analyst_target_price', 
                 'profit_margin', 'debt_to_equity', 'price_to_book', 'price_to_sales', 
                 'return_on_equity', 'current_ratio', 'revenue_growth', 'forward_pe']
found = [k for k in fund_features if k in features]

print(f'Latest symbol: {symbol}')
print(f'Total features: {len(features)}')
print(f'Fundamentals found: {len(found)}/12')
if len(found) == 12:
    print('SUCCESS - All fundamentals present!')
else:
    print('PARTIAL - Missing:', [k for k in fund_features if k not in features])

conn.close()

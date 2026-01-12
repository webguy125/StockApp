import sqlite3

conn = sqlite3.connect('backend/data/turbomode.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM active_signals')
active = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM signal_history')
history = cursor.fetchone()[0]

print(f'Active Signals: {active}')
print(f'Signal History: {history}')

cursor.execute('SELECT symbol, signal_type, confidence, entry_date, status FROM active_signals ORDER BY confidence DESC LIMIT 5')
print('\nTop 5 Active Signals in turbomode.db:')
for row in cursor.fetchall():
    print(f'  {row[0]:6s} {row[1]:4s} {row[2]:.1%} @ {row[3]} ({row[4]})')

conn.close()

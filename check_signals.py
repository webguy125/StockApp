import sqlite3

conn = sqlite3.connect('backend/data/turbomode.db')
cursor = conn.cursor()

# Check schema
cursor.execute('PRAGMA table_info(active_signals)')
cols = cursor.fetchall()
print('Columns:', [c[1] for c in cols])

# Check total active signals
cursor.execute('SELECT COUNT(*) FROM active_signals')
total = cursor.fetchone()[0]
print(f'\nTotal active signals: {total}')

# Check all signals
cursor.execute('SELECT * FROM active_signals')
signals = cursor.fetchall()

print('\nActive signals:')
for s in signals:
    print(s)

conn.close()

import sqlite3

conn = sqlite3.connect('backend/data/turbomode.db')
cursor = conn.cursor()

print('='*70)
print('CONFIDENCE ANALYSIS - Looking for identical confidence values')
print('='*70)

# Get unique confidence values
print('\nUnique confidence values in database:')
for row in cursor.execute('SELECT DISTINCT confidence FROM active_signals ORDER BY confidence DESC').fetchall():
    print(f'  {row[0]:.15f}')

# Count per confidence value
print('\nSignals per confidence value:')
for row in cursor.execute('SELECT confidence, COUNT(*) FROM active_signals GROUP BY confidence ORDER BY COUNT(*) DESC').fetchall():
    print(f'  {row[0]:.15f}: {row[1]} signals')

# Check if all signals from 3:47 AM have same confidence
print('\n' + '='*70)
print('Morning scan (3:47 AM) - First 10 signals:')
print('='*70)
for row in cursor.execute('''
    SELECT symbol, confidence, created_at
    FROM active_signals
    WHERE created_at LIKE '2025-12-27T03:47:%'
    ORDER BY id
    LIMIT 10
''').fetchall():
    print(f'  {row[0]}: {row[1]:.15f} - {row[2]}')

# Check if all signals from 5:30 PM have same confidence
print('\n' + '='*70)
print('Evening scan (5:30 PM) - First 10 signals:')
print('='*70)
for row in cursor.execute('''
    SELECT symbol, confidence, created_at
    FROM active_signals
    WHERE created_at LIKE '2025-12-27T17:30:%'
    ORDER BY id
    LIMIT 10
''').fetchall():
    print(f'  {row[0]}: {row[1]:.15f} - {row[2]}')

# Get variance in confidence for each scan time
print('\n' + '='*70)
print('Statistical analysis:')
print('='*70)
print('\nMorning scan stats:')
result = cursor.execute('''
    SELECT COUNT(DISTINCT confidence), MIN(confidence), MAX(confidence), AVG(confidence)
    FROM active_signals
    WHERE created_at LIKE '2025-12-27T03:47:%'
''').fetchone()
print(f'  Unique confidences: {result[0]}')
print(f'  Min: {result[1]:.15f}')
print(f'  Max: {result[2]:.15f}')
print(f'  Avg: {result[3]:.15f}')

print('\nEvening scan stats:')
result = cursor.execute('''
    SELECT COUNT(DISTINCT confidence), MIN(confidence), MAX(confidence), AVG(confidence)
    FROM active_signals
    WHERE created_at LIKE '2025-12-27T17:30:%'
''').fetchone()
print(f'  Unique confidences: {result[0]}')
print(f'  Min: {result[1]:.15f}')
print(f'  Max: {result[2]:.15f}')
print(f'  Avg: {result[3]:.15f}')

conn.close()

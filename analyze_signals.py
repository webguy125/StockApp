import sqlite3

conn = sqlite3.connect('backend/data/turbomode.db')
cursor = conn.cursor()

print('='*80)
print('CRITICAL ANALYSIS - Signal Distribution Issues')
print('='*80)

# 1. Check BUY vs SELL distribution
print('\n1. SIGNAL TYPE DISTRIBUTION:')
print('-'*80)
result = cursor.execute('''
    SELECT signal_type, COUNT(*)
    FROM active_signals
    GROUP BY signal_type
''').fetchall()

for row in result:
    print(f'  {row[0]}: {row[1]} signals')

# 2. Check by market cap
print('\n2. DISTRIBUTION BY MARKET CAP:')
print('-'*80)
result = cursor.execute('''
    SELECT signal_type, market_cap, COUNT(*)
    FROM active_signals
    GROUP BY signal_type, market_cap
    ORDER BY signal_type, market_cap
''').fetchall()

for row in result:
    print(f'  {row[0]} - {row[1]}: {row[2]} signals')

# 3. Check confidence distribution for BUY signals
print('\n3. BUY SIGNAL CONFIDENCE RANGES:')
print('-'*80)
result = cursor.execute('''
    SELECT
        CASE
            WHEN confidence >= 0.99 THEN '99%+'
            WHEN confidence >= 0.98 THEN '98-99%'
            WHEN confidence >= 0.90 THEN '90-98%'
            WHEN confidence >= 0.80 THEN '80-90%'
            ELSE 'Below 80%'
        END as conf_range,
        COUNT(*)
    FROM active_signals
    WHERE signal_type = 'BUY'
    GROUP BY conf_range
    ORDER BY conf_range DESC
''').fetchall()

for row in result:
    print(f'  {row[0]}: {row[1]} signals')

# 4. Check if ANY SELL signals exist
print('\n4. CHECKING FOR ANY SELL SIGNALS:')
print('-'*80)
result = cursor.execute('''
    SELECT COUNT(*)
    FROM active_signals
    WHERE signal_type = 'SELL'
''').fetchone()

print(f'  Total SELL signals: {result[0]}')

if result[0] == 0:
    print('  ⚠️  WARNING: NO SELL SIGNALS FOUND - This is statistically impossible!')
else:
    print('\n  Top 5 SELL signals:')
    for row in cursor.execute('''
        SELECT symbol, confidence, sector, market_cap, created_at
        FROM active_signals
        WHERE signal_type = 'SELL'
        ORDER BY confidence DESC
        LIMIT 5
    ''').fetchall():
        print(f'    {row[0]}: {row[1]:.4f} - {row[2]} ({row[3]}) - {row[4]}')

# 5. Check sector distribution
print('\n5. SECTOR DISTRIBUTION (BUY signals only):')
print('-'*80)
result = cursor.execute('''
    SELECT sector, COUNT(*), AVG(confidence) as avg_conf
    FROM active_signals
    WHERE signal_type = 'BUY'
    GROUP BY sector
    ORDER BY COUNT(*) DESC
''').fetchall()

for row in result:
    print(f'  {row[0]:<30} {row[1]:>3} signals @ {row[2]:.4f} avg confidence')

# 6. Check both scans separately
print('\n6. SIGNAL TYPE BY SCAN TIME:')
print('-'*80)
print('\nMorning scan (3:47 AM):')
result = cursor.execute('''
    SELECT signal_type, COUNT(*)
    FROM active_signals
    WHERE created_at LIKE '2025-12-27T03:47:%'
    GROUP BY signal_type
''').fetchall()
for row in result:
    print(f'  {row[0]}: {row[1]} signals')

print('\nEvening scan (5:30 PM):')
result = cursor.execute('''
    SELECT signal_type, COUNT(*)
    FROM active_signals
    WHERE created_at LIKE '2025-12-27T17:30:%'
    GROUP BY signal_type
''').fetchall()
for row in result:
    print(f'  {row[0]}: {row[1]} signals')

print('\n' + '='*80)
print('DIAGNOSIS:')
print('='*80)

# Get counts
buy_count = cursor.execute("SELECT COUNT(*) FROM active_signals WHERE signal_type='BUY'").fetchone()[0]
sell_count = cursor.execute("SELECT COUNT(*) FROM active_signals WHERE signal_type='SELL'").fetchone()[0]
conf_min = cursor.execute("SELECT MIN(confidence) FROM active_signals").fetchone()[0]
conf_max = cursor.execute("SELECT MAX(confidence) FROM active_signals").fetchone()[0]

print(f'\n📊 Summary:')
print(f'  - Total BUY signals: {buy_count}')
print(f'  - Total SELL signals: {sell_count}')
print(f'  - Confidence range: {conf_min:.6f} to {conf_max:.6f}')
print(f'  - Confidence spread: {(conf_max - conf_min)*100:.4f}%')

print(f'\n🚨 Issues Detected:')
if sell_count == 0:
    print('  ❌ ZERO SELL SIGNALS - Model is completely biased toward BUY')
if (conf_max - conf_min) < 0.01:
    print('  ❌ Confidence spread < 1% - Model is not discriminating between stocks')
if buy_count > 150:
    print('  ❌ Too many BUY signals - Model threshold may be too low')

conn.close()

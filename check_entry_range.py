"""
Quick check: Verify entry_min and entry_max are populated correctly
"""

import sqlite3
import os

# Get database path
db_path = os.path.join(os.path.dirname(__file__), "backend/data/turbomode.db")

print(f"Database: {db_path}\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get sample of active signals with entry range
cursor.execute("""
    SELECT symbol, entry_price, entry_min, entry_max, target_price, stop_price,
           confidence, signal_type, sector
    FROM active_signals
    WHERE status = 'ACTIVE'
    ORDER BY confidence DESC
    LIMIT 10
""")

results = cursor.fetchall()

print("=" * 100)
print("TOP 10 ACTIVE SIGNALS (Sorted by Confidence)")
print("=" * 100)
print(f"{'Symbol':<8} {'Type':<6} {'Conf%':<8} {'Entry Min':<12} {'Entry Signal':<12} {'Entry Max':<12} {'Target':<10} {'Stop':<10}")
print("-" * 100)

for row in results:
    symbol, entry, entry_min, entry_max, target, stop, conf, sig_type, sector = row

    # Calculate percentages to verify ±2%
    min_pct = ((entry_min / entry) - 1) * 100 if entry_min and entry else 0
    max_pct = ((entry_max / entry) - 1) * 100 if entry_max and entry else 0

    print(f"{symbol:<8} {sig_type:<6} {conf*100:<7.1f}% "
          f"${entry_min:<11.2f} ${entry:<11.2f} ${entry_max:<11.2f} "
          f"${target:<9.2f} ${stop:<9.2f}")
    print(f"{'':>23} Range: {min_pct:+.2f}% to {max_pct:+.2f}% from signal price")
    print()

# Count total active signals
cursor.execute("SELECT COUNT(*) FROM active_signals WHERE status = 'ACTIVE'")
total = cursor.fetchone()[0]

# Count BUY vs SELL
cursor.execute("SELECT signal_type, COUNT(*) FROM active_signals WHERE status = 'ACTIVE' GROUP BY signal_type")
type_counts = cursor.fetchall()

print("=" * 100)
print(f"Total Active Signals: {total}")
for sig_type, count in type_counts:
    print(f"  {sig_type}: {count}")
print("=" * 100)

conn.close()

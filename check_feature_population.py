import sqlite3

conn = sqlite3.connect('backend/data/turbomode.db')
cursor = conn.cursor()

# Check overall stats
cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN entry_features_json IS NOT NULL THEN 1 ELSE 0 END) as with_features,
        SUM(CASE WHEN entry_features_json IS NULL THEN 1 ELSE 0 END) as without_features
    FROM trades
""")
total, with_feat, without_feat = cursor.fetchone()

print("=" * 60)
print("FEATURE POPULATION ANALYSIS")
print("=" * 60)
print(f"Total samples:         {total:,}")
print(f"With features:         {with_feat:,} ({100*with_feat/total:.1f}%)")
print(f"Without features:      {without_feat:,} ({100*without_feat/total:.1f}%)")
print()

# Check per symbol
cursor.execute("""
    SELECT
        symbol,
        COUNT(*) as total,
        SUM(CASE WHEN entry_features_json IS NOT NULL THEN 1 ELSE 0 END) as with_features
    FROM trades
    GROUP BY symbol
    ORDER BY symbol
    LIMIT 20
""")

print("Sample breakdown by symbol (first 20):")
print(f"{'Symbol':<10} {'Total':>8} {'With Features':>15}")
print("-" * 40)
for row in cursor.fetchall():
    symbol, total, with_feat = row
    print(f"{symbol:<10} {total:>8,} {with_feat:>15,}")

conn.close()

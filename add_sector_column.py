"""
Add sector column to turbomode.db and populate it for all existing samples
OPTIMIZED: Uses single UPDATE with CASE statement instead of 230 separate UPDATEs
"""
import sqlite3
import sys
sys.path.insert(0, r'C:\StockApp\backend')

from turbomode.core_symbols import get_symbol_metadata

DB_PATH = r'C:\StockApp\backend\data\turbomode.db'

print("=" * 80)
print("ADD SECTOR COLUMN TO TURBOMODE.DB (OPTIMIZED)")
print("=" * 80)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Step 1: Add sector column if it doesn't exist
print("\n[STEP 1] Adding sector column...")
try:
    cursor.execute("ALTER TABLE trades ADD COLUMN sector TEXT")
    print("[OK] Added sector column")
except sqlite3.OperationalError as e:
    if "duplicate column" in str(e).lower():
        print("[OK] Sector column already exists")
    else:
        raise

# Step 2: Get all unique symbols from trades
cursor.execute("SELECT DISTINCT symbol FROM trades WHERE trade_type='backtest'")
symbols = [row[0] for row in cursor.fetchall()]
print(f"\n[STEP 2] Found {len(symbols)} unique symbols in database")

# Step 3: Build symbol-to-sector mapping
print("\n[STEP 3] Building symbol-to-sector mapping...")
symbol_sector_map = {}
failed = []

for symbol in symbols:
    metadata = get_symbol_metadata(symbol)
    sector = metadata.get('sector', 'unknown')

    if sector == 'unknown':
        failed.append(symbol)
        continue

    symbol_sector_map[symbol] = sector

print(f"[OK] Mapped {len(symbol_sector_map)} symbols to sectors")

if failed:
    print(f"[WARNING] Failed to get sector for {len(failed)} symbols:")
    for sym in failed:
        print(f"  - {sym}")

# Step 4: Generate and execute single UPDATE with CASE statement
print("\n[STEP 4] Updating database with single optimized query...")
print("  (This is MUCH faster than 230 separate UPDATEs)")

if symbol_sector_map:
    # Build CASE statement for all symbols
    case_clauses = []
    for symbol, sector in symbol_sector_map.items():
        case_clauses.append(f"WHEN symbol = '{symbol}' THEN '{sector}'")

    case_statement = "\n            ".join(case_clauses)

    update_query = f"""
        UPDATE trades
        SET sector = CASE
            {case_statement}
        END
        WHERE trade_type = 'backtest'
        AND symbol IN ({','.join([f"'{s}'" for s in symbol_sector_map.keys()])})
    """

    cursor.execute(update_query)
    updated = cursor.rowcount
    conn.commit()

    print(f"[OK] Updated {updated:,} samples with sector information")

# Step 5: Verify sector distribution
print("\n[STEP 5] Sector distribution:")
cursor.execute("""
    SELECT sector, COUNT(*)
    FROM trades
    WHERE trade_type = 'backtest'
    GROUP BY sector
    ORDER BY COUNT(*) DESC
""")

total = 0
sector_stats = []
for sector, count in cursor.fetchall():
    total += count
    sector_stats.append((sector, count))

for sector, count in sector_stats:
    pct = (count / total * 100) if total else 0
    print(f"  {sector:25s}: {count:8,} samples ({pct:.1f}%)")

print(f"\n  TOTAL: {total:,} samples")

conn.close()

print("\n" + "=" * 80)
print("SECTOR COLUMN ADDED AND POPULATED")
print("=" * 80)

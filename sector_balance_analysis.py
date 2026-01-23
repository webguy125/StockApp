"""
Analyze sector balance for training with available data
"""
import sys
import os

backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_path)

from turbomode.training_symbols import TRAINING_SYMBOLS, get_training_symbols, get_symbol_metadata
import sqlite3

# Connect to database
db_path = os.path.join(backend_path, 'data', 'turbomode.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get symbols with data
cursor.execute("""
    SELECT DISTINCT symbol FROM trades
    WHERE trade_type = 'backtest'
""")
symbols_with_data = set(row[0] for row in cursor.fetchall())
conn.close()

print("=" * 80)
print("SECTOR BALANCE ANALYSIS FOR TRAINING")
print("=" * 80)

print("\nOriginal Plan: 230 stocks across 11 sectors")
print("Actual Data: 167 stocks with historical data\n")

# Analyze each sector
sector_stats = {}

for sector in TRAINING_SYMBOLS.keys():
    # Count planned vs actual symbols per market cap
    large_planned = TRAINING_SYMBOLS[sector].get('large_cap', [])
    mid_planned = TRAINING_SYMBOLS[sector].get('mid_cap', [])
    small_planned = TRAINING_SYMBOLS[sector].get('small_cap', [])

    large_actual = [s for s in large_planned if s in symbols_with_data]
    mid_actual = [s for s in mid_planned if s in symbols_with_data]
    small_actual = [s for s in small_planned if s in symbols_with_data]

    total_planned = len(large_planned) + len(mid_planned) + len(small_planned)
    total_actual = len(large_actual) + len(mid_actual) + len(small_actual)

    sector_stats[sector] = {
        'large': {'planned': len(large_planned), 'actual': len(large_actual)},
        'mid': {'planned': len(mid_planned), 'actual': len(mid_actual)},
        'small': {'planned': len(small_planned), 'actual': len(small_actual)},
        'total': {'planned': total_planned, 'actual': total_actual},
        'coverage': total_actual / total_planned * 100 if total_planned > 0 else 0
    }

# Print sector-by-sector breakdown
print(f"{'SECTOR':<30} {'PLANNED':<10} {'ACTUAL':<10} {'COVERAGE':<12} {'STATUS':<10}")
print("-" * 80)

for sector in sorted(sector_stats.keys()):
    stats = sector_stats[sector]
    sector_name = sector.replace('_', ' ').title()
    planned = stats['total']['planned']
    actual = stats['total']['actual']
    coverage = stats['coverage']

    # Determine status
    if coverage >= 80:
        status = "GOOD"
    elif coverage >= 60:
        status = "FAIR"
    elif coverage >= 40:
        status = "WEAK"
    else:
        status = "CRITICAL"

    print(f"{sector_name:<30} {planned:<10} {actual:<10} {coverage:>6.1f}%     {status:<10}")

    # Show cap breakdown
    large_cov = stats['large']['actual'] / stats['large']['planned'] * 100 if stats['large']['planned'] > 0 else 0
    mid_cov = stats['mid']['actual'] / stats['mid']['planned'] * 100 if stats['mid']['planned'] > 0 else 0
    small_cov = stats['small']['actual'] / stats['small']['planned'] * 100 if stats['small']['planned'] > 0 else 0

    print(f"  └─ Large: {stats['large']['actual']}/{stats['large']['planned']} ({large_cov:.0f}%), "
          f"Mid: {stats['mid']['actual']}/{stats['mid']['planned']} ({mid_cov:.0f}%), "
          f"Small: {stats['small']['actual']}/{stats['small']['planned']} ({small_cov:.0f}%)")

# Overall statistics
total_planned = sum(s['total']['planned'] for s in sector_stats.values())
total_actual = sum(s['total']['actual'] for s in sector_stats.values())
overall_coverage = total_actual / total_planned * 100

print()
print("=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
print(f"Total Symbols Planned:  {total_planned}")
print(f"Total Symbols with Data: {total_actual}")
print(f"Overall Coverage:       {overall_coverage:.1f}%")

# Large cap coverage
large_planned_total = sum(s['large']['planned'] for s in sector_stats.values())
large_actual_total = sum(s['large']['actual'] for s in sector_stats.values())
large_coverage = large_actual_total / large_planned_total * 100
print(f"\nLarge Cap Coverage:     {large_actual_total}/{large_planned_total} ({large_coverage:.1f}%)")

# Mid cap coverage
mid_planned_total = sum(s['mid']['planned'] for s in sector_stats.values())
mid_actual_total = sum(s['mid']['actual'] for s in sector_stats.values())
mid_coverage = mid_actual_total / mid_planned_total * 100
print(f"Mid Cap Coverage:       {mid_actual_total}/{mid_planned_total} ({mid_coverage:.1f}%)")

# Small cap coverage
small_planned_total = sum(s['small']['planned'] for s in sector_stats.values())
small_actual_total = sum(s['small']['actual'] for s in sector_stats.values())
small_coverage = small_actual_total / small_planned_total * 100
print(f"Small Cap Coverage:     {small_actual_total}/{small_planned_total} ({small_coverage:.1f}%)")

# Sector with best/worst coverage
best_sector = max(sector_stats.items(), key=lambda x: x[1]['coverage'])
worst_sector = min(sector_stats.items(), key=lambda x: x[1]['coverage'])

print()
print(f"Best Coverage:  {best_sector[0].replace('_', ' ').title()} ({best_sector[1]['coverage']:.1f}%)")
print(f"Worst Coverage: {worst_sector[0].replace('_', ' ').title()} ({worst_sector[1]['coverage']:.1f}%)")

# Critical assessment
print()
print("=" * 80)
print("TRAINING READINESS ASSESSMENT")
print("=" * 80)

critical_sectors = [s for s, stats in sector_stats.items() if stats['coverage'] < 60]
weak_sectors = [s for s, stats in sector_stats.items() if 60 <= stats['coverage'] < 80]
good_sectors = [s for s, stats in sector_stats.items() if stats['coverage'] >= 80]

print(f"\nSectors with GOOD coverage (>=80%):     {len(good_sectors)}/11")
print(f"Sectors with FAIR coverage (60-79%):    {len(weak_sectors)}/11")
print(f"Sectors with WEAK/CRITICAL (<60%):      {len(critical_sectors)}/11")

if len(critical_sectors) > 0:
    print(f"\nCritical sectors: {', '.join(s.replace('_', ' ').title() for s in critical_sectors)}")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if len(critical_sectors) == 0 and large_coverage >= 90:
    print("\n✓ PROCEED WITH TRAINING")
    print(f"  - All sectors have >=60% coverage")
    print(f"  - Large cap coverage is excellent ({large_coverage:.1f}%)")
    print(f"  - {total_actual} stocks provide strong foundation for sector-specific models")
    print(f"  - Can retrain later after ingesting missing {total_planned - total_actual} symbols")
elif len(critical_sectors) <= 2 and overall_coverage >= 70:
    print("\n⚠ CONDITIONAL PROCEED")
    print(f"  - Most sectors have good coverage ({len(good_sectors)}/11 sectors >=80%)")
    print(f"  - {len(critical_sectors)} sectors need attention")
    print(f"  - Consider combining weak sectors temporarily")
    print(f"  - Or skip weak sectors and train {len(good_sectors) + len(weak_sectors)} sectors only")
else:
    print("\n✗ DO NOT PROCEED - INGEST MISSING DATA FIRST")
    print(f"  - Too many sectors with poor coverage ({len(critical_sectors)} critical)")
    print(f"  - Overall coverage too low ({overall_coverage:.1f}%)")
    print(f"  - Run master_market_data/ingest_market_data.py for missing symbols")

print()

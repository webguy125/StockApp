"""
Find additional mid-cap and small-cap stocks to balance the training set
Target: 230 stocks total
  - 75 large cap, 80 mid cap, 75 small cap
  - ~21 stocks per sector (7 large, 7 mid, 7 small per sector)
  - Spread across 11 sectors
"""
import json
import yfinance as yf
from backend.turbomode.scanning_symbols import SECTOR_MAPPING
from collections import defaultdict

# Load market cap data
with open('market_caps_data.json', 'r') as f:
    market_caps = json.load(f)

# Analyze current stock distribution
print("=" * 80)
print("CURRENT STOCK DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

sector_distribution = {}
for sector, symbols in SECTOR_MAPPING.items():
    sector_distribution[sector] = {
        'large_cap': [],
        'mid_cap': [],
        'small_cap': [],
        'total': len(symbols)
    }

    for symbol in symbols:
        if symbol in market_caps:
            tier_code = market_caps[symbol].get('tier_code', 2)
            if tier_code == 2:
                sector_distribution[sector]['large_cap'].append(symbol)
            elif tier_code == 1:
                sector_distribution[sector]['mid_cap'].append(symbol)
            elif tier_code == 0:
                sector_distribution[sector]['small_cap'].append(symbol)

# Print current distribution
for sector in sorted(sector_distribution.keys()):
    data = sector_distribution[sector]
    large = len(data['large_cap'])
    mid = len(data['mid_cap'])
    small = len(data['small_cap'])
    print(f"{sector:30s} Total: {data['total']:3d}  (L:{large:3d}, M:{mid:3d}, S:{small:3d})")

# Calculate totals
total_large = sum(len(d['large_cap']) for d in sector_distribution.values())
total_mid = sum(len(d['mid_cap']) for d in sector_distribution.values())
total_small = sum(len(d['small_cap']) for d in sector_distribution.values())
print()
print(f"Total: {total_large + total_mid + total_small} stocks")
print(f"  Large Cap: {total_large}")
print(f"  Mid Cap: {total_mid}")
print(f"  Small Cap: {total_small}")
print()

# Calculate what we need
target_total = 230
target_large = 75
target_mid = 80
target_small = 75
target_per_sector = 21  # 230 / 11 ≈ 21

print("=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)
print(f"Target: {target_total} stocks")
print(f"  Large Cap: {target_large} (need to remove {total_large - target_large})")
print(f"  Mid Cap: {target_mid} (need to add {target_mid - total_mid})")
print(f"  Small Cap: {target_small} (need to add {target_small - total_small})")
print(f"  Per Sector: ~{target_per_sector} (ideally 7L + 7M + 7S)")
print()

# Identify which sectors need more stocks
print("=" * 80)
print("SECTOR GAPS (Target: ~7 large, ~7 mid, ~7 small per sector)")
print("=" * 80)
print()

for sector in sorted(sector_distribution.keys()):
    data = sector_distribution[sector]
    large = len(data['large_cap'])
    mid = len(data['mid_cap'])
    small = len(data['small_cap'])

    need_mid = max(0, 7 - mid)
    need_small = max(0, 7 - small)
    can_remove_large = max(0, large - 7)

    if need_mid > 0 or need_small > 0 or can_remove_large > 0:
        print(f"{sector:30s}")
        print(f"  Current: L:{large:2d}, M:{mid:2d}, S:{small:2d}")
        print(f"  Actions: Remove {can_remove_large} large, Add {need_mid} mid, Add {need_small} small")
        print()

print("=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print()
print("1. This analysis shows which sectors need more mid/small cap stocks")
print("2. We need to manually curate high-quality mid/small cap stocks for each sector")
print("3. Criteria for new stocks:")
print("   - S&P MidCap 400 or SmallCap 600 constituents")
print("   - High liquidity (avg volume > 500K)")
print("   - Strong options market")
print("   - Clean historical data (no delisting, no flatlines)")
print("   - Institutional interest")
print()
print("4. For large caps, we'll keep the top performers from each sector")
print()
print("Would you like me to:")
print("  A) Search for candidate stocks automatically using sector screening")
print("  B) Provide a list of common mid/small cap tickers per sector for manual review")
print("  C) Both")

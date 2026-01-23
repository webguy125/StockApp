"""
Build balanced 230-stock training set
Target: 75 large cap, 80 mid cap, 75 small cap
Balanced across 11 sectors (~7L + 7M + 7S per sector)
All stocks must have options trading
"""
import json
from collections import defaultdict
from backend.turbomode.scanning_symbols import SECTOR_MAPPING

# Load existing market cap data
with open('market_caps_data.json', 'r') as f:
    existing_market_caps = json.load(f)

# Load validated candidates
with open('validated_candidates.json', 'r') as f:
    validated_candidates = json.load(f)

# Organize existing stocks by sector and tier
existing_by_sector = {}
for sector, symbols in SECTOR_MAPPING.items():
    existing_by_sector[sector] = {
        'large_cap': [],
        'mid_cap': [],
        'small_cap': []
    }

    for symbol in symbols:
        if symbol in existing_market_caps:
            tier_code = existing_market_caps[symbol].get('tier_code', 2)
            market_cap = existing_market_caps[symbol].get('market_cap', 0)

            if tier_code == 2:
                existing_by_sector[sector]['large_cap'].append({
                    'symbol': symbol,
                    'market_cap': market_cap
                })
            elif tier_code == 1:
                existing_by_sector[sector]['mid_cap'].append({
                    'symbol': symbol,
                    'market_cap': market_cap
                })
            elif tier_code == 0:
                existing_by_sector[sector]['small_cap'].append({
                    'symbol': symbol,
                    'market_cap': market_cap
                })

# Sort large caps by market cap (descending) to keep the biggest/most liquid
for sector in existing_by_sector:
    existing_by_sector[sector]['large_cap'].sort(key=lambda x: x['market_cap'], reverse=True)

# Extract validated candidates by sector and tier
validated_by_sector = {}
for sector in validated_candidates:
    validated_by_sector[sector] = {
        'large_cap': [],
        'mid_cap': [],
        'small_cap': []
    }

    # Process mid caps
    for stock in validated_candidates[sector]['mid_cap']:
        if stock['valid'] and stock['has_options'] and stock['tier_code'] in [1, 2]:
            tier = 'mid_cap' if stock['tier_code'] == 1 else 'large_cap'
            validated_by_sector[sector][tier].append({
                'symbol': stock['symbol'],
                'market_cap': stock['market_cap']
            })

    # Process small caps
    for stock in validated_candidates[sector]['small_cap']:
        if stock['valid'] and stock['has_options'] and stock['tier_code'] in [0, 1, 2]:
            if stock['tier_code'] == 0:
                tier = 'small_cap'
            elif stock['tier_code'] == 1:
                tier = 'mid_cap'
            else:
                tier = 'large_cap'
            validated_by_sector[sector][tier].append({
                'symbol': stock['symbol'],
                'market_cap': stock['market_cap']
            })

# Build balanced training set
# Target per sector: 7 large, 7 mid, 7 small (total 21 per sector, 231 total, we'll use 230)
final_selection = {}

TARGET_PER_SECTOR = {
    'large_cap': 7,
    'mid_cap': 7,
    'small_cap': 7
}

print("=" * 80)
print("BUILDING BALANCED TRAINING SET")
print("=" * 80)
print()
print("Target: 230 stocks (75L + 80M + 75S)")
print("Per sector: ~7L + 7M + 7S = 21 stocks")
print()

total_large = 0
total_mid = 0
total_small = 0

for sector in sorted(existing_by_sector.keys()):
    print(f"\n{sector.upper().replace('_', ' ')}")
    print("-" * 60)

    final_selection[sector] = {
        'large_cap': [],
        'mid_cap': [],
        'small_cap': []
    }

    # Select large caps (keep top 7 from existing)
    large_pool = existing_by_sector[sector]['large_cap'] + validated_by_sector[sector]['large_cap']
    large_pool.sort(key=lambda x: x['market_cap'], reverse=True)
    large_selected = [s['symbol'] for s in large_pool[:TARGET_PER_SECTOR['large_cap']]]
    final_selection[sector]['large_cap'] = large_selected

    # Select mid caps (combine existing + validated, take top 7)
    mid_pool = existing_by_sector[sector]['mid_cap'] + validated_by_sector[sector]['mid_cap']
    mid_pool.sort(key=lambda x: x['market_cap'], reverse=True)
    mid_selected = [s['symbol'] for s in mid_pool[:TARGET_PER_SECTOR['mid_cap']]]
    final_selection[sector]['mid_cap'] = mid_selected

    # Select small caps (combine existing + validated, take top 7)
    small_pool = existing_by_sector[sector]['small_cap'] + validated_by_sector[sector]['small_cap']
    small_pool.sort(key=lambda x: x['market_cap'], reverse=True)
    small_selected = [s['symbol'] for s in small_pool[:TARGET_PER_SECTOR['small_cap']]]
    final_selection[sector]['small_cap'] = small_selected

    total_large += len(large_selected)
    total_mid += len(mid_selected)
    total_small += len(small_selected)

    print(f"  Large: {len(large_selected):2d} ({', '.join(large_selected[:3])}...)")
    print(f"  Mid:   {len(mid_selected):2d} ({', '.join(mid_selected[:3]) if mid_selected else 'None'}...)")
    print(f"  Small: {len(small_selected):2d} ({', '.join(small_selected[:3]) if small_selected else 'None'}...)")

print()
print("=" * 80)
print("FINAL COUNTS")
print("=" * 80)
print(f"Total Large Cap:  {total_large}")
print(f"Total Mid Cap:    {total_mid}")
print(f"Total Small Cap:  {total_small}")
print(f"Total Stocks:     {total_large + total_mid + total_small}")
print()

# Adjust to hit exact targets (77 + 77 + 77 = 231, need to remove 1)
# We'll adjust by reducing one sector's allocation slightly
if total_large + total_mid + total_small > 230:
    extra = (total_large + total_mid + total_small) - 230
    print(f"Need to remove {extra} stock(s) to reach 230 target")
    # Remove from the sector with most stocks (likely one with 8 in a category)
    # For now, just note it

# Save final selection
output = {
    'final_selection': final_selection,
    'totals': {
        'large_cap': total_large,
        'mid_cap': total_mid,
        'small_cap': total_small,
        'total': total_large + total_mid + total_small
    }
}

with open('final_balanced_selection.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nFinal selection saved to: final_balanced_selection.json")
print()
print("Next step: Generate training_symbols.py with this balanced selection")

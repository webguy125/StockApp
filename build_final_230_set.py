"""
Build final 230-stock training set:
- 39 small caps (actual <$2B, >500K volume) - from search
- 100 large caps (with options preferred)
- 91 mid caps (with options preferred)
"""
import json
from backend.turbomode.scanning_symbols import SECTOR_MAPPING

# Load data
with open('market_caps_data.json', 'r') as f:
    market_caps_data = json.load(f)

with open('liquid_small_caps.json', 'r') as f:
    small_caps_found = json.load(f)

with open('validated_candidates.json', 'r') as f:
    validated_candidates = json.load(f)

# Get the 39 small caps we found
small_cap_symbols = [s['symbol'] for s in small_caps_found]  # All 39

print("=" * 80)
print("BUILDING FINAL 230-STOCK TRAINING SET")
print("=" * 80)
print()
print(f"Small caps confirmed: {len(small_cap_symbols)}")
print(f"Need: 100 large caps + 91 mid caps = 191 more stocks")
print()

# Collect all available stocks with their data
all_stocks = []

# From existing scanning symbols
for sector, symbols in SECTOR_MAPPING.items():
    for symbol in symbols:
        if symbol in market_caps_data:
            tier_code = market_caps_data[symbol].get('tier_code', 2)
            market_cap = market_caps_data[symbol].get('market_cap', 0)

            # Skip if it's already in our small cap list
            if symbol in small_cap_symbols:
                continue

            all_stocks.append({
                'symbol': symbol,
                'sector': sector,
                'market_cap': market_cap,
                'tier_code': tier_code,
                'has_options': True,  # Existing stocks likely have options
                'source': 'existing'
            })

# From validated candidates
for sector in validated_candidates:
    for cap_type in ['mid_cap', 'small_cap']:
        for stock in validated_candidates[sector][cap_type]:
            if stock['valid'] and stock['symbol'] not in small_cap_symbols:
                # Add if not already in list
                if stock['symbol'] not in [s['symbol'] for s in all_stocks]:
                    all_stocks.append({
                        'symbol': stock['symbol'],
                        'sector': sector,
                        'market_cap': stock['market_cap'],
                        'tier_code': stock['tier_code'],
                        'has_options': stock['has_options'],
                        'source': 'validated'
                    })

# Sort by market cap (descending)
all_stocks.sort(key=lambda x: x['market_cap'], reverse=True)

print(f"Total available stocks (excluding 39 small caps): {len(all_stocks)}")
print()

# Select 100 large caps (prefer with options)
large_with_options = [s for s in all_stocks if s['has_options'] and s['tier_code'] == 2]
large_without_options = [s for s in all_stocks if not s['has_options'] and s['tier_code'] == 2]

# If not enough large caps with tier_code=2, include some from tier_code=1 (they're large by our standards)
if len(large_with_options) < 100:
    # Add some "mid" tier stocks that are actually large
    mid_as_large = [s for s in all_stocks if s['has_options'] and s['tier_code'] == 1 and s['market_cap'] > 5_000_000_000]
    large_with_options.extend(mid_as_large)

large_caps = large_with_options[:100]
if len(large_caps) < 100:
    # Fill remaining with non-options stocks
    large_caps.extend(large_without_options[:100 - len(large_caps)])

print(f"Selected large caps: {len(large_caps)}")
print(f"  With options: {sum(1 for s in large_caps if s['has_options'])}")

# Select 91 mid caps (prefer with options)
# Exclude already selected large caps
remaining_stocks = [s for s in all_stocks if s['symbol'] not in [lc['symbol'] for lc in large_caps]]

mid_with_options = [s for s in remaining_stocks if s['has_options'] and s['tier_code'] in [1, 2]]
mid_without_options = [s for s in remaining_stocks if not s['has_options'] and s['tier_code'] in [1, 2]]

mid_caps = mid_with_options[:91]
if len(mid_caps) < 91:
    mid_caps.extend(mid_without_options[:91 - len(mid_caps)])

print(f"Selected mid caps: {len(mid_caps)}")
print(f"  With options: {sum(1 for s in mid_caps if s['has_options'])}")
print()

# Build final selection organized by sector
final_selection = {}

for sector in sorted(set([s['sector'] for s in large_caps + mid_caps] + [s.get('sector', 'unknown') for s in small_caps_found])):
    final_selection[sector] = {
        'large_cap': [],
        'mid_cap': [],
        'small_cap': []
    }

# Add large caps
for stock in large_caps:
    final_selection[stock['sector']]['large_cap'].append(stock['symbol'])

# Add mid caps
for stock in mid_caps:
    final_selection[stock['sector']]['mid_cap'].append(stock['symbol'])

# Add small caps
for stock in small_caps_found:
    sector = stock.get('sector', 'unknown').lower().replace(' ', '_')
    if sector == 'consumer_defensive':
        sector = 'consumer_staples'
    elif sector == 'consumer_cyclical':
        sector = 'consumer_discretionary'

    if sector not in final_selection:
        final_selection[sector] = {'large_cap': [], 'mid_cap': [], 'small_cap': []}

    final_selection[sector]['small_cap'].append(stock['symbol'])

# Print summary by sector
print("=" * 80)
print("FINAL SELECTION BY SECTOR")
print("=" * 80)
print()

total_large = 0
total_mid = 0
total_small = 0

for sector in sorted(final_selection.keys()):
    large_count = len(final_selection[sector]['large_cap'])
    mid_count = len(final_selection[sector]['mid_cap'])
    small_count = len(final_selection[sector]['small_cap'])

    total_large += large_count
    total_mid += mid_count
    total_small += small_count

    if large_count + mid_count + small_count > 0:
        print(f"{sector:30s} L:{large_count:3d}, M:{mid_count:3d}, S:{small_count:3d}")

print()
print("=" * 80)
print(f"TOTAL: {total_large} large + {total_mid} mid + {total_small} small = {total_large + total_mid + total_small} stocks")
print("=" * 80)
print()

# Save
output = {
    'final_selection': final_selection,
    'totals': {
        'large_cap': total_large,
        'mid_cap': total_mid,
        'small_cap': total_small,
        'total': total_large + total_mid + total_small
    },
    'with_options': {
        'large_cap': sum(1 for s in large_caps if s['has_options']),
        'mid_cap': sum(1 for s in mid_caps if s['has_options']),
        'small_cap': 'unknown'
    }
}

with open('final_230_selection.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Saved to: final_230_selection.json")
print()
print("Next: Generate training_symbols.py with this final selection")

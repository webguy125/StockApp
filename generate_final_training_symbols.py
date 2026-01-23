"""
Generate training_symbols.py from final 230-stock selection
"""
import json

# Load final selection
with open('final_230_selection.json', 'r') as f:
    data = json.load(f)

final_selection = data['final_selection']
totals = data['totals']

# Generate file content
file_content = f'''"""
Training Symbol List for TurboMode ML
Balanced list of 230 high-quality stocks for sector-specific model training

These symbols are used for:
- Sector-specific model training (11 sector models)
- Feature engineering baseline
- Data quality validation

Selection Criteria:
- Large caps (100 stocks): ALL have active options markets
- Mid caps (91 stocks): ALL have active options markets
- Small caps (39 stocks): <$2B market cap, >500K avg volume
- High liquidity across all tiers
- S&P 500 / MidCap 400 / SmallCap 600 constituents
- Clean historical data

Last Updated: 2026-01-14
Total: 230 stocks
  - Large Cap: {totals['large_cap']} (all with options)
  - Mid Cap: {totals['mid_cap']} (all with options)
  - Small Cap: {totals['small_cap']} (high liquidity)
"""

from typing import Dict, List

# GICS Sector Codes
SECTOR_CODES = {{
    'technology': 45,
    'financials': 40,
    'healthcare': 35,
    'consumer_discretionary': 25,
    'communication_services': 50,
    'industrials': 20,
    'consumer_staples': 30,
    'energy': 10,
    'materials': 15,
    'real_estate': 60,
    'utilities': 55
}}

# Training Symbols (230 stocks organized by sector and market cap)
TRAINING_SYMBOLS = {{
'''

# Add each sector
sector_order = [
    'technology', 'communication_services', 'consumer_discretionary',
    'consumer_staples', 'financials', 'healthcare', 'industrials',
    'energy', 'materials', 'utilities', 'real_estate'
]

for sector in sector_order:
    if sector not in final_selection:
        continue

    data = final_selection[sector]
    large_count = len(data['large_cap'])
    mid_count = len(data['mid_cap'])
    small_count = len(data['small_cap'])
    total_count = large_count + mid_count + small_count

    if total_count == 0:
        continue

    file_content += f'''
    # {"=" * 72}
    # {sector.upper().replace('_', ' ')} ({total_count} symbols: {large_count} large, {mid_count} mid, {small_count} small)
    # {"=" * 72}
    '{sector}': {{
'''

    # Large cap
    file_content += "        'large_cap': [\n"
    for symbol in sorted(data['large_cap']):
        file_content += f"            '{symbol}',\n"
    file_content += "        ],\n\n"

    # Mid cap
    file_content += "        'mid_cap': [\n"
    for symbol in sorted(data['mid_cap']):
        file_content += f"            '{symbol}',\n"
    file_content += "        ],\n\n"

    # Small cap
    file_content += "        'small_cap': [\n"
    for symbol in sorted(data['small_cap']):
        file_content += f"            '{symbol}',\n"
    file_content += "        ]\n"

    file_content += "    },\n"

file_content += '''}

# Cryptocurrency (3 symbols)
CRYPTO_SYMBOLS = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'SOL-USD',   # Solana
]


def get_training_symbols() -> List[str]:
    """
    Get flattened list of training symbols (230 stocks)

    Returns:
        Sorted list of 230 training symbol tickers
    """
    symbols = []

    for sector, cap_categories in TRAINING_SYMBOLS.items():
        for cap_category, symbol_list in cap_categories.items():
            symbols.extend(symbol_list)

    return sorted(list(set(symbols)))


def get_training_symbols_with_crypto() -> List[str]:
    """
    Get training symbols + crypto (233 total)

    Returns:
        List of 230 stocks + 3 crypto symbols
    """
    return get_training_symbols() + CRYPTO_SYMBOLS


def get_symbols_by_sector(sector: str) -> List[str]:
    """
    Get training symbols for a specific sector

    Args:
        sector: Sector name (e.g., 'technology', 'financials')

    Returns:
        List of symbols in that sector
    """
    if sector not in TRAINING_SYMBOLS:
        return []

    symbols = []
    for cap_category, symbol_list in TRAINING_SYMBOLS[sector].items():
        symbols.extend(symbol_list)

    return symbols


def get_symbols_by_market_cap(market_cap_category: str) -> List[str]:
    """
    Get training symbols for a specific market cap category

    Args:
        market_cap_category: 'large_cap', 'mid_cap', or 'small_cap'

    Returns:
        List of symbols in that category
    """
    symbols = []

    for sector, cap_categories in TRAINING_SYMBOLS.items():
        if market_cap_category in cap_categories:
            symbols.extend(cap_categories[market_cap_category])

    return list(set(symbols))


def get_symbol_metadata(symbol: str) -> Dict[str, str]:
    """
    Get sector and market cap metadata for a training symbol

    Args:
        symbol: Stock ticker

    Returns:
        Dict with 'symbol', 'sector', 'market_cap_category', 'sector_code' keys
    """
    for sector, cap_categories in TRAINING_SYMBOLS.items():
        for cap_category, symbol_list in cap_categories.items():
            if symbol in symbol_list:
                return {
                    'symbol': symbol,
                    'sector': sector,
                    'market_cap_category': cap_category,
                    'sector_code': SECTOR_CODES[sector]
                }

    return {
        'symbol': symbol,
        'sector': 'unknown',
        'market_cap_category': 'unknown',
        'sector_code': -1
    }


def get_statistics() -> Dict[str, int]:
    """
    Get statistics about training symbol list

    Returns:
        Dict with counts by sector and market cap
    """
    stats = {
        'total_training_symbols': len(get_training_symbols()),
        'total_crypto_symbols': len(CRYPTO_SYMBOLS),
        'by_sector': {},
        'by_market_cap': {
            'large_cap': len(get_symbols_by_market_cap('large_cap')),
            'mid_cap': len(get_symbols_by_market_cap('mid_cap')),
            'small_cap': len(get_symbols_by_market_cap('small_cap'))
        }
    }

    for sector in TRAINING_SYMBOLS.keys():
        stats['by_sector'][sector] = len(get_symbols_by_sector(sector))

    return stats


if __name__ == '__main__':
    print("=" * 80)
    print("TRAINING SYMBOLS FOR TURBOMODE ML")
    print("=" * 80)
    print()

    stats = get_statistics()

    print(f"Total Training Symbols: {stats['total_training_symbols']}")
    print(f"Total Crypto Symbols:   {stats['total_crypto_symbols']}")
    print()

    print(f"By Market Cap:")
    print(f"  Large Cap:  {stats['by_market_cap']['large_cap']}")
    print(f"  Mid Cap:    {stats['by_market_cap']['mid_cap']}")
    print(f"  Small Cap:  {stats['by_market_cap']['small_cap']}")
    print()

    print(f"By Sector:")
    for sector, count in sorted(stats['by_sector'].items()):
        print(f"  {sector.replace('_', ' ').title():30s} {count:2d} symbols")
    print()

    print(f"Training Symbol List (230 stocks):")
    symbols = get_training_symbols()
    for i, symbol in enumerate(symbols, 1):
        print(f"{symbol:8s}", end="")
        if i % 10 == 0:
            print()
    print("\\n")
'''

# Write the file
output_path = 'C:\\StockApp\\backend\\turbomode\\training_symbols.py'
with open(output_path, 'w') as f:
    f.write(file_content)

print("=" * 80)
print("TRAINING_SYMBOLS.PY GENERATED")
print("=" * 80)
print()
print(f"File: {output_path}")
print(f"Total symbols: {totals['total']}")
print(f"  Large cap: {totals['large_cap']}")
print(f"  Mid cap: {totals['mid_cap']}")
print(f"  Small cap: {totals['small_cap']}")
print()
print("DONE! Ready for sector-specific model training.")

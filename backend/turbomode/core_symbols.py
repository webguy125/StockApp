"""
Core Symbol List for TurboMode ML Training
Curated list of 40 high-quality stocks across major sectors + 3 crypto

Selection Criteria:
- High liquidity leaders in each sector
- Market-moving stocks with strong institutional following
- Representative of broad market behavior

Last Updated: 2026-01-06
Total Symbols: 40 (stocks) + 3 (crypto) = 43 total
"""

from typing import Dict, List

# GICS Sector Codes
SECTOR_CODES = {
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
}

# Core Symbol List (Curated 40 stocks)
CORE_SYMBOLS = {

    # ================================================================
    # TECHNOLOGY (10 symbols)
    # ================================================================
    'technology': {
        'large_cap': [
            'AAPL',   # Apple - Consumer Electronics
            'MSFT',   # Microsoft - Software
            'NVDA',   # NVIDIA - Semiconductors
            'AMD',    # AMD - Semiconductors
            'AVGO',   # Broadcom - Semiconductors
            'CRM',    # Salesforce - Cloud Software
            'ADBE',   # Adobe - Creative Software
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # COMMUNICATION SERVICES (4 symbols)
    # ================================================================
    'communication_services': {
        'large_cap': [
            'GOOGL',  # Alphabet - Internet/Search
            'META',   # Meta - Social Media
            'NFLX',   # Netflix - Streaming
            'DIS',    # Disney - Entertainment/Streaming
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # CONSUMER DISCRETIONARY (3 symbols)
    # ================================================================
    'consumer_discretionary': {
        'large_cap': [
            'AMZN',   # Amazon - E-commerce/Cloud
            'TSLA',   # Tesla - Electric Vehicles
            'HD',     # Home Depot - Home Improvement
            'MCD',    # McDonald's - Restaurants
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # CONSUMER STAPLES (3 symbols)
    # ================================================================
    'consumer_staples': {
        'large_cap': [
            'PG',     # Procter & Gamble - Consumer Products
            'KO',     # Coca-Cola - Beverages
            'COST',   # Costco - Retail
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # FINANCIALS (5 symbols)
    # ================================================================
    'financials': {
        'large_cap': [
            'JPM',    # JPMorgan Chase - Bank
            'BAC',    # Bank of America - Bank
            'GS',     # Goldman Sachs - Investment Bank
            'MS',     # Morgan Stanley - Investment Bank
            'BRK-B',  # Berkshire Hathaway - Diversified Holdings
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # HEALTHCARE (4 symbols)
    # ================================================================
    'healthcare': {
        'large_cap': [
            'UNH',    # UnitedHealth - Insurance
            'JNJ',    # Johnson & Johnson - Pharma/Consumer
            'ABBV',   # AbbVie - Pharma
            'MRK',    # Merck - Pharma
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # INDUSTRIALS (4 symbols)
    # ================================================================
    'industrials': {
        'large_cap': [
            'CAT',    # Caterpillar - Machinery
            'DE',     # Deere & Company - Agriculture Equipment
            'HON',    # Honeywell - Diversified Industrial
            'UPS',    # United Parcel Service - Logistics
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # ENERGY (3 symbols)
    # ================================================================
    'energy': {
        'large_cap': [
            'XOM',    # ExxonMobil - Integrated Oil
            'CVX',    # Chevron - Integrated Oil
            'SLB',    # Schlumberger - Oilfield Services
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # MATERIALS (2 symbols)
    # ================================================================
    'materials': {
        'large_cap': [
            'LIN',    # Linde - Industrial Gases
            'NEM',    # Newmont - Gold Mining
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # UTILITIES (2 symbols)
    # ================================================================
    'utilities': {
        'large_cap': [
            'NEE',    # NextEra Energy - Electric Utility
            'DUK',    # Duke Energy - Electric Utility
        ],
        'mid_cap': [],
        'small_cap': []
    },

    # ================================================================
    # REAL ESTATE (2 symbols)
    # ================================================================
    'real_estate': {
        'large_cap': [
            'PLD',    # Prologis - Industrial REIT
            'AMT',    # American Tower - Cell Tower REIT
        ],
        'mid_cap': [],
        'small_cap': []
    }
}

# Cryptocurrency (3 symbols)
CRYPTO_SYMBOLS = [
    'BTC-USD',   # Bitcoin
    'ETH-USD',   # Ethereum
    'SOL-USD',   # Solana
]


def get_all_core_symbols() -> List[str]:
    """
    Get flattened list of all core symbols

    Returns:
        List of all symbol tickers
    """
    symbols = []

    for sector, cap_categories in CORE_SYMBOLS.items():
        for cap_category, symbol_list in cap_categories.items():
            symbols.extend(symbol_list)

    # Remove duplicates (in case any appear in multiple sectors)
    symbols = list(set(symbols))

    return sorted(symbols)


def get_symbols_by_sector(sector: str) -> List[str]:
    """
    Get all symbols for a specific sector

    Args:
        sector: Sector name (e.g., 'technology', 'financials')

    Returns:
        List of symbols in that sector
    """
    if sector not in CORE_SYMBOLS:
        return []

    symbols = []
    for cap_category, symbol_list in CORE_SYMBOLS[sector].items():
        symbols.extend(symbol_list)

    return symbols


def get_symbols_by_market_cap(market_cap_category: str) -> List[str]:
    """
    Get all symbols for a specific market cap category

    Args:
        market_cap_category: 'large_cap', 'mid_cap', or 'small_cap'

    Returns:
        List of symbols in that category
    """
    symbols = []

    for sector, cap_categories in CORE_SYMBOLS.items():
        if market_cap_category in cap_categories:
            symbols.extend(cap_categories[market_cap_category])

    # Remove duplicates
    return list(set(symbols))


def get_symbol_metadata(symbol: str) -> Dict[str, str]:
    """
    Get sector and market cap category for a symbol

    Args:
        symbol: Stock ticker

    Returns:
        Dict with 'sector' and 'market_cap' keys
    """
    for sector, cap_categories in CORE_SYMBOLS.items():
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
    Get statistics about core symbol list

    Returns:
        Dict with counts by sector and market cap
    """
    stats = {
        'total_symbols': len(get_all_core_symbols()),
        'by_sector': {},
        'by_market_cap': {
            'large_cap': len(get_symbols_by_market_cap('large_cap')),
            'mid_cap': len(get_symbols_by_market_cap('mid_cap')),
            'small_cap': len(get_symbols_by_market_cap('small_cap'))
        }
    }

    for sector in CORE_SYMBOLS.keys():
        stats['by_sector'][sector] = len(get_symbols_by_sector(sector))

    return stats


if __name__ == '__main__':
    # Test and display core symbol list
    print("CORE SYMBOL LIST FOR TURBOMODE ML TRAINING")
    print()

    stats = get_statistics()

    print(f"Total Symbols: {stats['total_symbols']}")

    print(f"\nBy Market Cap:")
    print(f"  Large Cap:  {stats['by_market_cap']['large_cap']}")
    print(f"  Mid Cap:    {stats['by_market_cap']['mid_cap']}")
    print(f"  Small Cap:  {stats['by_market_cap']['small_cap']}")

    print(f"\nBy Sector:")
    for sector, count in sorted(stats['by_sector'].items()):
        print(f"  {sector.replace('_', ' ').title():30s} {count:2d} symbols")

    print(f"\nFull Symbol List:")
    all_symbols = get_all_core_symbols()
    for i, symbol in enumerate(all_symbols, 1):
        print(f"{symbol:8s}", end="")
        if i % 8 == 0:
            print()

    print("\n")
    print("Sample Metadata Lookup:")

    test_symbols = ['AAPL', 'JPM', 'JNJ', 'TSLA', 'NEE']
    for symbol in test_symbols:
        meta = get_symbol_metadata(symbol)
        print(f"{symbol:6s} -> {meta['sector']:25s} ({meta['market_cap_category']})")

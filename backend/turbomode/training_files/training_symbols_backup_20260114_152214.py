"""
Training Symbol List for TurboMode ML
Curated list of 40 high-quality stocks for model training

These symbols are used for:
- Model training (keeps training time manageable ~30-60 min)
- Feature engineering baseline
- Data quality validation

Selection Criteria:
- High liquidity leaders in each sector
- Market-moving stocks with strong institutional following
- Representative of broad market behavior
- Clean historical data

Last Updated: 2026-01-13
Total: 40 stocks + 3 crypto = 43 total
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

# Training Symbols (40 curated stocks)
TRAINING_SYMBOLS = {

    # ================================================================
    # TECHNOLOGY (7 symbols)
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
    # CONSUMER DISCRETIONARY (4 symbols)
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


def get_training_symbols() -> List[str]:
    """
    Get flattened list of training symbols (40 stocks)

    Returns:
        Sorted list of 40 training symbol tickers
    """
    symbols = []

    for sector, cap_categories in TRAINING_SYMBOLS.items():
        for cap_category, symbol_list in cap_categories.items():
            symbols.extend(symbol_list)

    return sorted(list(set(symbols)))


def get_training_symbols_with_crypto() -> List[str]:
    """
    Get training symbols + crypto (43 total)

    Returns:
        List of 40 stocks + 3 crypto symbols
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
        Dict with 'sector', 'market_cap_category', 'sector_code' keys
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

    print(f"Training Symbol List (40 stocks):")
    symbols = get_training_symbols()
    for i, symbol in enumerate(symbols, 1):
        print(f"{symbol:8s}", end="")
        if i % 8 == 0:
            print()
    print("\n")

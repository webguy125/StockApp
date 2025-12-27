"""
Core Symbol List for Advanced ML Training
Balanced representation across all 11 GICS sectors and 3 market cap categories

Selection Criteria:
- High liquidity (average volume > 500K daily)
- Established companies (listed > 1 year)
- Representative of sector behavior
- Mix of large cap (>$50B), mid cap ($10B-$50B), small cap ($2B-$10B)

Last Updated: 2025-12-21
Total Symbols: 77
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

# Market Cap Categories
MARKET_CAP_LARGE = 50_000_000_000   # $50B+
MARKET_CAP_MID = 10_000_000_000     # $10B - $50B
MARKET_CAP_SMALL = 2_000_000_000    # $2B - $10B

# Core Symbol List (Curated)
CORE_SYMBOLS = {

    # ================================================================
    # TECHNOLOGY (9 symbols)
    # ================================================================
    'technology': {
        'large_cap': [
            'AAPL',   # Apple - Consumer Electronics
            'MSFT',   # Microsoft - Software
            'NVDA',   # NVIDIA - Semiconductors
            'GOOGL',  # Alphabet - Internet Services
            'META',   # Meta - Social Media
        ],
        'mid_cap': [
            'PLTR',   # Palantir - Software/AI
            'SNOW',   # Snowflake - Cloud Software
            'CRWD',   # CrowdStrike - Cybersecurity
        ],
        'small_cap': [
            'SMCI',   # Super Micro Computer - Hardware
        ]
    },

    # ================================================================
    # FINANCIALS (8 symbols)
    # ================================================================
    'financials': {
        'large_cap': [
            'JPM',    # JPMorgan Chase - Bank
            'BAC',    # Bank of America - Bank
            'WFC',    # Wells Fargo - Bank
            'GS',     # Goldman Sachs - Investment Bank
        ],
        'mid_cap': [
            'SCHW',   # Charles Schwab - Brokerage
            'ALLY',   # Ally Financial - Online Bank
        ],
        'small_cap': [
            'GBCI',   # Glacier Bancorp - Regional Bank
            'CATY',   # Cathay General - Regional Bank
        ]
    },

    # ================================================================
    # HEALTHCARE (8 symbols)
    # ================================================================
    'healthcare': {
        'large_cap': [
            'JNJ',    # Johnson & Johnson - Pharma/Consumer
            'UNH',    # UnitedHealth - Insurance
            'LLY',    # Eli Lilly - Pharma
            'ABBV',   # AbbVie - Pharma
        ],
        'mid_cap': [
            'DXCM',   # Dexcom - Medical Devices
            'EXAS',   # Exact Sciences - Diagnostics
        ],
        'small_cap': [
            'TMDX',   # TransMedics - Medical Devices
            'KRYS',   # Krystal Biotech - Biotech
        ]
    },

    # ================================================================
    # CONSUMER DISCRETIONARY (8 symbols)
    # ================================================================
    'consumer_discretionary': {
        'large_cap': [
            'AMZN',   # Amazon - E-commerce/Cloud
            'TSLA',   # Tesla - Electric Vehicles
            'HD',     # Home Depot - Home Improvement
            'MCD',    # McDonald's - Restaurants
        ],
        'mid_cap': [
            'LULU',   # Lululemon - Athletic Apparel
            'DECK',   # Deckers - Footwear (HOKA, UGG)
        ],
        'small_cap': [
            'SHAK',   # Shake Shack - Restaurants
            'BOOT',   # Boot Barn - Retail
        ]
    },

    # ================================================================
    # COMMUNICATION SERVICES (7 symbols)
    # ================================================================
    'communication_services': {
        'large_cap': [
            'META',   # Meta - Social Media (also in tech)
            'GOOGL',  # Alphabet - Internet (also in tech)
            'DIS',    # Disney - Entertainment
            'NFLX',   # Netflix - Streaming
        ],
        'mid_cap': [
            'MTCH',   # Match Group - Dating Apps
        ],
        'small_cap': [
            'CARS',   # Cars.com - Automotive Classified
            'BMBL',   # Bumble - Dating App
        ]
    },

    # ================================================================
    # INDUSTRIALS (8 symbols)
    # ================================================================
    'industrials': {
        'large_cap': [
            'CAT',    # Caterpillar - Machinery
            'GE',     # General Electric - Aerospace/Power
            'BA',     # Boeing - Aerospace
            'HON',    # Honeywell - Diversified Industrial
        ],
        'mid_cap': [
            'UBER',   # Uber - Transportation
            'XPO',    # XPO Logistics - Freight
        ],
        'small_cap': [
            'CVCO',   # Cavco Industries - Manufactured Housing
            'ASTE',   # Astec Industries - Construction Equipment
        ]
    },

    # ================================================================
    # CONSUMER STAPLES (7 symbols)
    # ================================================================
    'consumer_staples': {
        'large_cap': [
            'PG',     # Procter & Gamble - Consumer Products
            'KO',     # Coca-Cola - Beverages
            'PEP',    # PepsiCo - Beverages/Snacks
            'WMT',    # Walmart - Retail
        ],
        'mid_cap': [
            'GIS',    # General Mills - Packaged Foods
        ],
        'small_cap': [
            'CHEF',   # The Chefs' Warehouse - Food Distribution
            'INGLES', # Ingles Markets - Grocery (IMKTA)
        ]
    },

    # ================================================================
    # ENERGY (7 symbols)
    # ================================================================
    'energy': {
        'large_cap': [
            'XOM',    # ExxonMobil - Integrated Oil
            'CVX',    # Chevron - Integrated Oil
            'COP',    # ConocoPhillips - E&P
            'SLB',    # Schlumberger - Oilfield Services
        ],
        'mid_cap': [
            'FANG',   # Diamondback Energy - E&P
        ],
        'small_cap': [
            'MTDR',   # Matador Resources - E&P
            'CTRA',   # Coterra Energy - E&P
        ]
    },

    # ================================================================
    # MATERIALS (7 symbols)
    # ================================================================
    'materials': {
        'large_cap': [
            'LIN',    # Linde - Industrial Gases
            'APD',    # Air Products - Industrial Gases
            'SHW',    # Sherwin-Williams - Paints
            'NEM',    # Newmont - Gold Mining
        ],
        'mid_cap': [
            'CF',     # CF Industries - Fertilizers
        ],
        'small_cap': [
            'CEIX',   # CONSOL Energy - Coal
            'IOSP',   # Innospec - Specialty Chemicals
        ]
    },

    # ================================================================
    # REAL ESTATE (7 symbols)
    # ================================================================
    'real_estate': {
        'large_cap': [
            'AMT',    # American Tower - Cell Towers
            'PLD',    # Prologis - Industrial REIT
            'EQIX',   # Equinix - Data Center REIT
            'SPG',    # Simon Property - Retail REIT
        ],
        'mid_cap': [
            'REXR',   # Rexford Industrial - Industrial REIT
        ],
        'small_cap': [
            'TRNO',   # Terreno Realty - Industrial REIT
            'STAG',   # STAG Industrial - Industrial REIT
        ]
    },

    # ================================================================
    # UTILITIES (6 symbols)
    # ================================================================
    'utilities': {
        'large_cap': [
            'NEE',    # NextEra Energy - Electric Utility
            'DUK',    # Duke Energy - Electric Utility
            'SO',     # Southern Company - Electric Utility
        ],
        'mid_cap': [
            'AES',    # AES Corporation - Power Generation
            'NRG',    # NRG Energy - Power Generation
        ],
        'small_cap': [
            'MGEE',   # MGE Energy - Electric Utility
        ]
    }
}

# Cryptocurrency (bonus - different behavior patterns)
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

    # Remove duplicates (META and GOOGL appear in multiple sectors)
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
    print("=" * 60)
    print("CORE SYMBOL LIST FOR ADVANCED ML TRAINING")
    print("=" * 60)

    stats = get_statistics()

    print(f"\nTotal Symbols: {stats['total_symbols']}")

    print(f"\nBy Market Cap:")
    print(f"  Large Cap (>$50B):  {stats['by_market_cap']['large_cap']}")
    print(f"  Mid Cap ($10B-$50B): {stats['by_market_cap']['mid_cap']}")
    print(f"  Small Cap ($2B-$10B): {stats['by_market_cap']['small_cap']}")

    print(f"\nBy Sector:")
    for sector, count in sorted(stats['by_sector'].items()):
        print(f"  {sector.replace('_', ' ').title():30s} {count:2d} symbols")

    print(f"\nFull Symbol List:")
    all_symbols = get_all_core_symbols()
    for i, symbol in enumerate(all_symbols, 1):
        print(f"{symbol:6s}", end="")
        if i % 10 == 0:
            print()

    print("\n\n" + "=" * 60)
    print("Sample Metadata Lookup:")
    print("=" * 60)

    test_symbols = ['AAPL', 'JPM', 'JNJ', 'TSLA', 'NEE']
    for symbol in test_symbols:
        meta = get_symbol_metadata(symbol)
        print(f"{symbol:6s} -> {meta['sector']:25s} ({meta['market_cap_category']})")

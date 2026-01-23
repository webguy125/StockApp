
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
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
  - Large Cap: 100 (all with options)
  - Mid Cap: 91 (all with options)
  - Small Cap: 39 (high liquidity)
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

# Training Symbols (230 stocks organized by sector and market cap)
TRAINING_SYMBOLS = {

    # ========================================================================
    # TECHNOLOGY (38 symbols: 25 large, 9 mid, 4 small)
    # ========================================================================
    'technology': {
        'large_cap': [
            'AAPL',
            'ADBE',
            'ADI',
            'AMAT',
            'AMD',
            'ANET',
            'AVGO',
            'CDNS',
            'CRM',
            'CRWD',
            'CSCO',
            'INTU',
            'KLAC',
            'LRCX',
            'MSFT',
            'MU',
            'NOW',
            'NVDA',
            'ORCL',
            'PANW',
            'PLTR',
            'QCOM',
            'SNPS',
            'TXN',
            'VRTX',
        ],

        'mid_cap': [
            'ADSK',
            'CRWV',
            'CTSH',
            'DDOG',
            'FTNT',
            'MDB',
            'NET',
            'SNOW',
            'ZS',
        ],

        'small_cap': [
            'COHU',
            'DOMO',
            'VMEO',
            'WOLF',
        ]
    },

    # ========================================================================
    # COMMUNICATION SERVICES (12 symbols: 4 large, 1 mid, 7 small)
    # ========================================================================
    'communication_services': {
        'large_cap': [
            'DIS',
            'GOOGL',
            'META',
            'NFLX',
        ],

        'mid_cap': [
            'WBD',
        ],

        'small_cap': [
            'AMC',
            'CCOI',
            'GOGO',
            'GTN',
            'IMAX',
            'PLAY',
            'WLY',
        ]
    },

    # ========================================================================
    # CONSUMER DISCRETIONARY (29 symbols: 10 large, 14 mid, 5 small)
    # ========================================================================
    'consumer_discretionary': {
        'large_cap': [
            'AMZN',
            'HD',
            'LOW',
            'MAR',
            'MCD',
            'NKE',
            'SBUX',
            'TJX',
            'TSLA',
            'WMT',
        ],

        'mid_cap': [
            'CCL',
            'CMG',
            'DG',
            'EBAY',
            'EXPE',
            'GM',
            'HLT',
            'LEN',
            'LVS',
            'ORLY',
            'RCL',
            'ROST',
            'TGT',
            'ULTA',
        ],

        'small_cap': [
            'CRI',
            'DBI',
            'IRBT',
            'PLCE',
            'TRIP',
        ]
    },

    # ========================================================================
    # CONSUMER STAPLES (8 symbols: 3 large, 1 mid, 4 small)
    # ========================================================================
    'consumer_staples': {
        'large_cap': [
            'COST',
            'KO',
            'PG',
        ],

        'mid_cap': [
            'HSY',
        ],

        'small_cap': [
            'EPC',
            'GO',
            'HAIN',
            'SMPL',
        ]
    },

    # ========================================================================
    # FINANCIALS (26 symbols: 17 large, 9 mid, 0 small)
    # ========================================================================
    'financials': {
        'large_cap': [
            'AXP',
            'BAC',
            'BK',
            'BLK',
            'BRK-B',
            'C',
            'CME',
            'COF',
            'GS',
            'ICE',
            'JPM',
            'MS',
            'PGR',
            'SCHW',
            'SPGI',
            'V',
            'WFC',
        ],

        'mid_cap': [
            'AIG',
            'ALL',
            'MET',
            'PRU',
            'PYPL',
            'SOFI',
            'TFC',
            'TRV',
            'USB',
        ],

        'small_cap': [
        ]
    },

    # ========================================================================
    # HEALTHCARE (35 symbols: 17 large, 11 mid, 7 small)
    # ========================================================================
    'healthcare': {
        'large_cap': [
            'ABBV',
            'ABT',
            'AMGN',
            'BMY',
            'CVS',
            'DHR',
            'GILD',
            'HCA',
            'ISRG',
            'JNJ',
            'LLY',
            'MDT',
            'MRK',
            'PFE',
            'SYK',
            'TMO',
            'UNH',
        ],

        'mid_cap': [
            'ALNY',
            'BDX',
            'CAH',
            'CI',
            'EW',
            'HUM',
            'IDXX',
            'MTD',
            'NTRA',
            'RMD',
            'VEEV',
        ],

        'small_cap': [
            'ACHC',
            'ATRC',
            'BCRX',
            'BTAI',
            'OMI',
            'SGRY',
            'TNDM',
        ]
    },

    # ========================================================================
    # INDUSTRIALS (38 symbols: 14 large, 21 mid, 3 small)
    # ========================================================================
    'industrials': {
        'large_cap': [
            'CAT',
            'DE',
            'EMR',
            'ETN',
            'GD',
            'GE',
            'HON',
            'HWM',
            'LMT',
            'NOC',
            'PH',
            'TT',
            'UNP',
            'UPS',
        ],

        'mid_cap': [
            'CARR',
            'CMI',
            'CSX',
            'CTAS',
            'FAST',
            'FDX',
            'GWW',
            'ITW',
            'JCI',
            'NSC',
            'ODFL',
            'OTIS',
            'PCAR',
            'PWR',
            'ROK',
            'ROL',
            'ROP',
            'RSG',
            'URI',
            'WAB',
            'XYL',
        ],

        'small_cap': [
            'ARLO',
            'NX',
            'WERN',
        ]
    },

    # ========================================================================
    # ENERGY (12 symbols: 2 large, 5 mid, 5 small)
    # ========================================================================
    'energy': {
        'large_cap': [
            'CVX',
            'XOM',
        ],

        'mid_cap': [
            'EQT',
            'FANG',
            'OXY',
            'SLB',
            'VLO',
        ],

        'small_cap': [
            'DK',
            'HLX',
            'REI',
            'VTLE',
            'WTI',
        ]
    },

    # ========================================================================
    # MATERIALS (10 symbols: 4 large, 6 mid, 0 small)
    # ========================================================================
    'materials': {
        'large_cap': [
            'FCX',
            'LIN',
            'NEM',
            'SHW',
        ],

        'mid_cap': [
            'A',
            'APD',
            'ECL',
            'MLM',
            'NUE',
            'VMC',
        ],

        'small_cap': [
        ]
    },

    # ========================================================================
    # UTILITIES (12 symbols: 3 large, 9 mid, 0 small)
    # ========================================================================
    'utilities': {
        'large_cap': [
            'DUK',
            'NEE',
            'SO',
        ],

        'mid_cap': [
            'AEP',
            'D',
            'ED',
            'ETR',
            'EXC',
            'PEG',
            'SRE',
            'WEC',
            'XEL',
        ],

        'small_cap': [
        ]
    },

    # ========================================================================
    # REAL ESTATE (10 symbols: 1 large, 5 mid, 4 small)
    # ========================================================================
    'real_estate': {
        'large_cap': [
            'PLD',
        ],

        'mid_cap': [
            'AMT',
            'DLR',
            'EXR',
            'PSA',
            'VTR',
        ],

        'small_cap': [
            'ALEX',
            'ESRT',
            'PDM',
            'UNIT',
        ]
    },
}

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
    print("\n")

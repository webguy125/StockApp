
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Scanning Symbol List for TurboMode
Extended list of 230 stocks for signal scanning

These symbols are used for:
- Overnight scanner (generates trading signals)
- Signal generation (fast model inference)
- More trading opportunities

Selection Criteria:
- ALL 40 training stocks (must scan stocks we trained on)
- S&P 500 constituents
- High liquidity and options volume
- Strong institutional interest
- Diverse sector representation
- Clean market cap data (no delisted/problematic stocks)

QUARANTINED (5 symbols removed):
- BLL, ETFC, NLSN, PKI (no market cap / insufficient data)
- SIVB (Silicon Valley Bank - delisted/collapsed)

Last Updated: 2026-01-14
Total: 230 stocks (40 training + 190 additional)
"""

from typing import List

# Scanning Symbols (230 stocks - includes all 40 training stocks + 190 additional)
SCANNING_SYMBOLS = [
    # Technology (31 stocks) [+AMD, +CRM, +CRWV, +IREN, +MU from additions]
    "AAPL", "ADBE", "ADI", "ADSK", "AKAM", "ALGN", "AMD", "AMAT", "ANET", "AVGO",
    "CDNS", "CRM", "CRWV", "CSCO", "CTSH", "FTNT", "INTU", "IREN", "KLAC", "LRCX",
    "MU", "MSFT", "NOW", "NVDA", "ORCL", "PANW", "QCOM", "SNPS", "TXN", "VRTX",
    "ZBRA",

    # Communication Services (4 stocks) [ALL FROM TRAINING]
    "DIS", "GOOGL", "META", "NFLX",

    # Financials (28 stocks) [+BAC, +BRK-B, +JPM from training]
    "AIG", "ALL", "AXP", "BAC", "BK", "BLK", "BRK-B", "C", "CFG", "CME", "COF",
    "GS", "ICE", "JPM", "MET", "MS", "PGR", "PRU", "PYPL", "SCHW", "SPGI",
    "TFC", "TROW", "TRV", "USB", "V", "WFC", "ZION",

    # Healthcare (36 stocks)
    "ABT", "ABBV", "AMGN", "BAX", "BDX", "BIIB", "BMY", "CAH", "CI", "CNC", "CVS",
    "DHR", "EW", "GILD", "HCA", "HOLX", "HSIC", "HUM", "IDXX", "ILMN", "ISRG", "JNJ",
    "LH", "LLY", "MDT", "MRK", "MTD", "PFE", "RMD", "STE", "SYK", "TFX", "TMO",
    "UNH", "VTRS", "ZBH",

    # Materials (23 stocks)
    "A", "APD", "AVY", "CE", "CF", "DD", "DOW", "ECL", "EMN", "FMC", "IFF",
    "IP", "LIN", "LYB", "MLM", "MOS", "NEM", "NUE", "PKG", "PPG", "SEE", "SHW", "VMC",

    # Consumer Discretionary (33 stocks) [+AMZN, +TSLA from training]
    "AMZN", "CCL", "CMG", "DG", "DLTR", "EBAY", "EXPE", "GM", "HAS", "HD", "HLT", "KMX",
    "LEN", "LOW", "LVS", "MAR", "MCD", "MGM", "NKE", "NWL", "ORLY", "POOL", "ROST",
    "SBUX", "TGT", "TJX", "TSCO", "TSLA", "ULTA", "VFC", "WHR", "WMT", "WYNN",

    # Consumer Staples (3 stocks) [ALL FROM TRAINING]
    "COST", "KO", "PG",

    # Energy (4 stocks) [+CVX, +OXY, +SLB, +XOM from training/additions]
    "CVX", "OXY", "SLB", "XOM",

    # Utilities (22 stocks)
    "AEP", "AES", "CNP", "D", "DTE", "DUK", "ED", "EIX", "ETR", "EVRG", "EXC", "FE",
    "NEE", "NI", "NRG", "PEG", "PNW", "PPL", "SO", "SRE", "WEC", "XEL",

    # Industrials (43 stocks) [+UPS from training]
    "APTV", "CAT", "CMI", "CSX", "CTAS", "DE", "DOV", "EMR", "ETN", "EXPD", "FAST",
    "FDX", "FLS", "GD", "GE", "GWW", "HON", "HWM", "IEX", "ITW", "JCI", "LMT", "MAS",
    "NOC", "NSC", "ODFL", "OTIS", "PCAR", "PH", "PNR", "PWR", "RHI", "ROK",
    "ROL", "ROP", "RSG", "SNA", "SWK", "TT", "TXT", "UNP", "UPS", "WAB", "XYL",

    # Real Estate (2 stocks) [ALL FROM TRAINING]
    "AMT", "PLD"
]

# Sector mapping for all 230 scanning symbols
SECTOR_MAPPING = {
    # Technology (31 stocks)
    'technology': [
        "AAPL", "ADBE", "ADI", "ADSK", "AKAM", "ALGN", "AMD", "AMAT", "ANET", "AVGO",
        "CDNS", "CRM", "CRWV", "CSCO", "CTSH", "FTNT", "INTU", "IREN", "KLAC", "LRCX",
        "MU", "MSFT", "NOW", "NVDA", "ORCL", "PANW", "QCOM", "SNPS", "TXN", "VRTX", "ZBRA"
    ],

    # Communication Services (4 stocks)
    'communication_services': ["DIS", "GOOGL", "META", "NFLX"],

    # Financials (28 stocks)
    'financials': [
        "AIG", "ALL", "AXP", "BAC", "BK", "BLK", "BRK-B", "C", "CFG", "CME", "COF",
        "GS", "ICE", "JPM", "MET", "MS", "PGR", "PRU", "PYPL", "SCHW", "SPGI",
        "TFC", "TROW", "TRV", "USB", "V", "WFC", "ZION"
    ],

    # Healthcare (36 stocks)
    'healthcare': [
        "ABT", "ABBV", "AMGN", "BAX", "BDX", "BIIB", "BMY", "CAH", "CI", "CNC", "CVS",
        "DHR", "EW", "GILD", "HCA", "HOLX", "HSIC", "HUM", "IDXX", "ILMN", "ISRG", "JNJ",
        "LH", "LLY", "MDT", "MRK", "MTD", "PFE", "RMD", "STE", "SYK", "TFX", "TMO",
        "UNH", "VTRS", "ZBH"
    ],

    # Materials (23 stocks)
    'materials': [
        "A", "APD", "AVY", "CE", "CF", "DD", "DOW", "ECL", "EMN", "FMC", "IFF",
        "IP", "LIN", "LYB", "MLM", "MOS", "NEM", "NUE", "PKG", "PPG", "SEE", "SHW", "VMC"
    ],

    # Consumer Discretionary (33 stocks)
    'consumer_discretionary': [
        "AMZN", "CCL", "CMG", "DG", "DLTR", "EBAY", "EXPE", "GM", "HAS", "HD", "HLT", "KMX",
        "LEN", "LOW", "LVS", "MAR", "MCD", "MGM", "NKE", "NWL", "ORLY", "POOL", "ROST",
        "SBUX", "TGT", "TJX", "TSCO", "TSLA", "ULTA", "VFC", "WHR", "WMT", "WYNN"
    ],

    # Consumer Staples (3 stocks)
    'consumer_staples': ["COST", "KO", "PG"],

    # Energy (4 stocks)
    'energy': ["CVX", "OXY", "SLB", "XOM"],

    # Utilities (22 stocks)
    'utilities': [
        "AEP", "AES", "CNP", "D", "DTE", "DUK", "ED", "EIX", "ETR", "EVRG", "EXC", "FE",
        "NEE", "NI", "NRG", "PEG", "PNW", "PPL", "SO", "SRE", "WEC", "XEL"
    ],

    # Industrials (43 stocks)
    'industrials': [
        "APTV", "CAT", "CMI", "CSX", "CTAS", "DE", "DOV", "EMR", "ETN", "EXPD", "FAST",
        "FDX", "FLS", "GD", "GE", "GWW", "HON", "HWM", "IEX", "ITW", "JCI", "LMT", "MAS",
        "NOC", "NSC", "ODFL", "OTIS", "PCAR", "PH", "PNR", "PWR", "RHI", "ROK",
        "ROL", "ROP", "RSG", "SNA", "SWK", "TT", "TXT", "UNP", "UPS", "WAB", "XYL"
    ],

    # Real Estate (2 stocks)
    'real_estate': ["AMT", "PLD"]
}

# GICS Sector codes (same as training_symbols.py and core_symbols.py)
SECTOR_CODES = {
    'technology': 45,
    'communication_services': 50,
    'financials': 40,
    'healthcare': 35,
    'materials': 15,
    'consumer_discretionary': 25,
    'consumer_staples': 30,
    'energy': 10,
    'utilities': 55,
    'industrials': 20,
    'real_estate': 60
}


def get_scanning_symbols() -> List[str]:
    """
    Get list of all scanning symbols (230 stocks)
    Includes ALL 40 training stocks + 190 additional stocks

    Returns:
        Sorted list of 230 scanning symbol tickers
    """
    return sorted(list(set(SCANNING_SYMBOLS)))


def get_symbol_metadata(symbol: str) -> dict:
    """
    Get sector metadata for a scanning symbol

    Args:
        symbol: Stock ticker

    Returns:
        Dict with 'sector', 'sector_code' keys
    """
    for sector, symbol_list in SECTOR_MAPPING.items():
        if symbol in symbol_list:
            return {
                'symbol': symbol,
                'sector': sector,
                'sector_code': SECTOR_CODES[sector]
            }

    # Symbol not found - return unknown
    return {
        'symbol': symbol,
        'sector': 'unknown',
        'sector_code': -1
    }


def get_statistics() -> dict:
    """
    Get statistics about scanning symbol list

    Returns:
        Dict with symbol count
    """
    return {
        'total_scanning_symbols': len(get_scanning_symbols()),
        'unique_symbols': len(set(SCANNING_SYMBOLS))
    }


if __name__ == '__main__':
    print("=" * 80)
    print("SCANNING SYMBOLS FOR TURBOMODE")
    print("=" * 80)
    print()

    stats = get_statistics()
    print(f"Total Scanning Symbols: {stats['total_scanning_symbols']}")
    print(f"Unique Symbols:         {stats['unique_symbols']}")
    print()

    print(f"Scanning Symbol List (230 stocks):")
    symbols = get_scanning_symbols()
    for i, symbol in enumerate(symbols, 1):
        print(f"{symbol:8s}", end="")
        if i % 10 == 0:
            print()
    print("\n")

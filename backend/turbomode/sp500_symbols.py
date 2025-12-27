"""
S&P 500 Symbol List with Market Cap and Sector Classifications
Used by TurboMode for overnight scanning and predictions
"""

# 11 GICS Sectors
SECTORS = {
    'Information Technology': 'IT',
    'Health Care': 'Healthcare',
    'Financials': 'Financials',
    'Consumer Discretionary': 'Consumer Discretionary',
    'Communication Services': 'Communication',
    'Industrials': 'Industrials',
    'Consumer Staples': 'Consumer Staples',
    'Energy': 'Energy',
    'Utilities': 'Utilities',
    'Real Estate': 'Real Estate',
    'Materials': 'Materials'
}

# S&P 500 symbols classified by market cap and sector
# Market Cap: Large (>$200B), Mid ($10B-$200B), Small (<$10B)
SP500_SYMBOLS = {
    'large_cap': {
        'Information Technology': [
            'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'CSCO',
            'ACN', 'AMD', 'IBM', 'INTC', 'NOW', 'TXN', 'QCOM', 'INTU',
            'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'PLTR'
        ],
        'Health Care': [
            'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'ABT', 'TMO', 'PFE',
            'DHR', 'AMGN', 'BMY', 'ISRG', 'GILD', 'CVS', 'CI', 'VRTX'
        ],
        'Financials': [
            'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS',
            'AXP', 'SPGI', 'BLK', 'C', 'SCHW', 'CB', 'PGR', 'MMC'
        ],
        'Consumer Discretionary': [
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX',
            'BKNG', 'ABNB', 'CMG', 'MAR', 'GM', 'F', 'ORLY', 'AZO'
        ],
        'Communication Services': [
            'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS'
        ],
        'Industrials': [
            'GE', 'CAT', 'RTX', 'BA', 'HON', 'UNP', 'UPS', 'LMT',
            'DE', 'ADP', 'GD', 'NOC', 'ETN', 'MMM', 'ITW', 'EMR'
        ],
        'Consumer Staples': [
            'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ',
            'CL', 'KMB', 'GIS', 'K', 'HSY', 'CHD', 'CLX', 'SJM'
        ],
        'Energy': [
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
            'OXY', 'WMB', 'KMI', 'HES', 'HAL', 'DVN', 'FANG', 'BKR'
        ],
        'Utilities': [
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PCG',
            'XEL', 'ED', 'WEC', 'ES', 'AWK', 'DTE', 'PPL', 'AEE'
        ],
        'Real Estate': [
            'AMT', 'PLD', 'EQIX', 'PSA', 'WELL', 'SPG', 'O', 'DLR',
            'VICI', 'AVB', 'EQR', 'VTR', 'INVH', 'ARE', 'MAA', 'KIM'
        ],
        'Materials': [
            'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'DOW',
            'DD', 'PPG', 'VMC', 'MLM', 'CTVA', 'IFF', 'ALB', 'BALL'
        ]
    },
    'mid_cap': {
        'Information Technology': [
            'ANSS', 'MPWR', 'NTAP', 'PTC', 'TYL', 'ZBRA', 'FFIV', 'JNPR',
            'AKAM', 'CRM', 'FTNT', 'GEN', 'IT', 'KEYS', 'EPAM', 'GDDY'
        ],
        'Health Care': [
            'EXAS', 'DXCM', 'ILMN', 'IDXX', 'MTD', 'A', 'ALGN', 'ZBH',
            'BIO', 'WAT', 'STE', 'TECH', 'HOLX', 'PODD', 'RVTY', 'VTRS'
        ],
        'Financials': [
            'ALLY', 'CFG', 'HBAN', 'RF', 'KEY', 'FITB', 'MTB', 'NTRS',
            'STT', 'TFC', 'USB', 'WBS', 'ZION', 'FRC', 'SIVB', 'SBNY'
        ],
        'Consumer Discretionary': [
            'LULU', 'DECK', 'POOL', 'ULTA', 'DRI', 'YUM', 'QSR', 'MHK',
            'WHR', 'LEN', 'DHI', 'PHM', 'TOL', 'KBH', 'TPX', 'GRMN'
        ],
        'Communication Services': [
            'MTCH', 'NWSA', 'NWS', 'PARA', 'DISH', 'OMC', 'IPG', 'LYV'
        ],
        'Industrials': [
            'UBER', 'CARR', 'OTIS', 'PCAR', 'ROK', 'DOV', 'FTV', 'GNRC',
            'IR', 'IEX', 'J', 'JCI', 'LDOS', 'MAS', 'NDSN', 'PWR'
        ],
        'Consumer Staples': [
            'KR', 'SYY', 'TSN', 'STZ', 'TAP', 'CAG', 'CPB', 'HRL',
            'MKC', 'LW', 'BG', 'SJM', 'BF.B', 'CHD', 'KMB', 'CLX'
        ],
        'Energy': [
            'MTDR', 'CTRA', 'APA', 'DVN', 'MRO', 'OVV', 'TRGP', 'LNG',
            'CHRD', 'EQT', 'CNX', 'RRC', 'SM', 'MGY', 'PR', 'CRK'
        ],
        'Utilities': [
            'AES', 'NRG', 'CEG', 'VST', 'CNP', 'CMS', 'PEG', 'ETR',
            'FE', 'NI', 'LNT', 'EVRG', 'ATO', 'PNW', 'OGE', 'NWE'
        ],
        'Real Estate': [
            'REXR', 'FR', 'EXR', 'CUBE', 'LSI', 'REG', 'UDR', 'CPT',
            'ESS', 'FRT', 'BXP', 'KRC', 'HIW', 'DEI', 'PDM', 'AIV'
        ],
        'Materials': [
            'CF', 'MOS', 'FMC', 'CE', 'IP', 'PKG', 'AMCR', 'SEE',
            'AVY', 'EMN', 'WRK', 'SLGN', 'HUN', 'OLN', 'NEU', 'ASH'
        ]
    },
    'small_cap': {
        'Information Technology': [
            'SMCI', 'CRWD', 'SNOW', 'PANW', 'DDOG', 'ZS', 'OKTA', 'NET',
            'S', 'DBX', 'BILL', 'DOCN', 'FROG', 'DT', 'WEX', 'FICO'
        ],
        'Health Care': [
            'TMDX', 'KRYS', 'NVCR', 'PRVA', 'VERV', 'RARE', 'FOLD', 'RGNX',
            'ARWR', 'BMRN', 'VCEL', 'ALNY', 'IONS', 'SRPT', 'NBIX', 'HALO'
        ],
        'Financials': [
            'GBCI', 'CATY', 'FIBK', 'WTFC', 'SFNC', 'CASH', 'CADE', 'FCNCA',
            'ONB', 'UBSI', 'CVBF', 'INDB', 'FULT', 'UCBI', 'PB', 'HWC'
        ],
        'Consumer Discretionary': [
            'SHAK', 'WING', 'TXRH', 'BLMN', 'CAKE', 'DIN', 'EAT', 'PLAY',
            'CHUY', 'BJRI', 'KRUS', 'RUTH', 'LOCO', 'FWRG', 'PZZA', 'DENN'
        ],
        'Communication Services': [
            'CARS', 'BMBL', 'RBLX', 'PINS', 'SNAP', 'YELP', 'QNST', 'MGNI',
            'TDS', 'USM', 'SHEN', 'CABO', 'GOGO', 'ATUS', 'LBRDA', 'TRIP'
        ],
        'Industrials': [
            'CVCO', 'ASTE', 'GVA', 'MLI', 'WLDN', 'RBC', 'ROLL', 'RUSHA',
            'JBHT', 'CHRW', 'LSTR', 'ODFL', 'XPO', 'KNX', 'R', 'ALK'
        ],
        'Consumer Staples': [
            'CHEF', 'INGLES', 'GO', 'SFM', 'ROST', 'TGT', 'DG', 'DLTR',
            'FIVE', 'OLLI', 'BIG', 'PSMT', 'ACI', 'KR', 'SYY', 'USF'
        ],
        'Energy': [
            'CEIX', 'BTU', 'ARCH', 'AMR', 'HCC', 'ARLP', 'METC', 'SUN',
            'CIVI', 'CPE', 'WTI', 'REI', 'VTLE', 'GPOR', 'NOG', 'CRC'
        ],
        'Utilities': [
            'MGEE', 'AVA', 'OTTR', 'NWE', 'SJW', 'YORW', 'MSEX', 'UTL',
            'ARTNA', 'BKH', 'CWEN.A', 'NJR', 'SR', 'SWX', 'AWR', 'CWT'
        ],
        'Real Estate': [
            'TRNO', 'STAG', 'COLD', 'REXR', 'EGP', 'NSA', 'SAFE', 'IRM',
            'PK', 'ROIC', 'VRE', 'EPRT', 'ADC', 'FCPT', 'GTY', 'NHI'
        ],
        'Materials': [
            'IOSP', 'SLVM', 'CENX', 'KALU', 'MP', 'NGVT', 'SXT', 'BCPC',
            'FUL', 'KWR', 'SCL', 'TROX', 'CBT', 'HWKN', 'HL', 'CDE'
        ]
    }
}


def get_all_symbols():
    """Get all S&P 500 symbols as flat list"""
    symbols = []
    for cap_size in SP500_SYMBOLS.values():
        for sector_stocks in cap_size.values():
            symbols.extend(sector_stocks)
    return sorted(list(set(symbols)))  # Remove duplicates, sort


def get_symbols_by_cap(cap_size):
    """Get symbols for specific market cap (large_cap, mid_cap, or small_cap)"""
    if cap_size not in SP500_SYMBOLS:
        raise ValueError(f"Invalid cap size: {cap_size}. Must be large_cap, mid_cap, or small_cap")

    symbols = []
    for sector_stocks in SP500_SYMBOLS[cap_size].values():
        symbols.extend(sector_stocks)
    return sorted(list(set(symbols)))


def get_symbols_by_sector(sector):
    """Get all symbols in a specific sector across all market caps"""
    symbols = []
    for cap_size in SP500_SYMBOLS.values():
        if sector in cap_size:
            symbols.extend(cap_size[sector])
    return sorted(list(set(symbols)))


def get_sector_for_symbol(symbol):
    """Find which sector a symbol belongs to"""
    for cap_size in SP500_SYMBOLS.values():
        for sector, stocks in cap_size.items():
            if symbol in stocks:
                return sector
    return None


def get_cap_size_for_symbol(symbol):
    """Find which market cap category a symbol belongs to"""
    for cap_size, sectors in SP500_SYMBOLS.items():
        for stocks in sectors.values():
            if symbol in stocks:
                return cap_size
    return None


if __name__ == "__main__":
    # Test the functions
    all_symbols = get_all_symbols()
    print(f"Total S&P 500 symbols: {len(all_symbols)}")

    large_cap = get_symbols_by_cap('large_cap')
    mid_cap = get_symbols_by_cap('mid_cap')
    small_cap = get_symbols_by_cap('small_cap')

    print(f"\nBreakdown:")
    print(f"  Large Cap: {len(large_cap)}")
    print(f"  Mid Cap: {len(mid_cap)}")
    print(f"  Small Cap: {len(small_cap)}")

    print(f"\nSample - AAPL:")
    print(f"  Sector: {get_sector_for_symbol('AAPL')}")
    print(f"  Cap Size: {get_cap_size_for_symbol('AAPL')}")

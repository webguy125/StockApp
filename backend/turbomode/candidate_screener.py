"""
Candidate Stock Screener for TurboMode Curation

Searches for potential symbols meeting the strict TurboMode criteria.

Sources:
- S&P 500 (large cap pool)
- S&P MidCap 400 (mid cap pool)
- S&P SmallCap 600 (small cap pool)
- High-volume NASDAQ stocks
"""

import pandas as pd
import yfinance as yf
from typing import List, Set

# EXPANDED S&P 500 + High-Volume Large Caps (~500+ symbols)
SP500_SAMPLE = [
    # Mega-cap tech (>$1T)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',

    # Large tech ($100B-$1T)
    'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'ACN', 'AMD', 'INTC', 'QCOM', 'TXN',
    'NOW', 'INTU', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'PANW', 'PLTR',
    'SHOP', 'SQ', 'PYPL', 'COIN', 'MRVL', 'ABNB', 'NXPI', 'ADI', 'ON', 'MCHP',
    'FTNT', 'WDAY', 'VEEV', 'DDOG', 'ZS', 'NET', 'OKTA', 'CRWD', 'SNOW', 'MDB',

    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'BLK', 'C', 'AXP', 'SPGI',
    'CB', 'PGR', 'MMC', 'ICE', 'CME', 'BX', 'KKR', 'AON', 'TRV', 'ALL', 'MET', 'PRU',
    'AIG', 'AFL', 'USB', 'PNC', 'TFC', 'COF', 'AMP', 'MKTX', 'MCO', 'MSCI', 'FIS',
    'FI', 'TROW', 'BEN', 'IVZ', 'STT', 'NTRS', 'RF', 'FITB', 'HBAN', 'KEY', 'CFG',

    # Healthcare
    'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'ISRG', 'DHR', 'PFE',
    'AMGN', 'VRTX', 'GILD', 'CVS', 'CI', 'MCK', 'ELV', 'HCA', 'BSX', 'MDT',
    'REGN', 'BMY', 'HUM', 'ZTS', 'SYK', 'BDX', 'EW', 'ILMN', 'BIIB', 'MRNA',
    'MOH', 'CNC', 'CAH', 'COR', 'IDXX', 'IQV', 'A', 'DXCM', 'ALGN', 'HOLX',
    'RMD', 'TECH', 'WST', 'PODD', 'EXAS', 'LH', 'DGX', 'STE', 'BAX', 'COO',

    # Consumer Discretionary
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'MAR',
    'F', 'GM', 'CMG', 'YUM', 'ORLY', 'AZO', 'RCL', 'CCL', 'NCLH', 'LVS',
    'WYNN', 'MGM', 'LULU', 'DECK', 'CROX', 'SKX', 'TPR', 'RL', 'PVH', 'GPS',
    'TGT', 'DG', 'DLTR', 'BBY', 'ULTA', 'ROST', 'TSCO', 'EBAY', 'ETSY', 'W',

    # Consumer Staples
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'CL', 'MDLZ', 'GIS',
    'KHC', 'KMB', 'SYY', 'ADM', 'HSY', 'K', 'TSN', 'HRL', 'CAG', 'CPB',
    'TAP', 'STZ', 'BF.B', 'DEO', 'SAM', 'MNST', 'KDP', 'CLX', 'CHD', 'EL',

    # Communication Services
    'META', 'GOOGL', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA',
    'ATVI', 'TTWO', 'RBLX', 'U', 'MTCH', 'PINS', 'SNAP', 'ROKU', 'PARA', 'WBD',
    'FOXA', 'FOX', 'NWSA', 'NWS', 'NYT', 'LYV', 'MSG', 'SIRI', 'SPOT',

    # Industrials
    'CAT', 'BA', 'HON', 'UNP', 'RTX', 'UPS', 'GE', 'LMT', 'DE', 'MMM',
    'GD', 'NOC', 'ITW', 'EMR', 'ETN', 'PH', 'CARR', 'WM', 'NSC', 'FDX',
    'CSX', 'ODFL', 'JBHT', 'EXPD', 'CHRW', 'UBER', 'DASH', 'LYFT', 'RSG', 'ROL',
    'PWR', 'J', 'FAST', 'WWD', 'WSO', 'GWW', 'IR', 'XYL', 'ROK', 'AME',

    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
    'HAL', 'BKR', 'DVN', 'FANG', 'MRO', 'HES', 'APA', 'CTRA', 'OVV', 'MTDR',
    'CNX', 'AR', 'MUR', 'RRC', 'SM', 'MGY', 'CHRD', 'PR', 'NOG', 'CLR',

    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'VMC', 'MLM', 'NUE', 'DOW',
    'DD', 'PPG', 'EMN', 'ALB', 'CE', 'FMC', 'CF', 'MOS', 'NTR', 'IFF',
    'STLD', 'RS', 'X', 'CLF', 'AA', 'CENX', 'MP', 'LAD', 'SEE', 'PKG',

    # Real Estate
    'AMT', 'PLD', 'EQIX', 'CCI', 'PSA', 'WELL', 'DLR', 'O', 'SPG', 'VICI',
    'AVB', 'EQR', 'DRE', 'MAA', 'CPT', 'UDR', 'ESS', 'IRM', 'SUI', 'CUBE',
    'VTR', 'PEAK', 'DOC', 'EXR', 'INVH', 'AMH', 'ACC', 'HST', 'RHP', 'BXP',

    # Utilities
    'NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'D', 'EXC', 'XEL', 'ES',
    'PCG', 'ED', 'WEC', 'PEG', 'EIX', 'AWK', 'DTE', 'CMS', 'PPL', 'FE',
    'NRG', 'VST', 'AES', 'CNP', 'ETR', 'EVRG', 'LNT', 'NI', 'PNW', 'OGE'
]

# EXPANDED Mid-cap candidates (~200+ symbols)
MIDCAP_SAMPLE = [
    # Technology
    'CRWD', 'DDOG', 'ZS', 'NET', 'OKTA', 'SNOW', 'MDB', 'ESTC', 'ZI', 'S',
    'FOUR', 'RNG', 'PCTY', 'GLBE', 'GDDY', 'MANH', 'WEX', 'CCOI', 'BR', 'GWRE',
    'TENB', 'SMAR', 'APPF', 'FFIV', 'VRNS', 'DUOL', 'BILL', 'ASAN', 'DOCN', 'FRSH',
    'PATH', 'FROG', 'CFLT', 'GTLB', 'IOT', 'AYX', 'PLAN', 'NCNO', 'DT', 'PCOR',

    # Financials
    'ALLY', 'HBAN', 'KEY', 'RF', 'CFG', 'FITB', 'MTB', 'NTRS', 'ZION', 'CBSH',
    'WAL', 'EWBC', 'ONB', 'UMBF', 'BOKF', 'SFNC', 'CADE', 'IBOC', 'TCBI', 'FFIN',
    'SOFI', 'HOOD', 'LC', 'AFRM', 'UPST', 'NU', 'VIRT', 'IBKR', 'LPLA', 'RJF',
    'SF', 'SNV', 'WTFC', 'ASB', 'VLY', 'PB', 'FHN', 'SIVB', 'CMA', 'GBCI',

    # Healthcare
    'DXCM', 'ALGN', 'HOLX', 'PODD', 'TECH', 'LNTH', 'RXRX', 'TWST', 'VCYT', 'NVST',
    'OMCL', 'GKOS', 'NEOG', 'NVCR', 'PRCT', 'OFIX', 'GMED', 'ATRC', 'IRTC', 'VCEL',
    'VEEV', 'TDOC', 'OSCR', 'DOCS', 'SDGR', 'PRVA', 'LEGN', 'ARWR', 'CRNX', 'CRSP',
    'NTLA', 'BLUE', 'FATE', 'EDIT', 'BEAM', 'VERV', 'IONS', 'FOLD', 'ACAD', 'ALNY',

    # Consumer Discretionary
    'LULU', 'DECK', 'ONON', 'CROX', 'SKX', 'FL', 'ASO', 'RVLV', 'FTCH', 'ETSY',
    'W', 'CVNA', 'CHWY', 'FIGS', 'VVV', 'SAM', 'TAP', 'WING', 'EAT', 'CAKE',
    'TXRH', 'BLMN', 'DNUT', 'BROS', 'CAVA', 'SHAK', 'DRI', 'BJRI', 'PLAY', 'PLNT',
    'BURL', 'FIVE', 'OLLI', 'GRMN', 'GPRO', 'SONO', 'AMWD', 'LEG', 'TPX', 'PRPL',

    # Communication Services
    'MTCH', 'PINS', 'SNAP', 'BMBL', 'IAC', 'ROKU', 'SIRI', 'WBD', 'PARA', 'FOXA',
    'RBLX', 'TTWO', 'ZG', 'Z', 'ANGI', 'YELP', 'TRIP', 'CARS', 'QUOT', 'ZD',

    # Industrials
    'UBER', 'DASH', 'LYFT', 'XPO', 'JBHT', 'ODFL', 'CHRW', 'KNX', 'EXPD', 'LSTR',
    'FAST', 'WWD', 'WSO', 'GWW', 'DCI', 'RUSHA', 'WERN', 'SAIA', 'JBLU', 'ALK',
    'DAL', 'UAL', 'AAL', 'LUV', 'SAVE', 'JBLU', 'SKYW', 'HA', 'MESA', 'ALGT',
    'JOBY', 'ACHR', 'EH', 'EVTL', 'LILM', 'BLDE', 'RDW', 'ARRY', 'GEV', 'PLUG',

    # Energy
    'FANG', 'DVN', 'MTDR', 'OVV', 'CHRD', 'SM', 'MGY', 'CTRA', 'PR', 'RRC',
    'CNX', 'AR', 'MUR', 'NOG', 'CLR', 'WLL', 'PDCE', 'VTLE', 'GPOR', 'CPE',

    # Materials
    'CF', 'MOS', 'NTR', 'FMC', 'ALB', 'SQM', 'SMG', 'IFF', 'CE', 'EMN',
    'MP', 'LAD', 'STLD', 'RS', 'X', 'CLF', 'AA', 'CENX', 'HCC', 'ATI',

    # Real Estate
    'REXR', 'FR', 'STAG', 'TRNO', 'EGP', 'KRG', 'EPR', 'BXP', 'VNO', 'HIW',
    'MAC', 'SKT', 'SLG', 'PDM', 'ROIC', 'GTY', 'KIM', 'REG', 'FRT', 'BRX',

    # Utilities
    'AES', 'NRG', 'VST', 'CNP', 'CMS', 'DTE', 'EVRG', 'FE', 'NI', 'PNW',
    'OGE', 'LNT', 'ATO', 'BKH', 'NJR', 'NWE', 'AVA', 'SJW', 'AWR', 'YORW'
]

# EXPANDED Small-cap candidates (~200+ symbols)
SMALLCAP_SAMPLE = [
    # Technology
    'SMCI', 'IONQ', 'RKLB', 'RDFN', 'APPS', 'CWAN', 'ASAN', 'DOCN', 'FRSH', 'BILL',
    'WOLF', 'MSTR', 'RIOT', 'MARA', 'HUT', 'CLSK', 'CIFR', 'BTBT', 'CAN', 'BITF',
    'EXAS', 'NTRA', 'PACB', 'NVTA', 'QDEL', 'GH', 'CDNA', 'TXG', 'MASS', 'PRFT',
    'PLTK', 'QLYS', 'CYBR', 'MIME', 'RPD', 'OSPN', 'ADEA', 'BRZE', 'JAMF', 'SWI',

    # Financials
    'CATY', 'GBCI', 'WTFC', 'UCBI', 'SFBS', 'FFBC', 'FULT', 'TBBK', 'HTLF', 'HWC',
    'BANR', 'WAFD', 'CASH', 'FFNW', 'FBNC', 'SBCF', 'INDB', 'NYCB', 'OZK', 'FBK',
    'PNFP', 'TFSL', 'ABCB', 'CIVB', 'ESSA', 'FMNB', 'BUSE', 'FCNCA', 'LKFN', 'GSBC',
    'PBCT', 'BHLB', 'WSFS', 'SASR', 'TOWN', 'VRTS', 'TFIN', 'FIBK', 'PPBI', 'CATY',

    # Healthcare
    'KRYS', 'TMDX', 'NUVA', 'STAA', 'ONEM', 'DVAX', 'XNCR', 'XENE', 'PCRX', 'RARE',
    'TNDM', 'NARI', 'NVCR', 'OMCL', 'AXNX', 'CYTK', 'INCY', 'SRPT', 'INSM', 'UTHR',
    'HALO', 'VKTX', 'KYMR', 'ADVM', 'ARDX', 'ARVN', 'AVDL', 'AXSM', 'BCRX', 'BGNE',
    'BPMC', 'CARA', 'CLDX', 'CORT', 'DAWN', 'EHTH', 'ELAN', 'ETNB', 'EVLO', 'EXEL',

    # Consumer Discretionary
    'BOOT', 'SHAK', 'TXRH', 'BLMN', 'DNUT', 'BROS', 'CAVA', 'DRI', 'BJRI',
    'YETI', 'GIII', 'SCVL', 'VRA', 'XPEL', 'COLM', 'CTOS', 'PENN', 'BALY', 'GTIM',
    'HELE', 'HIBB', 'KTB', 'LTH', 'MSGS', 'ODP', 'PCVX', 'PLCE', 'PRPL', 'SIG',
    'SKYW', 'SMRT', 'STMP', 'SBH', 'TLRY', 'VIAV', 'VSTO', 'WSBC', 'ZUMZ', 'AAP',

    # Communication Services
    'CARS', 'QUOT', 'YELP', 'TRIP', 'ZD', 'ALLO', 'CCOI', 'SHEN', 'LTRX', 'GOGO',
    'FUBO', 'MSGS', 'GENI', 'DKNG', 'RSI', 'BETZ', 'LGI', 'VZIO', 'MGNI', 'PUBM',

    # Industrials
    'CVCO', 'ASTE', 'PLOW', 'AGCO', 'TEX', 'AL', 'PLUG', 'FCEL', 'BLDP', 'BE',
    'FELE', 'AIT', 'ASTE', 'BOOM', 'BMI', 'CSWI', 'DY', 'ESE', 'FLS', 'FTV',
    'GNRC', 'HI', 'HOV', 'IESC', 'JJSF', 'KAI', 'KBR', 'KFY', 'LSTR', 'MTZ',
    'MTOR', 'NX', 'OSK', 'PRIM', 'RBA', 'RBC', 'RXO', 'SAIA', 'SSD', 'TRS',

    # Consumer Staples
    'CHEF', 'CASY', 'UNFI', 'SPTN', 'GO', 'KLG', 'JJSF', 'CALM', 'FARM', 'APAM',
    'AVO', 'BGS', 'CALM', 'CELH', 'CENTA', 'CENT', 'CHEF', 'CVGW', 'EPC', 'FDP',
    'FIZZ', 'FLO', 'FRPT', 'HAIN', 'INGR', 'IPAR', 'JBSS', 'LANC', 'LWAY', 'MGPI',

    # Energy
    'CTRA', 'SM', 'MGY', 'CRGY', 'RIG', 'VAL', 'WTI', 'MTDR', 'AR', 'TALO',
    'PTEN', 'NBR', 'NINE', 'HP', 'SLCA', 'SDRL', 'TDW', 'WTTR', 'WFRD', 'REI',
    'PUMP', 'LBRT', 'NEXT', 'PVAC', 'PARR', 'DK', 'TALO', 'SD', 'CRK', 'CPG',

    # Materials
    'IOSP', 'HUN', 'OLN', 'FUL', 'CBT', 'SON', 'SLGN', 'NEU', 'KWR', 'HWKN',
    'ASIX', 'ATI', 'BCPC', 'CBT', 'CMC', 'CRS', 'GFF', 'HAYN', 'HXL', 'IOSP',
    'KRA', 'KWR', 'MEC', 'MTX', 'NX', 'OEC', 'SXT', 'SLVM', 'TROX', 'WLK',

    # Real Estate
    'TRNO', 'STAG', 'LXP', 'FCPT', 'ELME', 'DEI', 'AHR', 'SAFE', 'ALEX', 'BNL',
    'BDN', 'BFS', 'BNL', 'BRX', 'BXP', 'CDP', 'CIO', 'CLDT', 'CTO', 'DEA',
    'ESRT', 'FCPT', 'FRT', 'GOOD', 'GPT', 'ILPT', 'JBGS', 'KRC', 'LXP', 'MPW',

    # Utilities
    'MGEE', 'AWR', 'SJW', 'YORW', 'OTTR', 'MSEX', 'NWE', 'AVA', 'BKH', 'NJR',
    'AGR', 'ALE', 'AQN', 'ARLP', 'AVA', 'BKH', 'CWEN', 'CPK', 'CWEN', 'NJR',
    'NWE', 'OGE', 'ORA', 'PNM', 'SJW', 'SR', 'UTL', 'WTRG', 'YORW', 'OTTR'
]


def get_candidate_universe() -> Set[str]:
    """
    Get comprehensive candidate universe from all sources

    Returns:
        Set of unique ticker symbols to screen
    """
    candidates = set()

    # Add all samples
    candidates.update(SP500_SAMPLE)
    candidates.update(MIDCAP_SAMPLE)
    candidates.update(SMALLCAP_SAMPLE)

    # Remove any invalid tickers
    candidates.discard('BRK.B')  # B-shares can cause issues
    candidates.discard('GOOG')  # Duplicate of GOOGL

    return candidates


def get_candidates_by_sector_and_cap(sector: str, market_cap_bucket: str, current_symbols: Set[str]) -> List[str]:
    """
    Get candidate symbols for a specific sector/market cap combination

    Args:
        sector: Target sector
        market_cap_bucket: 'large_cap', 'mid_cap', or 'small_cap'
        current_symbols: Set of symbols already in the list (to exclude)

    Returns:
        List of candidate tickers to evaluate
    """
    all_candidates = get_candidate_universe()

    # Exclude current symbols
    candidates = all_candidates - current_symbols

    # Filter by expected market cap bucket
    if market_cap_bucket == 'large_cap':
        # Prioritize SP500 for large caps
        bucket_candidates = [s for s in candidates if s in SP500_SAMPLE]
    elif market_cap_bucket == 'mid_cap':
        # Prioritize MidCap 400 for mid caps
        bucket_candidates = [s for s in candidates if s in MIDCAP_SAMPLE]
    else:  # small_cap
        # Prioritize SmallCap 600 for small caps
        bucket_candidates = [s for s in candidates if s in SMALLCAP_SAMPLE]

    return bucket_candidates


if __name__ == '__main__':
    # Test candidate generation
    universe = get_candidate_universe()
    print(f"Total candidate universe: {len(universe)} symbols")
    print(f"Large cap pool: {len([s for s in universe if s in SP500_SAMPLE])}")
    print(f"Mid cap pool: {len([s for s in universe if s in MIDCAP_SAMPLE])}")
    print(f"Small cap pool: {len([s for s in universe if s in SMALLCAP_SAMPLE])}")

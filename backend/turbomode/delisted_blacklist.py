"""
Blacklist of delisted/invalid stocks to skip during curation
Auto-generated from scan errors
"""

DELISTED_STOCKS = {
    # From 2026-01-04 scan (updated as scan progresses)
    'X', 'CLR', 'GPS', 'HES', 'BF.B', 'DRE', 'PEAK', 'PXD', 'ACC', 'MRO',
    'ATVI', 'SKX', 'SQ', 'PARA', 'MSG', 'AYX', 'CPE', 'PLAN', 'SJW', 'SAVE',
    'VERV', 'QUOT', 'HA', 'ZI', 'TPX', 'SMAR', 'BLDE', 'SIVB', 'FTCH', 'BLUE',
    'WLL', 'PDCE', 'FL', 'ROIC', 'RDFN', 'HAYN', 'PVAC', 'VSTO', 'SPTN', 'VZIO',
    'STMP', 'SLCA', 'ESSA', 'KLG', 'NUVA', 'NARI', 'PBCT', 'BHLB', 'CPG', 'CARA',
    'ETNB', 'MIME', 'AXNX', 'BPMC', 'AGR', 'HTLF', 'SWI', 'NYCB', 'MTOR', 'PPBI',
    'BGNE', 'KRA', 'CSWI', 'SASR', 'HIBB', 'NVTA', 'PNM', 'PRFT', 'ONEM', 'FFNW'
}

def is_delisted(symbol: str) -> bool:
    """Check if symbol is in the delisted blacklist"""
    return symbol.upper() in DELISTED_STOCKS

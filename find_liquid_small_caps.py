"""
Find 75 liquid small-cap stocks (<$2B market cap, >500K avg volume)
Options trading not required for small caps
"""
import yfinance as yf
import time
import json

# S&P SmallCap 600 constituent candidates (known liquid small caps)
# These are well-known small-cap stocks with institutional following
SMALL_CAP_CANDIDATES = [
    # Regional Banks
    'ABCB', 'BANR', 'BUSE', 'CADE', 'CATY', 'CBSH', 'FFIN', 'FNB', 'FULT',
    'HWC', 'IBOC', 'IBTX', 'MCB', 'NBHC', 'PB', 'PNFP', 'SASR', 'SFNC',
    'SNV', 'SSB', 'TCBI', 'UBSI', 'UCBI', 'WTFC',

    # Industrials
    'ACA', 'ATKR', 'BLDR', 'BXC', 'CEIX', 'CRS', 'CW', 'DY', 'EPC',
    'FELE', 'GATX', 'GVA', 'HI', 'ITGR', 'KAI', 'KBR', 'LSTR', 'MLI',
    'MTSI', 'NJR', 'NVST', 'PATK', 'PRIM', 'RBC', 'RUSHA', 'SSD', 'TRN',
    'TTEK', 'UNFI', 'WERN', 'WTS',

    # Technology
    'ARLO', 'AVAV', 'CALX', 'CIEN', 'COHU', 'DOMO', 'FORM', 'INFA', 'IRBT',
    'LSCC', 'LUMN', 'LVGO', 'MCHP', 'PSTG', 'PYCR', 'QLYS', 'RDNT', 'RIOT',
    'SWKS', 'SYNH', 'VZIO', 'WDC', 'WOLF',

    # Healthcare
    'ACHC', 'ADMA', 'AMED', 'ATRC', 'BCRX', 'BKD', 'BTAI', 'CORT', 'CPRX',
    'CRNX', 'CRVL', 'DNLI', 'ENSG', 'ETNB', 'GKOS', 'GMED', 'HALO', 'HIMS',
    'ICUI', 'IRTC', 'KRYS', 'LMAT', 'LPLA', 'MMSI', 'NEOG', 'NHC', 'OMI',
    'PDCO', 'PNTG', 'PTGX', 'RDUS', 'RVMD', 'SGRY', 'TNDM', 'TVTX', 'UHS',
    'USPH', 'XRAY',

    # Consumer Discretionary
    'AAP', 'ABG', 'AEO', 'ATGE', 'BIGG', 'BKE', 'BOOT', 'BURL', 'CAKE',
    'CCS', 'CHDN', 'CRI', 'CROX', 'CVNA', 'DBI', 'DBRG', 'DDS', 'FIVE',
    'FL', 'GCO', 'GPI', 'GRMN', 'GTLS', 'HBI', 'HIBB', 'HZO', 'JOUT',
    'KSS', 'KTB', 'LE', 'LESL', 'LEVI', 'M', 'MODG', 'OLLI', 'OXM',
    'PATK', 'PLAY', 'PLCE', 'PRKS', 'RVLV', 'SAH', 'SHAK', 'SKX', 'VSCO',
    'WINA', 'WOR', 'WSM', 'ZUMZ',

    # Consumer Staples
    'CALM', 'CENTA', 'ELF', 'GO', 'HAIN', 'JJSF', 'LANC', 'LWAY', 'MGPI',
    'SAM', 'SENEA', 'SMPL', 'SPB', 'TAP', 'UNFI', 'USFD', 'WMK',

    # Energy
    'APA', 'BTU', 'CHX', 'CRK', 'CVI', 'DK', 'HLX', 'LPI', 'MUR', 'NOG',
    'OII', 'PBF', 'PTEN', 'REI', 'TALO', 'VAL', 'VTLE', 'WTI',

    # Materials
    'AMBP', 'ARCH', 'ASH', 'ATI', 'CBT', 'CMP', 'GEF', 'HWKN', 'KWR',
    'MP', 'NEU', 'NX', 'RMBS', 'SCCO', 'SLVM', 'USLM', 'WLK',

    # Utilities
    'ALE', 'MGEE', 'NWN', 'SJW', 'SR', 'UTL',

    # Real Estate
    'AHR', 'ALEX', 'BNL', 'BRX', 'BXP', 'CTRE', 'CUZ', 'DEI', 'EPRT',
    'ESRT', 'GNL', 'HR', 'IRT', 'KRC', 'KRG', 'LXP', 'MAC', 'MPW',
    'NXRT', 'OUT', 'PDM', 'PIR', 'ROIC', 'SAFE', 'SBRA', 'SLG', 'UNIT',
    'UBA', 'VNO', 'WPC',

    # Communications
    'AMC', 'CCOI', 'CNK', 'DISH', 'GOGO', 'GTN', 'IMAX', 'LBRDA', 'NXST',
    'TRIP', 'VMEO', 'WLY',
]


def validate_small_cap(symbol):
    """Validate a small-cap candidate"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        market_cap = info.get('marketCap', 0)
        avg_volume = info.get('averageVolume', 0)

        # Must be < $2B market cap and > 500K volume
        if market_cap and market_cap < 2_000_000_000 and avg_volume > 500_000:
            return {
                'symbol': symbol,
                'market_cap': market_cap,
                'market_cap_billions': round(market_cap / 1_000_000_000, 3),
                'avg_volume': avg_volume,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'valid': True
            }
        else:
            return {'symbol': symbol, 'valid': False, 'reason': f'Cap: ${market_cap/1e9:.2f}B, Vol: {avg_volume:,}'}

    except Exception as e:
        return {'symbol': symbol, 'valid': False, 'error': str(e)}


def find_small_caps():
    """Find 75+ liquid small caps"""
    print("=" * 80)
    print("SEARCHING FOR LIQUID SMALL-CAP STOCKS")
    print("=" * 80)
    print(f"Criteria: Market cap < $2B, Avg volume > 500K")
    print(f"Searching {len(SMALL_CAP_CANDIDATES)} candidates...")
    print()

    valid_small_caps = []
    invalid = []

    for i, symbol in enumerate(SMALL_CAP_CANDIDATES, 1):
        result = validate_small_cap(symbol)

        if result.get('valid'):
            valid_small_caps.append(result)
        else:
            invalid.append(result)

        if i % 20 == 0:
            print(f"Progress: {i}/{len(SMALL_CAP_CANDIDATES)} ({i/len(SMALL_CAP_CANDIDATES)*100:.1f}%) - Found: {len(valid_small_caps)}")

        time.sleep(0.3)  # Rate limiting

    # Sort by market cap (largest first for liquidity)
    valid_small_caps.sort(key=lambda x: x['market_cap'], reverse=True)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Valid small caps found: {len(valid_small_caps)}")
    print(f"Need: 75")
    print()

    if len(valid_small_caps) >= 75:
        print("SUCCESS! Found enough liquid small caps")
        print()
        print("Top 75 by market cap:")
        for i, stock in enumerate(valid_small_caps[:75], 1):
            print(f"{i:3d}. {stock['symbol']:6s} ${stock['market_cap_billions']:.3f}B  Vol: {stock['avg_volume']:>10,}  {stock['sector']}")
    else:
        print(f"WARNING: Only found {len(valid_small_caps)}, need {75 - len(valid_small_caps)} more")
        print()
        print("All valid small caps:")
        for i, stock in enumerate(valid_small_caps, 1):
            print(f"{i:3d}. {stock['symbol']:6s} ${stock['market_cap_billions']:.3f}B  Vol: {stock['avg_volume']:>10,}  {stock['sector']}")

    # Save results
    with open('liquid_small_caps.json', 'w') as f:
        json.dump(valid_small_caps, f, indent=2)

    print()
    print(f"Results saved to: liquid_small_caps.json")

    return valid_small_caps


if __name__ == '__main__':
    find_small_caps()

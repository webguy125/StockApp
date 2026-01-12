"""
Check which stocks in blacklist were actually trading on Friday Jan 2, 2026
"""
import yfinance as yf

# All stocks from blacklist
blacklist = [
    'X', 'CLR', 'GPS', 'HES', 'BF.B', 'DRE', 'PEAK', 'PXD', 'ACC', 'MRO',
    'ATVI', 'SKX', 'SQ', 'PARA', 'MSG', 'AYX', 'CPE', 'PLAN', 'SJW', 'SAVE',
    'VERV', 'QUOT', 'HA', 'ZI', 'TPX', 'SMAR', 'BLDE', 'SIVB', 'FTCH', 'BLUE',
    'WLL', 'PDCE', 'FL', 'ROIC', 'RDFN', 'HAYN', 'PVAC', 'VSTO', 'SPTN', 'VZIO',
    'STMP', 'SLCA', 'ESSA', 'KLG', 'NUVA', 'NARI', 'PBCT', 'BHLB', 'CPG', 'CARA',
    'ETNB', 'MIME', 'AXNX', 'BPMC', 'AGR', 'HTLF', 'SWI', 'NYCB', 'MTOR', 'PPBI',
    'BGNE', 'KRA', 'CSWI', 'SASR', 'HIBB', 'NVTA', 'PNM', 'PRFT', 'ONEM', 'FFNW'
]

print('Checking blacklist stocks for trading on Jan 2, 2026:\n')

active_stocks = []
delisted_stocks = []

for symbol in blacklist:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start='2026-01-02', end='2026-01-03')

        if len(hist) > 0:
            price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            active_stocks.append((symbol, price, volume))
            print(f'{symbol}: ACTIVE - ${price:.2f}, Volume: {int(volume):,}')
        else:
            delisted_stocks.append(symbol)
    except:
        delisted_stocks.append(symbol)

print(f'\n{"="*60}')
print(f'SUMMARY:')
print(f'{"="*60}')
print(f'Active stocks (should NOT be in blacklist): {len(active_stocks)}')
print(f'Delisted stocks (correctly in blacklist): {len(delisted_stocks)}')

if active_stocks:
    print(f'\nACTIVE STOCKS TO REMOVE FROM BLACKLIST:')
    for symbol, price, volume in active_stocks:
        print(f'  {symbol}')

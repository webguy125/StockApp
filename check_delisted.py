"""
Quick script to verify which stocks are actually delisted vs just data errors
"""
import yfinance as yf

# Test symbols from the blacklist
test_symbols = ['X', 'GPS', 'MRO', 'SKX', 'SQ', 'FL', 'HES', 'PXD', 'CLR']

print('Checking if stocks are actually delisted or just data issues:\n')
print(f"{'Symbol':<8} {'Status':<15} {'Price':<10} {'Volume':<15}")
print('-' * 60)

for symbol in test_symbols:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')

        if len(hist) > 0:
            price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            print(f"{symbol:<8} {'ACTIVE':<15} ${price:<9.2f} {volume:>14,}")
        else:
            print(f"{symbol:<8} {'DELISTED':<15} {'N/A':<10} {'N/A':<15}")
    except Exception as e:
        print(f"{symbol:<8} {'ERROR':<15} {str(e)[:30]:<10}")

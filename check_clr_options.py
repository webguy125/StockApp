"""Check CLR for options availability"""
import yfinance as yf

ticker = yf.Ticker('CLR')
print('CLR Stock Info:')
print('=' * 60)

info = ticker.info
print(f'Company: {info.get("longName", "N/A")}')
print(f'Price: ${info.get("currentPrice", info.get("regularMarketPrice", "N/A"))}')
print(f'Exchange: {info.get("exchange", "N/A")}')
print(f'Market Cap: ${info.get("marketCap", 0)/1e9:.2f}B')

print(f'\nOptions Availability:')
print(f'Has options: {len(ticker.options) > 0}')
print(f'Options expirations: {len(ticker.options)}')
if len(ticker.options) > 0:
    print(f'First 5 expirations: {ticker.options[:5]}')
else:
    print('NO OPTIONS AVAILABLE - This is why it appears "delisted"!')

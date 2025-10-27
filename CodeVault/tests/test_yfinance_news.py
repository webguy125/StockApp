import yfinance as yf
import json
from datetime import datetime

# Test market news
print("=" * 80)
print("Testing Market News (S&P 500)")
print("=" * 80)
ticker_market = yf.Ticker("^GSPC")
market_news = ticker_market.news if hasattr(ticker_market, 'news') else []

print(f"\nFound {len(market_news)} market news items")
if len(market_news) > 0:
    print("\nFirst news item full structure:")
    print(json.dumps(market_news[0], indent=2, default=str))

    print("\nAll available keys:")
    print(list(market_news[0].keys()))

# Test symbol-specific news
print("\n" + "=" * 80)
print("Testing Symbol News (AAPL)")
print("=" * 80)
ticker_symbol = yf.Ticker("AAPL")
symbol_news = ticker_symbol.news if hasattr(ticker_symbol, 'news') else []

print(f"\nFound {len(symbol_news)} AAPL news items")
if len(symbol_news) > 0:
    print("\nFirst news item full structure:")
    print(json.dumps(symbol_news[0], indent=2, default=str))

    print("\nAll available keys:")
    print(list(symbol_news[0].keys()))

    # Check each field
    print("\nField values:")
    for key in symbol_news[0].keys():
        value = symbol_news[0][key]
        print(f"  {key}: {repr(value)[:100]}")

# Test BTC-USD as shown in console
print("\n" + "=" * 80)
print("Testing Symbol News (BTC-USD)")
print("=" * 80)
ticker_btc = yf.Ticker("BTC-USD")
btc_news = ticker_btc.news if hasattr(ticker_btc, 'news') else []

print(f"\nFound {len(btc_news)} BTC-USD news items")
if len(btc_news) > 0:
    print("\nFirst news item full structure:")
    print(json.dumps(btc_news[0], indent=2, default=str))

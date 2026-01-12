"""
Test what fundamental data is available from yfinance
"""

import yfinance as yf
import time

def test_fundamentals(symbol):
    """Test what fundamental data we can get from yfinance"""
    print(f"\n{'='*70}")
    print(f"Testing fundamentals for {symbol}")
    print(f"{'='*70}")

    start = time.time()
    ticker = yf.Ticker(symbol)

    # Get info (this is the main fundamental data source)
    info = ticker.info

    print(f"\nData fetch time: {time.time() - start:.2f}s")
    print(f"\nTotal info fields: {len(info)}")

    # Key fundamental metrics we care about
    fundamental_fields = {
        # Valuation
        'marketCap': 'Market Cap',
        'enterpriseValue': 'Enterprise Value',
        'trailingPE': 'P/E Ratio (TTM)',
        'forwardPE': 'Forward P/E',
        'priceToBook': 'Price/Book',
        'priceToSalesTrailing12Months': 'Price/Sales',

        # Profitability
        'profitMargins': 'Profit Margin',
        'operatingMargins': 'Operating Margin',
        'returnOnAssets': 'ROA',
        'returnOnEquity': 'ROE',

        # Growth
        'revenueGrowth': 'Revenue Growth',
        'earningsGrowth': 'Earnings Growth',

        # Financial Health
        'totalCash': 'Total Cash',
        'totalDebt': 'Total Debt',
        'debtToEquity': 'Debt/Equity',
        'currentRatio': 'Current Ratio',
        'quickRatio': 'Quick Ratio',

        # Dividend
        'dividendYield': 'Dividend Yield',

        # Volume/Liquidity
        'averageVolume': 'Avg Volume',
        'averageVolume10days': 'Avg Volume (10d)',
        'floatShares': 'Float Shares',
        'sharesOutstanding': 'Shares Outstanding',
        'sharesShort': 'Shares Short',
        'shortRatio': 'Short Ratio',
        'shortPercentOfFloat': 'Short % of Float',

        # Price metrics
        'beta': 'Beta',
        'fiftyTwoWeekHigh': '52-week High',
        'fiftyTwoWeekLow': '52-week Low',

        # Analyst data
        'targetMeanPrice': 'Analyst Target Price',
        'recommendationKey': 'Analyst Recommendation',
        'numberOfAnalystOpinions': 'Number of Analysts',
    }

    print(f"\n{'Field':<35} {'Value':<20} {'Available'}")
    print("-" * 70)

    available = []
    missing = []

    for field, label in fundamental_fields.items():
        value = info.get(field)
        if value is not None:
            available.append(field)
            print(f"{label:<35} {str(value)[:20]:<20} [YES]")
        else:
            missing.append(field)
            print(f"{label:<35} {'N/A':<20} [NO]")

    print(f"\n{'='*70}")
    print(f"Summary: {len(available)}/{len(fundamental_fields)} fields available")
    print(f"{'='*70}")

    return available, missing, info


# Test multiple stocks
test_symbols = ['AAPL', 'EXAS', 'TMDX', 'NVDA', 'TSLA']

all_results = {}
for symbol in test_symbols:
    available, missing, info = test_fundamentals(symbol)
    all_results[symbol] = {'available': available, 'missing': missing}
    time.sleep(0.5)  # Rate limit

# Find fields that are available for ALL stocks
print(f"\n\n{'='*70}")
print("FIELDS AVAILABLE FOR ALL TESTED STOCKS")
print(f"{'='*70}")

common_fields = set(all_results[test_symbols[0]]['available'])
for symbol in test_symbols[1:]:
    common_fields &= set(all_results[symbol]['available'])

print(f"\nFound {len(common_fields)} fields available for all stocks:")
for field in sorted(common_fields):
    print(f"  - {field}")

print(f"\n\n{'='*70}")
print("RECOMMENDED FEATURES TO ADD (Fast, Reliable, Relevant)")
print(f"{'='*70}")

recommended = {
    # Valuation (4 features)
    'trailingPE': 'Helps detect overvalued/undervalued',
    'priceToBook': 'Value metric',
    'priceToSalesTrailing12Months': 'Sales multiple',
    'marketCap': 'Already have tier, but actual value useful',

    # Profitability (2 features)
    'profitMargins': 'Profit efficiency',
    'returnOnEquity': 'Shareholder returns',

    # Financial Health (2 features)
    'debtToEquity': 'Leverage/risk',
    'currentRatio': 'Short-term liquidity',

    # Growth (2 features)
    'revenueGrowth': 'Top-line growth',
    'earningsGrowth': 'Bottom-line growth',

    # Short Interest (2 features)
    'shortPercentOfFloat': 'Short squeeze potential',
    'shortRatio': 'Days to cover',

    # Analyst (1 feature)
    'targetMeanPrice': 'Can compute % to target',

    # Volatility (1 feature)
    'beta': 'Market correlation/volatility'
}

print("\nTotal: 14 fundamental features")
print()
for field, reason in recommended.items():
    available_for = sum(1 for s in test_symbols if field in all_results[s]['available'])
    status = '[OK]' if available_for == len(test_symbols) else f'[MISSING] ({available_for}/{len(test_symbols)})'
    print(f"{status} {field:<30} - {reason}")

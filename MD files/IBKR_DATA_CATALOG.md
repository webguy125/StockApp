# IBKR API Data Catalog
**Complete Overview of Available Data**

---

## 1. Real-Time Market Data (Level I)

### Stock Quotes
**Endpoint**: `ib.reqMktData(contract)`
**Rate Limit**: 50 requests/second, 100 simultaneous subscriptions

**Data Fields Available:**
```python
{
    # Price Data
    'bid': 150.25,              # Current bid price
    'ask': 150.27,              # Current ask price
    'last': 150.26,             # Last trade price
    'close': 149.80,            # Previous close
    'open': 149.90,             # Today's open
    'high': 151.20,             # Today's high
    'low': 149.50,              # Today's low

    # Volume Data
    'volume': 45230000,         # Total volume today
    'bidSize': 500,             # Shares at bid
    'askSize': 800,             # Shares at ask
    'lastSize': 100,            # Size of last trade

    # Advanced Metrics
    'halted': False,            # Trading halted?
    'avgVolume': 52000000,      # Average daily volume (3mo)
    'vwap': 150.15,             # Volume-weighted average price
    'shortableShares': 1500000, # Shares available to short
    'shortingFeeRate': 0.0025,  # Annual fee to short (0.25%)

    # Options-Specific (for option contracts)
    'impliedVol': 0.285,        # Implied volatility
    'delta': 0.55,              # Delta
    'gamma': 0.025,             # Gamma
    'theta': -0.12,             # Theta (daily decay)
    'vega': 0.18,               # Vega (IV sensitivity)
    'optPrice': 5.50,           # Option theoretical price
    'pvDividend': 0.25,         # Present value of dividends
    'undPrice': 150.26          # Underlying stock price
}
```

**Latency**:
- Real-time: <100ms (requires $10-15/mo subscription)
- Delayed: 15 minutes (FREE)

---

## 2. Level II Market Data (Market Depth)

### Order Book / Market Depth
**Endpoint**: `ib.reqMktDepth(contract)`
**Rate Limit**: 3 simultaneous subscriptions

**Data Structure:**
```python
{
    'bids': [
        {'price': 150.25, 'size': 500, 'marketMaker': 'NSDQ'},
        {'price': 150.24, 'size': 1200, 'marketMaker': 'ARCA'},
        {'price': 150.23, 'size': 800, 'marketMaker': 'BATS'},
        {'price': 150.22, 'size': 2000, 'marketMaker': 'EDGX'},
        {'price': 150.21, 'size': 600, 'marketMaker': 'IEX'},
        # ... up to 10 levels deep
    ],
    'asks': [
        {'price': 150.27, 'size': 800, 'marketMaker': 'NSDQ'},
        {'price': 150.28, 'size': 1500, 'marketMaker': 'ARCA'},
        {'price': 150.29, 'size': 400, 'marketMaker': 'BATS'},
        # ... up to 10 levels deep
    ]
}
```

**Use Cases:**
- ✅ Verify actual liquidity at bid/ask levels
- ✅ Detect hidden orders or iceberg orders
- ✅ Analyze market maker behavior
- ✅ Optimize order placement (TurboOptions entry/exit)

---

## 3. Options Chains

### Complete Options Data
**Endpoints**:
1. `ib.reqSecDefOptParams()` - Get available strikes/expirations
2. `ib.qualifyContracts()` - Get full contract details
3. `ib.reqMktData()` - Get live quotes for specific contracts

**Available Expirations**: Weekly, monthly, quarterly (LEAPS)

**Per-Contract Data:**
```python
{
    # Contract Identification
    'symbol': 'AAPL',
    'strike': 150.0,
    'right': 'C',               # 'C' = call, 'P' = put
    'expiry': '20260117',       # Expiration date
    'exchange': 'SMART',        # Best execution venue
    'multiplier': '100',        # Contract size

    # Pricing
    'bid': 5.45,
    'ask': 5.50,
    'last': 5.48,
    'close': 5.20,              # Previous close
    'bidSize': 50,              # Contracts at bid
    'askSize': 75,              # Contracts at ask

    # Greeks (server-calculated)
    'delta': 0.5524,
    'gamma': 0.0247,
    'theta': -0.1185,           # Daily decay in dollars
    'vega': 0.1832,             # $change per 1% IV move
    'impliedVol': 0.2847,       # Implied volatility

    # Volume & Open Interest
    'volume': 12500,            # Contracts traded today
    'openInterest': 45230,      # Open contracts
    'avgVolume': 8500,          # 30-day average

    # Advanced Analytics
    'optPrice': 5.47,           # Theoretical fair value
    'undPrice': 150.26,         # Underlying stock price
    'pvDividend': 0.25,         # Dividend adjustment
    'histVol': 0.3120,          # Historical volatility (30d)
    'ivRank': 0.65,             # IV percentile (0-1)

    # Model-Specific
    'modelGreeks': {            # From IBKR's pricing model
        'delta': 0.5524,
        'gamma': 0.0247,
        'theta': -0.1185,
        'vega': 0.1832,
        'rho': 0.0432
    }
}
```

**Spread Calculation:**
```python
spread = ask - bid  # Absolute spread
spread_pct = (ask - bid) / ((ask + bid) / 2)  # Percentage spread
```

---

## 4. Historical Data

### Price History
**Endpoint**: `ib.reqHistoricalData()`
**Rate Limit**: 60 requests per 10 minutes (6/min sustained)

**Bar Sizes Available:**
- Intraday: 1 sec, 5 sec, 10 sec, 15 sec, 30 sec
- Minutes: 1 min, 2 min, 3 min, 5 min, 10 min, 15 min, 20 min, 30 min
- Hours: 1 hour, 2 hours, 3 hours, 4 hours, 8 hours
- Daily: 1 day
- Weekly: 1 week
- Monthly: 1 month

**Max Duration Per Request:**
| Bar Size | Max Duration |
|----------|--------------|
| 1 sec | 2000 seconds |
| 5 sec | 10,000 seconds |
| 1 min | 1 day |
| 5 min | 1 week |
| 1 hour | 1 month |
| 1 day | 1 year |
| 1 week | 5 years |

**Data Per Bar:**
```python
{
    'date': '2026-01-03 09:30:00',
    'open': 150.10,
    'high': 150.45,
    'low': 150.05,
    'close': 150.26,
    'volume': 1250000,
    'average': 150.22,      # VWAP for the bar
    'barCount': 3250        # Number of trades in bar
}
```

**Historical Options Data:**
```python
# Same bar structure, plus:
{
    'impliedVol': 0.285,    # IV at bar close
    'openInterest': 45230   # OI at bar close (end of day only)
}
```

---

## 5. Fundamental Data

### Company Fundamentals
**Endpoint**: `ib.reqFundamentalData(contract, reportType)`
**Rate Limit**: 50 requests/second

**Report Types Available:**

#### A. Company Overview (`ReportSnapshot`)
```python
{
    'companyName': 'Apple Inc.',
    'symbol': 'AAPL',
    'exchange': 'NASDAQ',
    'currency': 'USD',
    'sector': 'Technology',
    'industry': 'Consumer Electronics',
    'marketCap': 4021900000000,      # $4.02T
    'sharesOutstanding': 15204100000,
    'floatShares': 15100000000,
    'beta': 1.25,
    'peRatio': 35.8,
    'eps': 4.20,
    'dividendYield': 0.0045,         # 0.45%
    'exDividendDate': '2026-02-10',
    'dividendPayDate': '2026-02-17',
    'fiscalYearEnd': 'September',
    'description': 'Company description text...',
    'website': 'https://www.apple.com',
    'employees': 161000,
    'founded': 1976,
    'headquarters': 'Cupertino, California'
}
```

#### B. Financial Statements (`ReportsFinStatements`)
```python
{
    'income_statement': {
        'quarterly': [
            {
                'period': '2025-Q4',
                'revenue': 117154000000,
                'costOfRevenue': 66822000000,
                'grossProfit': 50332000000,
                'operatingExpenses': 14009000000,
                'operatingIncome': 36323000000,
                'netIncome': 30000000000,
                'eps': 1.97,
                'ebitda': 38500000000
            },
            # ... previous quarters
        ],
        'annual': [
            {
                'fiscalYear': 2025,
                'revenue': 394328000000,
                'netIncome': 99803000000,
                'eps': 6.56
            }
        ]
    },
    'balance_sheet': {
        'totalAssets': 352755000000,
        'totalLiabilities': 290020000000,
        'totalEquity': 62735000000,
        'cash': 61555000000,
        'totalDebt': 106621000000,
        'currentRatio': 0.98,
        'debtToEquity': 1.70
    },
    'cash_flow': {
        'operatingCashFlow': 110543000000,
        'capitalExpenditures': -10707000000,
        'freeCashFlow': 99836000000,
        'dividendsPaid': -14841000000,
        'stockRepurchases': -77550000000
    }
}
```

#### C. Analyst Estimates (`ReportsFinSummary`)
```python
{
    'consensusRatings': {
        'buy': 25,
        'hold': 10,
        'sell': 2,
        'meanRating': 'Strong Buy',
        'priceTarget': 185.50,
        'priceTargetHigh': 220.00,
        'priceTargetLow': 140.00
    },
    'earningsEstimates': {
        'nextQuarter': {
            'eps': 2.15,
            'revenue': 123000000000,
            'date': '2026-01-30'
        },
        'currentYear': {
            'eps': 8.45,
            'revenue': 450000000000
        }
    },
    'growthEstimates': {
        'epsGrowthNextYear': 0.085,     # 8.5%
        'revenueGrowth5Year': 0.072     # 7.2%
    }
}
```

#### D. Ownership Data (`ReportsOwnership`)
```python
{
    'institutionalOwnership': {
        'percentage': 0.6245,           # 62.45% institutional
        'shares': 9490000000,
        'topHolders': [
            {
                'name': 'Vanguard Group Inc',
                'shares': 1298000000,
                'percentage': 0.0854,   # 8.54%
                'value': 195000000000,
                'reportDate': '2025-12-31'
            },
            {
                'name': 'BlackRock Inc',
                'shares': 1045000000,
                'percentage': 0.0687
            },
            # ... top 10 holders
        ]
    },
    'insiderOwnership': {
        'percentage': 0.0007,           # 0.07%
        'shares': 10640000,
        'recentTransactions': [
            {
                'name': 'Tim Cook',
                'title': 'CEO',
                'transactionType': 'Sale',
                'shares': 5000,
                'price': 150.25,
                'date': '2026-01-02'
            }
        ]
    },
    'shortInterest': {
        'shares': 105000000,
        'percentage': 0.0069,           # 0.69% of float
        'ratio': 2.1,                   # Days to cover
        'previousShortInterest': 98000000,
        'reportDate': '2025-12-31'
    }
}
```

---

## 6. Scanner Data

### Real-Time Market Scanners
**Endpoint**: `ib.reqScannerSubscription()`
**Rate Limit**: 50 simultaneous scans

**Pre-Built Scans:**
- Most active by volume
- Top % gainers
- Top % losers
- Most volatile
- Hot by volume
- Hot by price
- Top option implied volatility
- Highest option volume

**Custom Scan Parameters:**
```python
scan = ScannerSubscription(
    instrument='STK',           # Stocks
    locationCode='STK.US.MAJOR',  # US exchanges
    scanCode='TOP_PERC_GAIN',   # Type of scan
    abovePrice=10.0,            # Min price
    belowPrice=500.0,           # Max price
    aboveVolume=1000000,        # Min volume
    marketCapAbove=2000000000,  # Min market cap ($2B)
    stockTypeFilter='ALL'       # ALL, CORP, ADR, ETF, REIT
)

# Returns top 50 results matching criteria
```

---

## 7. News & Events

### News Feed
**Endpoint**: `ib.reqNewsProviders()`, `ib.reqHistoricalNews()`

**Available News Sources:**
- Dow Jones Newswires
- Reuters
- PR Newswire
- BusinessWire
- MT Newswires

**News Article Data:**
```python
{
    'articleId': '1234567890',
    'headline': 'Apple announces new product',
    'provider': 'DJ-N',         # Dow Jones
    'sentimentScore': 0.75,     # 0-1 (positive sentiment)
    'time': '2026-01-03T10:30:00Z',
    'categories': ['Earnings', 'Technology'],
    'symbols': ['AAPL', 'MSFT'],
    'articleUrl': 'https://...',
    'body': 'Full article text...'
}
```

### Corporate Events
**Endpoint**: Included in fundamental data

**Event Types:**
- Earnings announcements (dates, estimates)
- Dividend ex-dates and pay dates
- Stock splits
- Merger/acquisition announcements
- Conference calls

---

## 8. Options-Specific Advanced Data

### Implied Volatility Surface
**Endpoint**: Custom calculation from options chain

**Available Data:**
```python
{
    'symbol': 'AAPL',
    'asOfDate': '2026-01-03',
    'surface': [
        {
            'strike': 140,
            'dte': 14,              # Days to expiration
            'callIV': 0.285,
            'putIV': 0.290,
            'callVolume': 5000,
            'putVolume': 7500,
            'callOI': 25000,
            'putOI': 32000
        },
        # ... all strikes × all expirations
    ],
    'ivSkew': {
        'atm': 0.285,
        'otm25Delta': 0.310,    # 25-delta OTM skew
        'itm25Delta': 0.265     # 25-delta ITM skew
    }
}
```

### Historical Implied Volatility
**Endpoint**: `ib.reqHistoricalData()` on option contracts

**Time Series Data:**
```python
{
    'symbol': 'AAPL',
    'strike': 150,
    'expiry': '2026-01-17',
    'history': [
        {
            'date': '2026-01-02',
            'iv': 0.2850,
            'stockPrice': 149.80,
            'optionPrice': 5.20,
            'volume': 8500,
            'openInterest': 45000
        },
        # ... historical daily data
    ]
}
```

### Greeks History
**Available**: Delta, gamma, theta, vega, rho historical values

---

## 9. What's NOT Available (vs Bloomberg Terminal)

**Limited/Missing Data:**
- ❌ Credit default swap (CDS) spreads
- ❌ Institutional-grade fixed income pricing
- ❌ Proprietary analyst research (only aggregated ratings)
- ❌ Full audit trail / trade reconstruction
- ❌ Some exotic derivatives pricing
- ⚠️ Some international markets have limited data

**But we DO have:**
- ✅ Everything needed for stock screening
- ✅ Professional-grade options data
- ✅ Comprehensive fundamentals
- ✅ Real-time and historical data
- ✅ Level 2 market depth
- ✅ Sufficient for 99% of retail/small fund use cases

---

## 10. Data for TurboMode Curation

### What We Need → What IBKR Provides

| Requirement | IBKR Endpoint | Quality |
|-------------|---------------|---------|
| **90-day avg volume** | Historical bars (1 day) | ✅ Exact |
| **Market cap** | Fundamental data | ✅ Real-time |
| **Sector classification** | Fundamental data | ✅ GICS standard |
| **ATM options spread** | Options chain bid/ask | ✅ Real-time |
| **Open interest** | Options chain | ✅ Daily updates |
| **Number of expirations** | reqSecDefOptParams() | ✅ Complete list |
| **Options volume** | Options chain | ✅ Real-time |
| **Bid/ask sizes** | Market depth | ✅ Level II |

**Everything we need for strict curation criteria is available!**

---

## 11. Data for TurboOptions Scoring

### Current vs IBKR Comparison

| Metric | Current (yfinance) | IBKR |
|--------|-------------------|------|
| **Live bid/ask** | 15-min delayed, often stale | Real-time or 15-min delayed |
| **Greeks** | We calculate manually | Server-calculated (more accurate) |
| **IV** | Manual calc from price | Direct from exchange |
| **Spread quality** | Unreliable | Accurate to the penny |
| **OI/Volume** | Daily snapshots | Real-time updates |
| **Historical IV** | Not available | Full history |
| **Level 2 depth** | Not available | 10-level order book |

**Verdict: IBKR provides everything we currently use, plus advanced data we can't get from yfinance**

---

## 12. Practical Example: Full Data for One Symbol

### Request: `AAPL` Complete Dataset

```python
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# 1. Basic quote
stock = Stock('AAPL', 'SMART', 'USD')
ticker = ib.reqMktData(stock)
ib.sleep(2)

print(f"Price: {ticker.last}, Bid: {ticker.bid}, Ask: {ticker.ask}")
print(f"Volume: {ticker.volume}, AvgVol: {ticker.avgVolume}")

# 2. Level 2 depth
depth = ib.reqMktDepth(stock)
ib.sleep(2)
print(f"Top bid: {depth.bids[0]}, Top ask: {depth.asks[0]}")

# 3. Historical data (90 days)
bars = ib.reqHistoricalData(
    stock,
    endDateTime='',
    durationStr='90 D',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True
)
print(f"Got {len(bars)} daily bars")

# 4. Fundamentals
fundamentals = ib.reqFundamentalData(stock, 'ReportSnapshot')
# Returns XML, parse for market cap, sector, etc.

# 5. Options chain
chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
strikes = chains[0].strikes
expirations = chains[0].expirations
print(f"Available: {len(expirations)} expirations, {len(strikes)} strikes")

# 6. Specific option contract
option = Option('AAPL', '20260117', 150, 'C', 'SMART')
ib.qualifyContracts(option)
opt_ticker = ib.reqMktData(option, '', False, False)
ib.sleep(2)

print(f"Call bid: {opt_ticker.bid}, ask: {opt_ticker.ask}")
print(f"Delta: {opt_ticker.modelGreeks.delta}")
print(f"IV: {opt_ticker.impliedVol}")
print(f"OI: {opt_ticker.openInterest}, Vol: {opt_ticker.volume}")
```

**Output Example:**
```
Price: 150.26, Bid: 150.25, Ask: 150.27
Volume: 45230000, AvgVol: 52000000
Top bid: {'price': 150.25, 'size': 500}, Top ask: {'price': 150.27, 'size': 800}
Got 90 daily bars
Available: 52 expirations, 145 strikes
Call bid: 5.45, ask: 5.50
Delta: 0.5524
IV: 0.2847
OI: 45230, Vol: 12500
```

---

## Summary: IBKR Data vs Our Needs

### ✅ Perfect Match:
- Real-time quotes (free 15-min delayed, or $10/mo real-time)
- Options chains with accurate spreads
- Historical data (90-day for volume calc)
- Fundamentals (market cap, sector)
- Greeks (server-calculated)

### ✅ Bonus Data (not currently using, but valuable):
- Level 2 depth (verify actual liquidity)
- Historical IV (better regime analysis)
- Analyst estimates (earnings predictions)
- Ownership data (institutional %)
- News feed (sentiment analysis)

### ⚠️ Rate Limit to Plan Around:
- Historical data: 6 requests/min (but can get 1 year per request)
- Solution: Batch requests, request longer periods

### 💰 Cost:
- **Free**: Everything except real-time quotes
- **$10-15/mo**: Real-time quotes (optional)

**Verdict: IBKR has all the data we need, plus way more than we're currently using from yfinance.**

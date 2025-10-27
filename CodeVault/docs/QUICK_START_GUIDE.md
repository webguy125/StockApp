# ThinkorSwim-Style Platform - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Start the Server
```bash
# From the StockApp directory
python app.py
```

### Step 2: Open the Interface
Navigate to:
```
http://localhost:8080/index_tos_style.html
```

### Step 3: Start Trading!
- Type a symbol (e.g., AAPL) in the watchlist search
- Press Enter to add to your list
- Click the symbol to load the chart

## 📊 Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ File | View | Tools | Analysis | Trade | Help      [Menu Bar]   │
├──────────┬────────────────────────────────────────┬──────────────┤
│ WATCHLIST│           CHART AREA                   │ ACTIVE TRADER│
│          │ ┌──────────────────────────────────┐   │              │
│ [Search] │ │Symbol [AAPL] [1D][5D][1M]...[1Y]│   │ Quick Order  │
│ ─────────│ └──────────────────────────────────┘   │ ────────────│
│ My List  │                                        │ Symbol: AAPL │
│ ────     │                                        │ Qty: [100]   │
│ AAPL ↑   │         📈 CANDLESTICK CHART          │ Type: Market │
│ MSFT ↑   │                                        │              │
│ GOOGL ↓  │                                        │ [BUY] [SELL] │
│ TSLA ↑   │                                        │              │
│          │                                        │ Positions    │
│ ─────────│                                        │ ────────────│
│ NEWS     │                                        │ AAPL  +$245 │
│ ────     │                                        │ MSFT  -$130 │
│ • Market │                                        │              │
│   Update │                                        │ Account      │
│ • AAPL   │                                        │ ────────────│
│   Earnings│                                       │ Value: $10K  │
├──────────┴────────────────────────────────────────┴──────────────┤
│ ● Connected | 12:34:56 | Market Data: Real-time    [Status Bar] │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

### 1. Watchlist (Left Panel)
- **Search**: Type symbol + Enter to add
- **Tabs**: My List, S&P 500, Tech, Crypto
- **Live Prices**: Updates every 5 seconds
- **Click**: Load chart for any symbol
- **Right-Click**: Context menu options

### 2. Chart Area (Center)
- **Symbol Input**: Type and load any symbol
- **Timeframes**: 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y
- **Intervals**: 1m, 5m, 15m, 30m, 1h, 1d, 1wk
- **Chart Types**: Candlestick, Line, Bar, OHLC
- **Tools**: Indicators, Drawing Tools, Compare

### 3. Active Trader (Right Panel)
- **Quick Order**: Fast buy/sell execution
- **Order Types**: Market, Limit, Stop, Stop-Limit
- **Positions**: View open positions with P&L
- **Account**: Real-time account summary

### 4. News Feed (Bottom Left)
- **Latest News**: Market updates
- **Symbol News**: Filter by selected stock
- **Timestamps**: Human-readable (e.g., "2 mins ago")
- **Expandable**: Click to read more

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save your layout |
| `Ctrl+E` | Export chart as PNG |
| `F1` | Show/hide watchlist |
| `F2` | Show/hide active trader |
| `D` | Enable drawing tools |
| `P` | Open pattern scanner |
| `Esc` | Close all menus |

## 🔧 Resizing Panels

### How to Resize
1. **Hover** over any panel divider (border between panels)
2. **Cursor changes** to resize arrows ↔️ or ↕️
3. **Click and drag** to resize
4. **Double-click** divider to reset to default size

### Which Panels Are Resizable?
- ✅ Left panel (watchlist) - Width: 200-500px
- ✅ Right panel (active trader) - Width: 250-600px
- ✅ News feed height - Split with watchlist
- ✅ All sizes saved automatically

## 💰 Placing Your First Trade

### Step-by-Step
1. **Select Symbol**
   - Click any symbol in watchlist
   - Or type in Active Trader panel

2. **Enter Details**
   - Quantity: How many shares
   - Order Type: Market (instant) or Limit (at price)
   - Price: Only for Limit orders

3. **Review**
   - Check "Estimated Cost"
   - Ensure you have buying power

4. **Execute**
   - Click **BUY** (green) or **SELL** (red)
   - Order confirmation shown

5. **Monitor**
   - Position appears in "Positions" section
   - P&L updates in real-time

## 📈 Loading Different Charts

### From Watchlist
```
1. Click any symbol → Chart loads automatically
```

### From Chart Toolbar
```
1. Type symbol in toolbar
2. Click "Load Chart"
```

### From URL
```
http://localhost:8080/index_tos_style.html?symbol=TSLA
```

### Change Timeframe
```
Click: [1D] [5D] [1M] [3M] [6M] [YTD] [1Y] [5Y]
```

### Change Interval
```
Dropdown: 1min → 5min → 15min → 30min → 1h → 1day → 1wk
```

## 🎨 Using Drawing Tools

1. **Enable Drawing Mode**
   - Click "Draw" button in toolbar
   - Or press `D` key

2. **Draw on Chart**
   - Click to start line
   - Click to end line
   - Drag to reposition

3. **Clear Drawings**
   - Menu: Analysis → Drawing Tools
   - Or use chart toolbar

## 📊 Adding Indicators

1. **Open Indicator Menu**
   - Click "+ Indicator" button
   - Or Menu: Analysis → Technical Studies

2. **Available Indicators**
   - SMA (Simple Moving Average)
   - EMA (Exponential Moving Average)
   - RSI (Relative Strength Index)
   - MACD
   - Bollinger Bands
   - VWAP
   - Volume

3. **Toggle Indicators**
   - Click to add/remove
   - Customize parameters (coming soon)

## 🔍 Using the Menu Bar

### File Menu
- **Save Layout**: Preserve panel sizes
- **Export Chart**: Download as PNG
- **Export Data**: Save chart data (CSV)

### View Menu
- **Show/Hide Panels**: Toggle watchlist, news, trader
- **Chart Settings**: Customize appearance
- **Layouts**: Switch between 1, 2x1, 2x2 charts

### Tools Menu
- **Pattern Scanner**: Detect chart patterns
- **Price Alerts**: Set price notifications
- **Trade Ideas**: Get AI suggestions
- **Backtester**: Test strategies

### Analysis Menu
- **Technical Studies**: Add indicators
- **Drawing Tools**: Enable drawings
- **Fundamentals**: Company data
- **Compare**: Overlay multiple symbols

### Trade Menu
- **Quick Trade**: Fast order entry
- **Trade History**: View past trades
- **Positions**: Open positions
- **Account Summary**: Full account view

## 🎯 Pro Tips

### Tip 1: Customize Your Watchlist
```
1. Switch to "My List" tab
2. Type symbols and press Enter
3. Saved automatically for next session
```

### Tip 2: Save Your Layout
```
1. Resize panels to your preference
2. Press Ctrl+S or File → Save Layout
3. Layout loads automatically next time
```

### Tip 3: Quick Symbol Switch
```
1. Click symbols in watchlist
2. No need to reload page
3. Chart updates instantly
```

### Tip 4: Monitor Multiple Stocks
```
1. Add symbols to watchlist
2. Prices update every 5 seconds
3. Green = up, Red = down
```

### Tip 5: Fast Trading
```
1. Click symbol → Auto-fills order form
2. Enter quantity → Shows estimated cost
3. One click buy/sell
```

## 🐛 Troubleshooting

### Chart Not Loading?
```
✓ Check: Server is running (python app.py)
✓ Check: Symbol is valid (e.g., AAPL, not Apple)
✓ Check: Browser console for errors (F12)
```

### Prices Not Updating?
```
✓ Wait 5 seconds (auto-refresh interval)
✓ Check internet connection
✓ Refresh page (F5)
```

### Panels Not Resizing?
```
✓ Hover over border until cursor changes
✓ Click and drag (don't just click)
✓ Try double-click to reset
```

### Orders Not Going Through?
```
✓ Check: Sufficient buying power
✓ Check: Valid quantity (positive number)
✓ Check: Server portfolio API is running
```

## 📱 Mobile Usage

The interface is optimized for desktop but works on mobile:
- Panels collapse on smaller screens
- Touch gestures for chart zoom/pan
- Vertical scrolling for all content
- Menu buttons adapted for touch

## 🎓 Learn More

- **Full Documentation**: See `TOS_STYLE_README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Keyboard Shortcuts**: Press `?` or Help → Shortcuts
- **About**: Help → About StockApp

## ⚡ Quick Reference Card

```
┌─────────────────────────────────────────┐
│  ESSENTIAL SHORTCUTS                    │
├─────────────────────────────────────────┤
│  Enter (in search)  → Add to watchlist  │
│  Click symbol       → Load chart        │
│  F1                 → Toggle watchlist  │
│  F2                 → Toggle trader     │
│  Ctrl+S             → Save layout       │
│  D                  → Drawing mode      │
│  Esc                → Close menus       │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  PANEL RESIZING                         │
├─────────────────────────────────────────┤
│  Hover border       → Resize cursor     │
│  Click & drag       → Resize panel      │
│  Double-click       → Reset size        │
│  Auto-save          → Remembers sizes   │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  TRADING WORKFLOW                       │
├─────────────────────────────────────────┤
│  1. Select symbol from watchlist        │
│  2. Chart loads automatically           │
│  3. Symbol fills order form             │
│  4. Enter quantity                      │
│  5. Click BUY or SELL                   │
│  6. View in Positions                   │
└─────────────────────────────────────────┘
```

## 🎉 Ready to Trade!

You're all set! Start by:
1. Adding symbols to your watchlist
2. Clicking a symbol to load the chart
3. Exploring the different timeframes
4. Placing your first practice trade

**Access the platform**: http://localhost:8080/index_tos_style.html

**Need help?** Check the full documentation in `TOS_STYLE_README.md`

---

Happy Trading! 📈💰

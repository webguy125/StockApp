# ThinkorSwim-Style Professional Trading Platform

A professional trading interface inspired by ThinkorSwim, featuring a sophisticated dark theme, resizable panels, and comprehensive trading tools.

## Overview

This implementation transforms StockApp into a professional-grade trading platform with a layout and visual design inspired by ThinkorSwim. The interface provides a seamless experience for stock analysis, trading, and portfolio management.

## Files Created

### CSS Theme
- **`frontend/css/tos-theme.css`** - Complete ThinkorSwim-inspired dark theme with professional styling

### JavaScript Modules

#### Layout Components
- **`frontend/js/layout/resizable-panels.js`** - Drag-to-resize functionality for all panels
- **`frontend/js/layout/menu-bar.js`** - Professional menu system with dropdown menus

#### Feature Components
- **`frontend/js/components/watchlist.js`** - Symbol watchlist with real-time price updates
- **`frontend/js/components/news-feed.js`** - Market news feed with filtering
- **`frontend/js/components/active-trader.js`** - Quick order entry and portfolio management

### Main Application
- **`frontend/index_tos_style.html`** - Main HTML file for ThinkorSwim-style interface
- **`frontend/js/tos-app.js`** - Main application controller that initializes all components

## Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  Menu Bar: File | View | Tools | Analysis | Trade | Help        │
├──────────────┬──────────────────────────────────┬───────────────┤
│              │                                  │               │
│  Watchlist   │        Chart Area                │ Active Trader │
│  ────────    │  ┌────────────────────────────┐  │               │
│  • AAPL      │  │ Toolbar: Symbol, Period,   │  │ Quick Order   │
│  • MSFT      │  │ Interval, Chart Type, etc. │  │ ───────────   │
│  • GOOGL     │  └────────────────────────────┘  │ Symbol: AAPL  │
│  • TSLA      │                                  │ Qty: 100      │
│              │                                  │ [BUY] [SELL]  │
│  ────────    │     Main Chart Display           │               │
│  News Feed   │                                  │ Positions     │
│  ────────    │                                  │ ───────────   │
│  • Latest    │                                  │ • AAPL +2.5%  │
│    Market    │                                  │ • MSFT -1.2%  │
│    News      │                                  │               │
│              │                                  │ Account       │
│              │                                  │ Summary       │
├──────────────┴──────────────────────────────────┴───────────────┤
│  Status Bar: Connected | 12:34:56 | Market Data: Real-time      │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### 1. Professional Menu Bar
- **File**: New Workspace, Save Layout, Export Chart/Data, Settings
- **View**: Toggle panels, Chart settings, Multiple layouts
- **Tools**: Pattern Scanner, Price Alerts, Trade Ideas, Backtester
- **Analysis**: Technical Studies, Drawing Tools, Fundamentals
- **Trade**: Quick Trade, Trade History, Positions, Account Summary
- **Help**: Documentation, Keyboard Shortcuts, About

### 2. Left Panel - Watchlist & News

#### Watchlist
- Symbol search with Enter to add
- Multiple tabs: My List, S&P 500, Tech, Crypto
- Real-time price updates every 5 seconds
- Color-coded gains (green) and losses (red)
- Click symbol to load chart
- Right-click context menu for actions

#### News Feed
- Real-time market news updates
- Symbol-specific filtering
- Timestamps in human-readable format ("2 mins ago")
- Click headlines to expand preview
- Auto-refresh every 2 minutes

### 3. Center Panel - Chart Area

#### Chart Toolbar
- Symbol input with autocomplete
- Period buttons: 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y
- Interval selector: 1m, 5m, 15m, 30m, 1h, 1d, 1wk
- Chart type: Candlestick, Line, Bar, OHLC
- Tools: Add Indicator, Drawing Tools, Compare

#### Main Chart
- Full Plotly.js candlestick chart
- Dark professional theme
- Integrated with existing StockApp functionality
- Drawing tools support
- Technical indicators support
- Zoom and pan capabilities

### 4. Right Panel - Active Trader

#### Quick Order Entry
- Symbol auto-populates from chart
- Order types: Market, Limit, Stop, Stop-Limit
- Quantity input with validation
- Price inputs (conditional based on order type)
- Time in Force: Day, GTC, GTD
- BUY/SELL buttons with validation
- Estimated cost calculation
- Real-time buying power check

#### Positions Section
- List of active positions
- Display: Symbol, Qty, Avg Price, Current P&L
- Color-coded P&L (green/red)
- Click to load symbol in chart
- Collapsible section

#### Orders Section
- Working orders display
- Filled orders history
- Collapsible section

#### Account Summary
- Buying Power
- Total P&L (Day)
- Account Value
- Margin Usage percentage

### 5. Resizable Panels

All panels support drag-to-resize:
- **Left Panel**: 200-500px width
- **Right Panel**: 250-600px width
- **Watchlist/News**: Adjustable height split
- **Double-click** resizer to reset to default size
- Panel sizes saved to localStorage

### 6. Status Bar
- Connection status indicator (green dot)
- Real-time server clock
- Market data status
- Notifications area

## Usage Instructions

### Accessing the ThinkorSwim Interface

1. **Open in browser**:
   ```
   http://localhost:8080/index_tos_style.html
   ```

2. **Or navigate from main page**:
   - Add a link to `index_tos_style.html` from your main index

### Basic Operations

#### Loading a Chart
1. Type symbol in watchlist search and press Enter
2. Click symbol in watchlist
3. Type symbol in chart toolbar and click "Load Chart"
4. Use URL parameter: `?symbol=AAPL`

#### Changing Timeframes
- Click period buttons (1D, 5D, 1M, etc.)
- Select interval from dropdown
- Chart automatically reloads

#### Placing Orders
1. Symbol auto-fills from chart (or type manually)
2. Select order type
3. Enter quantity
4. Enter price (for limit orders)
5. Click BUY or SELL
6. Order validated for sufficient funds

#### Resizing Panels
1. Hover over panel divider (vertical or horizontal)
2. Cursor changes to resize cursor
3. Click and drag to resize
4. Double-click to reset to default
5. Sizes saved automatically

#### Using Drawing Tools
- Click "Draw" button in toolbar
- Or press 'D' key
- Use Plotly drawing tools

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save layout |
| `Ctrl+E` | Export chart |
| `F1` | Toggle watchlist panel |
| `F2` | Toggle active trader panel |
| `D` | Enable drawing mode |
| `P` | Open pattern scanner |
| `Esc` | Close all menus |
| `Enter` | Submit symbol search |

## Integration with Existing StockApp

The ThinkorSwim interface integrates seamlessly with existing StockApp functionality:

### Reused Modules
- **Chart Loading**: `js/chart/loader.js`
- **Technical Indicators**: `js/chart/indicators.js`
- **Drawing Tools**: `js/trendlines/drawing.js`
- **Pattern Detection**: `js/analysis/patterns.js`
- **Trade Ideas**: `js/analysis/trade-ideas.js`
- **Portfolio API**: `/portfolio/buy`, `/portfolio/sell`
- **State Management**: `js/core/state.js`

### API Endpoints Used
- `GET /data/:symbol?interval=:interval&period=:period` - Chart data
- `POST /portfolio/buy` - Place buy order
- `POST /portfolio/sell` - Place sell order
- `GET /portfolio` - Get portfolio positions

## Customization

### Color Scheme
Edit `frontend/css/tos-theme.css` CSS variables:
```css
--tos-bg-primary: #1a1a1a;
--tos-accent-blue: #4a9eff;
--tos-accent-green: #00c851;
--tos-accent-red: #ff4444;
```

### Default Watchlist
Edit `frontend/js/components/watchlist.js`:
```javascript
this.predefinedLists = {
  mylist: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
  // Add more lists...
};
```

### Panel Sizes
Edit default sizes in `frontend/css/tos-theme.css`:
```css
.tos-left-panel {
  width: 300px;
  min-width: 200px;
  max-width: 500px;
}
```

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

## Performance Notes

- Watchlist updates every 5 seconds (configurable)
- News feed refreshes every 2 minutes (configurable)
- Chart renders with Plotly.js (hardware-accelerated)
- Panel resize positions saved to localStorage
- Minimal re-renders for optimal performance

## Known Limitations

1. **News Feed**: Currently uses mock data. Integrate with a real news API for production.
2. **Real-time Data**: Price updates use polling. Consider WebSocket for true real-time updates.
3. **Multi-chart Layout**: 2x1, 2x2 layouts show notification but not yet implemented.
4. **Advanced Order Types**: Stop and Stop-Limit orders accepted but need backend implementation.
5. **Mobile Responsive**: Panels collapse on mobile but experience is optimized for desktop.

## Future Enhancements

- [ ] Real news API integration (Alpha Vantage, Finnhub, etc.)
- [ ] WebSocket for real-time price streaming
- [ ] Multiple chart layouts (2x1, 2x2 grid)
- [ ] Technical indicator panel with custom parameters
- [ ] Advanced charting features (Fibonacci, Elliott Wave)
- [ ] Options chain display
- [ ] Level 2 market data
- [ ] Trade confirmation dialogs
- [ ] Historical trade analytics
- [ ] Custom theme builder

## Troubleshooting

### Chart not loading
- Check browser console for errors
- Verify API endpoint is running
- Ensure symbol is valid

### Panels not resizing
- Check if ResizablePanel is initialized
- Verify CSS classes are applied
- Clear localStorage and refresh

### Orders not executing
- Check if portfolio API is running
- Verify authentication token
- Check browser network tab for errors

### News not displaying
- Currently uses mock data by default
- Verify NewsFeed component is initialized
- Check browser console for errors

## Credits

Designed and implemented for StockApp based on ThinkorSwim trading platform design principles.

## License

Part of the StockApp project. All rights reserved.

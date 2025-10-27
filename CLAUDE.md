# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**StockApp - Professional Technical Analysis Platform** is a Flask-based web application that provides advanced stock charting with professional-grade technical indicators, drawing tools, and analysis capabilities. Features include:

- **Interactive candlestick charts** with real-time data from Yahoo Finance
- **Technical indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, VWAP
- **Advanced drawing tools**: Trendlines, rectangles, freehand drawing with volume calculation
- **Dark/Light theme** toggle for comfortable viewing
- **Watchlist sidebar** for quick symbol switching
- **Persistent annotations** - all drawings saved per symbol

## Architecture

### Backend (Flask API)
- **Main server**: `backend/api_server.py` - Flask server with CORS enabled
- **Port**: Default Flask development server (localhost:5127)
- **Data persistence**: JSON files stored in `backend/data/` directory
  - Format: `lines_{SYMBOL}.json` - stores trendlines with coordinates and volume data

### Frontend
- **Enhanced UI**: `frontend/index_enhanced.html` (default, served at `/`)
- **Classic UI**: `frontend/index.html` (available at `/classic`)
- **Charting library**: Plotly.js 2.27.1 (candlestick charts with interactive drawing)
- **No build step**: Pure HTML/JavaScript served directly by Flask
- **Modern design**: Responsive sidebar layout with dark/light theme support

### Key API Endpoints

- `GET /` - Serve enhanced frontend with all features
- `GET /classic` - Serve original simple frontend

- `GET /data/<symbol>` - Fetch OHLC stock data from yfinance
  - Query params: `start`, `end`, `period`, `interval`
  - Returns JSON array of candlestick data

- `POST /indicators` - Calculate technical indicators
  - Body: `{symbol, period, interval, indicators: [{type, params}]}`
  - Types: SMA, EMA, RSI, MACD, BB (Bollinger Bands), VWAP
  - Returns indicator data arrays aligned with chart dates

- `POST /volume` - Calculate average volume between two dates
  - Body: `{symbol, start_date, end_date, interval}`
  - Used when drawing trendlines to show volume stats

- `POST /save_line` - Persist a drawn trendline
  - Body: `{symbol, line: {id, x0, x1, y0, y1, volume}}`

- `GET /lines/<symbol>` - Load saved trendlines for a symbol

- `POST /delete_line` - Delete a specific trendline

- `POST /clear_lines` - Clear all trendlines for a symbol (deprecated in enhanced UI)

### Data Flow

1. User enters stock symbol and timeframe → Frontend fetches data from `/data/<symbol>`
2. User draws line on chart → Frontend calculates volume via `/volume` endpoint
3. Line coordinates + volume saved to `backend/data/lines_{SYMBOL}.json`
4. On chart reload → Frontend fetches saved lines from `/lines/<symbol>` and redraws them

## Development Commands

### Setup
```bash
# Create virtual environment and install dependencies
install_libs.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Start Flask server (runs on http://127.0.0.1:5000)
start_flask.bat

# Or manually:
cd backend
python api_server.py
```

The Flask server serves both the API and the frontend (`/` route serves `index.html`).

### Testing Specific Functionality
```bash
# Test candlestick plotting standalone
cd backend
python plot_chart.py

# Check saved lines for a symbol
python ../Check_lines.py
```

## Important Implementation Details

### Trendline Drawing System
- Uses Plotly's `dragmode: "drawline"` for interactive line drawing
- Each line gets a UUID assigned via `crypto.randomUUID()`
- Lines are stored with coordinates (x0, x1, y0, y1) where x values are ISO date strings
- Volume calculation happens asynchronously after line is drawn/modified
- `relayoutGuard` prevents duplicate volume calculations during rapid updates
- `suppressSave` flag prevents saving when loading existing lines

### Line Selection and Deletion
- Click near a line to select it (turns red)
- Press Delete key or right-click menu to delete selected line
- Distance calculation normalizes coordinates to handle different axis scales
- Threshold: 0.05 normalized units for click detection

### Timeframe System
- Predefined options: 1y/1d, 3y/1wk, 10y/1mo, 20y/1mo, plus intraday options
- Custom range: allows arbitrary start/end dates with interval selection
- Intervals: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo

### Data Storage
- Lines persisted per-symbol in JSON format
- File location: `backend/data/lines_{SYMBOL}.json`
- No database - simple file-based storage
- Multiple backup versions of api_server.py exist (numbered backups) - the main file is `api_server.py`

## New Features (10X Enhancement Implementation)

### Technical Indicators
- **Moving Averages**: SMA and EMA with customizable periods
- **RSI**: Relative Strength Index (14 period default) with separate subplot
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **Bollinger Bands**: Upper, middle, lower bands with standard deviation
- **VWAP**: Volume-Weighted Average Price for intraday analysis

### Enhanced UI/UX
- **Dark/Light Theme**: Toggle between themes with persistent preference
- **Modern Sidebar**: Organized controls with collapsible sections
- **Watchlist**: Quick-switch between frequently traded symbols
- **Drawing Tools**: Trendline, freehand, and rectangle drawing modes
- **Keyboard Shortcuts**: Delete key for quick line removal

### Smart Features
- **Automatic Label Positioning**: Volume labels avoid candle overlap
- **Theme-Aware Charts**: All chart elements adapt to light/dark theme
- **Indicator Subplots**: RSI and MACD display in dedicated panels below main chart
- **Responsive Layout**: Sidebar + main content for organized workspace

## Dependencies

Core packages (see requirements.txt):
- flask - Web framework
- flask-cors - Enable CORS for API
- yfinance - Yahoo Finance data fetching
- pandas - Data manipulation
- numpy - Numerical calculations for indicators
- plotly - Chart generation (used in plot_chart.py)

## File Organization

- `backend/api_server.py` - Main Flask application with indicator endpoint
- `backend/plot_chart.py` - Standalone candlestick plotting utility
- `backend/data/` - JSON storage for trendline data
- `frontend/index_enhanced.html` - **NEW**: Professional UI with indicators and themes (DEFAULT)
- `frontend/index.html` - Classic simple UI (available at `/classic`)
- `backend/backup*.py` - Old versions of api_server (can be ignored)
- `frontend/Backup*.html` - Old versions of frontend (can be ignored)
- `start_flask.bat` - Quick launcher for Windows
- `Check_lines.py` - Utility to inspect saved line data
- `PROJECT_ENHANCEMENT_PROMPT.json` - Full roadmap for 10X improvements

## Access URLs

- **Main App (Enhanced)**: http://127.0.0.1:5000/
- **Classic UI**: http://127.0.0.1:5000/classic

## Quick Start Guide

1. Start the server: `start_flask.bat` or `venv/Scripts/python.exe backend/api_server.py`
2. Open browser to http://127.0.0.1:5000/
3. Enter a stock symbol (e.g., GE, AAPL, TSLA)
4. Select a timeframe from the dropdown
5. Toggle indicators on/off in the sidebar
6. Draw trendlines by clicking and dragging on the chart
7. Switch to dark mode using the theme toggle button

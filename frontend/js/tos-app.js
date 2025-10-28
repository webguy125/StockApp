/**
 * ThinkorSwim-Style Trading Platform Main Application
 * Initializes all components and manages the trading interface
 */

import { state } from './core/state.js';
import { initializeResizablePanels, loadPanelStates } from './layout/resizable-panels.js';
import { initializeMenuBar } from './layout/menu-bar.js';
import { initializeWatchlist } from './components/watchlist.js';
import { initializeNewsFeed } from './components/news-feed.js';
import { initializeActiveTrader } from './components/active-trader.js';
import { initializeChartSettings } from './components/chart-settings.js';
import { loadChart, convertToCST } from './chart/loader.js';
import { setupPlotlyHandlers } from './trendlines/handlers.js';
import { CanvasRenderer } from './chart-renderers/canvas-renderer.js';
import { TimeframeRegistry } from './timeframes/TimeframeRegistry.js';
import { TickChartRegistry } from './tick-charts/TickChartRegistry.js';
import { TimeframeSelector } from './components/timeframe-selector.js';

/**
 * Main TOS Application Class
 */
class TOSApp {
  constructor() {
    this.menuBar = null;
    this.watchlist = null;
    this.newsFeed = null;
    this.activeTrader = null;
    this.resizablePanels = null;
    this.chartSettings = null; // Chart settings manager

    // Load last used settings from localStorage, or use defaults
    this.currentSymbol = localStorage.getItem('lastSymbol') || null;
    this.currentPeriod = 'max'; // Show all historical data
    this.currentInterval = '1d'; // Fixed to daily (will be replaced by timeframe system)
    // OLD 1D-ONLY RENDERER - COMMENTED OUT (now using timeframe registry)
    // this.chartRenderer = new CanvasRenderer();

    // Timeframe system (NEW)
    this.timeframeRegistry = new TimeframeRegistry();
    this.tickChartRegistry = new TickChartRegistry();
    this.timeframeSelector = null; // Initialized after DOM loads
    this.currentTimeframeId = localStorage.getItem('lastTimeframe') || '1d';
    this.currentTickChartId = null;
    this.activeChartType = 'timeframe'; // 'timeframe' or 'tick'

    this.activeIndicators = []; // Track indicators currently on chart: {id, type, name, params, traceIndices}
    this.liveUpdateInterval = null; // Timer for live data updates
    this.socket = null; // Socket.IO connection for real-time updates
    this.currentDragMode = 'pan'; // Track current drag mode (pan or zoom)
    this.ctrlKeyListenerAdded = false; // Track if Ctrl key listener is already added
    this.initializeWebSocket();
  }

  /**
   * Convert UTC date to CST (Central Standard Time) format
   * CST is America/Chicago timezone
   */
  formatDateToCST(date) {
    const d = new Date(date);
    // Format to CST using Intl.DateTimeFormat
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/Chicago',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });

    const parts = formatter.formatToParts(d);
    const values = {};
    parts.forEach(part => {
      values[part.type] = part.value;
    });

    return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second}`;
  }

  /**
   * Save current chart settings to localStorage for next session
   */
  saveChartSettings() {
    if (this.currentSymbol) {
      localStorage.setItem('lastSymbol', this.currentSymbol);
    }
    localStorage.setItem('lastPeriod', this.currentPeriod);
    localStorage.setItem('lastInterval', this.currentInterval);
    // console.log(`💾 Saved chart settings: ${this.currentSymbol} ${this.currentPeriod}/${this.currentInterval}`);
  }

  /**
   * Initialize WebSocket connection for real-time price updates
   */
  initializeWebSocket() {
    // Connect to Socket.IO server with reconnection settings
    // Connect to the same host that served the page (works for both localhost and 127.0.0.1)

    // Enable Socket.IO debug logging
    localStorage.debug = 'socket.io-client:*';

    this.socket = io({
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity,
      timeout: 20000,
      transports: ['websocket', 'polling']
    });

    // DEBUG: Catch-all listener to see ALL incoming Socket.IO events
    // this.socket.onAny((eventName, ...args) => {
    //   console.log(`🌐 [CATCH-ALL] Received event: "${eventName}"`, args);
    // });
    // console.log('✅ Registered catch-all event listener (onAny)');

    // Listen for connection
    this.socket.on('connect', () => {
      console.log('🔌 WebSocket connected - Socket ID:', this.socket.id);
      console.log('🔌 Transport:', this.socket.io.engine.transport.name);
      console.log('🔌 Connected:', this.socket.connected);

      // Restart live updates to switch from polling to WebSocket mode
      if (this.currentSymbol) {
        this.startLiveUpdates();
      }

      // Subscribe watchlist to all symbols now that WebSocket is connected
      if (this.watchlist) {
        this.watchlist.subscribeToAllSymbols();
      }
    });

    // Listen for connection response
    this.socket.on('connection_response', (data) => {
      console.log('📨 [CONNECTION_RESPONSE] Received:', data);
    });

    // Listen for Coinbase ticker updates (real-time price)
    this.socket.on('ticker_update', (data) => {
      // console.log('🔥 RECEIVED ticker_update event:', data);
      this.handleTickerUpdate(data);
    });
    // console.log('✅ Registered ticker_update event listener');

    // 1m candle updates removed - using ticker updates for all intervals

    // Listen for individual trade updates
    this.socket.on('trade_update', (data) => {
      // console.log('💎 RECEIVED trade_update event:', data);
      this.handleTradeUpdate(data);
    });
    // console.log('✅ Registered trade_update event listener');

    // Listen for subscription responses
    this.socket.on('subscription_response', (data) => {
      // console.log(`Subscription: ${data.message}`);
    });

    // Listen for connection errors
    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
    });

    // Listen for reconnection attempts
    this.socket.on('reconnect_attempt', (attemptNumber) => {
      console.log(`Attempting to reconnect... (attempt ${attemptNumber})`);
    });

    // Listen for successful reconnection
    this.socket.on('reconnect', (attemptNumber) => {
      console.log(`Reconnected after ${attemptNumber} attempts`);

      // Reload chart data to catch up on missed updates
      if (this.currentSymbol) {
        this.reloadChart();
      }
    });

    // Listen for disconnection
    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);

      // If disconnected by server, try to reconnect
      if (reason === 'io server disconnect') {
        this.socket.connect();
      }
    });

    // Setup page visibility detection to reload when tab becomes active
    this.setupPageVisibilityHandler();

    // Setup periodic keepalive ping (every 30 seconds)
    this.setupKeepalive();
  }

  /**
   * Setup page visibility handler to reload chart when tab becomes active
   * Also implements "always on" mode to prevent browser throttling
   */
  setupPageVisibilityHandler() {
    let lastActiveTime = Date.now();

    // Keep track of visibility changes for reconnection
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) {
        // Tab became visible
        const timeAway = Date.now() - lastActiveTime;

        // Always ensure WebSocket is connected when tab becomes visible
        if (!this.socket.connected) {
          console.log('Reconnecting WebSocket...');
          this.socket.connect();
        }

        // If away for a while, do a quick check to ensure data is current
        if (timeAway > 5 * 60 * 1000) {
          console.log(`Verifying chart data (away ${Math.round(timeAway / 60000)}m)...`);
          this.reloadChart();
        }
      } else {
        // Tab became hidden
        lastActiveTime = Date.now();
      }
    });

    // Setup "always on" mode to prevent browser throttling
    this.setupAlwaysOnMode();
  }

  /**
   * Keep the chart always active even in background tabs
   * Uses multiple techniques to prevent browser throttling
   */
  setupAlwaysOnMode() {
    // Technique 1: Silent audio context trick to keep tab "active"
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      gainNode.gain.value = 0.001; // Nearly silent
      oscillator.frequency.value = 20000; // Inaudible frequency
      oscillator.start(0);

      console.log('Audio context keepalive active');
    } catch (e) {
      console.log('Audio context not available');
    }

    // Technique 2: High-frequency requestAnimationFrame loop
    let lastFrameTime = Date.now();
    const frameLoop = () => {
      const now = Date.now();

      // Every 5 seconds, do a light check
      if (now - lastFrameTime > 5000) {
        lastFrameTime = now;

        // Ensure WebSocket is still connected
        if (this.socket && !this.socket.connected) {
          this.socket.connect();
        }
      }

      requestAnimationFrame(frameLoop);
    };
    frameLoop();

    // Technique 3: Periodic worker-like interval (browser can't throttle as much)
    setInterval(() => {
      // Force a tiny DOM update to keep JavaScript active
      const timestamp = Date.now();
      document.body.setAttribute('data-alive', timestamp);

      // Ensure WebSocket stays connected
      if (this.socket && !this.socket.connected) {
        console.log('Reconnecting WebSocket (keepalive)...');
        this.socket.connect();
      }
    }, 10000); // Every 10 seconds

    // Technique 4: Wake Lock API (if supported) - prevents display sleep
    this.setupWakeLock();
  }

  /**
   * Setup Wake Lock API to keep screen active (if supported)
   */
  async setupWakeLock() {
    if ('wakeLock' in navigator) {
      try {
        this.wakeLock = await navigator.wakeLock.request('screen');
        console.log('Wake Lock active - screen will stay on');

        // Reacquire wake lock if it's released
        this.wakeLock.addEventListener('release', async () => {
          console.log('Wake Lock released, reacquiring...');
          try {
            this.wakeLock = await navigator.wakeLock.request('screen');
          } catch (e) {
            console.log('Could not reacquire wake lock');
          }
        });
      } catch (e) {
        console.log('Wake Lock not available:', e.message);
      }
    }
  }

  /**
   * Setup periodic keepalive to prevent connection timeout
   */
  setupKeepalive() {
    setInterval(() => {
      if (this.socket && this.socket.connected) {
        this.socket.emit('ping', { timestamp: Date.now() });
      }
    }, 30000); // Ping every 30 seconds
  }

  /**
   * Update data bar with live price (works for all intervals)
   */
  updateDataBarWithLivePrice(currentPrice) {
    const plotDiv = document.getElementById('tos-plot');
    if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) {
      return;
    }

    const candlestickIndex = plotDiv.data.findIndex(trace => trace.type === 'candlestick');
    if (candlestickIndex === -1) return;

    const candlestickTrace = plotDiv.data[candlestickIndex];
    if (!candlestickTrace.close || candlestickTrace.close.length === 0) return;

    const lastIndex = candlestickTrace.close.length - 1;

    // Calculate daily change - use TradingView method (previous day's close at 00:00 UTC)
    let previousDayClose;

    if (this.currentInterval === '1d' || this.currentInterval === '3d' || this.currentInterval === '5d' ||
        this.currentInterval === '1wk' || this.currentInterval === '2wk' || this.currentInterval === '1mo') {
      // For daily+ charts, use previous candle's close
      previousDayClose = lastIndex > 0 ? candlestickTrace.close[lastIndex - 1] : candlestickTrace.open[lastIndex];
    } else {
      // For intraday charts, find the last candle BEFORE 00:00 UTC today (yesterday's close)
      // This matches TradingView's calculation method
      const now = new Date(candlestickTrace.x[lastIndex]);

      // Get today at 00:00 UTC
      const todayMidnightUTC = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), 0, 0, 0, 0));

      // Find the last candle BEFORE today's midnight UTC (this is yesterday's close)
      let yesterdayCloseIndex = -1;

      for (let i = lastIndex - 1; i >= Math.max(0, lastIndex - 500); i--) {
        const candleTime = new Date(candlestickTrace.x[i]);
        if (candleTime < todayMidnightUTC) {
          yesterdayCloseIndex = i;
          break;
        }
      }

      // Use yesterday's close (last candle before 00:00 UTC today)
      if (yesterdayCloseIndex >= 0) {
        previousDayClose = candlestickTrace.close[yesterdayCloseIndex];
      } else {
        // Fallback if we couldn't find yesterday's close
        previousDayClose = lastIndex > 0 ? candlestickTrace.close[lastIndex - 1] : candlestickTrace.open[lastIndex];
      }
    }

    const currentClose = currentPrice;
    const currentOpen = candlestickTrace.open[lastIndex];
    const currentHigh = Math.max(candlestickTrace.high[lastIndex], currentPrice);
    const currentLow = Math.min(candlestickTrace.low[lastIndex], currentPrice);
    const currentDate = candlestickTrace.x[lastIndex];

    const dataBarEl = document.getElementById('tos-chart-data-bar');
    if (dataBarEl && !window.userSelectedCandle) {
      // Use 24-hour change for consistency across all timeframes
      const change = currentClose - previousDayClose;
      const changePercent = ((change / previousDayClose) * 100).toFixed(2);
      const changeColor = change >= 0 ? '#00c851' : '#ff4444';

      // Format date to CST timezone
      let formattedDate = currentDate;
      try {
        formattedDate = this.formatDateToCST(currentDate);
      } catch (e) {
        formattedDate = currentDate;
      }

      // Get volume from volume trace (same way we get OHLC from candlestick trace)
      let volume = null;
      const volumeTrace = plotDiv.data.find(trace => trace.type === 'bar' && trace.yaxis === 'y2');
      if (volumeTrace && volumeTrace.y && volumeTrace.y[lastIndex] !== undefined) {
        volume = volumeTrace.y[lastIndex];
      }

      // Format volume
      const volumeStr = volume !== null && volume !== undefined
        ? `<span style="color: #a0a0a0;">Vol: <span style="color: #e0e0e0;">${volume.toLocaleString()}</span></span>`
        : '';

      dataBarEl.innerHTML = `
        <span style="color: #00bfff; margin-right: 10px;">LATEST:</span>
        <span style="color: #a0a0a0;">Date: <span style="color: #e0e0e0;">${formattedDate}</span></span>
        <span style="color: #a0a0a0;">O: <span style="color: #e0e0e0;">${currentOpen.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">H: <span style="color: #e0e0e0;">${currentHigh.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">L: <span style="color: #e0e0e0;">${currentLow.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">C: <span style="color: #e0e0e0;">${currentClose.toFixed(2)}</span></span>
        ${volumeStr}
        <span style="color: ${changeColor};">${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent}%)</span>
      `;
    }
  }

  /**
   * Handle real-time ticker updates from Coinbase
   */
  handleTickerUpdate(data) {
    // console.log(`📈 Chart received ticker: ${data.symbol} = $${data.price}`);

    // Only process if this is the current symbol
    // Handle both BTC-USD and BTC formats
    const symbolMatches = data.symbol && this.currentSymbol &&
      (data.symbol === this.currentSymbol ||
       data.symbol.includes(this.currentSymbol) ||
       this.currentSymbol.includes(data.symbol.split('-')[0]));

    if (!symbolMatches) {
      // console.log(`  ❌ Symbol mismatch: current=${this.currentSymbol}, ticker=${data.symbol}`);
      return;
    }

    // console.log(`  ✅ Symbol match! Updating chart...`);

    // Store latest ticker for this symbol (even if chart isn't loaded yet)
    this.lastTickerUpdate = data;

    // Update data bar for all intervals
    this.updateDataBarWithLivePrice(data.price);

    // Update candle with live price
    const now = Date.now();
    if (this.lastUpdateTime && (now - this.lastUpdateTime) < 250) {
      return; // Throttle: max 4 updates/sec
    }
    this.lastUpdateTime = now;

    // Update chart with price and volume
    // Volume now comes from WebSocket (initialized from REST + accumulated from trades)
    const volumeBTC = data.volume_today || 0;
    // OLD 1D-ONLY PATH - COMMENTED OUT (now using timeframe registry)
    // this.chartRenderer.updateLivePrice(data.price, volumeBTC);
    this.lastPrice = data.price;

    // Route to timeframe registry for timeframe-specific handling (NEW PATH)
    if (this.timeframeRegistry) {
      this.timeframeRegistry.handleTickerUpdate(data);
    }

    // Update the title for all intervals
    const plotDiv = document.getElementById('tos-plot');
    if (plotDiv && plotDiv.layout && plotDiv.layout.title) {
      const priceChange = data.price - (this.lastPrice || data.price);
      const changeSymbol = priceChange >= 0 ? '▲' : '▼';
      const changeColor = priceChange >= 0 ? '#00c853' : '#ff1744';

      const titleY = plotDiv.layout.title.y || 0.95;
      const titleText = `${data.symbol} - $${data.price.toFixed(2)} <span style="color: ${changeColor}">${changeSymbol}</span>`;
      console.log(`📝 Updating title: ${titleText}`);

      Plotly.relayout('tos-plot', {
        'title.text': titleText,
        'title.y': titleY,
        'title.yanchor': 'top',
        'title.xanchor': 'center',
        'title.x': 0.5
      }).then(() => {
        console.log('✅ Title updated');
      }).catch(err => {
        console.error('❌ Title update failed:', err);
      });
    }
    // Silently skip if chart not ready yet

    // Update the order entry display if it exists
    const orderSymbolDisplay = document.getElementById('order-symbol-display');
    if (orderSymbolDisplay && orderSymbolDisplay.textContent.includes(data.symbol)) {
      orderSymbolDisplay.textContent = `${data.symbol} - $${data.price.toFixed(2)}`;
    }

    this.lastPrice = data.price;
  }

  // handleCandleUpdate removed - 1m interval no longer supported

  /**
   * Handle individual trade updates (for tick/range charts)
   */
  handleTradeUpdate(data) {
    // Only for current symbol
    if (!this.currentSymbol || !data.symbol.includes(this.currentSymbol)) {
      return;
    }

    // Route based on active chart type
    if (this.activeChartType === 'tick') {
      // Route to tick chart registry
      if (this.tickChartRegistry) {
        this.tickChartRegistry.handleTradeUpdate(data);
      }
    } else {
      // Route to timeframe registry for tick/range chart aggregation
      if (this.timeframeRegistry) {
        this.timeframeRegistry.handleTradeUpdate(data);
      }
    }

    // Log for debugging
    // console.log('💹 Trade:', data.symbol, data.price, data.size, data.side);
  }

  /**
   * Start streaming for a symbol
   */
  async startStreaming(symbol) {
    if (!symbol) {
      // console.log('❌ Cannot start streaming - no symbol provided');
      return;
    }

    // console.log(`🚀 Starting WebSocket stream for ${symbol}...`);

    // Wait for Socket.IO to connect (with timeout)
    const waitForConnection = () => {
      return new Promise((resolve) => {
        if (this.socket && this.socket.connected) {
          resolve(true);
        } else {
          // Wait up to 2 seconds for connection
          let attempts = 0;
          const interval = setInterval(() => {
            attempts++;
            if (this.socket && this.socket.connected) {
              clearInterval(interval);
              resolve(true);
            } else if (attempts >= 20) { // 20 * 100ms = 2 seconds
              clearInterval(interval);
              resolve(false);
            }
          }, 100);
        }
      });
    };

    const isConnected = await waitForConnection();

    if (isConnected) {
      // Subscribe to live ticker updates (backend will also subscribe to matches for volume tracking)
      this.socket.emit('subscribe_ticker', { symbol });

      // console.log(`✅ Subscribed to live candles and ticker for ${symbol} via Socket.IO`);
      return;
    }

    // Socket.IO didn't connect in time - log warning
    console.warn('⚠️ Socket.IO not connected, real-time updates may not work');
    // console.log('Socket state:', {
    //   exists: !!this.socket,
    //   connected: this.socket?.connected,
    //   id: this.socket?.id
    // });
  }

  /**
   * Stop all streams
   */
  async stopAllStreams() {
    try {
      await fetch('/stop_all_streams', { method: 'POST' });
      // console.log('Stopped all streams');
    } catch (error) {
      console.error('Error stopping streams:', error);
    }
  }

  /**
   * Calculate which subplots are needed based on active indicators
   */
  getActiveSubplots() {
    const subplotIndicators = {
      'RSI': 'oscillator1',      // 0-100 range
      'STOCH': 'oscillator1',    // 0-100 range
      'ADX': 'oscillator1',      // 0-100 range
      'WILLR': 'oscillator2',    // -100-0 range
      'MACD': 'momentum',        // Unbounded
      'CCI': 'momentum',         // Unbounded
      'ATR': 'volatility',       // Unbounded
      'OBV': 'volume'            // Unbounded
    };

    const activeSubplots = new Set();

    this.activeIndicators.forEach(indicator => {
      const subplot = subplotIndicators[indicator.type];
      if (subplot) {
        activeSubplots.add(subplot);
      }
    });

    return Array.from(activeSubplots);
  }

  /**
   * Check if symbol is a cryptocurrency
   */
  isCryptoSymbol(symbol) {
    if (!symbol) return false;
    const cryptoSuffixes = ['-USD', '-USDT', '-BTC', '-ETH'];
    const cryptoSymbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'DOT', 'AVAX', 'SHIB', 'UNI', 'LINK', 'ATOM', 'LTC', 'BCH', 'XLM', 'ALGO', 'VET', 'ICP'];

    const upperSymbol = symbol.toUpperCase();

    // Check if ends with crypto suffix
    if (cryptoSuffixes.some(suffix => upperSymbol.endsWith(suffix))) {
      return true;
    }

    // Check if it's a known crypto symbol
    if (cryptoSymbols.includes(upperSymbol)) {
      return true;
    }

    return false;
  }

  /**
   * Calculate chart layout with dynamic subplots
   */
  calculateChartLayout() {
    const activeSubplots = this.getActiveSubplots();
    const numSubplots = activeSubplots.length;

    // Check if volume should be shown
    const showVolume = this.chartSettings && this.chartSettings.settings.showVolume;

    // Calculate domains for each subplot
    // Volume now overlays the main chart (no separate space needed)
    // Main chart should be at least 50% of the view
    // Each subplot gets equal space from the remaining area
    const subplotHeight = numSubplots > 0 ? 0.35 / numSubplots : 0;
    const mainChartBottom = (numSubplots > 0 ? 0.38 : 0);

    // Determine if we should add rangebreaks (only for non-crypto stocks)
    const isCrypto = this.isCryptoSymbol(this.currentSymbol);

    const layout = {
      title: {
        text: `${this.currentSymbol} - ${this.currentPeriod.toUpperCase()} / ${this.currentInterval.toUpperCase()}`,
        font: { color: '#e0e0e0', size: 16 }
      },
      xaxis: {
        gridcolor: '#404040',
        color: '#a0a0a0',
        rangeslider: { visible: false },
        domain: [0, 1],
        showspikes: false,  // DISABLED - using custom crosshair
        showticklabels: true
      },
      yaxis: {
        title: 'Price',
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: [mainChartBottom, 1],
        showspikes: false,  // DISABLED - using custom crosshair
        showticklabels: true
      },
      plot_bgcolor: '#1a1a1a',
      paper_bgcolor: '#1a1a1a',
      font: { color: '#e0e0e0' },
      margin: { l: 50, r: 70, t: 80, b: 50 },
      showlegend: true,
      legend: {
        x: 0,
        y: 1,
        bgcolor: 'rgba(0,0,0,0.3)',
        bordercolor: '#404040',
        borderwidth: 1
      },
      hovermode: 'closest',  // Enable hover detection for data updates (tooltips hidden via hoverinfo)
      hoverdistance: 100,
      spikedistance: -1,
      hoverlabel: {
        bgcolor: 'rgba(0,0,0,0)',
        bordercolor: 'rgba(0,0,0,0)',
        font: { size: 0, color: 'rgba(0,0,0,0)' },
        namelength: 0
      }
    };

    // Add subplot axes based on active subplots
    let currentBottom = 0.03;
    activeSubplots.forEach((subplot, idx) => {
      const yaxisNum = idx + 2;
      const domain = [currentBottom, currentBottom + subplotHeight - 0.02];

      layout[`yaxis${yaxisNum}`] = {
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: domain
      };

      // Set fixed range for oscillators
      if (subplot === 'oscillator1') {
        layout[`yaxis${yaxisNum}`].title = 'Oscillator';
        layout[`yaxis${yaxisNum}`].range = [0, 100];
      } else if (subplot === 'oscillator2') {
        layout[`yaxis${yaxisNum}`].title = 'Williams %R';
        layout[`yaxis${yaxisNum}`].range = [-100, 0];
      } else if (subplot === 'momentum') {
        layout[`yaxis${yaxisNum}`].title = 'Momentum';
      } else if (subplot === 'volatility') {
        layout[`yaxis${yaxisNum}`].title = 'Volatility';
      } else if (subplot === 'volume') {
        layout[`yaxis${yaxisNum}`].title = 'Volume';
      }

      currentBottom += subplotHeight;
    });

    // Add y2 axis for volume
    // Two modes: 'overlay' (layered on chart) or 'subgraph' (separate panel below)
    if (showVolume) {
      const volumeMode = this.chartSettings && this.chartSettings.settings.volumeMode === 'overlay' ? 'overlay' : 'subgraph';

      if (volumeMode === 'overlay') {
        // Overlay mode: volume bars layered beneath candles on same chart
        layout.yaxis2 = {
          title: '',
          gridcolor: 'transparent',
          color: '#a0a0a0',
          side: 'right',
          overlaying: 'y',  // Overlay on main chart
          showticklabels: false,
          tickformat: ',.0f',
          showgrid: false,
          fixedrange: false,
          layer: 'below traces'  // Volume bars appear below price candles
        };
      } else {
        // Subgraph mode: volume in separate panel below main chart
        // Adjust main chart domain to make room for volume
        layout.yaxis.domain = [0.28, 1];  // Main chart takes top 72%

        layout.yaxis2 = {
          title: '',
          gridcolor: '#404040',
          color: '#a0a0a0',
          side: 'right',
          domain: [0, 0.25],  // Volume subgraph takes bottom 25%
          showticklabels: true,
          tickfont: { size: 9 },
          tickformat: ',.0f',
          showgrid: true,
          gridwidth: 1,
          fixedrange: false
        };
      }
    }

    // Add rangebreaks for stocks (removes gaps for weekends and after-hours)
    // Skip for crypto which trades 24/7
    if (!isCrypto) {
      layout.xaxis.rangebreaks = [
        {
          bounds: ['sat', 'mon'],  // Hide weekends
          pattern: 'day of week'
        }
      ];

      // For intraday, also hide after-hours (4 PM to 9:30 AM next day)
      if (['1m', '5m', '15m', '30m', '1h'].includes(this.currentInterval)) {
        layout.xaxis.rangebreaks.push({
          bounds: [16, 9.5],  // 4:00 PM to 9:30 AM
          pattern: 'hour'
        });
      }
    }

    return layout;
  }

  /**
   * Get the y-axis assignment for an indicator type
   */
  getIndicatorYAxis(type) {
    const subplotIndicators = {
      'RSI': 'oscillator1',
      'STOCH': 'oscillator1',
      'ADX': 'oscillator1',
      'WILLR': 'oscillator2',
      'MACD': 'momentum',
      'CCI': 'momentum',
      'ATR': 'volatility',
      'OBV': 'volume'
    };

    const subplot = subplotIndicators[type];
    if (!subplot) {
      return 'y'; // Overlay on main chart
    }

    const activeSubplots = this.getActiveSubplots();
    const subplotIndex = activeSubplots.indexOf(subplot);

    if (subplotIndex === -1) {
      return 'y'; // Shouldn't happen, but fallback to main
    }

    return subplotIndex === 0 ? 'y2' : `y${subplotIndex + 2}`;
  }

  async initialize() {
    // console.log('Initializing ThinkorSwim-Style Trading Platform...');

    // Load saved panel states
    loadPanelStates();

    // Initialize resizable panels
    this.resizablePanels = initializeResizablePanels();

    // Initialize menu bar
    this.menuBar = initializeMenuBar();

    // Initialize watchlist with symbol selection callback AND WebSocket
    this.watchlist = initializeWatchlist((symbol) => {
      this.loadSymbol(symbol);
    }, this.socket);

    // Initialize news feed
    this.newsFeed = initializeNewsFeed();

    // Initialize active trader
    this.activeTrader = initializeActiveTrader();

    // Initialize chart settings
    this.chartSettings = initializeChartSettings();

    // Initialize chart controls
    this.initializeChartControls();

    // Initialize timeframe selector (NEW)
    this.timeframeSelector = new TimeframeSelector(
      this.timeframeRegistry,
      this.tickChartRegistry,
      (timeframeId) => this.handleTimeframeChange(timeframeId)
    );
    this.timeframeSelector.setCurrentTimeframe(this.currentTimeframeId);

    // Sync UI to match loaded settings from localStorage
    this.syncUIToSettings();

    // Initialize status bar
    this.initializeStatusBar();

    // Set up keyboard shortcuts
    this.initializeKeyboardShortcuts();

    // Subscribe to watchlist symbols if Socket.IO is already connected
    // (handles race condition where socket connected before watchlist was created)
    if (this.socket && this.socket.connected && this.watchlist) {
      console.log('🔔 Subscribing to watchlist symbols after initialization');
      this.watchlist.subscribeToAllSymbols();
    }

    // console.log('ThinkorSwim-Style Platform initialized successfully');
  }

  /**
   * Sync UI buttons and dropdowns to match current settings
   */
  syncUIToSettings() {
    // console.log(`🔄 Syncing UI to settings: ${this.currentPeriod}/${this.currentInterval}`);

    // Set timeframe dropdown to match current period:interval
    const timeframeSelect = document.getElementById('tos-timeframe-select');
    if (timeframeSelect) {
      const timeframeValue = `${this.currentPeriod}:${this.currentInterval}`;
      timeframeSelect.value = timeframeValue;

      // If the exact combination doesn't exist in dropdown, select closest match
      if (!timeframeSelect.value) {
        console.warn(`Timeframe ${timeframeValue} not found in dropdown, using default`);
        timeframeSelect.value = '1y:1d'; // Default fallback
      }
    }
  }

  initializeChartControls() {
    // Symbol input
    const symbolInput = document.getElementById('tos-symbol-input');
    if (symbolInput) {
      symbolInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          this.loadSymbol(e.target.value.trim().toUpperCase());
        }
      });
    }

    // Timeframe select (combines period and interval)
    const timeframeSelect = document.getElementById('tos-timeframe-select');
    if (timeframeSelect) {
      timeframeSelect.addEventListener('change', (e) => {
        // Parse value format: "period:interval" (e.g., "1y:1d")
        const [period, interval] = e.target.value.split(':');

        this.currentPeriod = period;
        this.currentInterval = interval;
        this.saveChartSettings();

        // console.log(`📅 Timeframe changed to: ${period} / ${interval}`);

        // Force recreate to recalculate indicators for new timeframe
        this.reloadChart(true);
        // Restart live updates with new interval timing
        this.startLiveUpdates();
      });
    }

    // Chart type select
    const chartTypeSelect = document.getElementById('tos-chart-type');
    if (chartTypeSelect) {
      chartTypeSelect.addEventListener('change', (e) => {
        this.updateChartType(e.target.value);
      });
    }

    // Load chart button
    const loadChartBtn = document.getElementById('btn-load-chart');
    if (loadChartBtn) {
      loadChartBtn.addEventListener('click', () => {
        const symbol = document.getElementById('tos-symbol-input')?.value.trim().toUpperCase();
        if (symbol) {
          this.loadSymbol(symbol);
        }
      });
    }

    // Add indicator button
    const addIndicatorBtn = document.getElementById('btn-add-indicator');
    if (addIndicatorBtn) {
      addIndicatorBtn.addEventListener('click', () => {
        this.showIndicatorPanel();
      });
    }

    // Drawing tools button
    const drawingToolsBtn = document.getElementById('btn-drawing-tools');
    if (drawingToolsBtn) {
      drawingToolsBtn.addEventListener('click', () => {
        if (window.enableDrawing) {
          window.enableDrawing();
        }
      });
    }

    // Compare button
    const compareBtn = document.getElementById('btn-compare');
    if (compareBtn) {
      compareBtn.addEventListener('click', () => {
        this.showCompareDialog();
      });
    }
  }

  async loadSymbol(symbol) {
    if (!symbol) return;

    console.log(`==================== LOADING NEW SYMBOL: ${symbol} ====================`);

    this.currentSymbol = symbol;
    state.currentSymbol = symbol;
    this.saveChartSettings();

    // Update symbol input
    const symbolInput = document.getElementById('tos-symbol-input');
    if (symbolInput) {
      symbolInput.value = symbol;
    }

    // Update active trader
    if (this.activeTrader) {
      this.activeTrader.setSymbol(symbol);
    }

    // Update news feed filter
    if (this.newsFeed) {
      this.newsFeed.setSymbolFilter(symbol);
    }

    // Load saved indicators for this symbol
    // console.log(`Loading saved indicators for ${symbol}`);
    this.loadSavedIndicators(symbol);
    // console.log(`After loadSavedIndicators: activeIndicators count = ${this.activeIndicators.length}`);
    if (this.activeIndicators.length > 0) {
      // console.log('Indicators to restore:', this.activeIndicators.map(i => i.type));
    }

    // === OLD PATH (1D ONLY) - COMMENTED OUT ===
    // Load the chart (force recreate when switching symbols)
    // console.log(`Calling reloadChart(true) for ${symbol}`);
    // await this.reloadChart(true);

    // === NEW PATH (ALL TIMEFRAMES) - USE TIMEFRAME REGISTRY ===
    // Load chart using the currently selected timeframe from dropdown
    console.log(`📊 Loading ${symbol} with timeframe: ${this.currentTimeframeId}`);
    if (this.timeframeRegistry) {
      // Load chart even if socket isn't connected yet (historical data doesn't need socket)
      // Socket will be used for live updates once connected
      await this.timeframeRegistry.switchTimeframe(this.currentTimeframeId, symbol, this.socket);
    } else {
      console.error('❌ Timeframe registry not initialized!');
    }

    // Apply chart settings after chart loads
    if (this.chartSettings) {
      this.chartSettings.applySettingsToChart();
    }

    // Start live updates for this symbol
    this.startLiveUpdates();

    // Start WebSocket streaming for real-time price updates
    await this.startStreaming(symbol);

    console.log(`==================== FINISHED LOADING SYMBOL: ${symbol} ====================`);
  }

  loadSavedIndicators(symbol) {
    try {
      // Load indicators globally (not per-symbol) so they persist across symbol changes
      const saved = localStorage.getItem(`tos-indicators-global`);
      if (saved) {
        this.activeIndicators = JSON.parse(saved);
        // console.log(`Loaded ${this.activeIndicators.length} saved indicators (global)`);
      } else {
        this.activeIndicators = [];
      }
    } catch (error) {
      console.error('Error loading saved indicators:', error);
      this.activeIndicators = [];
    }
  }

  saveIndicators() {
    try {
      // Save indicators globally (not per-symbol) so they persist across symbol changes
      const indicatorsToSave = this.activeIndicators.map(ind => ({
        id: ind.id,
        type: ind.type,
        name: ind.name,
        params: ind.params
      }));
      localStorage.setItem(`tos-indicators-global`, JSON.stringify(indicatorsToSave));
      // console.log(`Saved ${indicatorsToSave.length} indicators (global)`);
    } catch (error) {
      console.error('Error saving indicators:', error);
    }
  }

  // ============================================================
  // OLD 1D-ONLY RENDERING PATH - DEPRECATED
  // This function is no longer called (now using timeframe registry)
  // Kept here commented out for reference
  // ============================================================
  /*
  async reloadChart(forceRecreate = false) {
    if (!this.currentSymbol) return;

    console.log(`📊 Loading 1D chart for ${this.currentSymbol}`);

    try {
      // Fetch all historical data
      const fetchUrl = `/data/${this.currentSymbol}?period=${this.currentPeriod}&interval=${this.currentInterval}`;
      const response = await fetch(fetchUrl);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data || data.length === 0) {
        console.warn('⚠️ No data returned');
        return;
      }

      console.log(`✅ Received ${data.length} daily candles`);
      console.log(`📅 Date range: ${data[0].Date} to ${data[data.length - 1].Date}`);

      // Use all historical data
      state.chartData = data;

      // Render using canvas renderer with our chart data
      const success = await this.chartRenderer.render(data, this.currentSymbol);

      if (success) {
        console.log('✅ Chart rendered successfully');

        // Apply any ticker update that arrived during chart load
        if (this.lastTickerUpdate && this.lastTickerUpdate.symbol === this.currentSymbol) {
          console.log(`🔄 Applying pending ticker update: ${this.lastTickerUpdate.symbol} = $${this.lastTickerUpdate.price}`);
          const volumeBTC = this.lastTickerUpdate.volume_today || 0;
          this.chartRenderer.updateLivePrice(this.lastTickerUpdate.price, volumeBTC);
        }

        // Setup interactions
        // DISABLED: Plotly-specific interactions (using Canvas renderer now)
        // this.setupAdvancedInteractions();
        // this.removeCursorIcons();
        // this.injectHoverHideCSS();
        // this.setupTOSDataDisplay();
        // setupPlotlyHandlers('tos-plot');
        // this.setupChartScaling('tos-plot');
      }

    } catch (error) {
      console.error('❌ Error loading chart:', error);
    }
  }
  */

  async restoreSavedIndicators() {
    // console.log(`>>> restoreSavedIndicators called. activeIndicators count: ${this.activeIndicators?.length || 0}`);

    if (!this.activeIndicators || this.activeIndicators.length === 0) {
      // console.log('>>> No indicators to restore, returning early');
      return;
    }

    // console.log(`>>> Restoring ${this.activeIndicators.length} indicators for ${state.currentPeriod}/${state.currentInterval}`);
    // console.log('>>> Indicators:', this.activeIndicators.map(i => i.type));

    // Save the indicators to restore and clear the active array
    const indicatorsToRestore = [...this.activeIndicators];
    this.activeIndicators = [];
    // console.log(`>>> Cleared activeIndicators array, will restore ${indicatorsToRestore.length} indicators`);

    // Re-add each saved indicator with current period/interval
    for (const indicator of indicatorsToRestore) {
      try {
        // console.log(`>>> Re-adding ${indicator.type} with period=${state.currentPeriod}, interval=${state.currentInterval}`);
        await this.addIndicatorToChart(indicator.type, indicator.params, indicator.id);
        // console.log(`>>> Successfully added ${indicator.type}`);
      } catch (error) {
        console.error(`>>> Error restoring indicator ${indicator.type}:`, error);
      }
    }

    // console.log(`>>> Finished restoring ${indicatorsToRestore.length} indicators`);
    // console.log(`>>> Final activeIndicators count: ${this.activeIndicators.length}`);
  }

  // REMOVED: refreshAllIndicators() - No historical data fetching, indicators update from WebSocket data

  /**
   * Setup advanced chart interactions:
   * - Mouse drag to pan (default)
   * - Ctrl+drag to zoom
   * - Click and drag on axes to zoom specific axis
   */
  setupAdvancedInteractions() {
    const plotDiv = document.getElementById('tos-plot');
    if (!plotDiv) return;

    // Ensure chart starts in pan mode
    Plotly.relayout(plotDiv, { dragmode: this.currentDragMode });

    let isDraggingAxis = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragAxis = null;
    let startRange = null;
    let rafId = null;  // RequestAnimationFrame ID for throttling
    let lastUpdateTime = 0;  // Track last update time for throttling
    const updateInterval = 16;  // Minimum milliseconds between updates (~60fps for smooth axis zoom)
    let currentMouseEvent = null;  // Store current mouse position

    // Only add Ctrl key listener once (not on every chart load)
    if (!this.ctrlKeyListenerAdded) {
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Control') {
          const plotDiv = document.getElementById('tos-plot');
          if (!plotDiv) return;

          // Toggle mode
          this.currentDragMode = this.currentDragMode === 'pan' ? 'zoom' : 'pan';
          Plotly.relayout(plotDiv, { dragmode: this.currentDragMode });
          e.preventDefault();  // Prevent multiple triggers
        }
      });
      this.ctrlKeyListenerAdded = true;
    }

    // Listen for mousedown to check if clicking on axis areas
    const checkAxisDrag = (e) => {
      // Get plot area dimensions
      const plotRect = plotDiv.getBoundingClientRect();

      // Get click position relative to plot
      const clickX = e.clientX - plotRect.left;
      const clickY = e.clientY - plotRect.top;

      // console.log(`🖱️ Click detected at X:${clickX}, Y:${clickY}, plotHeight:${plotRect.height}, plotWidth:${plotRect.width}`);

      // X-axis area (bottom of chart) - LARGE bottom area for easier access
      const xAxisTop = plotRect.height - 250;  // Bottom 250px for x-axis (much larger!)
      const isXAxisClick = clickY >= xAxisTop && clickY <= plotRect.height && clickX >= 0 && clickX <= (plotRect.width - 100);

      // Y-axis area (right side of chart) - ENTIRE right area
      const yAxisLeft = plotRect.width - 100;  // Right 100px for y-axis
      const isYAxisClick = clickX >= yAxisLeft && clickX <= plotRect.width && clickY >= 0 && clickY <= (plotRect.height - 250);

      // console.log(`🎯 xAxisTop: ${xAxisTop}, isXAxisClick: ${isXAxisClick}, isYAxisClick: ${isYAxisClick}`);

      if (isXAxisClick || isYAxisClick) {
        // console.log(`✅ AXIS DRAG STARTED: ${isXAxisClick ? 'X-AXIS (HORIZONTAL)' : 'Y-AXIS (VERTICAL)'}`);

        isDraggingAxis = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        dragAxis = isXAxisClick ? 'x' : 'y';

        // Store current range
        if (dragAxis === 'x') {
          startRange = [...plotDiv.layout.xaxis.range];
        } else {
          startRange = [...plotDiv.layout.yaxis.range];
        }

        // COMPLETELY prevent Plotly's default panning on axis
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        // Change cursor to indicate dragging
        document.body.style.cursor = dragAxis === 'x' ? 'ew-resize' : 'ns-resize';

        // Only attach mousemove listener when actively dragging axis
        document.addEventListener('mousemove', handleAxisDrag, true);  // Use capture to intercept early
        document.addEventListener('mouseup', stopAxisDrag, true);

        return false;  // Additional prevention
      }
    };

    const handleAxisDrag = (e) => {
      if (!isDraggingAxis || !startRange) return;

      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();

      // Store current mouse event
      currentMouseEvent = e;

      // Use requestAnimationFrame to throttle updates for smooth performance
      if (rafId) return;  // Skip if already have a pending update

      rafId = requestAnimationFrame(() => {
        if (!currentMouseEvent || !isDraggingAxis) {
          rafId = null;
          return;
        }

        const now = Date.now();
        if (now - lastUpdateTime < updateInterval) {
          rafId = null;
          currentMouseEvent = null;
          return;
        }

        if (dragAxis === 'x') {
          // Horizontal drag on x-axis - EXPAND/COMPRESS by zooming time range
          const deltaX = currentMouseEvent.clientX - dragStartX;

          console.log(`📊 X-AXIS DRAG: deltaX=${deltaX}`);

          // Only update if drag is significant
          if (Math.abs(deltaX) < 2) {
            rafId = null;
            currentMouseEvent = null;
            return;
          }

          const plotWidth = plotDiv.offsetWidth;

          // Get current x-axis range
          const currentRange = plotDiv.layout.xaxis.range;
          if (!currentRange || currentRange.length !== 2) {
            rafId = null;
            currentMouseEvent = null;
            return;
          }

          // Convert to timestamps
          const start0 = new Date(currentRange[0]).getTime();
          const start1 = new Date(currentRange[1]).getTime();
          const rangeDiff = start1 - start0;

          // Drag RIGHT = expand candles (zoom in, fewer candles visible)
          // Drag LEFT = compress candles (zoom out, more candles visible)
          const sensitivity = 2;
          const zoomFactor = 1 - (deltaX / plotWidth) * sensitivity;
          const center = (start0 + start1) / 2;
          const newRange = rangeDiff * zoomFactor;

          const newStart = new Date(center - newRange / 2);
          const newEnd = new Date(center + newRange / 2);

          console.log(`📊 Zoom factor: ${zoomFactor.toFixed(3)}, new range: ${newRange / 1000 / 60} minutes`);

          Plotly.relayout(plotDiv, {
            'xaxis.range': [newStart, newEnd],
            'xaxis.autorange': false
          });

          // Update drag start for next frame
          dragStartX = currentMouseEvent.clientX;
        } else if (dragAxis === 'y') {
          // Vertical drag on y-axis
          const deltaY = currentMouseEvent.clientY - dragStartY;

          // Only update if drag is significant (reduces relayout calls)
          if (Math.abs(deltaY) < 2) {
            rafId = null;
            currentMouseEvent = null;
            return;
          }

          const plotHeight = plotDiv.offsetHeight;
          const rangeDiff = startRange[1] - startRange[0];

          // Drag up = zoom out (lengthen), drag down = zoom in (shorten)
          const zoomFactor = 1 - (deltaY / plotHeight) * 2;
          const center = (startRange[0] + startRange[1]) / 2;
          const newRange = rangeDiff * zoomFactor;

          Plotly.relayout(plotDiv, {
            'yaxis.range': [center - newRange / 2, center + newRange / 2],
            'yaxis.autorange': false
          });
        }

        lastUpdateTime = Date.now();  // Update timestamp after relayout
        currentMouseEvent = null;  // Clear stored event
        rafId = null;  // Clear RAF ID after update
      });
    };

    const stopAxisDrag = (e) => {
      if (isDraggingAxis) {
        if (e) {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
        }
        // Reset cursor
        document.body.style.cursor = '';
        // Cancel any pending RAF
        if (rafId) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        // Clean up event listeners (use capture: true to match how they were added)
        document.removeEventListener('mousemove', handleAxisDrag, true);
        document.removeEventListener('mouseup', stopAxisDrag, true);
      }
      isDraggingAxis = false;
      dragAxis = null;
      startRange = null;
      currentMouseEvent = null;
    };

    // Attach mousedown with capture phase to intercept before Plotly
    plotDiv.addEventListener('mousedown', checkAxisDrag, true);
  }

  /**
   * Create 100% custom crosshair with pure CSS/JS (no Plotly)
   */
  removeCursorIcons() {
    const plotDiv = document.getElementById('tos-plot');
    if (!plotDiv) return;

    // Store reference to this for use in event handlers
    const self = this;

    // Remove ALL Plotly images
    const removeAllImages = () => {
      plotDiv.querySelectorAll('image').forEach(img => img.remove());
    };
    setInterval(removeAllImages, 50);

    // Create custom crosshair lines
    const vLine = document.createElement('div');
    vLine.style.cssText = `
      position: absolute;
      width: 1px;
      height: 100%;
      background: #888;
      pointer-events: none;
      display: none;
      z-index: 9999;
    `;

    const hLine = document.createElement('div');
    hLine.style.cssText = `
      position: absolute;
      width: 100%;
      height: 1px;
      background: #888;
      pointer-events: none;
      display: none;
      z-index: 9999;
    `;

    plotDiv.style.position = 'relative';
    plotDiv.appendChild(vLine);
    plotDiv.appendChild(hLine);

    // Show/move crosshair on mouse move and update OHLCV data
    plotDiv.addEventListener('mousemove', (e) => {
      const rect = plotDiv.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      vLine.style.display = 'block';
      hLine.style.display = 'block';
      vLine.style.left = x + 'px';
      hLine.style.top = y + 'px';

      // Convert mouse X position to data coordinates to find the candle
      if (plotDiv.layout && plotDiv.data && plotDiv.data.length > 0) {
        try {
          // Get the x-axis range
          const xaxis = plotDiv.layout.xaxis;
          if (!xaxis || !xaxis.range) return;

          // Calculate which data point we're hovering over
          const plotWidth = rect.width;
          const xRange = xaxis.range;

          // Find the candlestick trace
          const candlestickTrace = plotDiv.data.find(t => t.type === 'candlestick');
          if (!candlestickTrace || !candlestickTrace.x) return;

          // Convert pixel position to date index
          const xMin = new Date(xRange[0]).getTime();
          const xMax = new Date(xRange[1]).getTime();
          const mouseTime = xMin + (x / plotWidth) * (xMax - xMin);

          // Find the closest candle
          let closestIdx = -1;
          let minDistance = Infinity;

          for (let i = 0; i < candlestickTrace.x.length; i++) {
            const candleTime = new Date(candlestickTrace.x[i]).getTime();
            const distance = Math.abs(candleTime - mouseTime);
            if (distance < minDistance) {
              minDistance = distance;
              closestIdx = i;
            }
          }

          // Update OHLCV data if we found a candle
          if (closestIdx >= 0 && window.displayOHLC) {
            const date = candlestickTrace.x[closestIdx];
            const open = candlestickTrace.open[closestIdx];
            const high = candlestickTrace.high[closestIdx];
            const low = candlestickTrace.low[closestIdx];
            const close = candlestickTrace.close[closestIdx];

            // Get volume
            let volume = null;
            const volumeTrace = plotDiv.data.find(t => t.type === 'bar' && t.yaxis === 'y2');
            if (volumeTrace && volumeTrace.y && volumeTrace.y[closestIdx] !== undefined) {
              volume = volumeTrace.y[closestIdx];
            }

            // Get previous close
            const previousClose = closestIdx > 0 ? candlestickTrace.close[closestIdx - 1] : null;

            // Update the data bar
            window.displayOHLC(date, open, high, low, close, volume, false, previousClose);
          }
        } catch (err) {
          // Silently ignore errors during mousemove
        }
      }
    });

    // Hide on mouse leave (and reset data bar)
    plotDiv.addEventListener('mouseleave', () => {
      vLine.style.display = 'none';
      hLine.style.display = 'none';
      if (window.updateTOSDataBar) {
        window.updateTOSDataBar();
      }
    });

    console.log('✅ Custom crosshair active');
  }

  injectHoverHideCSS() {
    // Check if style already exists
    if (document.getElementById('plotly-hover-hide-style')) return;

    const style = document.createElement('style');
    style.id = 'plotly-hover-hide-style';
    style.textContent = `
      /* Hide only the hover text boxes, keep spike lines */
      .hoverlayer g.hovertext {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
      }

      /* Ensure spike lines are visible */
      .hoverlayer line.spikeline {
        display: block !important;
        opacity: 1 !important;
        visibility: visible !important;
      }

      /* Hide the hover info on axes */
      .infolayer g.g-xtitle,
      .infolayer g.g-ytitle {
        pointer-events: none;
      }
    `;
    document.head.appendChild(style);
    // console.log('Hover hide CSS injected');
  }

  setupTOSDataDisplay() {
    const plotDiv = document.getElementById('tos-plot');
    if (!plotDiv) {
      console.error('Plot div not found');
      return;
    }

    // Create data display bar if it doesn't exist
    let dataBar = document.getElementById('tos-chart-data-bar');
    if (!dataBar) {
      dataBar = document.createElement('div');
      dataBar.id = 'tos-chart-data-bar';
      dataBar.style.cssText = `
        position: absolute;
        top: 10px;
        left: 10px;
        right: 10px;
        height: 28px;
        background: rgba(40, 40, 40, 0.95);
        border: 1px solid #555555;
        border-radius: 4px;
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        font-weight: 500;
        padding: 6px 16px;
        display: flex;
        gap: 24px;
        align-items: center;
        z-index: 1000;
        pointer-events: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      `;

      // Find the chart area container and make it relative
      const chartArea = document.querySelector('.tos-chart-area');
      if (chartArea) {
        chartArea.style.position = 'relative';
        chartArea.appendChild(dataBar);
        // console.log('Data bar added to chart area');
      } else {
        console.error('Chart area not found');
        return;
      }
    }

    // Function to display OHLC data
    const displayOHLC = (date, open, high, low, close, volume = null, isLatest = false, previousClose = null) => {
      const dataBarEl = document.getElementById('tos-chart-data-bar');
      if (!dataBarEl) return;

      // Validate OHLC data - return early if any values are undefined or NaN
      if (open === undefined || high === undefined || low === undefined || close === undefined ||
          isNaN(open) || isNaN(high) || isNaN(low) || isNaN(close)) {
        console.warn('Invalid OHLC data, skipping display:', {open, high, low, close});
        return;
      }

      // For latest candle, show change from previous close (real-time price change)
      // For historical candles, show change within that candle (open to close)
      const referencePrice = (isLatest && previousClose !== null) ? previousClose : open;
      const change = close - referencePrice;
      const changePercent = ((change / referencePrice) * 100).toFixed(2);
      const changeColor = change >= 0 ? '#00c851' : '#ff4444';
      const prefix = isLatest ? '<span style="color: #00bfff; margin-right: 10px;">LATEST:</span>' : '';

      // Format date to be more readable
      let formattedDate = date;
      if (typeof date === 'string') {
        try {
          const d = new Date(date);
          // Format as: "2025-10-20 16:10:02" (more readable)
          formattedDate = d.toISOString().replace('T', ' ').replace('Z', '').substring(0, 19);
        } catch (e) {
          // Keep original if parsing fails
          formattedDate = date;
        }
      }

      // Format volume with commas
      const volumeStr = volume !== null && volume !== undefined
        ? `<span style="color: #a0a0a0;">Vol: <span style="color: #e0e0e0;">${volume.toLocaleString()}</span></span>`
        : '';

      dataBarEl.innerHTML = `
        ${prefix}
        <span style="color: #a0a0a0;">Date: <span style="color: #e0e0e0;">${formattedDate}</span></span>
        <span style="color: #a0a0a0;">O: <span style="color: #e0e0e0;">${open.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">H: <span style="color: #e0e0e0;">${high.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">L: <span style="color: #e0e0e0;">${low.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">C: <span style="color: #e0e0e0;">${close.toFixed(2)}</span></span>
        ${volumeStr}
        <span style="color: ${changeColor};">${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent}%)</span>
      `;
    };

    // Function to show latest day's data
    const showLatestData = () => {
      const plotDiv = document.getElementById('tos-plot');
      if (!plotDiv || !plotDiv.data) {
        if (!state.chartData || state.chartData.length === 0) return;
        const latest = state.chartData[state.chartData.length - 1];
        const previousClose = state.chartData.length > 1 ? state.chartData[state.chartData.length - 2].Close : null;
        displayOHLC(
          latest.Date,
          latest.Open,
          latest.High,
          latest.Low,
          latest.Close,
          latest.Volume || null,
          true,
          previousClose
        );
        return;
      }

      // Get data from the chart trace (most up-to-date for live candles)
      const candlestickTrace = plotDiv.data.find(trace => trace.type === 'candlestick');
      if (!candlestickTrace || candlestickTrace.x.length === 0) return;

      const lastIndex = candlestickTrace.x.length - 1;
      const date = candlestickTrace.x[lastIndex];
      const open = candlestickTrace.open[lastIndex];
      const high = candlestickTrace.high[lastIndex];
      const low = candlestickTrace.low[lastIndex];
      const close = candlestickTrace.close[lastIndex];

      // Validate candle data before displaying
      if (open === undefined || high === undefined || low === undefined || close === undefined) {
        console.warn('Incomplete candle data at index', lastIndex);
        return;
      }

      // Get volume from volume trace (same way we get OHLC from candlestick trace)
      let volume = null;
      const volumeTrace = plotDiv.data.find(trace => trace.type === 'bar' && trace.yaxis === 'y2');
      if (volumeTrace && volumeTrace.y && volumeTrace.y[lastIndex] !== undefined) {
        volume = volumeTrace.y[lastIndex];
      }

      // Get previous close for change calculation
      const previousClose = lastIndex > 0 ? candlestickTrace.close[lastIndex - 1] : null;

      displayOHLC(
        date,
        open,
        high,
        low,
        close,
        volume,
        true,  // Mark as latest
        previousClose
      );
    };

    // Initialize global flag for tracking user candle selection
    if (typeof window.userSelectedCandle === 'undefined') {
      window.userSelectedCandle = false;
    }

    // Add hover event to update data bar based on vertical crosshair position
    plotDiv.on('plotly_hover', (eventData) => {
      if (!eventData.points || eventData.points.length === 0) return;

      // Get the x-coordinate of the hover (this is from the vertical crosshair)
      const point = eventData.points[0];
      const xValue = point.x;  // This is the x-axis value (date)

      // Find the candlestick trace (always trace 0)
      const candlestickTrace = plotDiv.data.find(trace => trace.type === 'candlestick');
      if (!candlestickTrace) return;

      // Find the index in the candlestick data that matches this x value
      let pointIndex = -1;
      for (let i = 0; i < candlestickTrace.x.length; i++) {
        if (candlestickTrace.x[i] === xValue ||
            new Date(candlestickTrace.x[i]).getTime() === new Date(xValue).getTime()) {
          pointIndex = i;
          break;
        }
      }

      // If we couldn't find exact match, use the point index from the event
      if (pointIndex === -1) {
        pointIndex = point.pointIndex;
      }

      if (pointIndex >= 0 && pointIndex < candlestickTrace.x.length) {
        const open = candlestickTrace.open[pointIndex];
        const high = candlestickTrace.high[pointIndex];
        const low = candlestickTrace.low[pointIndex];
        const close = candlestickTrace.close[pointIndex];
        const date = candlestickTrace.x[pointIndex];

        // Get volume from state.chartData
        const volume = state.chartData && state.chartData[pointIndex]
          ? state.chartData[pointIndex].Volume
          : null;

        // Check if this is the last candle
        const isLatest = pointIndex === (candlestickTrace.x.length - 1);

        // If user hovers over the latest candle, reset the flag so auto-updates resume
        // Otherwise, mark that user has selected a different candle
        window.userSelectedCandle = !isLatest;

        displayOHLC(date, open, high, low, close, volume, isLatest);
      }
    });

    // Return to latest data on unhover
    // DISABLED: Let the hovered candle stay displayed until user hovers over another candle
    // plotDiv.on('plotly_unhover', () => {
    //   showLatestData();
    // });

    // Initialize with latest data
    showLatestData();

    // Store functions globally so other parts can use them
    window.updateTOSDataBar = showLatestData;
    window.displayOHLC = displayOHLC;
  }

  setupChartScaling(plotId) {
    const plotDiv = document.getElementById(plotId);
    if (!plotDiv) return;

    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let originalXRange = null;
    let originalYRange = null;

    plotDiv.addEventListener('mousedown', (e) => {
      // Only activate on right-click or Ctrl+click
      if (e.button === 2 || (e.button === 0 && e.ctrlKey)) {
        e.preventDefault();
        isDragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;

        const layout = plotDiv._fullLayout;
        originalXRange = [...layout.xaxis.range];
        originalYRange = [...layout.yaxis.range];

        plotDiv.style.cursor = 'ew-resize';
      }
    });

    plotDiv.addEventListener('mousemove', (e) => {
      if (!isDragging) return;

      const deltaX = e.clientX - dragStartX;
      const deltaY = e.clientY - dragStartY;

      const layout = plotDiv._fullLayout;

      // Horizontal scaling (zoom time axis)
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        plotDiv.style.cursor = 'ew-resize';

        // Calculate zoom factor based on horizontal drag
        const zoomFactor = 1 - (deltaX / 500); // Sensitivity adjustment

        if (originalXRange && originalXRange[0] !== undefined) {
          const xCenter = (originalXRange[0] + originalXRange[1]) / 2;
          const xSpan = originalXRange[1] - originalXRange[0];
          const newXSpan = xSpan * zoomFactor;

          const newXRange = [
            xCenter - newXSpan / 2,
            xCenter + newXSpan / 2
          ];

          Plotly.relayout(plotId, {
            'xaxis.range': newXRange
          });
        }
      }
      // Vertical scaling (zoom price axis)
      else {
        plotDiv.style.cursor = 'ns-resize';

        // Calculate zoom factor based on vertical drag
        const zoomFactor = 1 + (deltaY / 500); // Sensitivity adjustment

        if (originalYRange && originalYRange[0] !== undefined) {
          const yCenter = (originalYRange[0] + originalYRange[1]) / 2;
          const ySpan = originalYRange[1] - originalYRange[0];
          const newYSpan = ySpan * zoomFactor;

          const newYRange = [
            yCenter - newYSpan / 2,
            yCenter + newYSpan / 2
          ];

          Plotly.relayout(plotId, {
            'yaxis.range': newYRange
          });
        }
      }
    });

    const endDrag = () => {
      if (isDragging) {
        isDragging = false;
        plotDiv.style.cursor = 'default';
      }
    };

    plotDiv.addEventListener('mouseup', endDrag);
    plotDiv.addEventListener('mouseleave', endDrag);

    // Prevent context menu on right-click
    plotDiv.addEventListener('contextmenu', (e) => {
      if (isDragging) {
        e.preventDefault();
      }
    });

    // console.log('TOS-style chart scaling enabled (Right-click + drag or Ctrl+click + drag)');
  }

  updateChartType(chartType) {
    // Update chart type (would need to reload chart with new type)
    // console.log(`Chart type changed to: ${chartType}`);
    // This would be implemented to change the trace type
  }

  showIndicatorPanel() {
    const indicators = [
      { name: 'SMA', fullName: 'Simple Moving Average', type: 'SMA', params: [{name: 'period', label: 'Period', default: 20, min: 1, max: 200}], category: 'Trend' },
      { name: 'EMA', fullName: 'Exponential Moving Average', type: 'EMA', params: [{name: 'period', label: 'Period', default: 20, min: 1, max: 200}], category: 'Trend' },
      { name: 'RSI', fullName: 'Relative Strength Index', type: 'RSI', params: [{name: 'period', label: 'Period', default: 14, min: 2, max: 50}], category: 'Momentum' },
      { name: 'MACD', fullName: 'Moving Average Convergence Divergence', type: 'MACD', params: [{name: 'fast', label: 'Fast Period', default: 12}, {name: 'slow', label: 'Slow Period', default: 26}, {name: 'signal', label: 'Signal Period', default: 9}], category: 'Momentum' },
      { name: 'BB', fullName: 'Bollinger Bands', type: 'BB', params: [{name: 'period', label: 'Period', default: 20}, {name: 'std', label: 'Std Dev', default: 2}], category: 'Volatility' },
      { name: 'VWAP', fullName: 'Volume-Weighted Average Price', type: 'VWAP', params: [], category: 'Volume' },
      { name: 'STOCH', fullName: 'Stochastic Oscillator', type: 'STOCH', params: [{name: 'period', label: 'Period', default: 14}, {name: 'k_period', label: '%K Period', default: 3}, {name: 'd_period', label: '%D Period', default: 3}], category: 'Momentum' },
      { name: 'ATR', fullName: 'Average True Range', type: 'ATR', params: [{name: 'period', label: 'Period', default: 14}], category: 'Volatility' },
      { name: 'ADX', fullName: 'Average Directional Index', type: 'ADX', params: [{name: 'period', label: 'Period', default: 14}], category: 'Trend' },
      { name: 'CCI', fullName: 'Commodity Channel Index', type: 'CCI', params: [{name: 'period', label: 'Period', default: 20}], category: 'Momentum' },
      { name: 'OBV', fullName: 'On Balance Volume', type: 'OBV', params: [], category: 'Volume' },
      { name: 'WILLR', fullName: 'Williams %R', type: 'WILLR', params: [{name: 'period', label: 'Period', default: 14}], category: 'Momentum' },
      { name: 'PSAR', fullName: 'Parabolic SAR', type: 'PSAR', params: [{name: 'af_start', label: 'AF Start', default: 0.02}, {name: 'af_increment', label: 'AF Increment', default: 0.02}, {name: 'af_max', label: 'AF Max', default: 0.2}], category: 'Trend' }
    ];

    // Show modal
    const modal = document.getElementById('indicator-modal');
    modal.style.display = 'block';

    // Populate indicator list
    const listContainer = document.getElementById('indicator-list');
    listContainer.innerHTML = '';

    indicators.forEach(ind => {
      const btn = document.createElement('button');
      btn.textContent = `${ind.name} - ${ind.fullName}`;
      btn.style.cssText = 'padding: 12px; background: var(--tos-bg-tertiary); color: var(--tos-text-primary); border: 1px solid var(--tos-border-color); border-radius: 4px; cursor: pointer; text-align: left; font-size: 13px;';
      btn.onmouseover = () => btn.style.background = 'var(--tos-bg-hover)';
      btn.onmouseout = () => btn.style.background = 'var(--tos-bg-tertiary)';
      btn.onclick = () => this.selectIndicator(ind);
      listContainer.appendChild(btn);
    });

    // Search functionality
    const searchInput = document.getElementById('indicator-search');
    searchInput.oninput = (e) => {
      const query = e.target.value.toLowerCase();
      const buttons = listContainer.querySelectorAll('button');
      buttons.forEach(btn => {
        const text = btn.textContent.toLowerCase();
        btn.style.display = text.includes(query) ? 'block' : 'none';
      });
    };

    // Wire up Clear All Indicators button
    const clearBtn = document.getElementById('btn-clear-indicators');
    if (clearBtn) {
      clearBtn.onclick = () => this.clearAllIndicators();
    }
  }

  clearAllIndicators() {
    // console.log('Clearing all indicators');

    // Clear the active indicators array
    this.activeIndicators = [];

    // Save empty indicators list
    this.saveIndicators();

    // Force recreate chart without indicators
    this.reloadChart(true);

    // Update UI
    this.updateCurrentIndicatorsUI();

    // Close the modal
    document.getElementById('indicator-modal').style.display = 'none';

    // Show success message
    const notif = document.getElementById('notifications-area');
    notif.textContent = '✓ All indicators cleared';
    notif.style.color = 'var(--tos-accent-green)';
    setTimeout(() => notif.textContent = '', 3000);
  }

  selectIndicator(indicator) {
    const paramsDiv = document.getElementById('indicator-params');
    const titleEl = document.getElementById('param-title');
    const inputsDiv = document.getElementById('param-inputs');

    titleEl.textContent = `${indicator.name} - ${indicator.fullName}`;
    inputsDiv.innerHTML = '';

    // Create parameter inputs
    indicator.params.forEach(param => {
      const label = document.createElement('label');
      label.textContent = param.label;
      label.style.cssText = 'display: block; margin-bottom: 5px; color: var(--tos-text-primary); font-size: 13px;';

      const input = document.createElement('input');
      input.type = 'number';
      input.value = param.default;
      input.min = param.min || 0;
      input.max = param.max || 1000;
      input.step = param.step || (param.default < 1 ? 0.01 : 1);
      input.dataset.paramName = param.name;
      input.style.cssText = 'width: 100%; padding: 8px; margin-bottom: 15px; background: var(--tos-bg-secondary); border: 1px solid var(--tos-border-color); color: var(--tos-text-primary); border-radius: 4px; font-size: 14px;';

      inputsDiv.appendChild(label);
      inputsDiv.appendChild(input);
    });

    paramsDiv.style.display = 'block';

    // Add confirm button handler
    document.getElementById('btn-add-indicator-confirm').onclick = () => {
      const params = {};
      const inputs = inputsDiv.querySelectorAll('input');
      inputs.forEach(input => {
        params[input.dataset.paramName] = parseFloat(input.value);
      });

      this.addIndicatorToChart(indicator.type, params);
      document.getElementById('indicator-modal').style.display = 'none';
      paramsDiv.style.display = 'none';
    };
  }

  async addIndicatorToChart(type, params, existingId = null) {
    if (!this.currentSymbol) {
      alert('Please load a chart first');
      return;
    }

    try {
      const response = await fetch('/indicators', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: this.currentSymbol,
          period: state.currentPeriod || '1y',
          interval: state.currentInterval || '1d',
          indicators: [{ type, params }]
        })
      });

      const data = await response.json();
      // console.log(`Received data for ${type}:`, Object.keys(data));
      // console.log(`Full response data:`, data);

      // Track the indicator BEFORE adding traces (so layout calculation works)
      const indicatorId = existingId || crypto.randomUUID();
      const tempIndicator = {
        id: indicatorId,
        type,
        name: type,
        params,
        traceIndices: [] // Will be filled after adding traces
      };

      // console.log(`Tracking indicator ${type} (id: ${indicatorId}), active count before: ${this.activeIndicators.length}`);
      this.activeIndicators.push(tempIndicator);
      // console.log(`Active indicators count after: ${this.activeIndicators.length}`);

      // Recalculate layout with new subplot if needed
      const newLayout = this.calculateChartLayout();
      await Plotly.relayout('tos-plot', newLayout);

      // Get the correct y-axis for this indicator
      const yaxis = this.getIndicatorYAxis(type);
      // console.log(`Indicator ${type} will use yaxis: ${yaxis}`);

      // Add indicator trace to chart
      const plotDiv = document.getElementById('tos-plot');
      if (!plotDiv) {
        console.error('Plot div not found!');
        return;
      }
      if (!plotDiv.data) {
        console.error('Plot data not initialized!');
        return;
      }
      // console.log(`Current plot has ${plotDiv.data.length} traces`);
      const newTraces = [];

      // Special handling for multi-line indicators
      if (type === 'MACD') {
        // MACD has 3 components: MACD line, Signal line, and Histogram
        if (data.MACD) {
          newTraces.push({
            x: data.dates,
            y: data.MACD,
            name: 'MACD',
            type: 'scatter',
            mode: 'lines',
            line: { width: 2, color: '#2196F3' },
            yaxis: yaxis !== 'y' ? yaxis : undefined,
            xaxis: yaxis !== 'y' ? 'x' : undefined
          });
        }
        if (data.MACD_signal) {
          newTraces.push({
            x: data.dates,
            y: data.MACD_signal,
            name: 'Signal',
            type: 'scatter',
            mode: 'lines',
            line: { width: 2, color: '#FF9800' },
            yaxis: yaxis !== 'y' ? yaxis : undefined,
            xaxis: yaxis !== 'y' ? 'x' : undefined
          });
        }
        if (data.MACD_histogram) {
          newTraces.push({
            x: data.dates,
            y: data.MACD_histogram,
            name: 'Histogram',
            type: 'bar',
            marker: { color: '#4CAF50' },
            yaxis: yaxis !== 'y' ? yaxis : undefined,
            xaxis: yaxis !== 'y' ? 'x' : undefined
          });
        }
      } else if (data[`${type}`]) {
        // Single-line indicator with exact type match (RSI, OBV, VWAP, ATR, CCI, WILLR, PSAR)
        // console.log(`Creating single-line trace for ${type} (exact match)`);
        // console.log(`Data length: ${data[type].length}, yaxis: ${yaxis}`);
        // console.log(`First 5 data points:`, data[type].slice(0, 5));

        // Choose color based on indicator type
        let lineColor = undefined;
        if (type === 'OBV') {
          lineColor = '#FF6B6B'; // Bright red for OBV
        } else if (type === 'VWAP') {
          lineColor = '#4ECDC4'; // Bright teal for VWAP
        }

        const trace = {
          x: data.dates,
          y: data[type],
          name: `${type}(${JSON.stringify(params).slice(1,-1)})`,
          type: 'scatter',
          mode: 'lines',
          line: { width: 2, color: lineColor }
        };

        // Assign to correct y-axis if it's a subplot indicator
        if (yaxis !== 'y') {
          trace.yaxis = yaxis;
          trace.xaxis = 'x';
          // console.log(`Assigned to subplot: yaxis=${trace.yaxis}, xaxis=${trace.xaxis}`);
        }

        newTraces.push(trace);
        // console.log(`Trace created:`, trace);
      } else if (type === 'SMA' || type === 'EMA') {
        // SMA/EMA with period suffix (SMA_20, EMA_20)
        const period_val = params.period || (type === 'SMA' ? 20 : 20);
        const dataKey = `${type}_${period_val}`;
        // console.log(`Creating ${type} trace with key: ${dataKey}`);

        if (data[dataKey]) {
          const trace = {
            x: data.dates,
            y: data[dataKey],
            name: `${type}(${period_val})`,
            type: 'scatter',
            mode: 'lines',
            line: { width: 2 }
          };

          // These are overlays on main chart (yaxis = 'y')
          newTraces.push(trace);
          // console.log(`Trace created for ${dataKey}:`, trace);
        } else {
          console.error(`Expected key ${dataKey} not found in data`);
        }
      } else {
        // Handle other multi-line indicators (BB, STOCH, ADX, PSAR, etc.)
        // console.log(`Creating multi-line traces for ${type}, available keys:`, Object.keys(data));
        Object.keys(data).forEach(key => {
          if (key !== 'dates' && data[key]) {
            // console.log(`Adding trace for key: ${key}`);
            const trace = {
              x: data.dates,
              y: data[key],
              name: key,
              type: 'scatter',
              mode: 'lines',
              line: { width: 2 }
            };

            // Assign to correct y-axis if it's a subplot indicator
            if (yaxis !== 'y') {
              trace.yaxis = yaxis;
              trace.xaxis = 'x';
            }

            newTraces.push(trace);
          }
        });
        // console.log(`Created ${newTraces.length} traces for ${type}`);
      }

      const plotData = plotDiv.data;
      const startIndex = plotData.length;

      console.log(`Adding ${newTraces.length} traces for ${type}, starting at index ${startIndex}`);
      await Plotly.addTraces('tos-plot', newTraces);

      // Update trace indices
      const traceIndices = [];
      for (let i = 0; i < newTraces.length; i++) {
        traceIndices.push(startIndex + i);
      }
      tempIndicator.traceIndices = traceIndices;
      tempIndicator.name = newTraces.length === 1 ? newTraces[0].name : type;
      // console.log(`Indicator ${type} added with trace indices:`, traceIndices);

      // Update the current indicators UI
      this.updateCurrentIndicatorsUI();

      // Save indicators to localStorage
      this.saveIndicators();

      // Show success message
      const notif = document.getElementById('notifications-area');
      notif.textContent = `✓ ${type} added`;
      notif.style.color = 'var(--tos-accent-green)';
      setTimeout(() => notif.textContent = '', 3000);

    } catch (error) {
      console.error('Error adding indicator:', error);
      alert('Error adding indicator: ' + error.message);
    }
  }

  updateCurrentIndicatorsUI() {
    const section = document.getElementById('current-indicators-section');
    const list = document.getElementById('current-indicators-list');

    if (this.activeIndicators.length === 0) {
      section.style.display = 'none';
      return;
    }

    section.style.display = 'block';
    list.innerHTML = '';

    this.activeIndicators.forEach(indicator => {
      const item = document.createElement('div');
      item.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; background: var(--tos-bg-secondary); border-radius: 4px; border: 1px solid var(--tos-border-color);';

      const label = document.createElement('span');
      label.textContent = indicator.name;
      label.style.cssText = 'color: var(--tos-text-primary); font-size: 13px;';

      const removeBtn = document.createElement('button');
      removeBtn.textContent = '×';
      removeBtn.style.cssText = 'background: #d32f2f; color: white; border: none; border-radius: 3px; width: 24px; height: 24px; cursor: pointer; font-size: 18px; line-height: 1; padding: 0;';
      removeBtn.onclick = () => this.removeIndicator(indicator.id);

      item.appendChild(label);
      item.appendChild(removeBtn);
      list.appendChild(item);
    });
  }

  async removeIndicator(indicatorId) {
    const indicator = this.activeIndicators.find(ind => ind.id === indicatorId);
    if (!indicator) return;

    try {
      const plotDiv = document.getElementById('tos-plot');
      if (!plotDiv || !plotDiv.data) {
        console.error('Plot not found or not initialized');
        return;
      }

      // Validate trace indices before removal
      const validIndices = indicator.traceIndices.filter(idx => idx >= 0 && idx < plotDiv.data.length);
      if (validIndices.length === 0) {
        console.warn('No valid trace indices to remove');
        // Still remove from tracking
        this.activeIndicators = this.activeIndicators.filter(ind => ind.id !== indicatorId);
        this.updateCurrentIndicatorsUI();
        return;
      }

      // Remove traces from chart (in reverse order to maintain correct indices)
      const sortedIndices = [...validIndices].sort((a, b) => b - a);
      for (const traceIndex of sortedIndices) {
        await Plotly.deleteTraces('tos-plot', traceIndex);
      }

      // Remove from active indicators
      this.activeIndicators = this.activeIndicators.filter(ind => ind.id !== indicatorId);

      // Update trace indices for remaining indicators
      this.activeIndicators.forEach(ind => {
        ind.traceIndices = ind.traceIndices.map(idx => {
          let newIdx = idx;
          sortedIndices.forEach(removedIdx => {
            if (idx > removedIdx) newIdx--;
          });
          return newIdx;
        });
      });

      // Recalculate layout - may remove subplot if no more indicators need it
      const newLayout = this.calculateChartLayout();
      await Plotly.relayout('tos-plot', newLayout);

      // Update UI
      this.updateCurrentIndicatorsUI();

      // Save indicators to localStorage
      this.saveIndicators();

      // Show success message
      const notif = document.getElementById('notifications-area');
      notif.textContent = `✓ ${indicator.type} removed`;
      notif.style.color = 'var(--tos-accent-green)';
      setTimeout(() => notif.textContent = '', 3000);
    } catch (error) {
      console.error('Error removing indicator:', error);
      // Clean up tracking even if removal failed
      this.activeIndicators = this.activeIndicators.filter(ind => ind.id !== indicatorId);
      this.updateCurrentIndicatorsUI();
      this.saveIndicators();
    }
  }

  showCompareDialog() {
    const symbol = prompt('Enter symbol to compare with current chart:');
    if (symbol) {
      alert(`Compare feature coming soon!\nWould compare ${this.currentSymbol} with ${symbol.toUpperCase()}`);
    }
  }

  initializeStatusBar() {
    // Update server time
    const updateTime = () => {
      const now = new Date();
      const timeString = now.toLocaleTimeString('en-US', { hour12: false });
      const timeElement = document.getElementById('server-time');
      if (timeElement) {
        timeElement.textContent = timeString;
      }
    };

    updateTime();
    setInterval(updateTime, 1000);

    // Set connection status
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');
    if (statusDot && statusText) {
      statusDot.classList.add('connected');
      statusText.textContent = 'Connected';
    }
  }

  initializeKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + S: Save layout
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        this.menuBar.saveLayout();
      }

      // Ctrl/Cmd + E: Export chart
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        this.menuBar.exportChart();
      }

      // F1: Toggle watchlist
      if (e.key === 'F1') {
        e.preventDefault();
        this.menuBar.togglePanel('tos-left-panel');
      }

      // F2: Toggle active trader
      if (e.key === 'F2') {
        e.preventDefault();
        this.menuBar.togglePanel('tos-right-panel');
      }

      // D: Drawing mode
      if (e.key === 'd' || e.key === 'D') {
        if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
          if (window.enableDrawing) {
            window.enableDrawing();
          }
        }
      }

      // P: Pattern scanner
      if (e.key === 'p' || e.key === 'P') {
        if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
          if (window.detectPatterns) {
            window.detectPatterns();
          }
        }
      }

      // Escape: Close menus
      if (e.key === 'Escape') {
        this.menuBar.closeAllMenus();
      }
    });
  }

  startLiveUpdates() {
    // WebSocket-only mode - no polling
    // console.log('WebSocket real-time mode active - no polling needed');

    // Update status bar to show WebSocket mode
    const statusText = document.getElementById('connection-text');
    if (statusText) {
      statusText.textContent = 'Live (WebSocket)';
    }
  }

  // REMOVED: updateChartData() - No longer needed, WebSocket provides real-time updates

  // REMOVED: stopLiveUpdates() - No longer needed, WebSocket handles connection lifecycle

  /**
   * Handle timeframe change from selector
   */
  async handleTimeframeChange(timeframeId) {
    console.log(`🔄 Chart changed to: ${timeframeId}`);

    // Detect if this is a tick chart (ends with 't') or timeframe
    const isTickChart = timeframeId.endsWith('t') &&
                        this.tickChartRegistry.get(timeframeId) !== undefined;

    if (isTickChart) {
      // Switch to tick chart
      console.log(`📊 Switching to tick chart: ${timeframeId}`);
      this.activeChartType = 'tick';
      this.currentTickChartId = timeframeId;
      localStorage.setItem('lastTickChart', timeframeId);

      if (this.currentSymbol && this.tickChartRegistry) {
        await this.tickChartRegistry.switchTickChart(timeframeId, this.currentSymbol, this.socket);
      }
    } else {
      // Switch to timeframe
      console.log(`📈 Switching to timeframe: ${timeframeId}`);
      this.activeChartType = 'timeframe';
      this.currentTimeframeId = timeframeId;
      localStorage.setItem('lastTimeframe', timeframeId);

      if (this.currentSymbol && this.timeframeRegistry) {
        await this.timeframeRegistry.switchTimeframe(timeframeId, this.currentSymbol, this.socket);
      }
    }
  }

  destroy() {
    // No polling to stop - WebSocket cleanup handled by socket.io
    if (this.watchlist) this.watchlist.destroy();
    if (this.newsFeed) this.newsFeed.destroy();
    if (this.activeTrader) this.activeTrader.destroy();
    if (this.timeframeRegistry) this.timeframeRegistry.destroy();
    if (this.tickChartRegistry) this.tickChartRegistry.destroy();
    if (this.timeframeSelector) this.timeframeSelector.destroy();
  }
}

/**
 * Initialize the application
 */
async function initializeApp() {
  // console.log('Starting ThinkorSwim-Style Platform...');

  const app = new TOSApp();
  await app.initialize();

  // Make app globally accessible
  window.tosApp = app;

  // Load symbol from URL param, localStorage, or default to BTC-USD
  const urlParams = new URLSearchParams(window.location.search);
  const defaultSymbol = urlParams.get('symbol') || app.currentSymbol || 'BTC-USD';

  // console.log(`🔍 [DEBUG] Initializing with symbol: ${defaultSymbol}, period: ${app.currentPeriod}, interval: ${app.currentInterval}`);

  setTimeout(() => {
    app.loadSymbol(defaultSymbol);
  }, 500);

  // console.log('Platform ready!');
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}

// Make functions globally accessible for inline onclick handlers
import { detectPatterns } from './analysis/patterns.js';
import { getPredictions } from './analysis/predictions.js';
import { getTradeIdeas } from './analysis/trade-ideas.js';
import { enableDrawing, enablePan, clearAllLines } from './trendlines/drawing.js';

window.detectPatterns = detectPatterns;
window.getPredictions = getPredictions;
window.getTradeIdeas = getTradeIdeas;
window.enableDrawing = enableDrawing;
window.enablePan = enablePan;
window.clearAllLines = clearAllLines;

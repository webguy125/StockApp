/**
 * 1 Day Timeframe
 * Independent implementation - displays daily candlestick chart with live updates
 * This is the default/main timeframe
 */
import { CanvasRenderer } from '../../chart-renderers/canvas-renderer.js';
import { volumeAccumulator } from '../../services/VolumeAccumulator.js';

export class Timeframe1d {
  constructor() {
    // Timeframe configuration
    this.id = '1d';
    this.name = '1 day';
    this.interval = '1d';
    this.period = 'max';  // Show all available daily data
    this.category = 'days';
    this.isCustom = false;

    // Chart renderer (independent instance)
    this.renderer = new CanvasRenderer('1d');

    // Data and state
    this.symbol = null;
    this.data = [];
    this.socket = null;
    this.isActive = false;
    this.lastTickerUpdate = null; // Store ticker that arrives before chart loads
    this.volumeCallback = null; // Callback for volume updates from shared accumulator
    this.newCandleCallback = null; // Callback for new candle detection
  }

  /**
   * Initialize the timeframe for a specific symbol
   */
  async initialize(symbol, socket) {
    console.log(`📊 [1D] Initializing for ${symbol}`);

    this.symbol = symbol;
    this.socket = socket;
    this.isActive = true;

    try {
      // Start shared volume accumulator
      volumeAccumulator.start(symbol, socket);

      // Load historical data
      await this.loadHistoricalData();

      // Fetch current candle's actual accumulated volume from backend
      if (this.data.length > 0) {
        const lastCandle = this.data[this.data.length - 1];

        try {
          const response = await fetch(`/current-candle-volume/${symbol}?interval=1d`);
          const currentCandleData = await response.json();

          console.log(`📊 [1D] Current candle data: O=${currentCandleData.open?.toFixed(2)} H=${currentCandleData.high?.toFixed(2)} L=${currentCandleData.low?.toFixed(2)} C=${currentCandleData.close?.toFixed(2)} V=${currentCandleData.volume?.toFixed(0)}`);

          // Update the last candle with current OHLCV data if available
          if (currentCandleData.open !== undefined) {
            lastCandle.Open = currentCandleData.open;
            lastCandle.High = currentCandleData.high;
            lastCandle.Low = currentCandleData.low;
            lastCandle.Close = currentCandleData.close;
            lastCandle.Volume = currentCandleData.volume;
          }

          volumeAccumulator.initializeCandleTimes('1d', currentCandleData.candle_start_time);
          volumeAccumulator.initializeVolume('1d', currentCandleData.volume);
        } catch (error) {
          console.error(`❌ [1D] Failed to fetch current candle data:`, error);
          // Fallback to 0 if fetch fails
          volumeAccumulator.initializeCandleTimes('1d', lastCandle.Date);
          volumeAccumulator.initializeVolume('1d', 0);
        }
      }

      // Register callback to receive volume updates
      this.volumeCallback = (volume) => {
        if (this.isActive && this.data.length > 0) {
          this.renderer.updateCurrentCandleVolume(volume);
        }
      };
      volumeAccumulator.registerCallback('1d', this.volumeCallback);

      // Register callback for new candle detection (critical for ORD Volume auto-update)
      this.newCandleCallback = (interval) => {
        if (this.isActive && this.data.length > 0 && interval === '1d') {
          console.log(`🕐 [1D] New candle detected - checking data array`);

          // Get the last candle to use as a reference
          const lastCandle = this.data[this.data.length - 1];

          // Get current price from last ticker update, or use last candle's close
          const currentPrice = this.lastTickerUpdate?.price || lastCandle.Close;

          // Get current timestamp (rounded down to daily boundary)
          const now = new Date();
          const candleTime = new Date(Math.floor(now.getTime() / (24 * 60 * 60000)) * (24 * 60 * 60000));

          // CRITICAL FIX: Check if last candle already has this timestamp (duplicate detection)
          const lastCandleTime = new Date(lastCandle.Date.includes('Z') ? lastCandle.Date : lastCandle.Date + 'Z');

          if (lastCandleTime.getTime() === candleTime.getTime()) {
            // Duplicate detected! Remove the flat candle and add the correct one
            console.log(`🗑️ [1D] Removing duplicate flat candle at ${candleTime.toLocaleTimeString()}`);
            this.data.pop(); // Remove the flat candle
          }

                    // Create new candle object with current price as OHLC
          const newCandle = {
            Date: candleTime.toISOString(),
            Open: currentPrice,
            High: currentPrice,
            Low: currentPrice,
            Close: currentPrice,
            Volume: 0  // Volume will accumulate via volumeCallback
          };

          // Add new candle to data array
          this.data.push(newCandle);

          console.log(`✅ [1D] Added candle #${this.data.length}: ${candleTime.toLocaleDateString()} @ $${currentPrice.toFixed(2)}`);

          // Trigger chart redraw to show the new candle
          if (this.renderer && this.renderer.draw) {
            this.renderer.draw();
          }
        }
      };
      volumeAccumulator.registerNewCandleCallback('1d', this.newCandleCallback);

      // Subscribe to WebSocket updates for price
      this.subscribeToLiveData();

      return true;
    } catch (error) {
      console.error(`❌ [1D] Initialization error:`, error);
      return false;
    }
  }

  /**
   * Load historical data from backend
   */
  async loadHistoricalData() {
    const url = `/data/${this.symbol}?interval=${this.interval}&period=${this.period}`;
    // console.log(`📥 [1D] Fetching: ${url}`);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.data = await response.json();

      if (this.data.length === 0) {
        throw new Error(`No data returned for ${this.symbol}`);
      }

      // console.log(`✅ [1D] Loaded ${this.data.length} daily candles`);

      // Render the data
      const success = await this.renderer.render(this.data, this.symbol);

      if (success) {
        // console.log('✅ [1D] Chart rendered successfully');

        // Apply any ticker update that arrived during chart load
        if (this.lastTickerUpdate && this.lastTickerUpdate.symbol === this.symbol) {
          // console.log(`🔄 [1D] Applying pending ticker update: ${this.lastTickerUpdate.symbol} = $${this.lastTickerUpdate.price}`);
          const volumeBTC = this.lastTickerUpdate.volume_today || 0;
          this.renderer.updateLivePrice(this.lastTickerUpdate.price, volumeBTC);
        }
      }

      return this.data;
    } catch (error) {
      console.error(`❌ [1D] Error loading data:`, error);
      throw error;
    }
  }

  /**
   * Subscribe to live WebSocket updates
   */
  subscribeToLiveData() {
    if (!this.socket) {
      // console.warn(`⚠️ [1D] No socket connection available`);
      return;
    }

    // Only subscribe to Coinbase ticker updates for crypto symbols
    // Stocks don't have real-time WebSocket feeds from Coinbase
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'];
    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith('-USD');

    if (isCrypto) {
      // Subscribe to ticker updates from Coinbase
      this.socket.emit('subscribe', {
        product_ids: [this.symbol],
        channels: ['ticker', 'matches']
      });

      console.log(`🔔 [1D] Subscribed to ${this.symbol} ticker`);
    } else {
      console.log(`📊 [1D] Skipping WebSocket subscription for stock symbol: ${this.symbol}`);
    }
  }

  /**
   * Handle live ticker update from WebSocket
   */
  handleTickerUpdate(data) {
    // console.log(`📈 [1D] Received ticker: ${data.symbol} = $${data.price}, isActive=${this.isActive}`);

    // Ignore ticker updates for stock symbols (they don't have real-time data from Coinbase)
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'];
    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith('-USD');

    if (!isCrypto) {
      console.log(`📊 [1D] Ignoring ticker update for stock symbol: ${this.symbol}`);
      return;
    }

    // Check if this ticker is for our symbol
    const symbolMatches = data.symbol && this.symbol &&
      (data.symbol === `${this.symbol}-USD` ||
       data.symbol.includes(this.symbol) ||
       this.symbol.includes(data.symbol.split('-')[0]));

    if (!this.isActive || !data || !symbolMatches) {
      console.log(`  ❌ [1D] Skipping ticker - isActive=${this.isActive}, symbolMatches=${symbolMatches}`);
      return;
    }

    const price = parseFloat(data.price);
    const volumeBTC = data.volume_today || 0;

    // console.log(`  ✅ [ update - price=${price}, volume=${volumeBTC}, data.length=${this.data.length}`);

    // Store latest ticker for this symbol (even if chart isn't loaded yet)
    this.lastTickerUpdate = {
      symbol: this.symbol,
      price: price,
      volume_today: volumeBTC
    };

    // Update the chart renderer with live price (NO volume - VolumeAccumulator handles that)
    if (this.data.length > 0) {
      // console.log(`  🖼️ [1D] Updating renderer with live price`);

      // CRITICAL: Update the current candle's OHLC in the data array
      // This ensures ORD Volume auto-update analyzes fresh price data
      const currentCandle = this.data[this.data.length - 1];
      currentCandle.Close = price;
      currentCandle.High = Math.max(currentCandle.High, price);
      currentCandle.Low = Math.min(currentCandle.Low, price);

      this.renderer.updateLivePrice(price, null);
    } else {
      // console.log(`  ⚠️ [, ticker stored for later`);
    }
  }

  /**
   * Handle trade update from WebSocket (placeholder)
   */
  handleTradeUpdate(data) {
    // Not needed for daily charts
  }

  /**
   * Deactivate this timeframe
   */
  deactivate() {
    console.log(`⏸️ [1D] Deactivating`);

    this.isActive = false;

    // Unregister volume callback
    if (this.volumeCallback) {
      volumeAccumulator.unregisterCallback('1d', this.volumeCallback);
      this.volumeCallback = null;
    }

    // Unregister new candle callback
    if (this.newCandleCallback) {
      volumeAccumulator.unregisterNewCandleCallback('1d', this.newCandleCallback);
      this.newCandleCallback = null;
    }

    // Destroy the renderer to remove the canvas from DOM
    if (this.renderer) {
      this.renderer.destroy();
    }

    // Unsubscribe from WebSocket
    if (this.socket && this.symbol) {
      this.socket.emit('unsubscribe', {
        product_ids: [this.symbol],
        channels: ['ticker']
      });
    }
  }

  /**
   * Clean up resources
   */
  destroy() {
    this.deactivate();

    if (this.renderer) {
      this.renderer.destroy();
    }

    this.data = [];
    this.symbol = null;
    this.socket = null;
    this.lastTickerUpdate = null;
  }

  /**
   * Reload chart data (full refresh)
   */
  async reload() {
    // console.log(`🔄 [1D] Reloading...`);
    await this.loadHistoricalData();
  }

  /**
   * Get current timeframe info
   */
  getInfo() {
    return {
      id: this.id,
      name: this.name,
      interval: this.interval,
      period: this.period,
      category: this.category,
      isCustom: this.isCustom,
      isActive: this.isActive,
      dataPoints: this.data.length,
      symbol: this.symbol
    };
  }
}

/**
 * 1 Minute Timeframe
 * Independent implementation - displays 1-minute candlestick chart with live updates
 */
import { CanvasRenderer } from '../../chart-renderers/canvas-renderer.js';
import { volumeAccumulator } from '../../services/VolumeAccumulator.js';

export class Timeframe1m {
  constructor() {
    // Timeframe configuration
    this.id = '1m';
    this.name = '1 minute';
    this.interval = '1m';
    this.period = '5d';  // Show 5 days of 1-minute candles
    this.category = 'minutes';
    this.isCustom = false;

    // Chart renderer (independent instance)
    this.renderer = new CanvasRenderer('1m');

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
    console.log(`📊 [1M] Initializing for ${symbol}`);

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
          const response = await fetch(`/current-candle-volume/${symbol}?interval=1m`);
          const currentCandleData = await response.json();

          console.log(`📊 [1M] Historical last candle: ${lastCandle.Date}`);
          console.log(`📊 [1M] Current candle time:   ${currentCandleData.candle_start_time}`);
          console.log(`📊 [1M] Current candle OHLCV: O=${currentCandleData.open?.toFixed(2)} H=${currentCandleData.high?.toFixed(2)} L=${currentCandleData.low?.toFixed(2)} C=${currentCandleData.close?.toFixed(2)} V=${currentCandleData.volume?.toFixed(2)}`);

          // Check if current candle time matches last historical candle time
          const lastCandleTime = new Date(lastCandle.Date.includes('Z') ? lastCandle.Date : lastCandle.Date + 'Z');
          const currentCandleTime = new Date(currentCandleData.candle_start_time);

          if (lastCandleTime.getTime() === currentCandleTime.getTime()) {
            // Same candle - update OHLCV
            console.log(`✅ [1M] Times match - updating last candle`);
            if (currentCandleData.open !== undefined) {
              lastCandle.Open = currentCandleData.open;
              lastCandle.High = currentCandleData.high;
              lastCandle.Low = currentCandleData.low;
              lastCandle.Close = currentCandleData.close;
              lastCandle.Volume = currentCandleData.volume;
            }
          } else {
            // Different candle - need to add new candle
            console.log(`⚠️ [1M] Time mismatch! Adding new candle for ${currentCandleData.candle_start_time}`);
            if (currentCandleData.open !== undefined) {
              const newCandle = {
                Date: currentCandleData.candle_start_time,
                Open: currentCandleData.open,
                High: currentCandleData.high,
                Low: currentCandleData.low,
                Close: currentCandleData.close,
                Volume: currentCandleData.volume
              };
              this.data.push(newCandle);
              console.log(`✅ [1M] Added current candle to data array, new length: ${this.data.length}`);
            }
          }

          volumeAccumulator.initializeCandleTimes('1m', currentCandleData.candle_start_time);
          volumeAccumulator.initializeVolume('1m', currentCandleData.volume);
        } catch (error) {
          console.error(`❌ [1M] Failed to fetch current candle data:`, error);
          // Fallback to 0 if fetch fails
          volumeAccumulator.initializeCandleTimes('1m', lastCandle.Date);
          volumeAccumulator.initializeVolume('1m', 0);
        }
      }

      // Register callback to receive volume updates
      this.volumeCallback = (volume) => {
        if (this.isActive && this.data.length > 0) {
          // console.log(`📊 [1M] Volume update: ${volume.toFixed(2)}`);
          this.renderer.updateCurrentCandleVolume(volume);
        }
      };
      volumeAccumulator.registerCallback('1m', this.volumeCallback);

      // Register callback for new candle detection (critical for ORD Volume auto-update)
      this.newCandleCallback = (interval) => {
        if (this.isActive && this.data.length > 0 && interval === '1m') {
          console.log(`🕐 [1M] New candle detected - checking data array`);

          // Get the last candle to use as a reference
          const lastCandle = this.data[this.data.length - 1];

          // Get current price from last ticker update, or use last candle's close
          const currentPrice = this.lastTickerUpdate?.price || lastCandle.Close;

          // Get current timestamp (rounded down to minute boundary)
          const now = new Date();
          const candleTime = new Date(Math.floor(now.getTime() / 60000) * 60000);

          // CRITICAL FIX: Check if last candle already has this timestamp (duplicate detection)
          const lastCandleTime = new Date(lastCandle.Date.includes('Z') ? lastCandle.Date : lastCandle.Date + 'Z');

          if (lastCandleTime.getTime() === candleTime.getTime()) {
            // Duplicate detected! Remove the flat candle and add the correct one
            console.log(`🗑️ [1M] Removing duplicate flat candle at ${candleTime.toLocaleTimeString()}`);
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

          console.log(`✅ [1M] Added candle #${this.data.length}: ${candleTime.toLocaleTimeString()} @ $${currentPrice.toFixed(2)}`);

          // Trigger chart redraw to show the new candle
          if (this.renderer && this.renderer.draw) {
            this.renderer.draw();
          }
        }
      };
      volumeAccumulator.registerNewCandleCallback('1m', this.newCandleCallback);

      // Subscribe to WebSocket updates for price
      this.subscribeToLiveData();

      return true;
    } catch (error) {
      console.error(`❌ [1M] Initialization error:`, error);
      return false;
    }
  }

  /**
   * Load historical data from backend
   */
  async loadHistoricalData() {
    const url = `/data/${this.symbol}?interval=${this.interval}&period=${this.period}`;
    console.log(`📥 [1M] Fetching: ${url}`);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.data = await response.json();

      if (this.data.length === 0) {
        throw new Error(`No data returned for ${this.symbol}`);
      }

      console.log(`✅ [1M] Loaded ${this.data.length} 1-minute candles`);

      // Render the data
      const success = await this.renderer.render(this.data, this.symbol);

      if (success) {
        console.log('✅ [1M] Chart rendered successfully');

        // Apply any ticker update that arrived during chart load
        if (this.lastTickerUpdate && this.lastTickerUpdate.symbol === this.symbol) {
          console.log(`🔄 [1M] Applying pending ticker update: ${this.lastTickerUpdate.symbol} = $${this.lastTickerUpdate.price}`);
          const bid = this.lastTickerUpdate.bid || null;
          const ask = this.lastTickerUpdate.ask || null;
          this.renderer.updateLivePrice(this.lastTickerUpdate.price, null, bid, ask);
        }
      }

      return this.data;
    } catch (error) {
      console.error(`❌ [1M] Error loading data:`, error);
      throw error;
    }
  }

  /**
   * Subscribe to live WebSocket updates
   */
  subscribeToLiveData() {
    if (!this.socket) {
      console.warn(`⚠️ [1M] No socket connection available`);
      return;
    }

    // Only subscribe to Coinbase ticker updates for crypto symbols
    // Stocks don't have real-time WebSocket feeds from Coinbase
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'];
    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith('-USD');

    if (isCrypto) {
      // Subscribe to ticker updates from Coinbase (matches handled by VolumeAccumulator)
      this.socket.emit('subscribe', {
        product_ids: [this.symbol],
        channels: ['ticker']
      });

      console.log(`🔔 [1M] Subscribed to ${this.symbol} ticker`);
    } else {
      console.log(`📊 [1M] Skipping WebSocket subscription for stock symbol: ${this.symbol}`);
    }
  }

  /**
   * Handle live ticker update from WebSocket
   */
  handleTickerUpdate(data) {
    // console.log(`📈 [1M] Received ticker: ${data.symbol} = $${data.price}, isActive=${this.isActive}`);

    // Ignore ticker updates for stock symbols (they don't have real-time data from Coinbase)
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'];
    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith('-USD');

    if (!isCrypto) {
      console.log(`📊 [1M] Ignoring ticker update for stock symbol: ${this.symbol}`);
      return;
    }

    // Check if this ticker is for our symbol
    const symbolMatches = data.symbol && this.symbol &&
      (data.symbol === `${this.symbol}-USD` ||
       data.symbol.includes(this.symbol) ||
       this.symbol.includes(data.symbol.split('-')[0]));

    if (!this.isActive || !data || !symbolMatches) {
      // console.log(`  ❌ [1M] Skipping ticker - isActive=${this.isActive}, symbolMatches=${symbolMatches}`);
      return;
    }

    const price = parseFloat(data.price);
    const bid = data.bid ? parseFloat(data.bid) : null;
    const ask = data.ask ? parseFloat(data.ask) : null;

    // console.log(`  ✅ [1M] Live update - price=${price}, bid=${bid}, ask=${ask}, data.length=${this.data.length}`);

    // Store latest ticker for this symbol (even if chart isn't loaded yet)
    this.lastTickerUpdate = {
      symbol: this.symbol,
      price: price,
      bid: bid,
      ask: ask
    };

    // Update the chart renderer with live price (NO volume - trade updates handle that)
    if (this.data.length > 0) {
      // console.log(`  🖼️ [1M] Updating renderer with live price`);

      // CRITICAL: Update the current candle's OHLC in the data array
      // This ensures ORD Volume auto-update analyzes fresh price data
      const currentCandle = this.data[this.data.length - 1];
      currentCandle.Close = price;
      currentCandle.High = Math.max(currentCandle.High, price);
      currentCandle.Low = Math.min(currentCandle.Low, price);

      this.renderer.updateLivePrice(price, null, bid, ask);
    } else {
      // console.log(`  ⚠️ [1M] No data yet, ticker stored for later`);
    }
  }

  /**
   * Deactivate this timeframe
   */
  deactivate() {
    console.log(`⏸️ [1M] Deactivating`);

    this.isActive = false;

    // Unregister volume callback
    if (this.volumeCallback) {
      volumeAccumulator.unregisterCallback('1m', this.volumeCallback);
      this.volumeCallback = null;
    }

    // Unregister new candle callback
    if (this.newCandleCallback) {
      volumeAccumulator.unregisterNewCandleCallback('1m', this.newCandleCallback);
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
        channels: ['ticker', 'matches']
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
    console.log(`🔄 [1M] Reloading...`);
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

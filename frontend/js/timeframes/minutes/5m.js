/**
 * 5 Minute Timeframe
 * Independent implementation - displays 5-minute candlestick chart with live updates
 */
import { CanvasRenderer } from '../../chart-renderers/canvas-renderer.js';
import { volumeAccumulator } from '../../services/VolumeAccumulator.js';

export class Timeframe5m {
  constructor() {
    // Timeframe configuration
    this.id = '5m';
    this.name = '5 minutes';
    this.interval = '5m';
    this.period = '1d';
    this.category = 'minutes';
    this.isCustom = false;

    // Chart renderer (independent instance)
    this.renderer = new CanvasRenderer('5m');

    // Data and state
    this.symbol = null;
    this.data = [];
    this.socket = null;
    this.isActive = false;
    this.lastTickerUpdate = null; // Store ticker that arrives before chart loads
    this.volumeCallback = null; // Callback for volume updates from shared accumulator
  }

  /**
   * Initialize the timeframe for a specific symbol
   */
  async initialize(symbol, socket) {
    console.log(`📊 [5M] Initializing for ${symbol}`);

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
          const response = await fetch(`/current-candle-volume/${symbol}?interval=5m`);
          const currentCandleData = await response.json();

          console.log(`📊 [5M] Current candle data: O=${currentCandleData.open?.toFixed(2)} H=${currentCandleData.high?.toFixed(2)} L=${currentCandleData.low?.toFixed(2)} C=${currentCandleData.close?.toFixed(2)} V=${currentCandleData.volume?.toFixed(0)}`);

          // Update the last candle with current OHLCV data if available
          if (currentCandleData.open !== undefined) {
            lastCandle.Open = currentCandleData.open;
            lastCandle.High = currentCandleData.high;
            lastCandle.Low = currentCandleData.low;
            lastCandle.Close = currentCandleData.close;
            lastCandle.Volume = currentCandleData.volume;
          }

          volumeAccumulator.initializeCandleTimes('5m', currentCandleData.candle_start_time);
          volumeAccumulator.initializeVolume('5m', currentCandleData.volume);
        } catch (error) {
          console.error(`❌ [5M] Failed to fetch current candle data:`, error);
          // Fallback to 0 if fetch fails
          volumeAccumulator.initializeCandleTimes('5m', lastCandle.Date);
          volumeAccumulator.initializeVolume('5m', 0);
        }
      }

      // Register callback to receive volume updates
      this.volumeCallback = (volume) => {
        if (this.isActive && this.data.length > 0) {
          this.renderer.updateCurrentCandleVolume(volume);
        }
      };
      volumeAccumulator.registerCallback('5m', this.volumeCallback);

      // Subscribe to WebSocket updates for price
      this.subscribeToLiveData();

      return true;
    } catch (error) {
      console.error(`❌ [5M] Initialization error:`, error);
      return false;
    }
  }

  /**
   * Load historical data from backend
   */
  async loadHistoricalData() {
    const url = `/data/${this.symbol}?interval=${this.interval}&period=${this.period}`;
    // console.log(`📥 [5M] Fetching: ${url}`);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.data = await response.json();

      if (this.data.length === 0) {
        throw new Error(`No data returned for ${this.symbol}`);
      }

      // console.log(`✅ [5M] Loaded ${this.data.length} 5-minute candles`);

      // Render the data
      const success = await this.renderer.render(this.data, this.symbol);

      if (success) {
        // console.log('✅ [5M] Chart rendered successfully');

        // Apply any ticker update that arrived during chart load
        if (this.lastTickerUpdate && this.lastTickerUpdate.symbol === this.symbol) {
          // console.log(`🔄 [5M] Applying pending ticker update: ${this.lastTickerUpdate.symbol} = $${this.lastTickerUpdate.price}`);
          const bid = this.lastTickerUpdate.bid || null;
          const ask = this.lastTickerUpdate.ask || null;
          this.renderer.updateLivePrice(this.lastTickerUpdate.price, null, bid, ask);
        }
      }

      return this.data;
    } catch (error) {
      console.error(`❌ [5M] Error loading data:`, error);
      throw error;
    }
  }

  /**
   * Subscribe to live WebSocket updates
   */
  subscribeToLiveData() {
    if (!this.socket) {
      console.warn(`⚠️ [5M] No socket connection available`);
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

      console.log(`🔔 [5M] Subscribed to ${this.symbol} ticker`);
    } else {
      console.log(`📊 [5M] Skipping WebSocket subscription for stock symbol: ${this.symbol}`);
    }
  }

  /**
   * Handle live ticker update from WebSocket
   */
  handleTickerUpdate(data) {
    // console.log(`📈 [5M] Received ticker: ${data.symbol} = $${data.price}, isActive=${this.isActive}`);

    // Ignore ticker updates for stock symbols (they don't have real-time data from Coinbase)
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'];
    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith('-USD');

    if (!isCrypto) {
      console.log(`📊 [5M] Ignoring ticker update for stock symbol: ${this.symbol}`);
      return;
    }

    // Check if this ticker is for our symbol
    const symbolMatches = data.symbol && this.symbol &&
      (data.symbol === `${this.symbol}-USD` ||
       data.symbol.includes(this.symbol) ||
       this.symbol.includes(data.symbol.split('-')[0]));

    if (!this.isActive || !data || !symbolMatches) {
      console.log(`  ❌ [5M] Skipping ticker - isActive=${this.isActive}, symbolMatches=${symbolMatches}`);
      return;
    }

    const price = parseFloat(data.price);
    const bid = data.bid ? parseFloat(data.bid) : null;
    const ask = data.ask ? parseFloat(data.ask) : null;

    // console.log(`  ✅ [5M] Live update - price=${price}, bid=${bid}, ask=${ask}, data.length=${this.data.length}`);

    // Store latest ticker for this symbol (even if chart isn't loaded yet)
    this.lastTickerUpdate = {
      symbol: this.symbol,
      price: price,
      bid: bid,
      ask: ask
    };

    // Update the chart renderer with live price (NO volume - trade updates handle that)
    if (this.data.length > 0) {
      // console.log(`  🖼️ [5M] Updating renderer with live price`);
      this.renderer.updateLivePrice(price, null, bid, ask);
    } else {
      // console.log(`  ⚠️ [5M] Chart not loaded yet, ticker stored for later`);
    }
  }

  /**
   * Deactivate this timeframe
   */
  deactivate() {
    console.log(`⏸️ [5M] Deactivating`);

    this.isActive = false;

    // Unregister volume callback
    if (this.volumeCallback) {
      volumeAccumulator.unregisterCallback('5m', this.volumeCallback);
      this.volumeCallback = null;
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
    // console.log(`🔄 [5M] Reloading...`);
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

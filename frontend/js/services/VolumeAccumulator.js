/**
 * Volume Accumulator Service
 * Runs in the background and accumulates volume for all timeframes simultaneously
 * This ensures accurate volume data when switching between timeframes
 */

export class VolumeAccumulator {
  constructor() {
    this.symbol = null;
    this.socket = null;
    this.isActive = false;

    // Volume accumulators for each timeframe (keyed by interval)
    this.volumes = {
      '1m': { volume: 0, candleStartTime: null },
      '5m': { volume: 0, candleStartTime: null },
      '15m': { volume: 0, candleStartTime: null },
      '30m': { volume: 0, candleStartTime: null },
      '1h': { volume: 0, candleStartTime: null },
      '2h': { volume: 0, candleStartTime: null },
      '4h': { volume: 0, candleStartTime: null },
      '6h': { volume: 0, candleStartTime: null },
      '1d': { volume: 0, candleStartTime: null }
    };

    // Interval durations in milliseconds
    this.intervals = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '2h': 2 * 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '6h': 6 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000
    };

    // Callbacks to notify timeframes when volume updates
    this.callbacks = {};

    // Callbacks to notify when a new candle starts (for ORD Volume auto-update)
    this.newCandleCallbacks = {};
  }

  /**
   * Start accumulating volume for a symbol
   */
  start(symbol, socket) {
    if (this.isActive && this.symbol === symbol) {
      console.log(`📊 [VolumeAccumulator] Already running for ${symbol}`);
      return;
    }

    console.log(`📊 [VolumeAccumulator] Starting for ${symbol}`);
    this.symbol = symbol;
    this.socket = socket;
    this.isActive = true;

    // Reset all volumes
    for (const interval in this.volumes) {
      this.volumes[interval] = { volume: 0, candleStartTime: null };
    }

    // Only subscribe to WebSocket for crypto symbols
    // Stock symbols don't have real-time trade data from Coinbase
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC', 'MATIC', 'UNI'];
    const isCrypto = cryptoSymbols.includes(symbol) || symbol.endsWith('-USD');

    if (this.socket && isCrypto) {
      this.socket.emit('subscribe', {
        product_ids: [`${symbol}-USD`],
        channels: ['matches']
      });
      console.log(`📊 [VolumeAccumulator] Subscribed to matches for ${symbol}-USD`);
    } else {
      console.log(`📊 [VolumeAccumulator] Skipping WebSocket subscription for stock symbol: ${symbol}`);
    }
  }

  /**
   * Stop accumulating (called when changing symbols)
   */
  stop() {
    if (!this.isActive) return;

    console.log(`📊 [VolumeAccumulator] Stopping for ${this.symbol}`);

    if (this.socket && this.symbol) {
      this.socket.emit('unsubscribe', {
        product_ids: [`${this.symbol}-USD`],
        channels: ['matches']
      });
    }

    this.isActive = false;
    this.symbol = null;
    this.callbacks = {};
  }

  /**
   * Initialize candle start times from historical data
   */
  initializeCandleTimes(interval, lastCandleDate) {
    if (this.volumes[interval]) {
      this.volumes[interval].candleStartTime = new Date(lastCandleDate);
      console.log(`📊 [VolumeAccumulator] Initialized ${interval} candle time: ${lastCandleDate}`);
    }
  }

  /**
   * Initialize volume from the last historical candle
   */
  initializeVolume(interval, lastCandleVolume) {
    if (this.volumes[interval]) {
      this.volumes[interval].volume = lastCandleVolume || 0;
      console.log(`📊 [VolumeAccumulator] Initialized ${interval} volume: ${this.volumes[interval].volume.toFixed(4)} BTC`);
    }
  }

  /**
   * Register a callback to be notified when volume updates for a specific interval
   */
  registerCallback(interval, callback) {
    if (!this.callbacks[interval]) {
      this.callbacks[interval] = [];
    }
    this.callbacks[interval].push(callback);
  }

  /**
   * Unregister a callback
   */
  unregisterCallback(interval, callback) {
    if (this.callbacks[interval]) {
      this.callbacks[interval] = this.callbacks[interval].filter(cb => cb !== callback);
    }
  }

  /**
   * Register a callback to be notified when a new candle starts
   */
  registerNewCandleCallback(interval, callback) {
    if (!this.newCandleCallbacks[interval]) {
      this.newCandleCallbacks[interval] = [];
    }
    this.newCandleCallbacks[interval].push(callback);
  }

  /**
   * Unregister a new candle callback
   */
  unregisterNewCandleCallback(interval, callback) {
    if (this.newCandleCallbacks[interval]) {
      this.newCandleCallbacks[interval] = this.newCandleCallbacks[interval].filter(cb => cb !== callback);
    }
  }

  /**
   * Handle trade update from WebSocket
   */
  handleTradeUpdate(data) {
    if (!this.isActive || !data) return;

    // Ignore trade updates for stock symbols (they don't have real-time data from Coinbase)
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC', 'MATIC', 'UNI'];
    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith('-USD');

    if (!isCrypto) {
      // console.log(`📊 [VolumeAccumulator] Ignoring trade update for stock symbol: ${this.symbol}`);
      return;
    }

    // Check if this trade is for our symbol
    const symbolMatches = data.product_id && this.symbol &&
      (data.product_id === `${this.symbol}-USD` ||
       data.product_id.includes(this.symbol) ||
       this.symbol.includes(data.product_id.split('-')[0]));

    if (!symbolMatches || !data.size) return;

    const tradeSize = parseFloat(data.size);
    if (isNaN(tradeSize)) return;

    const now = new Date();

    // Update volume for each timeframe
    for (const interval in this.volumes) {
      const vol = this.volumes[interval];

      // Initialize candle start time if not set
      if (!vol.candleStartTime) {
        vol.candleStartTime = this.roundDownToInterval(now, interval);
      }

      // Check if we've moved to a new candle period
      let newCandleStarted = false;

      // Use millisecond-based logic for ALL intervals (including 1d)
      // The candle start time is set correctly by the backend based on exchange-specific boundaries
      const timeSinceCandle = now - vol.candleStartTime;
      const intervalMs = this.intervals[interval];

      if (timeSinceCandle >= intervalMs) {
        newCandleStarted = true;
        vol.candleStartTime = new Date(vol.candleStartTime.getTime() + intervalMs);
      }

      if (newCandleStarted) {
        // New candle started - reset volume
        vol.volume = 0;
        console.log(`📊 [VolumeAccumulator] ${interval} new candle started, reset volume`);

        // Notify new candle callbacks (for ORD Volume auto-update)
        if (this.newCandleCallbacks[interval]) {
          this.newCandleCallbacks[interval].forEach(callback => callback(interval));
        }
      }

      // Accumulate the trade
      vol.volume += tradeSize;

      // Notify registered callbacks for this interval
      if (this.callbacks[interval]) {
        this.callbacks[interval].forEach(callback => callback(vol.volume));
      }
    }
  }

  /**
   * Round down a timestamp to the start of the interval
   */
  roundDownToInterval(date, interval) {
    // Use millisecond-based rounding for all intervals
    // Note: For daily candles, the backend provides the accurate start time via initializeCandleTimes()
    const ms = date.getTime();
    const intervalMs = this.intervals[interval];
    return new Date(Math.floor(ms / intervalMs) * intervalMs);
  }

  /**
   * Get current volume for a specific interval
   */
  getVolume(interval) {
    return this.volumes[interval]?.volume || 0;
  }

  /**
   * Get current candle start time for a specific interval
   */
  getCandleStartTime(interval) {
    return this.volumes[interval]?.candleStartTime;
  }
}

// Create singleton instance
export const volumeAccumulator = new VolumeAccumulator();

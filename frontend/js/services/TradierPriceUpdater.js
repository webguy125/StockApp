/**
 * Tradier Real-Time Price Updater (WebSocket Version)
 * Receives real-time quotes from Tradier WebSocket via Flask Socket.IO
 * No more polling - true real-time push updates!
 */
export class TradierPriceUpdater {
  constructor() {
    this.symbol = null;
    this.isActive = false;
    this.updateCallbacks = new Map(); // timeframe -> callback
    this.lastPrice = null;
    this.lastUpdate = null;
    this.socket = null; // Will be set by chart system

    console.log(`🔴 [TRADIER] WebSocket-based price updater initialized`);
  }

  /**
   * Set Socket.IO connection (called by chart system)
   */
  setSocket(socket) {
    if (this.socket) {
      console.warn(`🔴 [TRADIER] Socket already set, ignoring`);
      return;
    }

    this.socket = socket;

    // Listen for ticker_update events from backend
    this.socket.on('ticker_update', (data) => {
      this.handleTickerUpdate(data);
    });

    console.log(`🔴 [TRADIER] Socket.IO listener registered`);
  }

  /**
   * Handle ticker update from Socket.IO (pushed from backend)
   */
  handleTickerUpdate(data) {
    // Only process updates for our current symbol
    if (!this.isActive || !this.symbol || data.symbol !== this.symbol) {
      return;
    }

    try {
      const quote = {
        price: data.price || data.last,
        bid: data.bid,
        ask: data.ask,
        volume: data.volume,
        change: data.change || 0,
        change_percentage: data.changePercent || 0,
        timestamp: new Date()
      };

      // Update cached price
      this.lastPrice = quote.price;
      this.lastUpdate = quote.timestamp;

      // Log update (throttled to avoid spam)
      const now = Date.now();
      if (!this._lastLogTime || (now - this._lastLogTime) > 10000) {
        console.log(`🔴 [TRADIER WS] Real-time price: ${this.symbol} = $${quote.price.toFixed(2)}`);
        this._lastLogTime = now;
      }

      // Notify all registered callbacks
      this.updateCallbacks.forEach((callback, timeframeId) => {
        try {
          callback(quote);
        } catch (error) {
          console.error(`❌ [TRADIER] Error in callback for ${timeframeId}:`, error);
        }
      });

    } catch (error) {
      console.error(`❌ [TRADIER] Error processing ticker update:`, error);
    }
  }

  /**
   * Start receiving real-time updates for a symbol
   */
  start(symbol) {
    // Prevent duplicate starts for the same symbol
    if (this.isActive && this.symbol === symbol) {
      console.log(`🔴 [TRADIER] Already running for ${symbol}, skipping start`);
      return;
    }

    console.log(`🔴 [TRADIER] Starting real-time WebSocket updates for ${symbol}`);

    // Stop existing updates if switching symbols
    if (this.isActive && this.symbol && this.symbol !== symbol) {
      this.stop();
    }

    this.symbol = symbol;
    this.isActive = true;

    // Subscribe via Socket.IO (backend will forward Tradier WebSocket updates)
    if (this.socket) {
      this.socket.emit('subscribe_ticker', { symbol: symbol });
      console.log(`🔴 [TRADIER] Subscribed to ${symbol} via Socket.IO`);
    } else {
      console.warn(`⚠️ [TRADIER] Socket not set, cannot subscribe to ${symbol}`);
    }
  }

  /**
   * Stop receiving updates
   */
  stop() {
    console.log(`⏹️ [TRADIER] Stopping real-time price updates for ${this.symbol}`);

    this.isActive = false;

    // Unsubscribe from Socket.IO
    if (this.socket && this.symbol) {
      this.socket.emit('unsubscribe', { symbols: [this.symbol] });
    }

    this.updateCallbacks.clear();
    this.symbol = null;
  }

  /**
   * Register a callback to receive price updates
   */
  registerCallback(timeframeId, callback) {
    this.updateCallbacks.set(timeframeId, callback);

    // If we have a cached price, immediately call the callback
    if (this.lastPrice !== null) {
      callback({
        price: this.lastPrice,
        timestamp: this.lastUpdate
      });
    }
  }

  /**
   * Unregister a callback
   */
  unregisterCallback(timeframeId) {
    this.updateCallbacks.delete(timeframeId);
  }

  /**
   * Get the last fetched price
   */
  getLastPrice() {
    return this.lastPrice;
  }

  /**
   * Check if updater is active
   */
  isRunning() {
    return this.isActive;
  }
}

// Singleton instance
export const tradierPriceUpdater = new TradierPriceUpdater();

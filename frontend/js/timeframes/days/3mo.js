/**
 * 3 Month Timeframe
 * Independent implementation following 1D pattern
 * Shows 3 months of daily candles
 */
import { CanvasRenderer } from '../../chart-renderers/canvas-renderer.js';

export class Timeframe3mo {
  constructor() {
    // Timeframe configuration
    this.id = '3mo';
    this.name = '3 months';
    this.interval = '3mo';  // Each candle represents 3 months
    this.period = 'max';
    this.category = 'days';
    this.isCustom = false;

    // Chart renderer (independent instance) - use '3mo' for wider gridlines
    this.renderer = new CanvasRenderer('3mo');

    // Data and state
    this.symbol = null;
    this.data = [];
    this.socket = null;
    this.isActive = false;

    // Destroy the renderer to remove the canvas from DOM
    if (this.renderer) {
      this.renderer.destroy();
    }
    this.lastTickerUpdate = null; // Store ticker that arrives before chart loads
  }

  /**
   * Initialize the timeframe for a specific symbol
   */
  async initialize(symbol, socket) {
    // console.log(`📊 [3MO] Initializing for ${symbol}`);

    this.symbol = symbol;
    this.socket = socket;
    this.isActive = true;

    try {
      // Load historical data
      await this.loadHistoricalData();

      // Subscribe to WebSocket updates
      this.subscribeToLiveData();

      return true;
    } catch (error) {
      console.error(`❌ [3MO] Initialization error:`, error);
      return false;
    }
  }

  /**
   * Load historical data from backend (1D pattern)
   */
  async loadHistoricalData() {
    const url = `/data/${this.symbol}?interval=${this.interval}&period=${this.period}`;
    // console.log(`📥 [3MO] Fetching: ${url}`);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.data = await response.json();

      if (this.data.length === 0) {
        throw new Error(`No data returned for ${this.symbol}`);
      }

      // console.log(`✅ [3MO] Loaded ${this.data.length} 3-month candles`);

      // Render the data
      const success = await this.renderer.render(this.data, this.symbol);

      if (success) {
        // console.log('✅ [3MO] Chart rendered successfully');

        // Apply any ticker update that arrived during chart load (NO volume - VolumeAccumulator handles that)
        if (this.lastTickerUpdate && this.lastTickerUpdate.symbol === this.symbol) {
          // console.log(`🔄 [3MO] Applying pending ticker update: ${this.lastTickerUpdate.symbol} = $${this.lastTickerUpdate.price}`);
          this.renderer.updateLivePrice(this.lastTickerUpdate.price, null);
        }
      }

      return this.data;
    } catch (error) {
      console.error(`❌ [3MO] Error loading data:`, error);
      throw error;
    }
  }

  /**
   * Subscribe to live WebSocket updates
   */
  subscribeToLiveData() {
    if (!this.socket) {
      // console.warn(`⚠️ [3MO] No socket connection available`);
      return;
    }

    // Subscribe to ticker updates from Coinbase
    this.socket.emit('subscribe', {
      product_ids: [this.symbol],
      channels: ['ticker', 'matches']
    });

    // console.log(`🔔 [3MO] Subscribed to ${this.symbol}`);
  }

  /**
   * Handle live ticker update from WebSocket (1D pattern)
   */
  handleTickerUpdate(data) {
    // console.log(`📈 [3MO] Received ticker: ${data.symbol} = $${data.price}, isActive=${this.isActive}`);

    // Ignore ticker updates for stock symbols (they don't have real-time data from Coinbase)
    const cryptoSymbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'];
    const isCrypto = cryptoSymbols.includes(this.symbol) || this.symbol.endsWith('-USD');

    if (!isCrypto) {
      console.log(`📊 [3MO] Ignoring ticker update for stock symbol: ${this.symbol}`);
      return;
    }

    // Check if this ticker is for our symbol
    const symbolMatches = data.symbol && this.symbol &&
      (data.symbol === `${this.symbol}-USD` ||
       data.symbol.includes(this.symbol) ||
       this.symbol.includes(data.symbol.split('-')[0]));

    if (!this.isActive || !data || !symbolMatches) {
      console.log(`  ❌ [3MO] Skipping ticker - isActive=${this.isActive}, symbolMatches=${symbolMatches}`);
      return;
    }

    const price = parseFloat(data.price);
    const volumeBTC = data.volume_today || 0;

    // console.log(`  ✅ [ update - price=${price}, volume=${volumeBTC}, data.length=${this.data.length}`);

    // Store latest ticker for this symbol (even if chart isn't loaded yet) - 1D pattern
    this.lastTickerUpdate = {
      symbol: this.symbol,
      price: price,
      volume_today: volumeBTC
    };

    // Update the chart renderer with live price (NO volume - VolumeAccumulator handles that)
    if (this.data.length > 0) {
      // console.log(`  🖼️ [3MO] Updating renderer with live price`);
      this.renderer.updateLivePrice(price, null);
    } else {
      // console.log(`  ⚠️ [, ticker stored for later`);
    }
  }

  /**
   * Handle trade update from WebSocket (placeholder)
   */
  handleTradeUpdate(data) {
    // Not needed for 3 month daily charts
  }

  /**
   * Deactivate this timeframe
   */
  deactivate() {
    // console.log(`⏸️ [3MO] Deactivating`);

    this.isActive = false;

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
    // console.log(`🔄 [3MO] Reloading...`);
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

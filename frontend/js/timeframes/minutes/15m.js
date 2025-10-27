/**
 * 15 Minute Timeframe
 * Independent implementation - displays 15-minute candlestick chart with live updates
 */
import { CanvasRenderer } from '../../chart-renderers/canvas-renderer.js';

export class Timeframe15m {
  constructor() {
    // Timeframe configuration
    this.id = '15m';
    this.name = '15 minutes';
    this.interval = '15m';
    this.period = '5d';
    this.category = 'minutes';
    this.isCustom = false;

    // Chart renderer (independent instance)
    this.renderer = new CanvasRenderer('15m');

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
    console.log(`📊 [15M] Initializing for ${symbol}`);

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
      console.error(`❌ [15M] Initialization error:`, error);
      return false;
    }
  }

  /**
   * Load historical data from backend
   */
  async loadHistoricalData() {
    const url = `/data/${this.symbol}?interval=${this.interval}&period=${this.period}`;
    console.log(`📥 [15M] Fetching: ${url}`);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.data = await response.json();

      if (this.data.length === 0) {
        throw new Error(`No data returned for ${this.symbol}`);
      }

      console.log(`✅ [15M] Loaded ${this.data.length} 15-minute candles`);

      // Render the data
      const success = await this.renderer.render(this.data, this.symbol);

      if (success) {
        console.log('✅ [15M] Chart rendered successfully');

        // Apply any ticker update that arrived during chart load
        if (this.lastTickerUpdate && this.lastTickerUpdate.symbol === this.symbol) {
          console.log(`🔄 [15M] Applying pending ticker update: ${this.lastTickerUpdate.symbol} = $${this.lastTickerUpdate.price}`);
          const volumeBTC = this.lastTickerUpdate.volume_today || 0;
          this.renderer.updateLivePrice(this.lastTickerUpdate.price, volumeBTC);
        }
      }

      return this.data;
    } catch (error) {
      console.error(`❌ [15M] Error loading data:`, error);
      throw error;
    }
  }

  /**
   * Subscribe to live WebSocket updates
   */
  subscribeToLiveData() {
    if (!this.socket) {
      console.warn(`⚠️ [15M] No socket connection available`);
      return;
    }

    // Subscribe to ticker updates from Coinbase
    this.socket.emit('subscribe', {
      product_ids: [this.symbol],
      channels: ['ticker', 'matches']
    });

    console.log(`🔔 [15M] Subscribed to ${this.symbol}`);
  }

  /**
   * Handle live ticker update from WebSocket
   */
  handleTickerUpdate(data) {
    console.log(`📈 [15M] Received ticker: ${data.symbol} = $${data.price}, isActive=${this.isActive}`);

    // Check if this ticker is for our symbol
    const symbolMatches = data.symbol && this.symbol &&
      (data.symbol === `${this.symbol}-USD` ||
       data.symbol.includes(this.symbol) ||
       this.symbol.includes(data.symbol.split('-')[0]));

    if (!this.isActive || !data || !symbolMatches) {
      console.log(`  ❌ [15M] Skipping ticker - isActive=${this.isActive}, symbolMatches=${symbolMatches}`);
      return;
    }

    const price = parseFloat(data.price);
    const volumeBTC = data.volume_today || 0;

    console.log(`  ✅ [15M] Processing ticker update - price=${price}, volume=${volumeBTC}, data.length=${this.data.length}`);

    // Store latest ticker for this symbol (even if chart isn't loaded yet)
    this.lastTickerUpdate = {
      symbol: this.symbol,
      price: price,
      volume_today: volumeBTC
    };

    // Update the chart renderer with live price
    if (this.data.length > 0) {
      console.log(`  🖼️ [15M] Updating renderer with live price`);
      this.renderer.updateLivePrice(price, volumeBTC);
    } else {
      console.log(`  ⚠️ [15M] Chart not loaded yet, ticker stored for later`);
    }
  }

  /**
   * Handle trade update from WebSocket (placeholder)
   */
  handleTradeUpdate(data) {
    // Not needed for 15-minute charts
  }

  /**
   * Deactivate this timeframe
   */
  deactivate() {
    console.log(`⏸️ [15M] Deactivating`);

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
    console.log(`🔄 [15M] Reloading...`);
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

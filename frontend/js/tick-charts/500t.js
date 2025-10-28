/**
 * 500-Tick Chart
 * Generates OHLCV bars from every 500 trades using Coinbase WebSocket
 */
import { CanvasRenderer } from '../chart-renderers/canvas-renderer.js';

export class TickChart500t {
  constructor() {
    // Tick chart configuration
    this.id = '500t';
    this.name = '500 ticks';
    this.tickThreshold = 500;
    this.category = 'ticks';
    this.isCustom = false;

    // Chart renderer (independent instance)
    this.renderer = new CanvasRenderer('500t');

    // Data and state
    this.symbol = null;
    this.data = []; // Completed bars
    this.socket = null;
    this.isActive = false;

    // Trade accumulator for current bar
    this.currentBar = {
      trades: [],
      tickCount: 0
    };
  }

  /**
   * Initialize the tick chart for a specific symbol
   */
  async initialize(symbol, socket) {
    console.log(`📊 [500T] Initializing for ${symbol}`);

    this.symbol = symbol;
    this.socket = socket;
    this.isActive = true;

    try {
      // Load historical tick bars from backend
      await this.loadHistoricalBars();

      // Subscribe to WebSocket trade updates
      this.subscribeToTrades();

      return true;
    } catch (error) {
      console.error(`❌ [500T] Initialization error:`, error);
      return false;
    }
  }

  /**
   * Load historical tick bars from backend
   */
  async loadHistoricalBars() {
    const url = `/data/tick/${this.symbol}/${this.tickThreshold}`;
    console.log(`📥 [500T] Fetching: ${url}`);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.data = await response.json();

      // console.log(`✅ [500T] Loaded ${this.data.length} tick bars`);

      // Render the data
      if (this.data.length > 0) {
        const success = await this.renderer.render(this.data, this.symbol);
        if (success) {
          console.log('✅ [500T] Chart rendered successfully');
        }
      } else {
        console.log('⚠️ [500T] No historical data, starting fresh');
        // Initialize empty chart
        await this.renderer.render([], this.symbol);
      }

      return this.data;
    } catch (error) {
      console.error(`❌ [500T] Error loading data:`, error);
      // Start with empty data if file doesn't exist
      this.data = [];
      await this.renderer.render([], this.symbol);
      return this.data;
    }
  }

  /**
   * Subscribe to live WebSocket trade updates
   * Note: Backend automatically subscribes to 'matches' channel for all symbols
   */
  subscribeToTrades() {
    if (!this.socket) {
      console.warn(`⚠️ [500T] No socket connection available`);
      return;
    }

    // No need to emit subscribe - backend handles Coinbase subscription centrally
    // Trade updates will arrive via 'trade_update' events routed through handleTradeUpdate()

    console.log(`🔔 [500T] Ready to receive ${this.symbol} trades`);
  }

  /**
   * Handle live trade update from WebSocket
   */
  handleTradeUpdate(data) {
    if (!this.isActive) {
      return;
    }

    // Check if this trade is for our symbol
    const symbolMatches = data.product_id === this.symbol;

    if (!symbolMatches) {
      return;
    }

    // console.log(`📈 [500T] Trade received: ${data.product_id} price=${data.price} size=${data.size}`);

    // Add trade to current bar
    this.currentBar.trades.push({
      price: parseFloat(data.price),
      size: parseFloat(data.size),
      time: data.time
    });
    this.currentBar.tickCount++;

    console.log(`  📊 [500T] Current bar: ${this.currentBar.tickCount}/${this.tickThreshold} trades`);

    // Check if we've reached the threshold
    if (this.currentBar.tickCount >= this.tickThreshold) {
      this.completeBar();
    }
  }

  /**
   * Complete the current bar and start a new one
   */
  async completeBar() {
    // console.log(`✅ [500T] Completing bar with ${this.currentBar.tickCount} trades`);

    const trades = this.currentBar.trades;

    // Construct OHLCV bar
    const bar = {
      Date: trades[trades.length - 1].time, // Use last trade timestamp
      Open: trades[0].price,
      High: Math.max(...trades.map(t => t.price)),
      Low: Math.min(...trades.map(t => t.price)),
      Close: trades[trades.length - 1].price,
      Volume: trades.reduce((sum, t) => sum + t.size, 0),
      TickCount: trades.length
    };

    console.log(`  📦 [500T] Bar created: O=${bar.Open} H=${bar.High} L=${bar.Low} C=${bar.Close} V=${bar.Volume.toFixed(4)}`);

    // Add to data array (keep last 300 bars)
    this.data.push(bar);
    if (this.data.length > 300) {
      this.data.shift(); // Remove oldest bar
    }

    // Update chart renderer
    await this.renderer.render(this.data, this.symbol);

    // Persist to backend (async, don't wait)
    this.saveBarToBackend(bar);

    // Reset accumulator
    this.currentBar = {
      trades: [],
      tickCount: 0
    };

    console.log(`🔄 [500T] Bar complete, accumulator reset`);
  }

  /**
   * Save bar to backend
   */
  async saveBarToBackend(bar) {
    const url = `/data/tick/${this.symbol}/${this.tickThreshold}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bar)
      });

      const result = await response.json();
      if (result.success) {
        console.log(`💾 [500T] Bar saved to backend (total: ${result.total_bars})`);
      } else {
        console.error(`❌ [500T] Failed to save bar:`, result.error);
      }
    } catch (error) {
      console.error(`❌ [500T] Error saving bar:`, error);
    }
  }

  /**
   * Deactivate this tick chart
   */
  deactivate() {
    console.log(`⏸️ [500T] Deactivating`);

    this.isActive = false;

    // Destroy the renderer to remove the canvas from DOM
    if (this.renderer) {
      this.renderer.destroy();
    }

    // No need to unsubscribe - backend manages Coinbase connection centrally
    // Just stop processing incoming trade updates by setting isActive = false
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
    this.currentBar = { trades: [], tickCount: 0 };
    this.symbol = null;
    this.socket = null;
  }

  /**
   * Reload chart data (full refresh)
   */
  async reload() {
    console.log(`🔄 [500T] Reloading...`);
    await this.loadHistoricalBars();
  }

  /**
   * Get current tick chart info
   */
  getInfo() {
    return {
      id: this.id,
      name: this.name,
      tickThreshold: this.tickThreshold,
      category: this.category,
      isCustom: this.isCustom,
      isActive: this.isActive,
      dataPoints: this.data.length,
      symbol: this.symbol,
      currentBarTicks: this.currentBar.tickCount
    };
  }
}

/**
 * 10-Tick Chart
 * Generates OHLCV bars from every 10 trades using Coinbase WebSocket
 */
import { CanvasRenderer } from '../chart-renderers/canvas-renderer.js';

export class TickChart10t {
  constructor() {
    // Tick chart configuration
    this.id = '10t';
    this.name = '10 ticks';
    this.tickThreshold = 10;
    this.category = 'ticks';
    this.isCustom = false;

    // Chart renderer (independent instance)
    this.renderer = new CanvasRenderer('10t');

    // Data and state
    this.symbol = null;
    this.data = []; // Completed bars
    this.socket = null;
    this.isActive = false; // Chart is visible and rendering
    this.isAccumulating = false; // Chart is processing trades (can be true even when not visible)

    // Trade accumulator for current bar (incremental OHLCV calculation - memory efficient)
    this.currentBar = {
      tickCount: 0,
      open: null,
      high: -Infinity,
      low: Infinity,
      close: null,
      volume: 0,
      firstTime: null,
      lastTime: null
    };
  }

  /**
   * Initialize the tick chart for a specific symbol
   */
  async initialize(symbol, socket) {
    // console.log(`📊 [10T] Initializing for ${symbol}`);

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
      console.error(`❌ [10T] Initialization error:`, error);
      return false;
    }
  }

  /**
   * Load historical tick bars from backend
   */
  async loadHistoricalBars() {
    const url = `/data/tick/${this.symbol}/${this.tickThreshold}`;
    // console.log(`📥 [10T] Fetching: ${url}`);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      this.data = await response.json();

      // console.log(`✅ [10T] Loaded ${this.data.length} tick bars`);

      // Render the data
      if (this.data.length > 0) {
        const success = await this.renderer.render(this.data, this.symbol);
        if (success) {
          // console.log('✅ [10T] Chart rendered successfully');
        }
      } else {
        // console.log('⚠️ [10T] No historical data, starting fresh');
        // Initialize empty chart
        await this.renderer.render([], this.symbol);
      }

      return this.data;
    } catch (error) {
      console.error(`❌ [10T] Error loading data:`, error);
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
      // console.warn(`⚠️ [10T] No socket connection available`);
      return;
    }

    // No need to emit subscribe - backend handles Coinbase subscription centrally
    // Trade updates will arrive via 'trade_update' events routed through handleTradeUpdate()

    // console.log(`🔔 [10T] Ready to receive ${this.symbol} trades`);
  }

  /**
   * Start accumulating trades in the background (without rendering)
   * Used for background streaming so charts are ready when user switches to them
   */
  async startAccumulating(symbol, socket) {
    // console.log(`🔄 [10T] Starting background accumulation for ${symbol}`);

    this.symbol = symbol;
    this.socket = socket;
    this.isAccumulating = true;

    try {
      // Load historical tick bars from backend
      const url = `/data/tick/${this.symbol}/${this.tickThreshold}`;
      const response = await fetch(url);
      if (response.ok) {
        this.data = await response.json();
        // console.log(`✅ [10T] Loaded ${this.data.length} tick bars (background)`);
      } else {
        this.data = [];
      }

      // Subscribe to WebSocket trade updates
      this.subscribeToTrades();

      return true;
    } catch (error) {
      console.error(`❌ [10T] Background init error:`, error);
      this.data = [];
      return false;
    }
  }

  /**
   * Activate this chart (make it visible and start rendering)
   */
  async activate() {
    // console.log(`▶️ [10T] Activating chart`);

    this.isActive = true;

    // Render the accumulated data
    if (this.data.length > 0) {
      const success = await this.renderer.render(this.data, this.symbol);
      if (success) {
        // console.log(`✅ [10T] Chart rendered with ${this.data.length} bars`);
      }
      return success;
    } else {
      // console.log(`⚠️ [10T] No data yet, rendering empty chart`);
      await this.renderer.render([], this.symbol);
      return true;
    }
  }

  /**
   * Handle live trade update from WebSocket
   */
  handleTradeUpdate(data) {
    // Check accumulation flag instead of isActive
    if (!this.isAccumulating) {
      return;
    }

    // Check if this trade is for our symbol
    const symbolMatches = data.product_id === this.symbol;

    if (!symbolMatches) {
      return;
    }

    // console.log(`📈 [10T] Trade received: ${data.product_id} price=${data.price} size=${data.size}`);

    // Incrementally update OHLCV (memory-efficient - no array storage)
    const price = parseFloat(data.price);
    const size = parseFloat(data.size);
    const time = data.time;

    if (this.currentBar.tickCount === 0) {
      // First trade in bar
      this.currentBar.open = price;
      this.currentBar.firstTime = time;
    }

    // Update high/low
    this.currentBar.high = Math.max(this.currentBar.high, price);
    this.currentBar.low = Math.min(this.currentBar.low, price);

    // Update close and volume
    this.currentBar.close = price;
    this.currentBar.volume += size;
    this.currentBar.lastTime = time;
    this.currentBar.tickCount++;

    // console.log(`  📊 [10T] Current bar: ${this.currentBar.tickCount}/${this.tickThreshold} trades`);

    // Check if we've reached the threshold
    if (this.currentBar.tickCount >= this.tickThreshold) {
      this.completeBar();
    }
  }

  /**
   * Complete the current bar and start a new one
   */
  async completeBar() {
    // console.log(`✅ [10T] Completing bar with ${this.currentBar.tickCount} trades`);

    // Construct OHLCV bar from incremental data (already calculated)
    const bar = {
      Date: this.currentBar.lastTime, // Use last trade timestamp
      Open: this.currentBar.open,
      High: this.currentBar.high,
      Low: this.currentBar.low,
      Close: this.currentBar.close,
      Volume: this.currentBar.volume,
      TickCount: this.currentBar.tickCount
    };

    // console.log(`  📦 [10T] Bar created: O=${bar.Open} H=${bar.High} L=${bar.Low} C=${bar.Close} V=${bar.Volume.toFixed(4)}`);

    // Add to data array (keep last 300 bars)
    this.data.push(bar);
    if (this.data.length > 300) {
      this.data.shift(); // Remove oldest bar
    }

    // Update chart renderer (only if this chart is currently active/visible)
    if (this.isActive) {
      await this.renderer.render(this.data, this.symbol);
    }

    // Persist to backend (async, don't wait)
    this.saveBarToBackend(bar);

    // Reset accumulator (memory-efficient reset)
    this.currentBar = {
      tickCount: 0,
      open: null,
      high: -Infinity,
      low: Infinity,
      close: null,
      volume: 0,
      firstTime: null,
      lastTime: null
    };

    // console.log(`🔄 [10T] Bar complete, accumulator reset`);
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
        // console.log(`💾 [10T] Bar saved to backend (total: ${result.total_bars})`);
      } else {
        console.error(`❌ [10T] Failed to save bar:`, result.error);
      }
    } catch (error) {
      console.error(`❌ [10T] Error saving bar:`, error);
    }
  }

  /**
   * Deactivate this tick chart (hide it, but continue accumulating in background)
   */
  deactivate() {
    // console.log(`⏸️ [10T] Deactivating (still accumulating in background)`);

    this.isActive = false;

    // Destroy the renderer to remove the canvas from DOM
    if (this.renderer) {
      this.renderer.destroy();
    }

    // Keep accumulating trades (isAccumulating stays true)
    // No need to unsubscribe - backend manages Coinbase connection centrally
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
    this.currentBar = {
      tickCount: 0,
      open: null,
      high: -Infinity,
      low: Infinity,
      close: null,
      volume: 0,
      firstTime: null,
      lastTime: null
    };
    this.symbol = null;
    this.socket = null;
    this.isActive = false;
    this.isAccumulating = false;
  }

  /**
   * Reload chart data (full refresh)
   */
  async reload() {
    // console.log(`🔄 [10T] Reloading...`);
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
      isAccumulating: this.isAccumulating,
      dataPoints: this.data.length,
      symbol: this.symbol,
      currentBarTicks: this.currentBar.tickCount
    };
  }
}

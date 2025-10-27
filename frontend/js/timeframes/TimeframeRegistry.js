/**
 * Timeframe Registry
 * Central registry for all timeframe implementations
 * Manages timeframe instances and provides switching logic
 */

// Import working timeframes (standalone implementations only)
import { Timeframe1m } from './minutes/1m.js';
import { Timeframe5m } from './minutes/5m.js';
import { Timeframe15m } from './minutes/15m.js';
import { Timeframe1h } from './minutes/1h.js';
import { Timeframe1d } from './days/1d.js';
import { Timeframe1w } from './days/1w.js';
import { Timeframe1mo } from './days/1mo.js';
import { Timeframe3mo } from './days/3mo.js';

export class TimeframeRegistry {
  constructor() {
    this.timeframes = new Map();
    this.currentTimeframe = null;
    this.currentSymbol = null;
    this.socket = null;

    // Initialize all timeframes
    this.registerAllTimeframes();
  }

  /**
   * Register all timeframe instances (only working standalone implementations)
   */
  registerAllTimeframes() {
    // Minutes (standalone implementations with ticker timing fix)
    this.register(new Timeframe1m());
    this.register(new Timeframe5m());
    this.register(new Timeframe15m());
    this.register(new Timeframe1h());

    // Days (standalone implementations with ticker timing fix)
    this.register(new Timeframe1d());
    this.register(new Timeframe1w());
    this.register(new Timeframe1mo());
    this.register(new Timeframe3mo());

    console.log(`📚 Registered ${this.timeframes.size} timeframes`);
  }

  /**
   * Register a single timeframe
   */
  register(timeframe) {
    this.timeframes.set(timeframe.id, timeframe);
  }

  /**
   * Get a timeframe by ID
   */
  get(timeframeId) {
    return this.timeframes.get(timeframeId);
  }

  /**
   * Get all timeframes grouped by category
   */
  getAllGrouped() {
    const grouped = {
      ticks: [],
      seconds: [],
      minutes: [],
      hours: [],
      days: [],
      ranges: []
    };

    for (const timeframe of this.timeframes.values()) {
      grouped[timeframe.category].push(timeframe);
    }

    return grouped;
  }

  /**
   * Switch to a different timeframe
   */
  async switchTimeframe(timeframeId, symbol, socket) {
    console.log(`🔄 Switching to timeframe: ${timeframeId} for ${symbol}`);

    const newTimeframe = this.get(timeframeId);
    if (!newTimeframe) {
      console.error(`❌ Timeframe not found: ${timeframeId}`);
      return false;
    }

    // Deactivate current timeframe
    if (this.currentTimeframe) {
      this.currentTimeframe.deactivate();
    }

    // Activate new timeframe
    this.currentTimeframe = newTimeframe;
    this.currentSymbol = symbol;
    this.socket = socket;

    const success = await newTimeframe.initialize(symbol, socket);

    if (success) {
      console.log(`✅ Switched to ${newTimeframe.name}`);
      return true;
    } else {
      console.error(`❌ Failed to switch to ${newTimeframe.name}`);
      return false;
    }
  }

  /**
   * Handle ticker update - route to current timeframe
   */
  handleTickerUpdate(data) {
    if (this.currentTimeframe) {
      this.currentTimeframe.handleTickerUpdate(data);
    }
  }

  /**
   * Handle trade update - route to current timeframe
   */
  handleTradeUpdate(data) {
    if (this.currentTimeframe) {
      this.currentTimeframe.handleTradeUpdate(data);
    }
  }

  /**
   * Reload current timeframe
   */
  async reloadCurrent() {
    if (this.currentTimeframe) {
      await this.currentTimeframe.reload();
    }
  }

  /**
   * Get current timeframe info
   */
  getCurrentInfo() {
    if (this.currentTimeframe) {
      return this.currentTimeframe.getInfo();
    }
    return null;
  }

  /**
   * Clean up all timeframes
   */
  destroy() {
    for (const timeframe of this.timeframes.values()) {
      timeframe.destroy();
    }
    this.timeframes.clear();
    this.currentTimeframe = null;
  }
}

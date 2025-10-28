/**
 * Tick Chart Registry
 * Central registry for all tick chart implementations
 * Manages tick chart instances and provides switching logic
 */

// Tick charts
import { TickChart10t } from './10t.js';
import { TickChart50t } from './50t.js';
import { TickChart100t } from './100t.js';
import { TickChart250t } from './250t.js';
import { TickChart500t } from './500t.js';
import { TickChart1000t } from './1000t.js';

export class TickChartRegistry {
  constructor() {
    this.tickCharts = new Map();
    this.currentTickChart = null;
    this.currentSymbol = null;
    this.socket = null;

    // Initialize all tick charts
    this.registerAllTickCharts();
  }

  /**
   * Register all tick chart instances
   */
  registerAllTickCharts() {
    // Ticks (trade-based charts)
    this.register(new TickChart10t());
    this.register(new TickChart50t());
    this.register(new TickChart100t());
    this.register(new TickChart250t());
    this.register(new TickChart500t());
    this.register(new TickChart1000t());

    console.log(`📊 Registered ${this.tickCharts.size} tick charts`);
  }

  /**
   * Register a single tick chart
   */
  register(tickChart) {
    this.tickCharts.set(tickChart.id, tickChart);
  }

  /**
   * Get a tick chart by ID
   */
  get(tickChartId) {
    return this.tickCharts.get(tickChartId);
  }

  /**
   * Get all tick charts grouped by category
   */
  getAllGrouped() {
    const grouped = {
      ticks: []
    };

    for (const tickChart of this.tickCharts.values()) {
      grouped.ticks.push(tickChart);
    }

    return grouped;
  }

  /**
   * Initialize ALL tick charts to accumulate trades in background
   * Called when a new symbol is loaded
   */
  async initializeAllForSymbol(symbol, socket) {
    console.log(`🚀 Initializing background accumulation for all tick charts: ${symbol}`);

    this.currentSymbol = symbol;
    this.socket = socket;

    // Start all tick charts accumulating in the background
    for (const tickChart of this.tickCharts.values()) {
      await tickChart.startAccumulating(symbol, socket);
    }

    console.log(`✅ All tick charts now accumulating ${symbol} trades in background`);
  }

  /**
   * Switch to a different tick chart (for display)
   * All charts continue accumulating in background, but only one renders
   */
  async switchTickChart(tickChartId, symbol, socket) {
    console.log(`🔄 Switching to tick chart: ${tickChartId} for ${symbol}`);

    const newTickChart = this.get(tickChartId);
    if (!newTickChart) {
      console.error(`❌ Tick chart not found: ${tickChartId}`);
      return false;
    }

    // If symbol changed, reinitialize all tick charts
    if (symbol !== this.currentSymbol) {
      await this.initializeAllForSymbol(symbol, socket);
    }

    // Deactivate current tick chart (hide it, but keep accumulating)
    if (this.currentTickChart) {
      this.currentTickChart.deactivate();
    }

    // Activate new tick chart (show it)
    this.currentTickChart = newTickChart;
    this.currentSymbol = symbol;
    this.socket = socket;

    const success = await newTickChart.activate();

    if (success) {
      console.log(`✅ Switched to ${newTickChart.name}`);
      return true;
    } else {
      console.error(`❌ Failed to switch to ${newTickChart.name}`);
      return false;
    }
  }

  /**
   * Handle trade update - route to ALL tick charts for background accumulation
   * This is called when a new trade comes from the 'matches' channel
   */
  handleTradeUpdate(data) {
    // Route to ALL tick charts so they can accumulate in background
    for (const tickChart of this.tickCharts.values()) {
      if (tickChart.isAccumulating) {
        tickChart.handleTradeUpdate(data);
      }
    }
  }

  /**
   * Reload current tick chart
   */
  async reloadCurrent() {
    if (this.currentTickChart) {
      await this.currentTickChart.reload();
    }
  }

  /**
   * Get current tick chart info
   */
  getCurrentInfo() {
    if (this.currentTickChart) {
      return this.currentTickChart.getInfo();
    }
    return null;
  }

  /**
   * Clean up all tick charts
   */
  destroy() {
    for (const tickChart of this.tickCharts.values()) {
      tickChart.destroy();
    }
    this.tickCharts.clear();
    this.currentTickChart = null;
  }
}

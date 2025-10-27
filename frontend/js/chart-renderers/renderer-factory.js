/**
 * Renderer Factory
 * Automatically selects the correct renderer based on timeframe
 * DO NOT MODIFY - This manages renderer selection
 */

import { IntradayRenderer } from './intraday-renderer.js';
import { DailyRenderer } from './daily-renderer.js';

export class RendererFactory {
  static renderers = [
    new IntradayRenderer(),
    new DailyRenderer()
  ];

  /**
   * Get the appropriate renderer for a given interval
   * @param {string} interval - The timeframe interval (e.g., '1m', '5m', '1d')
   * @returns {BaseChartRenderer} The renderer instance for this interval
   */
  static getRenderer(interval) {
    const renderer = this.renderers.find(r => r.supports(interval));

    if (!renderer) {
      console.error(`No renderer found for interval: ${interval}`);
      // Fallback to intraday renderer
      return this.renderers[0];
    }

    console.log(`📊 Using ${renderer.constructor.name} for ${interval}`);
    return renderer;
  }

  /**
   * Render chart using the appropriate renderer
   * @param {Array} data - Chart data
   * @param {string} symbol - Stock symbol
   * @param {string} interval - Timeframe interval
   * @param {Object} options - Rendering options
   * @returns {Promise<boolean>} Success status
   */
  static async renderChart(data, symbol, interval, options = {}) {
    const renderer = this.getRenderer(interval);
    return await renderer.render(data, symbol, interval, options);
  }

  /**
   * Update chart with live data
   * @param {string} interval - Current timeframe
   * @param {Object} tickerData - Live ticker data
   * @returns {Promise<boolean>} Success status
   */
  static async updateChart(interval, tickerData) {
    const renderer = this.getRenderer(interval);
    return await renderer.updateWithTicker(tickerData);
  }

  /**
   * Clean up all renderers
   */
  static destroyAll() {
    this.renderers.forEach(r => r.destroy());
  }
}

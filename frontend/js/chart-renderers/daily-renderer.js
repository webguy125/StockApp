/**
 * Daily/Weekly/Monthly Chart Renderer
 * Handles: 1d, 1wk, 1mo timeframes
 * DO NOT MODIFY - This is the stable daily+ rendering logic
 */

import { BaseChartRenderer } from './base-renderer.js';
import { VolumeHandler } from './volume-handler.js';

export class DailyRenderer extends BaseChartRenderer {
  constructor(plotId = 'tos-plot') {
    super(plotId);
    this.supportedIntervals = ['1d', '1wk', '1mo'];
  }

  /**
   * Check if this renderer supports the given interval
   */
  supports(interval) {
    return this.supportedIntervals.includes(interval);
  }

  /**
   * Create daily-specific chart layout
   */
  createLayout(symbol, interval, volumeMode = 'subgraph', showVolume = true) {
    const layout = this.createBaseLayout(symbol, interval);

    // Daily-specific settings
    layout.xaxis.tickformat = '%Y-%m-%d';
    layout.xaxis.nticks = 15;
    layout.xaxis.tickfont = { size: 10 };

    // Add rangebreaks only for daily charts (hide weekends)
    if (interval === '1d') {
      layout.xaxis.rangebreaks = [
        {
          bounds: ['sat', 'mon'],
          pattern: 'day of week'
        }
      ];
    }

    // Add volume axis if enabled
    if (showVolume) {
      layout.yaxis2 = VolumeHandler.createVolumeAxis(volumeMode, 0);

      // Adjust main chart domain if volume is in subgraph mode
      if (volumeMode === 'subgraph') {
        layout.yaxis.domain = [0.28, 1];  // Leave room for volume at bottom
      }
    }

    return layout;
  }

  /**
   * Render daily chart
   */
  async render(data, symbol, interval, options = {}) {
    const {
      volumeMode = 'subgraph',
      showVolume = true
    } = options;

    this.initialize(symbol, interval, data);

    // Prepare data arrays
    const dates = data.map(d => d.Date);
    const opens = data.map(d => d.Open);
    const highs = data.map(d => d.High);
    const lows = data.map(d => d.Low);
    const closes = data.map(d => d.Close);
    const volumes = data.map(d => d.Volume || 0);

    // Create traces
    const traces = [
      this.createCandlestickTrace(dates, opens, highs, lows, closes, symbol)
    ];

    // Add volume trace if enabled
    if (showVolume) {
      traces.push(VolumeHandler.createVolumeTrace(dates, volumes, volumeMode));
    }

    // Create layout
    const layout = this.createLayout(symbol, interval, volumeMode, showVolume);

    // Render
    return await this.renderChart(traces, layout);
  }

  /**
   * Update chart with live ticker data (for real-time updates)
   */
  async updateWithTicker(tickerData) {
    const plotDiv = this.getPlotDiv();
    if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) return false;

    const candleIndex = plotDiv.data.findIndex(trace => trace.type === 'candlestick');
    if (candleIndex === -1) return false;

    const candleTrace = plotDiv.data[candleIndex];
    if (!candleTrace.close || candleTrace.close.length === 0) return false;

    const lastIndex = candleTrace.close.length - 1;
    const price = tickerData.price;

    try {
      // Update last candle with new price
      candleTrace.close[lastIndex] = price;
      candleTrace.high[lastIndex] = Math.max(candleTrace.high[lastIndex], price);
      candleTrace.low[lastIndex] = Math.min(candleTrace.low[lastIndex], price);

      // Update the chart
      await Plotly.restyle(plotDiv, {
        open: [candleTrace.open],
        high: [candleTrace.high],
        low: [candleTrace.low],
        close: [candleTrace.close]
      }, [candleIndex]);

      return true;
    } catch (error) {
      console.error('Error updating daily chart:', error);
      return false;
    }
  }
}

/**
 * Base Chart Renderer
 * Foundation class that all timeframe-specific renderers extend
 * DO NOT MODIFY - This provides core chart rendering functionality
 */

export class BaseChartRenderer {
  constructor(plotId = 'tos-plot') {
    this.plotId = plotId;
    this.currentSymbol = null;
    this.currentInterval = null;
    this.chartData = null;
  }

  /**
   * Get the plot div element
   */
  getPlotDiv() {
    return document.getElementById(this.plotId);
  }

  /**
   * Create basic candlestick trace
   * Override in subclasses for timeframe-specific customization
   */
  createCandlestickTrace(dates, opens, highs, lows, closes, symbol) {
    return {
      x: dates,
      open: opens,
      high: highs,
      low: lows,
      close: closes,
      type: 'candlestick',
      name: symbol,
      increasing: { line: { color: '#00c851' } },
      decreasing: { line: { color: '#ff4444' } },
      hoverinfo: 'none',
      hovertemplate: '<extra></extra>'
    };
  }

  /**
   * Create base chart layout
   * Override in subclasses for timeframe-specific layout
   */
  createBaseLayout(symbol, interval) {
    return {
      title: {
        text: `${symbol} - ${interval.toUpperCase()}`,
        font: { color: '#e0e0e0', size: 16 }
      },
      xaxis: {
        gridcolor: '#404040',
        color: '#a0a0a0',
        rangeslider: { visible: false },
        domain: [0, 1],
        showspikes: false,
        showticklabels: true,
        type: 'date'
      },
      yaxis: {
        title: 'Price',
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: [0, 1],
        showspikes: false,
        showticklabels: true
      },
      plot_bgcolor: '#1a1a1a',
      paper_bgcolor: '#1a1a1a',
      font: { color: '#e0e0e0' },
      margin: { l: 50, r: 70, t: 80, b: 50 },
      showlegend: true,
      legend: {
        x: 0,
        y: 1,
        bgcolor: 'rgba(0,0,0,0.3)',
        bordercolor: '#404040',
        borderwidth: 1
      },
      hovermode: 'closest',
      hoverdistance: 100
    };
  }

  /**
   * Render chart with Plotly
   * Common rendering logic for all timeframes
   */
  async renderChart(traces, layout) {
    const plotDiv = this.getPlotDiv();
    if (!plotDiv) {
      console.error(`Plot div ${this.plotId} not found`);
      return false;
    }

    try {
      await Plotly.newPlot(plotDiv, traces, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
        scrollZoom: true,
        editable: true,
        dragmode: 'pan'
      });

      console.log(`✅ Chart rendered successfully for ${this.currentSymbol} (${this.currentInterval})`);
      return true;
    } catch (error) {
      console.error(`❌ Error rendering chart:`, error);
      return false;
    }
  }

  /**
   * Update existing chart with new data
   * Override in subclasses for timeframe-specific update logic
   */
  async updateChart(newData) {
    const plotDiv = this.getPlotDiv();
    if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) {
      console.warn('Cannot update chart - not initialized');
      return false;
    }

    // Find candlestick trace
    const candleIndex = plotDiv.data.findIndex(trace => trace.type === 'candlestick');
    if (candleIndex === -1) {
      console.warn('No candlestick trace found');
      return false;
    }

    try {
      // Basic update - override in subclasses for specific behavior
      await Plotly.restyle(plotDiv, {
        x: [newData.dates],
        open: [newData.opens],
        high: [newData.highs],
        low: [newData.lows],
        close: [newData.closes]
      }, [candleIndex]);

      return true;
    } catch (error) {
      console.error('Error updating chart:', error);
      return false;
    }
  }

  /**
   * Initialize the renderer with symbol and interval
   */
  initialize(symbol, interval, data) {
    this.currentSymbol = symbol;
    this.currentInterval = interval;
    this.chartData = data;
    console.log(`🎯 Initialized ${this.constructor.name} for ${symbol} (${interval})`);
  }

  /**
   * Clean up renderer resources
   */
  destroy() {
    const plotDiv = this.getPlotDiv();
    if (plotDiv) {
      Plotly.purge(plotDiv);
    }
    this.chartData = null;
    console.log(`🗑️ Destroyed renderer for ${this.currentSymbol}`);
  }
}

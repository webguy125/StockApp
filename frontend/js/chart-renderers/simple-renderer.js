/**
 * Simple Chart Renderer
 * Basic 1D chart with live updates - NO COMPLEX LOGIC
 */

import { Crosshair } from '../chart-interactions/crosshair.js';
import { PanZoom } from '../chart-interactions/pan-zoom.js';
import { KeyboardControls } from '../chart-interactions/keyboard-controls.js';

export class SimpleRenderer {
  constructor() {
    this.plotId = 'tos-plot';

    // Initialize interaction modules
    this.crosshair = new Crosshair(this.plotId);
    this.panZoom = new PanZoom(this.plotId);
    this.keyboardControls = new KeyboardControls(this.plotId, {
      panZoom: this.panZoom
    });
  }

  /**
   * Render a simple 1D candlestick chart
   */
  async render(data, symbol) {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv) {
      console.error(`❌ Plot div '${this.plotId}' not found in DOM`);
      return false;
    }
    console.log(`✅ Found plot div: ${this.plotId}`);

    // Extract data - keep dates exactly as received from backend
    const dates = data.map(d => d.Date);
    const opens = data.map(d => d.Open);
    const highs = data.map(d => d.High);
    const lows = data.map(d => d.Low);
    const closes = data.map(d => d.Close);
    const volumes = data.map(d => d.Volume || 0);

    console.log(`📊 Rendering ${dates.length} candles`);
    console.log(`📅 First date: ${dates[0]}, Last date: ${dates[dates.length - 1]}`);
    console.log(`📅 Last 5 dates:`, dates.slice(-5));
    console.log(`📊 Last 3 closes:`, closes.slice(-3));

    // Fix backend date issue: backend uses UTC which may be tomorrow in your timezone
    // Remove future dates and ensure we have today's candle based on LOCAL time
    const now = new Date();
    const todayLocal = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const todayStr = todayLocal.toISOString().split('T')[0]; // Local date as YYYY-MM-DD

    console.log(`📅 Today (local): ${todayStr}`);

    // Remove any future dates (beyond today local time)
    while (dates.length > 0 && dates[dates.length - 1] > todayStr) {
      console.warn(`⚠️ Removing future date: ${dates[dates.length - 1]}`);
      dates.pop();
      opens.pop();
      highs.pop();
      lows.pop();
      closes.pop();
      volumes.pop();
    }

    // Add today's candle if missing
    if (dates.length > 0 && dates[dates.length - 1] < todayStr) {
      const lastClose = closes[closes.length - 1];
      console.log(`🆕 Adding today's candle (${todayStr})`);
      dates.push(todayStr);
      opens.push(lastClose);
      highs.push(lastClose);
      lows.push(lastClose);
      closes.push(lastClose);
      volumes.push(0);
    }

    console.log(`📅 After date fix - Last 5 dates:`, dates.slice(-5));

    // Create candlestick trace
    const candleTrace = {
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

    // Create volume trace
    const volumeTrace = {
      x: dates,
      y: volumes,
      type: 'bar',
      name: 'Volume',
      yaxis: 'y2',
      marker: { color: 'rgba(0, 188, 212, 0.7)' },
      hoverinfo: 'none',
      hovertemplate: '<extra></extra>'
    };

    // Get crosshair and pan/zoom configurations
    const crosshairConfig = this.crosshair.getConfig();
    const panZoomLayoutConfig = this.panZoom.getLayoutConfig();
    const panZoomConfig = this.panZoom.getConfig();

    // Simple layout
    const layout = {
      title: {
        text: `${symbol} - Daily`,
        font: { color: '#e0e0e0', size: 16 }
      },
      xaxis: {
        type: 'category',
        gridcolor: '#404040',
        color: '#a0a0a0',
        rangeslider: { visible: false },
        tickangle: -45,
        ...crosshairConfig.xaxis  // Apply crosshair config
      },
      yaxis: {
        title: 'Price',
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: [0.28, 1],
        ...crosshairConfig.yaxis  // Apply crosshair config
      },
      yaxis2: {
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: [0, 0.25]
      },
      plot_bgcolor: '#1a1a1a',
      paper_bgcolor: '#1a1a1a',
      font: { color: '#e0e0e0' },
      margin: { l: 50, r: 70, t: 80, b: 50 },
      showlegend: false,
      ...panZoomLayoutConfig  // Apply pan/zoom layout config
    };

    try {
      await Plotly.newPlot(plotDiv, [candleTrace, volumeTrace], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
        ...panZoomConfig  // Apply pan/zoom config
      });

      console.log('✅ Simple chart rendered successfully');
      return true;
    } catch (error) {
      console.error('❌ Error rendering chart:', error);
      return false;
    }
  }

  /**
   * Update the last candle with new price
   */
  updateLivePrice(price) {
    console.log(`🔴 updateLivePrice called with price: ${price}`);

    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) {
      console.log('❌ No plot div or data');
      return false;
    }

    const candleTrace = plotDiv.data[0];
    if (!candleTrace || !candleTrace.close) {
      console.log('❌ No candle trace');
      return false;
    }

    const lastIndex = candleTrace.close.length - 1;
    console.log(`📊 Updating candle ${lastIndex}: close ${candleTrace.close[lastIndex]} -> ${price}`);

    // Update last candle
    candleTrace.close[lastIndex] = price;
    candleTrace.high[lastIndex] = Math.max(candleTrace.high[lastIndex], price);
    candleTrace.low[lastIndex] = Math.min(candleTrace.low[lastIndex], price);

    Plotly.restyle(plotDiv, {
      open: [candleTrace.open],
      high: [candleTrace.high],
      low: [candleTrace.low],
      close: [candleTrace.close]
    }, [0]);

    console.log('✅ Chart updated');
    return true;
  }

  /**
   * Clean up interaction modules
   */
  destroy() {
    if (this.keyboardControls) {
      this.keyboardControls.destroy();
    }
  }

  /**
   * Get references to interaction modules for external control
   */
  getInteractions() {
    return {
      crosshair: this.crosshair,
      panZoom: this.panZoom,
      keyboardControls: this.keyboardControls
    };
  }
}

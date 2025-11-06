/**
 * ORD Volume Integration Script
 * Completely segregated integration for ORD Volume feature
 * Wires up button to ORD Volume system
 *
 * NO SHARED CODE - Standalone initialization
 */

import { ORDVolumeController } from './ORDVolumeController.js';
import { ORDVolumeRenderer } from './ORDVolumeRenderer.js';
import { ORDVolumeBridge } from './ord-volume-bridge.js';

/**
 * Initialize ORD Volume feature
 * Call this function when the app loads
 */
export function initializeORDVolume() {
  console.log('[ORD Volume] Initializing segregated ORD Volume system...');

  // Create controller and renderer instances
  const ordVolumeController = new ORDVolumeController();
  let ordVolumeRenderer = null;

  // Wire up the ORD Volume button
  const ordVolumeBtn = document.getElementById('btn-ord-volume');

  if (!ordVolumeBtn) {
    console.error('[ORD Volume] Button not found: btn-ord-volume');
    return;
  }

  ordVolumeBtn.addEventListener('click', () => {
    console.log('[ORD Volume] Button clicked');

    // Get current chart data (need to extract from existing chart)
    const candles = extractCandleData();
    const symbol = extractCurrentSymbol();
    const canvas = extractChartCanvas();
    const chartState = extractChartState();

    if (!candles || candles.length === 0) {
      alert('Please load chart data before using ORD Volume');
      return;
    }

    // ALWAYS recreate renderer to ensure we have the latest canvas/chartState
    if (canvas) {
      ordVolumeRenderer = new ORDVolumeRenderer(canvas, chartState);

      // Register renderer with bridge for draw mode
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.setORDVolumeRenderer(ordVolumeRenderer);
        console.log('[ORD Volume] Renderer created and registered with bridge');
      }
    } else {
      alert('Chart canvas not found');
      return;
    }

    // Open ORD Volume modal
    ordVolumeController.open(candles, symbol, ordVolumeRenderer);
  });

  console.log('[ORD Volume] Initialization complete');
}

/**
 * Extract candle data from existing chart
 * This function needs to be customized based on how the main app stores candle data
 * @returns {Array} Array of OHLCV objects
 */
function extractCandleData() {
  // Method 1: Get data from active timeframe
  if (window.tosApp && window.tosApp.activeChartType === 'timeframe') {
    const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
    if (currentTimeframe && currentTimeframe.data && currentTimeframe.data.length > 0) {
      console.log(`[ORD Volume] Found ${currentTimeframe.data.length} candles from timeframe ${window.tosApp.currentTimeframeId}`);
      return convertToOHLCV(currentTimeframe.data);
    }
  }

  // Method 2: Get data from active tick chart
  if (window.tosApp && window.tosApp.activeChartType === 'tick') {
    const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
    if (currentTickChart && currentTickChart.data && currentTickChart.data.length > 0) {
      console.log(`[ORD Volume] Found ${currentTickChart.data.length} candles from tick chart ${window.tosApp.currentTickChartId}`);
      return convertToOHLCV(currentTickChart.data);
    }
  }

  // Method 3: Check if there's a global chartData variable
  if (window.chartData) {
    return convertToOHLCV(window.chartData);
  }

  // Method 4: Try to extract from Plotly chart
  try {
    const chartDiv = document.getElementById('chart');
    if (chartDiv && chartDiv.data && chartDiv.data[0]) {
      const trace = chartDiv.data[0];
      if (trace.open && trace.high && trace.low && trace.close) {
        const candles = [];
        for (let i = 0; i < trace.open.length; i++) {
          candles.push({
            open: trace.open[i],
            high: trace.high[i],
            low: trace.low[i],
            close: trace.close[i],
            volume: trace.volume ? trace.volume[i] : 0
          });
        }
        return candles;
      }
    }
  } catch (e) {
    console.error('[ORD Volume] Error extracting candle data:', e);
  }

  console.error('[ORD Volume] Could not find candle data');
  return [];
}

/**
 * Convert chart data to OHLCV format
 * @param {Object} chartData - Raw chart data
 * @returns {Array} OHLCV array
 */
function convertToOHLCV(chartData) {
  if (!chartData) return [];

  // If already in correct format (array of objects with OHLCV)
  if (Array.isArray(chartData) && chartData.length > 0) {
    const first = chartData[0];

    // Check if it has OHLCV properties
    if (first.open !== undefined && first.high !== undefined &&
        first.low !== undefined && first.close !== undefined) {
      // Already in correct format, just ensure volume exists
      return chartData.map(candle => ({
        open: parseFloat(candle.open) || candle.Open || 0,
        high: parseFloat(candle.high) || candle.High || 0,
        low: parseFloat(candle.low) || candle.Low || 0,
        close: parseFloat(candle.close) || candle.Close || 0,
        volume: parseFloat(candle.volume) || candle.Volume || 0
      }));
    }

    // Check for capitalized keys (Date, Open, High, Low, Close, Volume)
    if (first.Open !== undefined && first.High !== undefined) {
      return chartData.map(candle => ({
        open: parseFloat(candle.Open) || 0,
        high: parseFloat(candle.High) || 0,
        low: parseFloat(candle.Low) || 0,
        close: parseFloat(candle.Close) || 0,
        volume: parseFloat(candle.Volume) || 0
      }));
    }
  }

  // Handle object with arrays
  if (chartData.open && Array.isArray(chartData.open)) {
    const candles = [];
    for (let i = 0; i < chartData.open.length; i++) {
      candles.push({
        open: parseFloat(chartData.open[i]) || 0,
        high: parseFloat(chartData.high[i]) || 0,
        low: parseFloat(chartData.low[i]) || 0,
        close: parseFloat(chartData.close[i]) || 0,
        volume: parseFloat(chartData.volume ? chartData.volume[i] : 0) || 0
      });
    }
    return candles;
  }

  return [];
}

/**
 * Extract current symbol from the chart
 * @returns {String} Current symbol
 */
function extractCurrentSymbol() {
  // Method 1: Check input field
  const symbolInput = document.getElementById('tos-symbol-input');
  if (symbolInput && symbolInput.value) {
    return symbolInput.value;
  }

  // Method 2: Check tosApp
  if (window.tosApp && window.tosApp.currentSymbol) {
    return window.tosApp.currentSymbol;
  }

  // Method 3: Default
  return 'BTC-USD';
}

/**
 * Extract chart canvas element
 * @returns {HTMLCanvasElement} Canvas element
 */
function extractChartCanvas() {
  // Method 1: Get from active timeframe renderer
  if (window.tosApp && window.tosApp.activeChartType === 'timeframe') {
    const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
    if (currentTimeframe && currentTimeframe.renderer && currentTimeframe.renderer.canvas) {
      console.log('[ORD Volume] Found canvas from timeframe renderer');
      return currentTimeframe.renderer.canvas;
    }
  }

  // Method 2: Get from active tick chart renderer
  if (window.tosApp && window.tosApp.activeChartType === 'tick') {
    const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
    if (currentTickChart && currentTickChart.renderer && currentTickChart.renderer.canvas) {
      console.log('[ORD Volume] Found canvas from tick chart renderer');
      return currentTickChart.renderer.canvas;
    }
  }

  // Method 3: Direct canvas lookup
  const canvas = document.getElementById('chartCanvas');
  if (canvas) return canvas;

  // Method 4: Query selector
  const canvasQuery = document.querySelector('canvas');
  if (canvasQuery) return canvasQuery;

  console.error('[ORD Volume] Could not find canvas element');
  return null;
}

/**
 * Extract chart state (for coordinate conversion)
 * @returns {Object} Chart state object with conversion functions
 */
function extractChartState() {
  // Method 1: Get from active timeframe renderer
  if (window.tosApp && window.tosApp.activeChartType === 'timeframe') {
    const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
    if (currentTimeframe && currentTimeframe.renderer) {
      const renderer = currentTimeframe.renderer;
      console.log('[ORD Volume] Found chart state from timeframe renderer');
      return {
        xToIndex: renderer.xToIndex ? renderer.xToIndex.bind(renderer) : null,
        yToPrice: renderer.yToPrice ? renderer.yToPrice.bind(renderer) : null,
        indexToX: renderer.indexToX ? renderer.indexToX.bind(renderer) : null,
        priceToY: renderer.priceToY ? renderer.priceToY.bind(renderer) : null
      };
    }
  }

  // Method 2: Get from active tick chart renderer
  if (window.tosApp && window.tosApp.activeChartType === 'tick') {
    const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
    if (currentTickChart && currentTickChart.renderer) {
      const renderer = currentTickChart.renderer;
      console.log('[ORD Volume] Found chart state from tick chart renderer');
      return {
        xToIndex: renderer.xToIndex ? renderer.xToIndex.bind(renderer) : null,
        yToPrice: renderer.yToPrice ? renderer.yToPrice.bind(renderer) : null,
        indexToX: renderer.indexToX ? renderer.indexToX.bind(renderer) : null,
        priceToY: renderer.priceToY ? renderer.priceToY.bind(renderer) : null
      };
    }
  }

  console.warn('[ORD Volume] Could not find chart state, using fallback');
  // Return empty state (renderer will use fallback)
  return {};
}

/**
 * Load saved ORD Volume analysis when chart loads
 * @param {String} symbol - Symbol to load analysis for
 */
export async function loadSavedORDVolume(symbol) {
  // This function can be called when chart loads to restore saved ORD Volume overlays
  console.log(`[ORD Volume] Loading saved analysis for ${symbol}...`);

  // Implementation would fetch from backend and render
  // For now, this is a placeholder for future enhancement
}

// Auto-initialize when script loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeORDVolume);
} else {
  // DOM already loaded
  initializeORDVolume();
}

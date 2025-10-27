/**
 * Chart Loader Module
 * Handles chart initialization and data loading
 */

import { state } from '../core/state.js';
import { buildIndicatorTraces } from './indicators.js';
import { createChartLayout } from './layout.js';
import { setupChartEvents } from './events.js';
import { loadSavedLines } from '../trendlines/handlers.js';

/**
 * Convert UTC timestamp to CST display string
 * Returns the date in CST timezone for chart display
 */
export function convertToCST(utcDateStr) {
  const date = new Date(utcDateStr);
  // Convert to CST (America/Chicago) - Plotly expects ISO strings
  // We'll adjust the time by -6 hours (CST) or -5 hours (CDT)
  const cstDate = new Date(date.toLocaleString('en-US', { timeZone: 'America/Chicago' }));
  return cstDate;
}

/**
 * Main chart loading function
 * Fetches data and initializes the chart with indicators
 */
export async function loadChart() {
  const symbol = document.getElementById("symbolInput").value.trim().toUpperCase();
  if (!symbol) return;

  state.currentSymbol = symbol;

  let fetchUrl = `/data/${symbol}?interval=${state.currentInterval}`;
  if (state.currentPeriod) fetchUrl += `&period=${state.currentPeriod}`;

  try {
    const response = await fetch(fetchUrl);
    const data = await response.json();

    if (data.length === 0) {
      alert("No data found for symbol: " + symbol);
      return;
    }

    state.chartData = data;

    const traces = [{
      x: data.map(row => convertToCST(row.Date)),  // Convert to CST for display
      open: data.map(row => row.Open),
      high: data.map(row => row.High),
      low: data.map(row => row.Low),
      close: data.map(row => row.Close),
      type: "candlestick",
      name: symbol,
      xaxis: "x",
      yaxis: "y"
    }];

    // Load indicators
    if (Object.keys(state.activeIndicators).length > 0) {
      const indicatorTraces = await buildIndicatorTraces(symbol, data);
      traces.push(...indicatorTraces);
    }

    const layout = createChartLayout(symbol);

    await Plotly.newPlot("plot", traces, layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d'],
      modeBarButtonsToAdd: ['drawopenpath', 'eraseshape'],
      scrollZoom: true,
      editable: true
    });

    // Load saved trendlines and setup event handlers
    await loadSavedLines(symbol);
    setupChartEvents();

    // Show/hide resize controls based on active indicators
    const hasRSI = state.activeIndicators.RSI;
    const hasMACD = state.activeIndicators.MACD;
    document.getElementById('rsiHeightControl').style.display = hasRSI ? 'block' : 'none';
    document.getElementById('macdHeightControl').style.display = hasMACD ? 'block' : 'none';

  } catch (error) {
    console.error("Error loading chart:", error);
    alert("Error loading chart data");
  }
}

// Make globally accessible for onclick handlers
window.loadChart = loadChart;

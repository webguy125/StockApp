/**
 * Chart Layout Module
 * Handles chart layout configuration and panel sizing
 */

import { state } from '../core/state.js';

/**
 * Create chart layout configuration
 */
export function createChartLayout(symbol) {
  const hasRSI = state.activeIndicators.RSI;
  const hasMACD = state.activeIndicators.MACD;

  const layout = {
    title: {
      text: `${symbol} - Technical Analysis`,
      y: 0.95,  // Move title down more (lower value = further down)
      yanchor: 'top',
      xanchor: 'center',
      x: 0.5
    },
    dragmode: 'pan',
    xaxis: {
      rangeslider: { visible: false },
      fixedrange: false,
      showspikes: true,
      spikemode: 'across',
      type: 'date',
      tickformat: '%Y-%m-%d %H:%M',
      nticks: 20, // More time labels like TradingView
      tickfont: { size: 10 }, // Smaller text
      showgrid: true,
      gridwidth: 1,
      gridcolor: state.currentTheme === 'light' ? '#e0e0e0' : '#333333'
    },
    yaxis: {
      title: "",  // Remove title for cleaner look
      side: 'right',
      domain: hasRSI && hasMACD ? [0.4, 1] : (hasRSI || hasMACD ? [0.3, 1] : [0, 1]),
      fixedrange: false,
      showspikes: true,
      spikemode: 'across',
      nticks: 15, // More price labels like TradingView
      tickfont: { size: 10 }, // Smaller text
      showgrid: true,
      gridwidth: 1,
      gridcolor: state.currentTheme === 'light' ? '#e0e0e0' : '#333333',
      tickformat: '.2f' // 2 decimal places
    },
    paper_bgcolor: state.currentTheme === 'light' ? '#ffffff' : '#1a1a1a',
    plot_bgcolor: state.currentTheme === 'light' ? '#ffffff' : '#1a1a1a',
    font: { color: state.currentTheme === 'light' ? '#1a1a1a' : '#e0e0e0' },
    showlegend: true,
    legend: {
      x: 0,
      y: 1.08,
      orientation: 'h',
      font: { size: 10 }
    },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: state.currentTheme === 'light' ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.9)',
      font: { size: 11 }
    },
    margin: {
      l: 10,  // Minimal left margin
      r: 60,  // Room for price labels on right
      t: 50,  // Top margin for title
      b: 40,  // Bottom margin for time labels
      pad: 0
    },
    newshape: {
      line: { color: state.currentTheme === 'light' ? 'black' : '#9ca3af', width: 2 }
    },
    activeshape: {
      fillcolor: 'rgba(239, 68, 68, 0.1)',
      opacity: 0.5
    }
  };

  if (hasRSI && hasMACD) {
    layout.yaxis2 = {
      title: 'RSI',
      side: 'right',
      domain: [0.2, 0.35],
      range: [0, 100],
      fixedrange: false,
      tickfont: { size: 9 },
      nticks: 5,
      showgrid: true,
      gridwidth: 1,
      gridcolor: state.currentTheme === 'light' ? '#e0e0e0' : '#333333'
    };
    layout.yaxis3 = {
      title: 'MACD',
      side: 'right',
      domain: [0, 0.15],
      fixedrange: false,
      tickfont: { size: 9 },
      nticks: 5,
      showgrid: true,
      gridwidth: 1,
      gridcolor: state.currentTheme === 'light' ? '#e0e0e0' : '#333333'
    };
  } else if (hasRSI) {
    layout.yaxis2 = {
      title: 'RSI',
      side: 'right',
      domain: [0, 0.25],
      range: [0, 100],
      fixedrange: false,
      tickfont: { size: 9 },
      nticks: 5,
      showgrid: true,
      gridwidth: 1,
      gridcolor: state.currentTheme === 'light' ? '#e0e0e0' : '#333333'
    };
  } else if (hasMACD) {
    layout.yaxis3 = {
      title: 'MACD',
      side: 'right',
      domain: [0, 0.25],
      fixedrange: false,
      tickfont: { size: 9 },
      nticks: 5,
      showgrid: true,
      gridwidth: 1,
      gridcolor: state.currentTheme === 'light' ? '#e0e0e0' : '#333333'
    };
  }

  return layout;
}

/**
 * Update panel heights dynamically
 */
export function updatePanelHeights() {
  const plotDiv = document.getElementById("plot");
  if (!plotDiv || !plotDiv.layout) return;

  const hasRSI = state.activeIndicators.RSI;
  const hasMACD = state.activeIndicators.MACD;

  const mainHeight = parseInt(document.getElementById('mainChartHeight').value) / 100;

  let newLayout = {
    'yaxis.domain': [hasRSI && hasMACD ? 0.4 : (hasRSI || hasMACD ? 0.3 : 0), mainHeight]
  };

  if (hasRSI && hasMACD) {
    const rsiHeight = 0.15;
    const macdHeight = 0.15;
    newLayout['yaxis2.domain'] = [0.2, 0.35];
    newLayout['yaxis3.domain'] = [0, 0.15];
  } else if (hasRSI) {
    const remaining = 1 - mainHeight;
    newLayout['yaxis2.domain'] = [0, remaining];
  } else if (hasMACD) {
    const remaining = 1 - mainHeight;
    newLayout['yaxis3.domain'] = [0, remaining];
  }

  Plotly.relayout("plot", newLayout);
}

// Make globally accessible for onclick handlers
window.updatePanelHeights = updatePanelHeights;

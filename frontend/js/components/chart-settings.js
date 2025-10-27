/**
 * Chart Settings Module
 * Handles chart configuration and display settings
 */

import { state } from '../core/state.js';

export class ChartSettings {
  constructor() {
    this.settings = {
      timezone: localStorage.getItem('chartTimezone') || 'America/Chicago',
      theme: localStorage.getItem('chartTheme') || 'dark',
      gridLines: localStorage.getItem('chartGridLines') !== 'false', // default true (fixed typo)
      showHighLow: localStorage.getItem('chartHighLow') !== 'false', // default true
      priceAxisPosition: localStorage.getItem('chartPriceAxis') || 'right',
      showCrosshair: localStorage.getItem('chartCrosshair') !== 'false', // default true
      showSupportResistance: localStorage.getItem('chartSupportResistance') === 'true', // default false
      showVolume: localStorage.getItem('chartShowVolume') !== 'false', // default true
      volumeMode: localStorage.getItem('chartVolumeMode') || 'subgraph' // 'overlay' or 'subgraph'
    };

    this.modal = null;
    this.timezoneSelect = null;
    this.themeToggle = null;
    this.gridToggle = null;
    this.highLowToggle = null;
    this.priceAxisSelect = null;
    this.crosshairToggle = null;
    this.supportResistanceToggle = null;
    this.volumeToggle = null;
    this.volumeModeSelect = null;
  }

  /**
   * Initialize chart settings UI
   */
  initialize() {
    this.modal = document.getElementById('chartSettingsModal');
    this.timezoneSelect = document.getElementById('timezoneSelect');
    this.themeToggle = document.getElementById('themeToggle');
    this.gridToggle = document.getElementById('gridToggle');
    this.highLowToggle = document.getElementById('highLowToggle');
    this.priceAxisSelect = document.getElementById('priceAxisSelect');
    this.crosshairToggle = document.getElementById('crosshairToggle');
    this.supportResistanceToggle = document.getElementById('supportResistanceToggle');
    this.volumeToggle = document.getElementById('volumeToggle');
    this.volumeModeSelect = document.getElementById('volumeModeSelect');

    // Load saved settings
    this.loadSettings();

    // Close modal when clicking outside
    this.modal.addEventListener('click', (e) => {
      if (e.target === this.modal) {
        this.closeModal();
      }
    });

    // ESC key to close modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.modal.classList.contains('active')) {
        this.closeModal();
      }
    });
  }

  /**
   * Load settings into UI
   */
  loadSettings() {
    // Sync global state with saved theme
    state.currentTheme = this.settings.theme;

    if (this.timezoneSelect) {
      this.timezoneSelect.value = this.settings.timezone;
    }

    if (this.themeToggle) {
      this.themeToggle.checked = this.settings.theme === 'dark';
      document.getElementById('themeLabel').textContent =
        this.settings.theme === 'dark' ? 'Dark Mode' : 'Light Mode';
    }

    if (this.gridToggle) {
      this.gridToggle.checked = this.settings.gridLines;
    }

    if (this.highLowToggle) {
      this.highLowToggle.checked = this.settings.showHighLow;
    }

    if (this.priceAxisSelect) {
      this.priceAxisSelect.value = this.settings.priceAxisPosition;
    }

    if (this.crosshairToggle) {
      this.crosshairToggle.checked = this.settings.showCrosshair;
    }

    if (this.supportResistanceToggle) {
      this.supportResistanceToggle.checked = this.settings.showSupportResistance;
    }

    if (this.volumeToggle) {
      this.volumeToggle.checked = this.settings.showVolume;
    }

    if (this.volumeModeSelect) {
      this.volumeModeSelect.value = this.settings.volumeMode;
    }

    // Don't apply settings here - chart doesn't exist yet
    // Settings will be applied after chart loads
  }

  /**
   * Apply settings to chart (call this AFTER chart is loaded)
   */
  applySettingsToChart() {
    // Use requestAnimationFrame to apply settings after next render
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        this.applySettings();
      });
    });
  }

  /**
   * Toggle settings modal
   */
  toggleModal() {
    if (this.modal.classList.contains('active')) {
      this.closeModal();
    } else {
      this.openModal();
    }
  }

  /**
   * Open settings modal
   */
  openModal() {
    this.modal.classList.add('active');
  }

  /**
   * Close settings modal
   */
  closeModal() {
    this.modal.classList.remove('active');
  }

  /**
   * Update timezone setting
   */
  updateTimezone(timezone) {
    this.settings.timezone = timezone;
    localStorage.setItem('chartTimezone', timezone);

    // Update all chart timestamps
    this.reloadChart();
  }

  /**
   * Toggle theme (dark/light)
   */
  toggleTheme(isDark) {
    this.settings.theme = isDark ? 'dark' : 'light';
    localStorage.setItem('chartTheme', this.settings.theme);
    document.getElementById('themeLabel').textContent = isDark ? 'Dark Mode' : 'Light Mode';

    // Update global state
    state.currentTheme = this.settings.theme;

    // Apply theme to chart
    this.applyTheme();
  }

  /**
   * Toggle grid lines
   */
  toggleGrid(showGrid) {
    this.settings.gridLines = showGrid;
    localStorage.setItem('chartGridLines', showGrid);

    // Update chart grid
    this.updateChartGrid();
  }

  /**
   * Toggle high/low markers
   */
  toggleHighLow(show) {
    this.settings.showHighLow = show;
    localStorage.setItem('chartHighLow', show);

    // Update price markers
    this.updateHighLowMarkers();
  }

  /**
   * Update price axis position (left/right)
   */
  updatePriceAxisPosition(position) {
    this.settings.priceAxisPosition = position;
    localStorage.setItem('chartPriceAxis', position);

    // Update chart layout
    this.updatePriceAxis();
  }

  /**
   * Toggle crosshair price display
   */
  toggleCrosshair(show) {
    this.settings.showCrosshair = show;
    localStorage.setItem('chartCrosshair', show);

    // Update crosshair behavior
    this.updateCrosshair();
  }

  /**
   * Toggle support/resistance levels
   */
  toggleSupportResistance(show) {
    this.settings.showSupportResistance = show;
    localStorage.setItem('chartSupportResistance', show);

    // Update support/resistance display
    this.updateSupportResistance();
  }

  /**
   * Toggle volume overlay
   */
  toggleVolume(show) {
    this.settings.showVolume = show;
    localStorage.setItem('chartShowVolume', show);

    // Update volume display
    this.updateVolume();
  }

  /**
   * Change volume display mode (overlay or subgraph)
   */
  setVolumeMode(mode) {
    this.settings.volumeMode = mode;
    localStorage.setItem('chartVolumeMode', mode);

    // Reload chart to apply new mode
    this.reloadChart();
  }

  /**
   * Save all settings
   */
  saveSettings() {
    // All settings are saved in real-time via localStorage
    // This just closes the modal
    this.closeModal();

    // Show confirmation
    console.log('Chart settings saved successfully');
  }

  /**
   * Reset to default settings
   */
  resetSettings() {
    this.settings = {
      timezone: 'America/Chicago',
      theme: 'dark',
      gridLines: true,
      showHighLow: true,
      priceAxisPosition: 'right',
      showCrosshair: true,
      showSupportResistance: false,
      showVolume: true
    };

    // Clear localStorage
    localStorage.removeItem('chartTimezone');
    localStorage.removeItem('chartTheme');
    localStorage.removeItem('chartGridLines');
    localStorage.removeItem('chartHighLow');
    localStorage.removeItem('chartPriceAxis');
    localStorage.removeItem('chartCrosshair');
    localStorage.removeItem('chartSupportResistance');
    localStorage.removeItem('chartShowVolume');

    // Reload UI
    this.loadSettings();
  }

  /**
   * Apply all settings to chart
   */
  applySettings() {
    const plotDiv = document.getElementById('tos-plot') || document.getElementById('plot');
    if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) return;

    // Find candlestick trace and get date range
    const candleTrace = plotDiv.data.find(trace => trace.type === 'candlestick');
    if (!candleTrace || !candleTrace.x || candleTrace.x.length === 0) return;

    // Store the current x-axis range before applying settings
    const xRange = [candleTrace.x[0], candleTrace.x[candleTrace.x.length - 1]];

    // FIRST: Lock the x-axis range immediately and keep it locked
    Plotly.relayout(plotDiv, {
      'xaxis.range': xRange,
      'xaxis.autorange': false,
      'xaxis.fixedrange': false  // Allow zooming/panning
    }).then(() => {
      // Now apply all settings while range is locked
      this.applyTheme();
      this.updateChartGrid();
      this.updateHighLowMarkers();
      this.updatePriceAxis();
      this.updateCrosshair();
      this.updateSupportResistance();
      this.updateVolume();
    });
  }

  /**
   * Apply theme to chart
   */
  applyTheme() {
    const plotDiv = document.getElementById('tos-plot') || document.getElementById('plot');
    if (!plotDiv || !plotDiv.layout) return;

    const isDark = this.settings.theme === 'dark';
    const update = {
      paper_bgcolor: isDark ? '#1a1a1a' : '#ffffff',
      plot_bgcolor: isDark ? '#1a1a1a' : '#ffffff',
      'font.color': isDark ? '#e0e0e0' : '#1a1a1a',
      'xaxis.gridcolor': isDark ? '#333333' : '#e0e0e0',
      'yaxis.gridcolor': isDark ? '#333333' : '#e0e0e0'
    };

    Plotly.relayout(plotDiv, update);
  }

  /**
   * Update chart grid visibility
   */
  updateChartGrid() {
    const plotDiv = document.getElementById('tos-plot') || document.getElementById('plot');
    if (!plotDiv || !plotDiv.layout) return;

    const update = {
      'xaxis.showgrid': this.settings.gridLines,
      'yaxis.showgrid': this.settings.gridLines
    };

    Plotly.relayout(plotDiv, update);
  }

  /**
   * Update high/low price markers
   */
  updateHighLowMarkers() {
    const plotDiv = document.getElementById('tos-plot') || document.getElementById('plot');
    if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) return;

    if (!this.settings.showHighLow) {
      // Remove existing high/low annotations and lines
      const annotations = (plotDiv.layout.annotations || []).filter(a => !a._highLowMarker);
      const shapes = (plotDiv.layout.shapes || []).filter(s => !s._highLowLine);
      Plotly.relayout(plotDiv, { annotations, shapes });
      return;
    }

    // Find candlestick trace
    const candleTrace = plotDiv.data.find(trace => trace.type === 'candlestick');
    if (!candleTrace) return;

    // Calculate high and low
    const high = Math.max(...candleTrace.high);
    const low = Math.min(...candleTrace.low);

    // Get existing annotations (excluding high/low)
    const existingAnnotations = (plotDiv.layout.annotations || []).filter(a => !a._highLowMarker);

    // Add high/low annotations with TradingView-style transparent background
    const newAnnotations = [
      ...existingAnnotations,
      {
        x: 1.002, // Slightly outside the chart
        y: high,
        xref: 'paper',
        yref: 'y',
        text: `${high.toFixed(2)}`,
        showarrow: false,
        xanchor: 'left',
        font: {
          color: '#ffffff',
          size: 10,
          family: 'Arial, sans-serif'
        },
        bgcolor: 'rgba(0, 200, 81, 0.75)', // Semi-transparent green
        borderpad: 3,
        borderwidth: 0,
        _highLowMarker: true
      },
      {
        x: 1.002,
        y: low,
        xref: 'paper',
        yref: 'y',
        text: `${low.toFixed(2)}`,
        showarrow: false,
        xanchor: 'left',
        font: {
          color: '#ffffff',
          size: 10,
          family: 'Arial, sans-serif'
        },
        bgcolor: 'rgba(255, 68, 68, 0.75)', // Semi-transparent red
        borderpad: 3,
        borderwidth: 0,
        _highLowMarker: true
      }
    ];

    // Add horizontal lines at high/low prices
    const shapes = (plotDiv.layout.shapes || []).filter(s => !s._highLowLine);

    shapes.push(
      // High price line
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: high,
        y1: high,
        line: {
          color: 'rgba(0, 200, 81, 0.4)',
          width: 1,
          dash: 'dot'
        },
        _highLowLine: true
      },
      // Low price line
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: low,
        y1: low,
        line: {
          color: 'rgba(255, 68, 68, 0.4)',
          width: 1,
          dash: 'dot'
        },
        _highLowLine: true
      }
    );

    Plotly.relayout(plotDiv, {
      annotations: newAnnotations,
      shapes: shapes
    });
  }

  /**
   * Update price axis position
   */
  updatePriceAxis() {
    const plotDiv = document.getElementById('tos-plot') || document.getElementById('plot');
    if (!plotDiv || !plotDiv.layout) return;

    const update = {
      'yaxis.side': this.settings.priceAxisPosition
    };

    Plotly.relayout(plotDiv, update);
  }

  /**
   * Update crosshair behavior
   * DISABLED - Using custom crosshair implementation instead of Plotly spikes
   */
  updateCrosshair() {
    // Do nothing - custom crosshair is handled in tos-app.js
    return;
  }

  /**
   * Update support/resistance levels
   * Calculates and displays recent support and resistance levels
   */
  updateSupportResistance() {
    const plotDiv = document.getElementById('tos-plot') || document.getElementById('plot');
    if (!plotDiv || !plotDiv.data || plotDiv.data.length === 0) return;

    // Remove existing S/R lines
    let shapes = (plotDiv.layout.shapes || []).filter(s => !s._supportResistance);

    if (!this.settings.showSupportResistance) {
      Plotly.relayout(plotDiv, { shapes });
      return;
    }

    // Find candlestick trace
    const candleTrace = plotDiv.data.find(trace => trace.type === 'candlestick');
    if (!candleTrace || candleTrace.close.length < 20) return;

    // Simple S/R calculation using pivot points
    const levels = this.calculateSupportResistance(candleTrace);

    // Add S/R lines
    levels.forEach(level => {
      shapes.push({
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: level.price,
        y1: level.price,
        line: {
          color: level.type === 'resistance' ? '#ff4444' : '#00c851',
          width: 1,
          dash: 'dot'
        },
        _supportResistance: true
      });
    });

    Plotly.relayout(plotDiv, { shapes });
  }

  /**
   * Calculate support and resistance levels
   */
  calculateSupportResistance(candleTrace) {
    const highs = candleTrace.high;
    const lows = candleTrace.low;
    const closes = candleTrace.close;
    const length = closes.length;

    // Get recent data (last 50 candles)
    const recentLength = Math.min(50, length);
    const recentHighs = highs.slice(-recentLength);
    const recentLows = lows.slice(-recentLength);

    // Find resistance (recent high)
    const resistance = Math.max(...recentHighs);

    // Find support (recent low)
    const support = Math.min(...recentLows);

    return [
      { type: 'resistance', price: resistance },
      { type: 'support', price: support }
    ];
  }

  /**
   * Update volume overlay visibility
   */
  updateVolume() {
    const plotDiv = document.getElementById('tos-plot') || document.getElementById('plot');
    if (!plotDiv || !plotDiv.data) return;

    // Find volume bar trace (type: 'bar' and yaxis: 'y2')
    let volumeTraceIndex = -1;
    for (let i = 0; i < plotDiv.data.length; i++) {
      if (plotDiv.data[i].type === 'bar' && plotDiv.data[i].yaxis === 'y2') {
        volumeTraceIndex = i;
        break;
      }
    }

    if (volumeTraceIndex >= 0 && !this.settings.showVolume) {
      // Volume trace exists and we want to hide it - just hide the trace
      Plotly.restyle(plotDiv, { visible: false }, [volumeTraceIndex]);
    } else {
      // Either no volume trace exists, or we're turning it back on
      // In both cases, reload chart to ensure proper layout with y2 axis
      this.reloadChart();
    }
  }

  /**
   * Reload chart with current settings
   */
  reloadChart() {
    // Trigger chart reload through the main app with forceRecreate=true
    // This ensures volume is properly added/removed from the chart
    if (window.tosApp && window.tosApp.reloadChart) {
      window.tosApp.reloadChart(true);
    }
  }
}

/**
 * Initialize chart settings
 */
export function initializeChartSettings() {
  const chartSettings = new ChartSettings();
  chartSettings.initialize();
  return chartSettings;
}

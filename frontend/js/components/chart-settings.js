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
   * Canvas renderer handles settings directly, no Plotly needed
   */
  applySettings() {
    // Canvas renderer applies settings during render
    // Settings are read from localStorage in the renderer
    console.log('Canvas renderer will apply settings on next draw');
  }

  /**
   * Apply theme to chart
   * Canvas renderer reads theme from settings
   */
  applyTheme() {
    // Canvas renderer will use this.settings.theme on next draw
    this.reloadChart();
  }

  /**
   * Update chart grid visibility
   * Canvas renderer reads grid setting
   */
  updateChartGrid() {
    // Canvas renderer will apply grid setting on next draw
    this.reloadChart();
  }

  /**
   * Update high/low price markers
   * Canvas renderer draws these natively
   */
  updateHighLowMarkers() {
    // Canvas renderer draws high/low markers in drawPriceMarkers() method
    // Setting is already saved to localStorage
    this.reloadChart();
  }

  /**
   * Update price axis position
   * Canvas renderer handles axis positioning
   */
  updatePriceAxis() {
    // Canvas renderer will use priceAxisPosition setting
    this.reloadChart();
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
   * Canvas renderer can draw these if needed
   */
  updateSupportResistance() {
    // Canvas renderer will handle support/resistance levels
    // Setting is saved to localStorage
    this.reloadChart();
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
   * Canvas renderer handles volume drawing
   */
  updateVolume() {
    // Canvas renderer will use showVolume setting
    this.reloadChart();
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

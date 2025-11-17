/**
 * Base Indicator Class
 * All indicators inherit from this base class
 * Provides common functionality: settings management, alerts, rendering hooks
 */

export class IndicatorBase {
  constructor(config) {
    // Metadata
    this.name = config.name || 'Unnamed Indicator';
    this.version = config.version || '1.0.0';
    this.description = config.description || '';
    this.tags = config.tags || [];
    this.dependencies = config.dependencies || [];
    this.outputType = config.output_type || 'line'; // 'line', 'histogram', 'overlay', 'oscillator'
    this.helpText = config.help_text || '';

    // Settings
    this.defaultSettings = config.default_settings || {};
    this.currentSettings = { ...this.defaultSettings };

    // Alerts
    this.alertsConfig = config.alerts || { enabled: false, conditions: [] };
    this.activeAlerts = [];

    // State
    this.enabled = false;
    this.cachedData = null;
    this.lastCalculation = null;

    // Rendering
    this.canvas = null;
    this.plotData = null;
  }

  /**
   * Calculate indicator values from OHLCV data
   * Must be implemented by each indicator
   * @param {Array} candles - Array of OHLCV candles
   * @returns {Array} Calculated indicator values
   */
  calculate(candles) {
    throw new Error(`calculate() must be implemented by ${this.name}`);
  }

  /**
   * Update settings and recalculate if needed
   * @param {Object} newSettings - New settings to apply
   */
  updateSettings(newSettings) {
    this.currentSettings = { ...this.currentSettings, ...newSettings };
    this.invalidateCache();
  }

  /**
   * Get current settings
   * @returns {Object} Current settings
   */
  getSettings() {
    return { ...this.currentSettings };
  }

  /**
   * Export settings to JSON
   * @returns {Object} Exportable settings object
   */
  exportSettings() {
    return {
      indicator: this.name,
      version: this.version,
      settings: this.currentSettings,
      alerts: this.alertsConfig
    };
  }

  /**
   * Import settings from JSON
   * @param {Object} data - Settings to import
   */
  importSettings(data) {
    if (data.indicator !== this.name) {
      console.warn(`Settings mismatch: expected ${this.name}, got ${data.indicator}`);
    }
    this.currentSettings = { ...this.defaultSettings, ...data.settings };
    this.alertsConfig = { ...this.alertsConfig, ...data.alerts };
    this.invalidateCache();
  }

  /**
   * Reset to default settings
   */
  resetSettings() {
    this.currentSettings = { ...this.defaultSettings };
    this.invalidateCache();
  }

  /**
   * Enable the indicator
   */
  enable() {
    this.enabled = true;
  }

  /**
   * Disable the indicator
   */
  disable() {
    this.enabled = false;
    this.clearPlot();
  }

  /**
   * Toggle indicator on/off
   * @returns {boolean} New enabled state
   */
  toggle() {
    this.enabled = !this.enabled;
    if (!this.enabled) {
      this.clearPlot();
    }
    return this.enabled;
  }

  /**
   * Invalidate cached data (forces recalculation)
   */
  invalidateCache() {
    this.cachedData = null;
    this.lastCalculation = null;
  }

  /**
   * Check if cached data is valid
   * @param {Array} candles - Current candles
   * @returns {boolean} True if cache is valid
   */
  isCacheValid(candles) {
    if (!this.cachedData || !this.lastCalculation) {
      return false;
    }
    // Simple check: same length and same last candle timestamp
    if (candles.length !== this.lastCalculation.candleCount) {
      return false;
    }
    const lastCandle = candles[candles.length - 1];
    return lastCandle.Date === this.lastCalculation.lastCandleDate;
  }

  /**
   * Get or calculate indicator data (with caching)
   * @param {Array} candles - OHLCV candles
   * @returns {Array} Indicator values
   */
  getData(candles) {
    if (!this.enabled) {
      return null;
    }

    if (this.isCacheValid(candles)) {
      return this.cachedData;
    }

    // Calculate fresh data
    const data = this.calculate(candles);

    // Cache it
    this.cachedData = data;
    this.lastCalculation = {
      timestamp: Date.now(),
      candleCount: candles.length,
      lastCandleDate: candles[candles.length - 1]?.Date
    };

    // Check alerts
    if (this.alertsConfig.enabled) {
      this.checkAlerts(data, candles);
    }

    return data;
  }

  /**
   * Check alert conditions
   * @param {Array} data - Indicator data
   * @param {Array} candles - OHLCV candles
   */
  checkAlerts(data, candles) {
    if (!this.alertsConfig.conditions || this.alertsConfig.conditions.length === 0) {
      return;
    }

    const latestValue = data[data.length - 1];

    this.alertsConfig.conditions.forEach(condition => {
      let triggered = false;

      switch (condition.type) {
        case 'greater_than':
          triggered = latestValue[condition.field] > condition.threshold;
          break;
        case 'less_than':
          triggered = latestValue[condition.field] < condition.threshold;
          break;
        case 'cross_over':
          if (data.length >= 2) {
            const prev = data[data.length - 2];
            triggered = prev[condition.field] < prev[condition.target] &&
                       latestValue[condition.field] > latestValue[condition.target];
          }
          break;
        case 'cross_under':
          if (data.length >= 2) {
            const prev = data[data.length - 2];
            triggered = prev[condition.field] > prev[condition.target] &&
                       latestValue[condition.field] < latestValue[condition.target];
          }
          break;
      }

      if (triggered) {
        this.fireAlert(condition.message, latestValue);
      }
    });
  }

  /**
   * Fire an alert
   * @param {string} message - Alert message
   * @param {Object} data - Associated data
   */
  fireAlert(message, data) {
    const alert = {
      indicator: this.name,
      message,
      data,
      timestamp: new Date().toISOString()
    };

    this.activeAlerts.push(alert);

    // Dispatch custom event
    window.dispatchEvent(new CustomEvent('indicator-alert', { detail: alert }));

    console.log(`🔔 [${this.name}] ${message}`, data);
  }

  /**
   * Clear all active alerts
   */
  clearAlerts() {
    this.activeAlerts = [];
  }

  /**
   * Get info about this indicator
   * @returns {Object} Indicator metadata
   */
  getInfo() {
    return {
      name: this.name,
      version: this.version,
      description: this.description,
      tags: this.tags,
      dependencies: this.dependencies,
      outputType: this.outputType,
      enabled: this.enabled,
      settings: this.currentSettings,
      helpText: this.helpText
    };
  }

  /**
   * Clear plot from canvas (to be implemented by renderer)
   */
  clearPlot() {
    // Will be overridden by renderer integration
    this.plotData = null;
  }

  /**
   * Render indicator on canvas (to be implemented by renderer)
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Object} bounds - Rendering bounds
   */
  render(ctx, bounds) {
    // Will be overridden by specific indicator renderers
  }
}

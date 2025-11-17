/**
 * Indicator Registry
 * Central registry for all available indicators
 * Manages indicator lifecycle, settings, and rendering
 */

export class IndicatorRegistry {
  constructor() {
    this.indicators = new Map(); // name -> indicator instance
    this.renderQueue = []; // Indicators to render on next frame
    this.listeners = new Map(); // event -> callbacks
  }

  /**
   * Register a new indicator
   * @param {IndicatorBase} indicator - Indicator instance
   */
  register(indicator) {
    if (this.indicators.has(indicator.name)) {
      console.warn(`Indicator ${indicator.name} already registered, replacing`);
    }

    this.indicators.set(indicator.name, indicator);
    console.log(`✅ Registered indicator: ${indicator.name} v${indicator.version}`);

    this.emit('indicator-registered', { indicator });
  }

  /**
   * Unregister an indicator
   * @param {string} name - Indicator name
   */
  unregister(name) {
    const indicator = this.indicators.get(name);
    if (indicator) {
      indicator.disable();
      this.indicators.delete(name);
      console.log(`❌ Unregistered indicator: ${name}`);
      this.emit('indicator-unregistered', { name });
    }
  }

  /**
   * Get an indicator by name
   * @param {string} name - Indicator name
   * @returns {IndicatorBase|null}
   */
  get(name) {
    return this.indicators.get(name) || null;
  }

  /**
   * Get all registered indicators
   * @returns {Array<IndicatorBase>}
   */
  getAll() {
    return Array.from(this.indicators.values());
  }

  /**
   * Get all enabled indicators
   * @returns {Array<IndicatorBase>}
   */
  getEnabled() {
    return this.getAll().filter(ind => ind.enabled);
  }

  /**
   * Get indicators by tag
   * @param {string} tag - Tag to filter by
   * @returns {Array<IndicatorBase>}
   */
  getByTag(tag) {
    return this.getAll().filter(ind => ind.tags.includes(tag));
  }

  /**
   * Get indicators by output type
   * @param {string} type - Output type ('line', 'histogram', 'overlay', 'oscillator')
   * @returns {Array<IndicatorBase>}
   */
  getByType(type) {
    return this.getAll().filter(ind => ind.outputType === type);
  }

  /**
   * Enable an indicator
   * @param {string} name - Indicator name
   */
  enable(name) {
    const indicator = this.get(name);
    if (indicator) {
      indicator.enable();
      this.emit('indicator-enabled', { indicator });
      console.log(`▶️ Enabled indicator: ${name}`);
    }
  }

  /**
   * Disable an indicator
   * @param {string} name - Indicator name
   */
  disable(name) {
    const indicator = this.get(name);
    if (indicator) {
      indicator.disable();
      this.emit('indicator-disabled', { indicator });
      console.log(`⏸️ Disabled indicator: ${name}`);
    }
  }

  /**
   * Toggle an indicator
   * @param {string} name - Indicator name
   * @returns {boolean} New enabled state
   */
  toggle(name) {
    const indicator = this.get(name);
    if (indicator) {
      const newState = indicator.toggle();
      this.emit(newState ? 'indicator-enabled' : 'indicator-disabled', { indicator });
      console.log(`${newState ? '▶️' : '⏸️'} ${newState ? 'Enabled' : 'Disabled'} indicator: ${name}`);
      return newState;
    }
    return false;
  }

  /**
   * Update indicator settings
   * @param {string} name - Indicator name
   * @param {Object} settings - New settings
   */
  updateSettings(name, settings) {
    const indicator = this.get(name);
    if (indicator) {
      indicator.updateSettings(settings);
      this.emit('indicator-settings-updated', { indicator, settings });
      console.log(`⚙️ Updated settings for ${name}`);
    }
  }

  /**
   * Calculate all enabled indicators for given candles
   * @param {Array} candles - OHLCV candles
   * @returns {Map} name -> calculated data
   */
  calculateAll(candles) {
    const results = new Map();

    this.getEnabled().forEach(indicator => {
      try {
        const data = indicator.getData(candles);
        if (data) {
          results.set(indicator.name, data);
        }
      } catch (error) {
        console.error(`❌ Error calculating ${indicator.name}:`, error);
      }
    });

    return results;
  }

  /**
   * Export all indicator settings
   * @returns {Object} Settings for all indicators
   */
  exportAllSettings() {
    const settings = {};

    this.getAll().forEach(indicator => {
      settings[indicator.name] = indicator.exportSettings();
    });

    return {
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      indicators: settings
    };
  }

  /**
   * Import indicator settings
   * @param {Object} data - Settings data to import
   */
  importAllSettings(data) {
    if (!data.indicators) {
      console.error('Invalid settings data');
      return;
    }

    Object.entries(data.indicators).forEach(([name, settingsData]) => {
      const indicator = this.get(name);
      if (indicator) {
        indicator.importSettings(settingsData);
        console.log(`📥 Imported settings for ${name}`);
      } else {
        console.warn(`⚠️ Indicator ${name} not found, skipping import`);
      }
    });

    this.emit('settings-imported', { data });
  }

  /**
   * Reset all indicators to default settings
   */
  resetAll() {
    this.getAll().forEach(indicator => {
      indicator.resetSettings();
      indicator.disable();
    });

    this.emit('all-reset');
    console.log('🔄 Reset all indicators to default settings');
  }

  /**
   * Register an event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Unregister an event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit an event
   * @param {string} event - Event name
   * @param {Object} data - Event data
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Get registry statistics
   * @returns {Object} Statistics
   */
  getStats() {
    const all = this.getAll();
    const enabled = this.getEnabled();

    return {
      total: all.length,
      enabled: enabled.length,
      disabled: all.length - enabled.length,
      byType: {
        line: this.getByType('line').length,
        histogram: this.getByType('histogram').length,
        overlay: this.getByType('overlay').length,
        oscillator: this.getByType('oscillator').length
      }
    };
  }
}

// Create singleton instance
export const indicatorRegistry = new IndicatorRegistry();

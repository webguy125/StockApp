/**
 * Indicator System Initialization
 * Registers all indicators and wires up the indicators button
 */

import { indicatorRegistry } from './IndicatorRegistry.js';
import { indicatorSettingsModal } from './IndicatorSettingsModal.js';
import { RSI } from './RSI/RSI.js';
import { MACD } from './MACD/MACD.js';
import { BollingerBands} from './BollingerBands/BollingerBands.js';
import { TriadTrendPulse } from './TriadTrendPulse/TriadTrendPulse.js';

/**
 * Initialize the indicator system
 */
export function initIndicators() {
  console.log('📊 Initializing Indicator System...');

  // Create and register all indicators
  const rsi = new RSI();
  const macd = new MACD();
  const bollingerBands = new BollingerBands();
  const triadTrendPulse = new TriadTrendPulse();

  console.log('📊 Registering RSI indicator...');
  indicatorRegistry.register(rsi);
  console.log('📊 Registering MACD indicator...');
  indicatorRegistry.register(macd);
  console.log('📊 Registering Bollinger Bands indicator...');
  indicatorRegistry.register(bollingerBands);
  console.log('📊 Registering Triad Trend Pulse indicator...');
  indicatorRegistry.register(triadTrendPulse);

  console.log(`📊 Total registered indicators: ${indicatorRegistry.getAll().length}`);

  // Wire up indicators button (ID: btn-add-indicator in index_tos_style.html)
  const indicatorsButton = document.getElementById('btn-add-indicator');
  if (indicatorsButton) {
    indicatorsButton.addEventListener('click', () => {
      indicatorSettingsModal.open();
    });
    console.log('✅ Indicators button wired up');
  } else {
    console.warn('⚠️ Indicators button (btn-add-indicator) not found');
  }

  // Listen for indicator changes
  indicatorRegistry.on('indicator-enabled', ({ indicator }) => {
    console.log(`▶️ Indicator enabled: ${indicator.name}`);
    // Trigger chart re-render if needed
    triggerChartUpdate();
  });

  indicatorRegistry.on('indicator-disabled', ({ indicator }) => {
    console.log(`⏸️ Indicator disabled: ${indicator.name}`);
    // Trigger chart re-render if needed
    triggerChartUpdate();
  });

  indicatorRegistry.on('indicator-settings-updated', ({ indicator }) => {
    console.log(`⚙️ Settings updated: ${indicator.name}`);
    // Trigger chart re-render if needed
    triggerChartUpdate();
  });

  // Load saved settings from localStorage
  loadSavedSettings();

  console.log('✅ Indicator system initialized');
  console.log(`📊 Registered ${indicatorRegistry.getAll().length} indicators`);
}

/**
 * Trigger chart update/redraw
 */
function triggerChartUpdate() {
  // Dispatch custom event that the chart renderer can listen to
  window.dispatchEvent(new CustomEvent('indicators-changed'));
}

/**
 * Load saved settings from localStorage
 */
function loadSavedSettings() {
  try {
    const savedSettings = localStorage.getItem('indicator_settings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings);
      indicatorRegistry.importAllSettings(settings);
      console.log('📥 Loaded saved indicator settings');
    }
  } catch (error) {
    console.error('❌ Error loading saved settings:', error);
  }
}

/**
 * Save settings to localStorage
 */
export function saveIndicatorSettings() {
  try {
    const settings = indicatorRegistry.exportAllSettings();
    localStorage.setItem('indicator_settings', JSON.stringify(settings));
    console.log('💾 Saved indicator settings');
  } catch (error) {
    console.error('❌ Error saving settings:', error);
  }
}

// Auto-save settings when indicators change
indicatorRegistry.on('indicator-enabled', saveIndicatorSettings);
indicatorRegistry.on('indicator-disabled', saveIndicatorSettings);
indicatorRegistry.on('indicator-settings-updated', saveIndicatorSettings);

// Export registry for external access
export { indicatorRegistry, indicatorSettingsModal };

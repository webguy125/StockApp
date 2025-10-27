/**
 * Chart Events Module
 * Handles chart interaction events and user input
 */

import { state } from '../core/state.js';
import { loadChart } from './loader.js';

/**
 * Setup all chart event listeners
 */
export function setupChartEvents() {
  // This function is called after chart is initialized
  // Event handlers for plotly events are registered in handlers.js
}

/**
 * Initialize timeframe and symbol input handlers
 */
export function initializeInputHandlers() {
  // Symbol input - Enter key
  document.getElementById('symbolInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') loadChart();
  });

  // Timeframe selection
  document.getElementById('timeframeSelect').addEventListener('change', function() {
    const val = this.value;
    if (val !== '') {
      const parts = val.split('_');
      state.currentPeriod = parts[0];
      state.currentInterval = parts[1];
      loadChart();
    }
  });

  // Indicator toggles
  document.querySelectorAll('.indicator-toggle').forEach(toggle => {
    toggle.addEventListener('change', function() {
      const indicator = this.getAttribute('data-indicator');
      const period = this.getAttribute('data-period');

      if (this.checked) {
        state.activeIndicators[indicator] = { period: period || null };
      } else {
        delete state.activeIndicators[indicator];
      }

      if (state.currentSymbol) loadChart();
    });
  });
}

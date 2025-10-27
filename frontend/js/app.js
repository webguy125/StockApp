/**
 * Main Application Entry Point
 * Initializes the StockApp and wires up all event handlers
 */

import { state } from './core/state.js';
import { toggleTheme, initializeThemeToggle } from './core/theme.js';
import { initializeTabSwitching } from './core/tabs.js';
import { initializeInputHandlers } from './chart/events.js';
import { setupPlotlyHandlers } from './trendlines/handlers.js';
import { loadPlugins } from './plugins/executor.js';
import { loadPortfolio } from './portfolio/manager.js';

/**
 * Initialize the application
 */
function initializeApp() {
  console.log('Initializing StockApp...');

  // Initialize theme toggle
  initializeThemeToggle();

  // Initialize tab switching with portfolio/plugin loading hooks
  initializeTabSwitching((tabName) => {
    if (tabName === 'portfolio') loadPortfolio();
    if (tabName === 'plugins') loadPlugins();
  });

  // Initialize input handlers for chart controls
  initializeInputHandlers();

  // Close popup when clicking outside
  document.addEventListener('click', function(e) {
    const popup = document.getElementById("popupMenu");
    if (popup && !popup.contains(e.target)) {
      const hidePopup = () => {
        popup.style.display = "none";
      };
      hidePopup();
    }
  });

  // Load plugins on startup
  loadPlugins();

  console.log('StockApp initialized successfully');
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}

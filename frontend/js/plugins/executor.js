/**
 * Plugin Executor Module
 * Handles plugin loading and execution
 */

import { state } from '../core/state.js';

/**
 * Load available plugins
 */
export async function loadPlugins() {
  try {
    const response = await fetch("/plugins");
    const data = await response.json();

    const select = document.getElementById('pluginSelect');
    select.innerHTML = '<option value="">Select a plugin...</option>';

    if (data.plugins && data.plugins.length > 0) {
      data.plugins.forEach(plugin => {
        const option = document.createElement('option');
        option.value = plugin.name;
        option.textContent = `${plugin.name} v${plugin.version}`;
        select.appendChild(option);
      });
    } else {
      select.innerHTML = '<option value="">No plugins available</option>';
    }
  } catch (error) {
    console.error("Error loading plugins:", error);
  }
}

/**
 * Execute selected plugin
 */
export async function executePlugin() {
  const pluginName = document.getElementById('pluginSelect').value;
  const symbol = document.getElementById('pluginSymbol').value.trim().toUpperCase();
  const period = document.getElementById('pluginPeriod').value;

  if (!pluginName || !symbol) {
    alert("Please select a plugin and enter a symbol");
    return;
  }

  document.getElementById('pluginResults').innerHTML = '<p class="loading">Executing plugin...</p>';

  try {
    // Get the plugin filename (need to map from display name to filename)
    const pluginsResponse = await fetch("/plugins");
    const pluginsData = await pluginsResponse.json();

    // Find the plugin file name by matching the name
    let pluginFileName = null;
    for (let i = 0; i < pluginsData.plugins.length; i++) {
      if (pluginsData.plugins[i].name === pluginName) {
        // The plugin filename is typically snake_case version
        pluginFileName = pluginName.toLowerCase().replace(/\s+/g, '_') + '_plugin';
        break;
      }
    }

    // Try common plugin names
    if (!pluginFileName) {
      if (pluginName.includes('WMA') || pluginName.includes('Weighted')) pluginFileName = 'wma_plugin';
      if (pluginName.includes('ATR') || pluginName.includes('True Range')) pluginFileName = 'atr_plugin';
    }

    const params = period ? { period: parseInt(period) } : {};

    const response = await fetch("/plugins/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        plugin: pluginFileName,
        symbol: symbol,
        period: "1y",
        interval: "1d",
        params: params
      })
    });

    const data = await response.json();

    if (data.error) {
      document.getElementById('pluginResults').innerHTML = `<div class="error">${data.error}</div>`;
    } else {
      let html = `
        <h3>Plugin Results: ${pluginName}</h3>
        <div class="result-item">
          <strong>Symbol:</strong> ${data.symbol}<br>
          <strong>Data Points:</strong> ${data.result.length}<br>
          <strong>Last Value:</strong> ${data.result[data.result.length - 1]?.toFixed(2) || 'N/A'}
        </div>
        <div class="success">Plugin executed successfully!</div>
      `;
      document.getElementById('pluginResults').innerHTML = html;
    }
  } catch (error) {
    document.getElementById('pluginResults').innerHTML = '<div class="error">Error executing plugin</div>';
  }
}

/**
 * Show plugins tab
 */
export function showPlugins() {
  document.querySelector('[data-tab="plugins"]').click();
}

// Make globally accessible for onclick handlers
window.loadPlugins = loadPlugins;
window.executePlugin = executePlugin;
window.showPlugins = showPlugins;

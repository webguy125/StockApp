/**
 * Pattern Detection Module
 * Handles chart pattern detection and display
 */

import { state } from '../core/state.js';

/**
 * Detect chart patterns for current symbol
 */
export async function detectPatterns() {
  if (!state.currentSymbol) {
    alert("Please load a chart first");
    return;
  }

  try {
    const response = await fetch("/patterns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol: state.currentSymbol,
        period: state.currentPeriod || "1y",
        interval: state.currentInterval
      })
    });

    const data = await response.json();

    let html = '<h3 style="margin-bottom: 15px;">Pattern Detection Results</h3>';

    if (data.patterns && data.patterns.length > 0) {
      data.patterns.forEach(pattern => {
        html += `
          <div class="result-item">
            <strong>${pattern.type}</strong> (${(pattern.confidence * 100).toFixed(0)}% confidence)<br>
            <small>${pattern.description}</small><br>
            <small>Support: $${pattern.support?.toFixed(2) || 'N/A'} | Resistance: $${pattern.resistance?.toFixed(2) || 'N/A'}</small>
          </div>
        `;
      });
    } else {
      html += '<p>No patterns detected in current timeframe.</p>';
    }

    document.getElementById('analysisResults').innerHTML = html;
    document.querySelector('[data-tab="analysis"]').click();
  } catch (error) {
    alert("Error detecting patterns");
  }
}

// Make globally accessible for onclick handlers
window.detectPatterns = detectPatterns;

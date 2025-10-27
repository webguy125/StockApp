/**
 * Price Prediction Module
 * Handles ML-based price predictions
 */

import { state } from '../core/state.js';

/**
 * Get price predictions for current symbol
 */
export async function getPredictions() {
  if (!state.currentSymbol) {
    alert("Please load a chart first");
    return;
  }

  document.getElementById('analysisResults').innerHTML = '<p class="loading">Generating predictions...</p>';
  document.querySelector('[data-tab="analysis"]').click();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol: state.currentSymbol,
        period: "1y",
        interval: "1d",
        days: 30
      })
    });

    const data = await response.json();

    let html = `
      <h3 style="margin-bottom: 15px;">30-Day Price Prediction</h3>
      <div class="result-item">
        <strong>Current Price:</strong> $${data.current_price.toFixed(2)}<br>
        <strong>Trend:</strong> ${data.trend} (${data.trend_strength.toFixed(2)}% strength)<br>
        <strong>Model:</strong> ${data.model} (R² = ${data.r2_score.toFixed(3)})<br>
        <strong>30-Day Forecast:</strong> $${data.predictions[29].toFixed(2)}<br>
        <strong>Confidence:</strong> ${(data.confidence[29] * 100).toFixed(0)}%
      </div>
      <div class="result-item">
        <strong>Prediction Chart:</strong>
        <div id="predictionChart" style="height: 300px; margin-top: 10px;"></div>
      </div>
    `;

    document.getElementById('analysisResults').innerHTML = html;

    // Plot prediction chart
    const trace = {
      x: data.dates,
      y: data.predictions,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Predicted Price',
      line: { color: '#3b82f6' }
    };

    const layout = {
      title: 'Price Forecast',
      xaxis: { title: 'Date' },
      yaxis: { title: 'Price ($)' },
      paper_bgcolor: state.currentTheme === 'light' ? '#ffffff' : '#2d3748',
      plot_bgcolor: state.currentTheme === 'light' ? '#ffffff' : '#2d3748',
      font: { color: state.currentTheme === 'light' ? '#1a1a1a' : '#e0e0e0' },
      height: 300
    };

    Plotly.newPlot('predictionChart', [trace], layout, { responsive: true });
  } catch (error) {
    document.getElementById('analysisResults').innerHTML = '<div class="error">Error generating predictions</div>';
  }
}

// Make globally accessible for onclick handlers
window.getPredictions = getPredictions;

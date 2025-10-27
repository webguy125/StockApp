/**
 * Trade Ideas Module
 * Handles AI-generated trade ideas
 */

import { state } from '../core/state.js';

/**
 * Get AI-generated trade ideas for current symbol
 */
export async function getTradeIdeas() {
  if (!state.currentSymbol) {
    alert("Please load a chart first");
    return;
  }

  document.getElementById('analysisResults').innerHTML = '<p class="loading">Generating trade ideas...</p>';
  document.querySelector('[data-tab="analysis"]').click();

  try {
    const response = await fetch("/trade_ideas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol: state.currentSymbol,
        period: "3mo",
        interval: "1d"
      })
    });

    const data = await response.json();

    let html = `
      <h3 style="margin-bottom: 15px;">AI Trade Ideas for ${data.symbol}</h3>
      <div class="result-item">
        <strong>Current Price:</strong> $${data.current_price.toFixed(2)}<br>
        <strong>Market Condition:</strong><br>
        RSI: ${data.market_condition.rsi.toFixed(1)} |
        Trend: ${data.market_condition.trend}<br>
        Support: $${data.market_condition.support.toFixed(2)} |
        Resistance: $${data.market_condition.resistance.toFixed(2)}
      </div>
    `;

    if (data.ideas && data.ideas.length > 0) {
      data.ideas.forEach(idea => {
        const profitPct = ((idea.target - idea.entry) / idea.entry * 100).toFixed(2);
        html += `
          <div class="trade-idea ${idea.type.toLowerCase()}">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
              <strong>${idea.type} - ${idea.strategy}</strong>
              <span>${(idea.confidence * 100).toFixed(0)}% confidence</span>
            </div>
            <div style="font-size: 13px;">
              <strong>Entry:</strong> $${idea.entry.toFixed(2)} |
              <strong>Target:</strong> $${idea.target.toFixed(2)} (${profitPct}%)<br>
              <strong>Stop Loss:</strong> $${idea.stop_loss.toFixed(2)} |
              <strong>R/R:</strong> ${idea.risk_reward}:1<br>
              <strong>Timeframe:</strong> ${idea.timeframe}<br>
              <em>${idea.reason}</em>
            </div>
          </div>
        `;
      });
    } else {
      html += '<p>No trade ideas generated for current market conditions.</p>';
    }

    document.getElementById('analysisResults').innerHTML = html;
  } catch (error) {
    document.getElementById('analysisResults').innerHTML = '<div class="error">Error generating trade ideas</div>';
  }
}

// Make globally accessible for onclick handlers
window.getTradeIdeas = getTradeIdeas;

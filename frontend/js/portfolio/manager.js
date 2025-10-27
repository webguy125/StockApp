/**
 * Portfolio Manager Module
 * Handles portfolio viewing and trading operations
 */

import { state } from '../core/state.js';

/**
 * Load and display portfolio
 */
export async function loadPortfolio() {
  try {
    const response = await fetch("/portfolio");
    const data = await response.json();

    let html = `
      <div class="result-item">
        <div style="display: flex; justify-content: space-between;">
          <div><strong>Cash:</strong> $${data.cash.toFixed(2)}</div>
          <div><strong>Total Value:</strong> $${data.total_value.toFixed(2)}</div>
        </div>
        <div style="margin-top: 8px;">
          <strong>Total P&L:</strong>
          <span class="${data.total_pnl >= 0 ? 'profit' : 'loss'}">
            $${data.total_pnl.toFixed(2)}
          </span>
        </div>
      </div>
    `;

    if (data.positions && data.positions.length > 0) {
      html += '<h4 style="margin: 15px 0 10px 0;">Positions:</h4>';
      data.positions.forEach(pos => {
        html += `
          <div class="portfolio-position">
            <div>
              <strong>${pos.symbol}</strong><br>
              <small>${pos.shares} shares @ $${pos.avg_cost.toFixed(2)}</small>
            </div>
            <div style="text-align: right;">
              <div>${pos.current_price ? '$' + pos.current_price.toFixed(2) : 'N/A'}</div>
              <div class="${pos.pnl >= 0 ? 'profit' : 'loss'}">
                ${pos.pnl >= 0 ? '+' : ''}$${pos.pnl?.toFixed(2) || '0.00'}
                (${pos.pnl_pct >= 0 ? '+' : ''}${pos.pnl_pct?.toFixed(2) || '0.00'}%)
              </div>
            </div>
          </div>
        `;
      });
    } else {
      html += '<p style="margin-top: 15px;">No positions. Use the trade form above to buy stocks.</p>';
    }

    document.getElementById('portfolioDisplay').innerHTML = html;
  } catch (error) {
    document.getElementById('portfolioDisplay').innerHTML = '<div class="error">Error loading portfolio</div>';
  }
}

/**
 * Buy stock
 */
export async function buyStock() {
  const symbol = document.getElementById('tradeSymbol').value.trim().toUpperCase();
  const shares = parseInt(document.getElementById('tradeShares').value);

  if (!symbol || !shares || shares < 1) {
    alert("Please enter valid symbol and shares");
    return;
  }

  try {
    const response = await fetch("/portfolio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "buy",
        symbol: symbol,
        shares: shares
      })
    });

    const data = await response.json();

    if (data.success) {
      alert(`Bought ${shares} shares of ${symbol}`);
      loadPortfolio();
      document.getElementById('tradeSymbol').value = '';
      document.getElementById('tradeShares').value = '';
    } else {
      alert(data.error || "Error buying stock");
    }
  } catch (error) {
    alert("Error buying stock");
  }
}

/**
 * Sell stock
 */
export async function sellStock() {
  const symbol = document.getElementById('tradeSymbol').value.trim().toUpperCase();
  const shares = parseInt(document.getElementById('tradeShares').value);

  if (!symbol || !shares || shares < 1) {
    alert("Please enter valid symbol and shares");
    return;
  }

  try {
    const response = await fetch("/portfolio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "sell",
        symbol: symbol,
        shares: shares
      })
    });

    const data = await response.json();

    if (data.success) {
      alert(`Sold ${shares} shares of ${symbol}`);
      loadPortfolio();
      document.getElementById('tradeSymbol').value = '';
      document.getElementById('tradeShares').value = '';
    } else {
      alert(data.error || "Error selling stock");
    }
  } catch (error) {
    alert("Error selling stock");
  }
}

// Make globally accessible for onclick handlers
window.loadPortfolio = loadPortfolio;
window.buyStock = buyStock;
window.sellStock = sellStock;

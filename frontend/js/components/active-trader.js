/**
 * Active Trader Component
 * Quick order entry, positions, and account management
 */

import { state } from '../core/state.js';

export class ActiveTrader {
  constructor() {
    this.currentSymbol = null;
    this.positions = [];
    this.orders = [];
    this.accountData = {
      buyingPower: 10000,
      totalPL: 0,
      accountValue: 10000,
      marginUsage: 0
    };

    this.init();
  }

  init() {
    this.bindEvents();
    this.loadAccount();
    this.loadPositions();
  }

  bindEvents() {
    // Order type change
    const orderTypeSelect = document.getElementById('order-type');
    if (orderTypeSelect) {
      orderTypeSelect.addEventListener('change', (e) => {
        this.handleOrderTypeChange(e.target.value);
      });
    }

    // Buy button
    const buyBtn = document.getElementById('btn-buy');
    if (buyBtn) {
      buyBtn.addEventListener('click', () => this.placeBuyOrder());
    }

    // Sell button
    const sellBtn = document.getElementById('btn-sell');
    if (sellBtn) {
      sellBtn.addEventListener('click', () => this.placeSellOrder());
    }

    // Quantity input validation
    const qtyInput = document.getElementById('order-quantity');
    if (qtyInput) {
      qtyInput.addEventListener('input', () => this.validateOrder());
    }

    // Price input validation
    const priceInput = document.getElementById('order-price');
    if (priceInput) {
      priceInput.addEventListener('input', () => this.validateOrder());
    }

    // Section toggles
    const sectionToggles = document.querySelectorAll('.tos-section-toggle');
    sectionToggles.forEach(toggle => {
      toggle.addEventListener('click', (e) => {
        const content = e.currentTarget.nextElementSibling;
        if (content) {
          content.classList.toggle('collapsed');
        }
      });
    });
  }

  setSymbol(symbol) {
    this.currentSymbol = symbol;
    const symbolDisplay = document.getElementById('order-symbol-display');
    if (symbolDisplay) {
      symbolDisplay.textContent = symbol || 'SELECT SYMBOL';
    }

    // Auto-populate symbol in input
    const symbolInput = document.getElementById('order-symbol');
    if (symbolInput && symbol) {
      symbolInput.value = symbol;
    }

    this.validateOrder();
  }

  handleOrderTypeChange(orderType) {
    const priceGroup = document.getElementById('price-group');
    const stopGroup = document.getElementById('stop-group');

    if (priceGroup) {
      // Show price input for limit and stop-limit orders
      priceGroup.style.display = ['limit', 'stop-limit'].includes(orderType) ? 'block' : 'none';
    }

    if (stopGroup) {
      // Show stop price for stop and stop-limit orders
      stopGroup.style.display = ['stop', 'stop-limit'].includes(orderType) ? 'block' : 'none';
    }
  }

  validateOrder() {
    const symbolInput = document.getElementById('order-symbol');
    const qtyInput = document.getElementById('order-quantity');
    const priceInput = document.getElementById('order-price');
    const buyBtn = document.getElementById('btn-buy');
    const sellBtn = document.getElementById('btn-sell');

    const symbol = symbolInput?.value.trim();
    const quantity = parseInt(qtyInput?.value) || 0;
    const price = parseFloat(priceInput?.value) || 0;
    const orderType = document.getElementById('order-type')?.value;

    let isValid = symbol && quantity > 0;

    if (orderType === 'limit' || orderType === 'stop-limit') {
      isValid = isValid && price > 0;
    }

    // Check buying power for buy orders
    const estimatedCost = quantity * (price || 100); // Use price or estimate
    const hasFunds = estimatedCost <= this.accountData.buyingPower;

    if (buyBtn) {
      buyBtn.disabled = !isValid || !hasFunds;
    }

    if (sellBtn) {
      sellBtn.disabled = !isValid;
    }

    // Update estimated cost display
    this.updateEstimatedCost(quantity, price);
  }

  updateEstimatedCost(quantity, price) {
    const costDisplay = document.getElementById('estimated-cost');
    if (!costDisplay) return;

    if (quantity > 0 && price > 0) {
      const cost = quantity * price;
      costDisplay.textContent = `Est. Cost: $${cost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      costDisplay.style.color = cost <= this.accountData.buyingPower ? 'var(--tos-text-primary)' : 'var(--tos-accent-red)';
    } else {
      costDisplay.textContent = '';
    }
  }

  async placeBuyOrder() {
    const symbol = document.getElementById('order-symbol')?.value.trim().toUpperCase();
    const quantity = parseInt(document.getElementById('order-quantity')?.value) || 0;
    const orderType = document.getElementById('order-type')?.value || 'market';

    if (!symbol || quantity <= 0) {
      this.showNotification('Please enter valid symbol and quantity', 'error');
      return;
    }

    try {
      // Use existing portfolio API
      const response = await fetch('/portfolio/buy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${state.authToken || 'demo'}`
        },
        body: JSON.stringify({
          symbol: symbol,
          shares: quantity
        })
      });

      const result = await response.json();

      if (response.ok) {
        this.showNotification(`Buy order placed: ${quantity} shares of ${symbol}`, 'success');
        this.clearOrderForm();
        this.loadPositions();
        this.loadAccount();
      } else {
        this.showNotification(result.error || 'Order failed', 'error');
      }
    } catch (error) {
      console.error('Error placing buy order:', error);
      this.showNotification('Error placing order', 'error');
    }
  }

  async placeSellOrder() {
    const symbol = document.getElementById('order-symbol')?.value.trim().toUpperCase();
    const quantity = parseInt(document.getElementById('order-quantity')?.value) || 0;

    if (!symbol || quantity <= 0) {
      this.showNotification('Please enter valid symbol and quantity', 'error');
      return;
    }

    try {
      // Use existing portfolio API
      const response = await fetch('/portfolio/sell', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${state.authToken || 'demo'}`
        },
        body: JSON.stringify({
          symbol: symbol,
          shares: quantity
        })
      });

      const result = await response.json();

      if (response.ok) {
        this.showNotification(`Sell order placed: ${quantity} shares of ${symbol}`, 'success');
        this.clearOrderForm();
        this.loadPositions();
        this.loadAccount();
      } else {
        this.showNotification(result.error || 'Order failed', 'error');
      }
    } catch (error) {
      console.error('Error placing sell order:', error);
      this.showNotification('Error placing order', 'error');
    }
  }

  clearOrderForm() {
    const qtyInput = document.getElementById('order-quantity');
    const priceInput = document.getElementById('order-price');

    if (qtyInput) qtyInput.value = '';
    if (priceInput) priceInput.value = '';

    this.validateOrder();
  }

  async loadAccount() {
    try {
      // Use existing portfolio API
      const response = await fetch('/portfolio', {
        headers: {
          'Authorization': `Bearer ${state.authToken || 'demo'}`
        }
      });

      const data = await response.json();

      if (data.portfolio) {
        // Calculate account metrics from portfolio
        let totalValue = 0;
        let totalCost = 0;

        data.portfolio.forEach(position => {
          totalValue += position.shares * position.current_price;
          totalCost += position.shares * position.avg_price;
        });

        this.accountData.totalPL = totalValue - totalCost;
        this.accountData.accountValue = 10000 + this.accountData.totalPL; // Starting with $10k
        this.accountData.buyingPower = this.accountData.accountValue - totalCost;
        this.accountData.marginUsage = (totalCost / this.accountData.accountValue) * 100;
      }
    } catch (error) {
      console.error('Error loading account:', error);
      // Use default values
    }

    this.renderAccountSummary();
  }

  async loadPositions() {
    try {
      // Use existing portfolio API
      const response = await fetch('/portfolio', {
        headers: {
          'Authorization': `Bearer ${state.authToken || 'demo'}`
        }
      });

      const data = await response.json();

      if (data.portfolio) {
        this.positions = data.portfolio.map(p => ({
          symbol: p.symbol,
          quantity: p.shares,
          avgPrice: p.avg_price,
          currentPrice: p.current_price,
          pnl: (p.current_price - p.avg_price) * p.shares,
          pnlPercent: ((p.current_price - p.avg_price) / p.avg_price) * 100
        }));
      }
    } catch (error) {
      console.error('Error loading positions:', error);
      this.positions = [];
    }

    this.renderPositions();
  }

  renderPositions() {
    const container = document.getElementById('positions-content');
    if (!container) return;

    container.innerHTML = '';

    if (this.positions.length === 0) {
      const emptyMessage = document.createElement('div');
      emptyMessage.style.cssText = 'padding: 16px; text-align: center; color: var(--tos-text-secondary); font-size: 11px;';
      emptyMessage.textContent = 'No open positions';
      container.appendChild(emptyMessage);
      return;
    }

    this.positions.forEach(position => {
      const item = this.createPositionItem(position);
      container.appendChild(item);
    });
  }

  createPositionItem(position) {
    const item = document.createElement('div');
    item.className = 'tos-position-item';

    const header = document.createElement('div');
    header.className = 'tos-position-header';

    const symbol = document.createElement('div');
    symbol.className = 'tos-position-symbol';
    symbol.textContent = position.symbol;

    const pnl = document.createElement('div');
    pnl.className = 'tos-position-pnl ' + (position.pnl >= 0 ? 'positive' : 'negative');
    pnl.textContent = (position.pnl >= 0 ? '+' : '') + position.pnl.toFixed(2);

    header.appendChild(symbol);
    header.appendChild(pnl);

    const details = document.createElement('div');
    details.className = 'tos-position-details';

    details.innerHTML = `
      <div>Qty: ${position.quantity}</div>
      <div>Avg: $${position.avgPrice.toFixed(2)}</div>
      <div>Current: $${position.currentPrice.toFixed(2)}</div>
      <div class="${position.pnlPercent >= 0 ? 'tos-text-green' : 'tos-text-red'}">${(position.pnlPercent >= 0 ? '+' : '')}${position.pnlPercent.toFixed(2)}%</div>
    `;

    item.appendChild(header);
    item.appendChild(details);

    // Click to load symbol
    item.addEventListener('click', () => {
      this.setSymbol(position.symbol);
      if (window.tosApp && window.tosApp.watchlist) {
        window.tosApp.watchlist.selectSymbol(position.symbol);
      }
    });

    return item;
  }

  renderAccountSummary() {
    const summary = document.getElementById('account-summary');
    if (!summary) return;

    summary.innerHTML = `
      <div class="tos-account-row">
        <div class="tos-account-label">Buying Power</div>
        <div class="tos-account-value">$${this.accountData.buyingPower.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
      </div>
      <div class="tos-account-row">
        <div class="tos-account-label">Total P&L (Day)</div>
        <div class="tos-account-value ${this.accountData.totalPL >= 0 ? 'positive' : 'negative'}">
          ${this.accountData.totalPL >= 0 ? '+' : ''}$${this.accountData.totalPL.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
      </div>
      <div class="tos-account-row">
        <div class="tos-account-label">Account Value</div>
        <div class="tos-account-value">$${this.accountData.accountValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
      </div>
      <div class="tos-account-row">
        <div class="tos-account-label">Margin Usage</div>
        <div class="tos-account-value">${this.accountData.marginUsage.toFixed(1)}%</div>
      </div>
    `;
  }

  showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 50px;
      right: 20px;
      background: var(--tos-bg-tertiary);
      color: var(--tos-text-primary);
      padding: 12px 20px;
      border-radius: 4px;
      border-left: 3px solid ${type === 'success' ? 'var(--tos-accent-green)' : type === 'error' ? 'var(--tos-accent-red)' : 'var(--tos-accent-blue)'};
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      z-index: 10000;
      font-size: 13px;
      animation: slideIn 0.3s ease;
      max-width: 300px;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
      notification.style.animation = 'slideOut 0.3s ease';
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  }

  destroy() {
    // Cleanup if needed
  }
}

export function initializeActiveTrader() {
  return new ActiveTrader();
}

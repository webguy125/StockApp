/**
 * Watchlist Component
 * Manages symbol watchlists with real-time WebSocket price updates
 */

export class Watchlist {
  constructor(onSymbolSelect, socket) {
    this.symbols = [];
    this.activeTab = 'mylist';
    this.onSymbolSelect = onSymbolSelect;
    this.socket = socket; // WebSocket connection for real-time updates
    this.subscribedSymbols = new Set(); // Track which symbols we've subscribed to

    this.predefinedLists = {
      mylist: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD'],
      majors: ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD'],
      defi: ['UNI-USD', 'AAVE-USD', 'LINK-USD', 'MKR-USD', 'COMP-USD', 'SNX-USD', 'CRV-USD', 'SUSHI-USD'],
      meme: ['DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'FLOKI-USD', 'BONK-USD']
    };

    this.init();
  }

  init() {
    this.loadWatchlist();
    this.bindEvents();
    this.setupWebSocketListeners();
    // Don't subscribe here - wait for WebSocket to connect
    // Main app will call subscribeToAllSymbols() after connection
  }

  bindEvents() {
    // Search input
    const searchInput = document.getElementById('watchlist-search');
    if (searchInput) {
      searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          this.addSymbol(e.target.value.trim().toUpperCase());
          e.target.value = '';
        }
      });
    }

    // Tab switching
    const tabs = document.querySelectorAll('.tos-watchlist-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        this.switchTab(tabName);
      });
    });
  }

  switchTab(tabName) {
    this.activeTab = tabName;

    // Update active tab UI
    document.querySelectorAll('.tos-watchlist-tab').forEach(tab => {
      tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Load symbols for this tab
    this.symbols = [...this.predefinedLists[tabName]];
    this.render();
    this.subscribeToAllSymbols(); // Subscribe to WebSocket updates for new symbols
  }

  loadWatchlist() {
    // Check if we need to migrate from stocks to crypto
    const migrated = localStorage.getItem('tos-watchlist-migrated-to-crypto');

    if (!migrated) {
      // Clear old stock watchlist and set crypto defaults
      localStorage.removeItem('tos-watchlist-mylist');
      localStorage.setItem('tos-watchlist-migrated-to-crypto', 'true');
      console.log('Migrated watchlist from stocks to crypto');
    }

    // Load saved watchlist from localStorage
    const saved = localStorage.getItem('tos-watchlist-mylist');
    if (saved) {
      try {
        this.predefinedLists.mylist = JSON.parse(saved);
      } catch (error) {
        console.error('Error loading watchlist:', error);
      }
    }

    this.symbols = [...this.predefinedLists[this.activeTab]];
    this.render();
    // WebSocket subscriptions handled by subscribeToAllSymbols() called from init()
  }

  saveWatchlist() {
    if (this.activeTab === 'mylist') {
      localStorage.setItem('tos-watchlist-mylist', JSON.stringify(this.predefinedLists.mylist));
    }
  }

  addSymbol(symbol) {
    if (!symbol || this.symbols.includes(symbol)) return;

    if (this.activeTab === 'mylist') {
      this.predefinedLists.mylist.push(symbol);
      this.symbols.push(symbol);
      this.saveWatchlist();
      this.render();
      this.subscribeToSymbol(symbol); // Subscribe to WebSocket updates for new symbol
    }
  }

  removeSymbol(symbol) {
    if (this.activeTab === 'mylist') {
      this.predefinedLists.mylist = this.predefinedLists.mylist.filter(s => s !== symbol);
      this.symbols = this.symbols.filter(s => s !== symbol);
      this.saveWatchlist();
      this.render();
    }
  }

  render() {
    const container = document.getElementById('watchlist-content');
    if (!container) return;

    container.innerHTML = '';

    this.symbols.forEach(symbol => {
      const item = this.createWatchlistItem(symbol);
      container.appendChild(item);
    });
  }

  createWatchlistItem(symbol) {
    const item = document.createElement('div');
    item.className = 'tos-watchlist-item';
    item.dataset.symbol = symbol;

    const symbolEl = document.createElement('div');
    symbolEl.className = 'tos-watchlist-symbol';
    symbolEl.textContent = symbol;

    const priceEl = document.createElement('div');
    priceEl.className = 'tos-watchlist-price';
    priceEl.textContent = '--';

    const changeEl = document.createElement('div');
    changeEl.className = 'tos-watchlist-change';
    changeEl.textContent = '--';

    const percentEl = document.createElement('div');
    percentEl.className = 'tos-watchlist-percent';
    percentEl.textContent = '--';

    item.appendChild(symbolEl);
    item.appendChild(priceEl);
    item.appendChild(changeEl);
    item.appendChild(percentEl);

    // Click to load symbol in chart
    item.addEventListener('click', () => {
      this.selectSymbol(symbol);
    });

    // Right-click menu
    item.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this.showContextMenu(symbol, e.clientX, e.clientY);
    });

    return item;
  }

  selectSymbol(symbol) {
    // Update selected state
    document.querySelectorAll('.tos-watchlist-item').forEach(item => {
      item.classList.toggle('selected', item.dataset.symbol === symbol);
    });

    // Notify parent (load chart)
    if (this.onSymbolSelect) {
      this.onSymbolSelect(symbol);
    }
  }

  showContextMenu(symbol, x, y) {
    // Remove existing context menu
    const existing = document.querySelector('.watchlist-context-menu');
    if (existing) existing.remove();

    const menu = document.createElement('div');
    menu.className = 'watchlist-context-menu';
    menu.style.cssText = `
      position: fixed;
      left: ${x}px;
      top: ${y}px;
      background: var(--tos-bg-secondary);
      border: 1px solid var(--tos-border-color);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
      z-index: 10000;
      border-radius: 3px;
      overflow: hidden;
    `;

    const actions = [
      { label: 'Load Chart', action: () => this.selectSymbol(symbol) },
      { label: 'View Details', action: () => this.viewDetails(symbol) }
    ];

    if (this.activeTab === 'mylist') {
      actions.push({ label: 'Remove', action: () => this.removeSymbol(symbol) });
    }

    actions.forEach(({ label, action }) => {
      const item = document.createElement('div');
      item.className = 'tos-dropdown-item';
      item.textContent = label;
      item.addEventListener('click', () => {
        action();
        menu.remove();
      });
      menu.appendChild(item);
    });

    document.body.appendChild(menu);

    // Close on click outside
    const closeMenu = (e) => {
      if (!menu.contains(e.target)) {
        menu.remove();
        document.removeEventListener('click', closeMenu);
      }
    };
    setTimeout(() => document.addEventListener('click', closeMenu), 0);
  }

  viewDetails(symbol) {
    alert(`Detailed view for ${symbol}\n\nFeature coming soon!`);
  }

  /**
   * Setup WebSocket event listeners for real-time price updates
   */
  setupWebSocketListeners() {
    if (!this.socket) {
      console.warn('No WebSocket connection available for watchlist');
      return;
    }

    // Listen for ticker updates (real-time price changes)
    this.socket.on('ticker_update', (data) => {
      // console.log('💰🔥 WATCHLIST RECEIVED ticker_update event:', data);
      this.handleTickerUpdate(data);
    });

    // console.log('✅ Watchlist WebSocket listeners setup, socket connected:', this.socket.connected);
  }

  /**
   * Handle incoming ticker update from WebSocket
   */
  handleTickerUpdate(data) {
    const symbol = data.symbol;

    // Only update if this symbol is in the watchlist
    if (!this.symbols.includes(symbol)) {
      return;
    }

    // console.log(`💰 Watchlist received ticker for ${symbol}: $${data.price}`);

    // Calculate price change (we'll track previous price per symbol)
    if (!this.previousPrices) {
      this.previousPrices = {};
    }

    const currentPrice = data.price;
    const previousPrice = this.previousPrices[symbol] || currentPrice;
    const change = currentPrice - previousPrice;
    const changePercent = previousPrice > 0 ? (change / previousPrice) * 100 : 0;

    // Update the watchlist item display
    this.updateWatchlistItem(symbol, currentPrice, change, changePercent);

    // Store current price as previous for next update
    this.previousPrices[symbol] = currentPrice;
  }

  /**
   * Subscribe to WebSocket ticker updates for all symbols in watchlist
   */
  subscribeToAllSymbols() {
    if (!this.socket || !this.socket.connected) {
      console.warn('WebSocket not connected, cannot subscribe to symbols');
      return;
    }

    this.symbols.forEach(symbol => {
      this.subscribeToSymbol(symbol);
    });
  }

  /**
   * Subscribe to WebSocket ticker updates for a single symbol
   */
  subscribeToSymbol(symbol) {
    if (!this.socket || !this.socket.connected) {
      console.warn(`WebSocket not connected, cannot subscribe to ${symbol}`);
      return;
    }

    // Don't subscribe twice
    if (this.subscribedSymbols.has(symbol)) {
      return;
    }

    this.socket.emit('subscribe_ticker', { symbol });
    this.subscribedSymbols.add(symbol);
  }

  updateWatchlistItem(symbol, price, change, changePercent) {
    const item = document.querySelector(`.tos-watchlist-item[data-symbol="${symbol}"]`);
    if (!item) return;

    const priceEl = item.querySelector('.tos-watchlist-price');
    const changeEl = item.querySelector('.tos-watchlist-change');
    const percentEl = item.querySelector('.tos-watchlist-percent');

    if (priceEl) priceEl.textContent = price.toFixed(2);

    const isPositive = change >= 0;

    if (changeEl) {
      changeEl.textContent = (isPositive ? '+' : '') + change.toFixed(2);
      changeEl.className = 'tos-watchlist-change ' + (isPositive ? 'positive' : 'negative');
    }

    if (percentEl) {
      percentEl.textContent = (isPositive ? '+' : '') + changePercent.toFixed(2) + '%';
      percentEl.className = 'tos-watchlist-percent ' + (isPositive ? 'positive' : 'negative');
    }
  }

  destroy() {
    // WebSocket cleanup handled by main app
    console.log('Watchlist destroyed');
  }
}

export function initializeWatchlist(onSymbolSelect, socket) {
  return new Watchlist(onSymbolSelect, socket);
}

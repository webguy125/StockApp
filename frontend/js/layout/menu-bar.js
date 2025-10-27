/**
 * Menu Bar Component
 * Professional menu system with dropdown menus
 */

export class MenuBar {
  constructor() {
    this.activeMenu = null;
    this.menuItems = this.defineMenuStructure();
    this.init();
  }

  defineMenuStructure() {
    return {
      file: {
        title: 'File',
        items: [
          { label: 'New Workspace', action: 'newWorkspace' },
          { label: 'Save Layout', action: 'saveLayout' },
          { separator: true },
          { label: 'Export Chart', action: 'exportChart' },
          { label: 'Export Data', action: 'exportData' },
          { separator: true },
          { label: 'Settings', action: 'settings' },
          { separator: true },
          { label: 'Exit', action: 'exit' }
        ]
      },
      view: {
        title: 'View',
        items: [
          { label: 'Show/Hide Watchlist', action: 'toggleWatchlist' },
          { label: 'Show/Hide News', action: 'toggleNews' },
          { label: 'Show/Hide Active Trader', action: 'toggleActiveTrader' },
          { separator: true },
          { label: 'Chart Settings', action: 'chartSettings' },
          { label: 'Layout: 1 Chart', action: 'layout1' },
          { label: 'Layout: 2x1', action: 'layout2x1' },
          { label: 'Layout: 2x2', action: 'layout2x2' }
        ]
      },
      tools: {
        title: 'Tools',
        items: [
          { label: 'Pattern Scanner', action: 'patternScanner' },
          { label: 'Price Alerts', action: 'priceAlerts' },
          { label: 'Trade Ideas', action: 'tradeIdeas' },
          { label: 'Backtester', action: 'backtester' },
          { separator: true },
          { label: 'Market Screener', action: 'marketScreener' }
        ]
      },
      analysis: {
        title: 'Analysis',
        items: [
          { label: 'Technical Studies', action: 'technicalStudies' },
          { label: 'Drawing Tools', action: 'drawingTools' },
          { label: 'Fundamentals', action: 'fundamentals' },
          { separator: true },
          { label: 'Compare Symbols', action: 'compareSymbols' },
          { label: 'Correlation Matrix', action: 'correlationMatrix' }
        ]
      },
      trade: {
        title: 'Trade',
        items: [
          { label: 'Quick Trade', action: 'quickTrade' },
          { label: 'Trade History', action: 'tradeHistory' },
          { label: 'Positions', action: 'positions' },
          { separator: true },
          { label: 'Account Summary', action: 'accountSummary' }
        ]
      },
      help: {
        title: 'Help',
        items: [
          { label: 'Documentation', action: 'documentation' },
          { label: 'Keyboard Shortcuts', action: 'shortcuts' },
          { separator: true },
          { label: 'About StockApp', action: 'about' }
        ]
      }
    };
  }

  init() {
    this.bindEvents();
  }

  bindEvents() {
    // Close dropdowns when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.tos-menu-item')) {
        this.closeAllMenus();
      }
    });

    // Close dropdowns on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.closeAllMenus();
      }
    });
  }

  toggleMenu(menuName) {
    const menuItem = document.querySelector(`[data-menu="${menuName}"]`);
    const dropdown = menuItem?.querySelector('.tos-dropdown');

    if (!menuItem || !dropdown) return;

    if (this.activeMenu === menuName) {
      this.closeMenu(menuName);
    } else {
      this.closeAllMenus();
      this.openMenu(menuName);
    }
  }

  openMenu(menuName) {
    const menuItem = document.querySelector(`[data-menu="${menuName}"]`);
    const dropdown = menuItem?.querySelector('.tos-dropdown');

    if (!menuItem || !dropdown) return;

    menuItem.classList.add('active');
    dropdown.classList.add('active');
    this.activeMenu = menuName;
  }

  closeMenu(menuName) {
    const menuItem = document.querySelector(`[data-menu="${menuName}"]`);
    const dropdown = menuItem?.querySelector('.tos-dropdown');

    if (!menuItem || !dropdown) return;

    menuItem.classList.remove('active');
    dropdown.classList.remove('active');
    if (this.activeMenu === menuName) {
      this.activeMenu = null;
    }
  }

  closeAllMenus() {
    const activeMenus = document.querySelectorAll('.tos-menu-item.active');
    activeMenus.forEach(menu => menu.classList.remove('active'));

    const activeDropdowns = document.querySelectorAll('.tos-dropdown.active');
    activeDropdowns.forEach(dropdown => dropdown.classList.remove('active'));

    this.activeMenu = null;
  }

  handleMenuAction(action) {
    this.closeAllMenus();

    // Define menu action handlers
    const actions = {
      // File menu
      newWorkspace: () => this.newWorkspace(),
      saveLayout: () => this.saveLayout(),
      exportChart: () => this.exportChart(),
      exportData: () => this.exportData(),
      settings: () => this.showSettings(),
      exit: () => window.close(),

      // View menu
      toggleWatchlist: () => this.togglePanel('tos-left-panel'),
      toggleNews: () => this.toggleNewsPanel(),
      toggleActiveTrader: () => this.togglePanel('tos-right-panel'),
      chartSettings: () => this.showChartSettings(),
      layout1: () => this.setLayout(1),
      layout2x1: () => this.setLayout('2x1'),
      layout2x2: () => this.setLayout('2x2'),

      // Tools menu
      patternScanner: () => this.openPatternScanner(),
      priceAlerts: () => this.openPriceAlerts(),
      tradeIdeas: () => this.openTradeIdeas(),
      backtester: () => this.openBacktester(),
      marketScreener: () => this.openMarketScreener(),

      // Analysis menu
      technicalStudies: () => this.openTechnicalStudies(),
      drawingTools: () => this.openDrawingTools(),
      fundamentals: () => this.openFundamentals(),
      compareSymbols: () => this.openCompareSymbols(),
      correlationMatrix: () => this.openCorrelationMatrix(),

      // Trade menu
      quickTrade: () => this.focusQuickTrade(),
      tradeHistory: () => this.showTradeHistory(),
      positions: () => this.showPositions(),
      accountSummary: () => this.showAccountSummary(),

      // Help menu
      documentation: () => window.open('https://github.com/yourusername/stockapp', '_blank'),
      shortcuts: () => this.showKeyboardShortcuts(),
      about: () => this.showAbout()
    };

    if (actions[action]) {
      actions[action]();
    } else {
      console.log(`Menu action not implemented: ${action}`);
    }
  }

  // Menu action implementations
  newWorkspace() {
    if (confirm('Create a new workspace? Current layout will be saved.')) {
      this.saveLayout();
      localStorage.setItem('tos-current-workspace', Date.now().toString());
      window.location.reload();
    }
  }

  saveLayout() {
    const layout = {
      timestamp: Date.now(),
      panels: {
        leftWidth: document.querySelector('.tos-left-panel')?.style.width,
        rightWidth: document.querySelector('.tos-right-panel')?.style.width,
        watchlistHeight: document.querySelector('.tos-watchlist')?.style.height
      }
    };

    localStorage.setItem('tos-saved-layout', JSON.stringify(layout));
    this.showNotification('Layout saved successfully');
  }

  exportChart() {
    const plot = document.getElementById('tos-plot');
    if (plot) {
      Plotly.downloadImage(plot, {
        format: 'png',
        width: 1920,
        height: 1080,
        filename: `chart-${Date.now()}`
      });
    }
  }

  exportData() {
    // Trigger data export (could be implemented with existing chart data)
    console.log('Export data functionality');
    this.showNotification('Export data - Feature coming soon');
  }

  showSettings() {
    this.showNotification('Settings panel - Feature coming soon');
  }

  togglePanel(panelClass) {
    const panel = document.querySelector(`.${panelClass}`);
    if (panel) {
      panel.classList.toggle('collapsed');
      const isCollapsed = panel.classList.contains('collapsed');
      localStorage.setItem(`${panelClass}-collapsed`, isCollapsed);
    }
  }

  toggleNewsPanel() {
    const newsPanel = document.querySelector('.tos-news-feed');
    if (newsPanel) {
      newsPanel.style.display = newsPanel.style.display === 'none' ? 'flex' : 'none';
    }
  }

  showChartSettings() {
    this.showNotification('Chart settings - Feature coming soon');
  }

  setLayout(layoutType) {
    this.showNotification(`Layout ${layoutType} - Feature coming soon`);
  }

  openPatternScanner() {
    // Trigger pattern detection from existing functionality
    if (window.detectPatterns) {
      window.detectPatterns();
    }
  }

  openPriceAlerts() {
    this.showNotification('Price alerts - Feature coming soon');
  }

  openTradeIdeas() {
    // Trigger trade ideas from existing functionality
    if (window.getTradeIdeas) {
      window.getTradeIdeas();
    }
  }

  openBacktester() {
    this.showNotification('Backtester - Feature coming soon');
  }

  openMarketScreener() {
    this.showNotification('Market screener - Feature coming soon');
  }

  openTechnicalStudies() {
    this.showNotification('Technical studies panel - Feature coming soon');
  }

  openDrawingTools() {
    if (window.enableDrawing) {
      window.enableDrawing();
    }
  }

  openFundamentals() {
    this.showNotification('Fundamentals - Feature coming soon');
  }

  openCompareSymbols() {
    this.showNotification('Compare symbols - Feature coming soon');
  }

  openCorrelationMatrix() {
    this.showNotification('Correlation matrix - Feature coming soon');
  }

  focusQuickTrade() {
    const orderEntry = document.querySelector('.tos-order-entry input');
    if (orderEntry) {
      orderEntry.focus();
    }
  }

  showTradeHistory() {
    this.showNotification('Trade history - Feature coming soon');
  }

  showPositions() {
    const positionsSection = document.querySelector('.tos-positions .tos-section-content');
    if (positionsSection) {
      positionsSection.classList.remove('collapsed');
    }
  }

  showAccountSummary() {
    const accountSection = document.querySelector('.tos-account-summary');
    if (accountSection) {
      accountSection.scrollIntoView({ behavior: 'smooth' });
    }
  }

  showKeyboardShortcuts() {
    const shortcuts = `
Keyboard Shortcuts:

Chart Navigation:
  Arrow Keys - Pan chart
  +/- - Zoom in/out
  Space - Pan mode
  Esc - Cancel drawing

Tools:
  D - Drawing mode
  P - Pattern scanner
  T - Trade ideas
  I - Add indicator

View:
  F1 - Toggle watchlist
  F2 - Toggle active trader
  F11 - Fullscreen

General:
  Ctrl+S - Save layout
  Ctrl+E - Export chart
    `.trim();

    alert(shortcuts);
  }

  showAbout() {
    alert('StockApp ThinkorSwim-Style Trading Platform\nVersion 1.0\n\nProfessional stock analysis and trading interface.');
  }

  showNotification(message) {
    // Simple notification system
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 50px;
      right: 20px;
      background: var(--tos-bg-tertiary);
      color: var(--tos-text-primary);
      padding: 12px 20px;
      border-radius: 4px;
      border: 1px solid var(--tos-border-color);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      z-index: 10000;
      font-size: 13px;
      animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
      notification.style.animation = 'slideOut 0.3s ease';
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  }
}

export function initializeMenuBar() {
  return new MenuBar();
}

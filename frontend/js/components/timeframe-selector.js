/**
 * Timeframe Selector UI Component
 * TradingView-style dropdown menu for selecting chart timeframes
 */

export class TimeframeSelector {
  constructor(registry, tickRegistry, onTimeframeChange) {
    this.registry = registry;
    this.tickRegistry = tickRegistry;
    this.onTimeframeChange = onTimeframeChange;
    this.currentTimeframeId = '1d'; // Default to 1 day
    this.isOpen = false;

    this.initializeUI();
  }

  /**
   * Initialize the UI component
   */
  initializeUI() {
    // Find or create the container
    let container = document.getElementById('timeframe-selector-container');
    if (!container) {
      // Create container if it doesn't exist
      container = document.createElement('div');
      container.id = 'timeframe-selector-container';
      container.className = 'timeframe-selector-container';

      // Insert into the chart controls area
      const chartControls = document.querySelector('.chart-controls');
      if (chartControls) {
        chartControls.insertBefore(container, chartControls.firstChild);
      }
    }

    this.container = container;
    this.render();
    this.attachEventListeners();
  }

  /**
   * Render the selector UI
   */
  render() {
    const grouped = this.registry.getAllGrouped();
    const tickGrouped = this.tickRegistry.getAllGrouped();

    this.container.innerHTML = `
      <div class="timeframe-selector">
        <button class="timeframe-button" id="timeframe-button">
          <span class="timeframe-label">${this.getCurrentLabel()}</span>
          <svg width="10" height="6" viewBox="0 0 10 6" fill="currentColor">
            <path d="M0 0l5 6 5-6z"/>
          </svg>
        </button>

        <div class="timeframe-dropdown ${this.isOpen ? 'open' : ''}" id="timeframe-dropdown">
          ${this.renderCategory('TICKS', tickGrouped.ticks)}
          ${this.renderCategory('SECONDS', grouped.seconds)}
          ${this.renderCategory('MINUTES', grouped.minutes)}
          ${this.renderCategory('HOURS', grouped.hours)}
          ${this.renderCategory('DAYS', grouped.days)}
          ${this.renderCategory('RANGES', grouped.ranges)}
        </div>
      </div>
    `;
  }

  /**
   * Render a category section
   */
  renderCategory(title, timeframes) {
    if (timeframes.length === 0) return '';

    const items = timeframes.map(tf => `
      <div class="timeframe-item ${tf.id === this.currentTimeframeId ? 'active' : ''}"
           data-timeframe-id="${tf.id}">
        ${tf.name}
      </div>
    `).join('');

    return `
      <div class="timeframe-category">
        <div class="timeframe-category-header">${title}</div>
        <div class="timeframe-category-items">
          ${items}
        </div>
      </div>
    `;
  }

  /**
   * Get current timeframe label
   */
  getCurrentLabel() {
    // Check timeframe registry first
    let current = this.registry.get(this.currentTimeframeId);
    if (current) return current.name;

    // Check tick chart registry
    current = this.tickRegistry.get(this.currentTimeframeId);
    if (current) return current.name;

    return '1 day';
  }

  /**
   * Attach event listeners
   */
  attachEventListeners() {
    // Toggle dropdown
    const button = this.container.querySelector('#timeframe-button');
    if (button) {
      button.addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggleDropdown();
      });
    }

    // Handle timeframe selection
    const items = this.container.querySelectorAll('.timeframe-item');
    items.forEach(item => {
      item.addEventListener('click', (e) => {
        const timeframeId = e.target.dataset.timeframeId;
        this.selectTimeframe(timeframeId);
      });
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (this.isOpen && !this.container.contains(e.target)) {
        this.closeDropdown();
      }
    });
  }

  /**
   * Toggle dropdown open/closed
   */
  toggleDropdown() {
    this.isOpen = !this.isOpen;
    const dropdown = this.container.querySelector('#timeframe-dropdown');
    if (dropdown) {
      dropdown.classList.toggle('open', this.isOpen);
    }
  }

  /**
   * Close dropdown
   */
  closeDropdown() {
    this.isOpen = false;
    const dropdown = this.container.querySelector('#timeframe-dropdown');
    if (dropdown) {
      dropdown.classList.remove('open');
    }
  }

  /**
   * Select a timeframe
   */
  async selectTimeframe(timeframeId) {
    console.log(`📊 Selected timeframe: ${timeframeId}`);

    this.currentTimeframeId = timeframeId;
    this.closeDropdown();
    this.render();
    this.attachEventListeners();

    // Notify callback
    if (this.onTimeframeChange) {
      await this.onTimeframeChange(timeframeId);
    }
  }

  /**
   * Update current timeframe programmatically
   */
  setCurrentTimeframe(timeframeId) {
    this.currentTimeframeId = timeframeId;
    this.render();
    this.attachEventListeners();
  }

  /**
   * Destroy component
   */
  destroy() {
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }
}

/**
 * Keyboard Controls Module
 * Manages keyboard shortcuts for chart interactions
 */

export class KeyboardControls {
  constructor(plotId, options = {}) {
    this.plotId = plotId;
    this.panZoom = options.panZoom || null; // Reference to PanZoom instance
    this.enabled = true;
    this.listeners = new Map(); // Track registered listeners

    this.init();
  }

  /**
   * Initialize keyboard event listeners
   */
  init() {
    // Ctrl key toggles between pan and zoom
    this.registerListener('ctrl-toggle', (e) => {
      if (e.key === 'Control' && this.panZoom) {
        this.panZoom.toggleMode();
        e.preventDefault();
      }
    });

    // R key resets zoom
    this.registerListener('reset-zoom', (e) => {
      if (e.key === 'r' || e.key === 'R') {
        if (this.panZoom) {
          this.panZoom.resetZoom();
          e.preventDefault();
        }
      }
    });

    // Plus/Minus keys for zoom in/out
    this.registerListener('zoom-keys', (e) => {
      const plotDiv = document.getElementById(this.plotId);
      if (!plotDiv) return;

      if (e.key === '+' || e.key === '=') {
        // Zoom in
        this.zoomIn();
        e.preventDefault();
      } else if (e.key === '-' || e.key === '_') {
        // Zoom out
        this.zoomOut();
        e.preventDefault();
      }
    });
  }

  /**
   * Register a keyboard listener
   * @param {string} name - Unique name for the listener
   * @param {Function} handler - Event handler function
   */
  registerListener(name, handler) {
    if (this.listeners.has(name)) {
      document.removeEventListener('keydown', this.listeners.get(name));
    }

    const wrappedHandler = (e) => {
      if (!this.enabled) return;
      handler(e);
    };

    this.listeners.set(name, wrappedHandler);
    document.addEventListener('keydown', wrappedHandler);
  }

  /**
   * Unregister a keyboard listener
   * @param {string} name - Name of the listener to remove
   */
  unregisterListener(name) {
    if (this.listeners.has(name)) {
      document.removeEventListener('keydown', this.listeners.get(name));
      this.listeners.delete(name);
    }
  }

  /**
   * Enable keyboard controls
   */
  enable() {
    this.enabled = true;
  }

  /**
   * Disable keyboard controls
   */
  disable() {
    this.enabled = false;
  }

  /**
   * Toggle keyboard controls on/off
   */
  toggle() {
    this.enabled = !this.enabled;
  }

  /**
   * Zoom in by 20%
   */
  zoomIn() {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv || !plotDiv._fullLayout) return;

    const xRange = plotDiv._fullLayout.xaxis.range;
    const yRange = plotDiv._fullLayout.yaxis.range;

    const xCenter = (xRange[0] + xRange[1]) / 2;
    const yCenter = (yRange[0] + yRange[1]) / 2;
    const xSpan = (xRange[1] - xRange[0]) * 0.8;
    const ySpan = (yRange[1] - yRange[0]) * 0.8;

    Plotly.relayout(this.plotId, {
      'xaxis.range': [xCenter - xSpan / 2, xCenter + xSpan / 2],
      'yaxis.range': [yCenter - ySpan / 2, yCenter + ySpan / 2]
    });
  }

  /**
   * Zoom out by 20%
   */
  zoomOut() {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv || !plotDiv._fullLayout) return;

    const xRange = plotDiv._fullLayout.xaxis.range;
    const yRange = plotDiv._fullLayout.yaxis.range;

    const xCenter = (xRange[0] + xRange[1]) / 2;
    const yCenter = (yRange[0] + yRange[1]) / 2;
    const xSpan = (xRange[1] - xRange[0]) * 1.2;
    const ySpan = (yRange[1] - yRange[0]) * 1.2;

    Plotly.relayout(this.plotId, {
      'xaxis.range': [xCenter - xSpan / 2, xCenter + xSpan / 2],
      'yaxis.range': [yCenter - ySpan / 2, yCenter + ySpan / 2]
    });
  }

  /**
   * Clean up all event listeners
   */
  destroy() {
    for (const [name, handler] of this.listeners) {
      document.removeEventListener('keydown', handler);
    }
    this.listeners.clear();
  }
}

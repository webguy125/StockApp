/**
 * Pan and Zoom Module
 * Manages chart pan and zoom interactions
 */

export class PanZoom {
  constructor(plotId) {
    this.plotId = plotId;
    this.dragMode = 'pan'; // 'pan' or 'zoom'
    this.scrollZoomEnabled = true;
  }

  /**
   * Get pan/zoom configuration for Plotly config
   */
  getConfig() {
    return {
      scrollZoom: this.scrollZoomEnabled,
      dragmode: this.dragMode
    };
  }

  /**
   * Get layout configuration for pan/zoom
   */
  getLayoutConfig() {
    return {
      dragmode: this.dragMode,
      hovermode: 'x' // Show data for all traces at same x-position
    };
  }

  /**
   * Enable scroll zoom
   */
  enableScrollZoom() {
    this.scrollZoomEnabled = true;
    this.update();
  }

  /**
   * Disable scroll zoom
   */
  disableScrollZoom() {
    this.scrollZoomEnabled = false;
    this.update();
  }

  /**
   * Set drag mode to pan
   */
  setPan() {
    this.dragMode = 'pan';
    this.update();
  }

  /**
   * Set drag mode to zoom
   */
  setZoom() {
    this.dragMode = 'zoom';
    this.update();
  }

  /**
   * Toggle between pan and zoom modes
   */
  toggleMode() {
    this.dragMode = this.dragMode === 'pan' ? 'zoom' : 'pan';
    this.update();
  }

  /**
   * Update the chart with current pan/zoom settings
   */
  update() {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv) return;

    Plotly.relayout(this.plotId, {
      dragmode: this.dragMode
    });
  }

  /**
   * Reset zoom to show all data
   */
  resetZoom() {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv) return;

    Plotly.relayout(this.plotId, {
      'xaxis.autorange': true,
      'yaxis.autorange': true
    });
  }

  /**
   * Zoom to a specific range
   * @param {Object} xRange - {min, max} for x-axis
   * @param {Object} yRange - {min, max} for y-axis
   */
  zoomToRange(xRange = null, yRange = null) {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv) return;

    const updates = {};
    if (xRange) {
      updates['xaxis.range'] = [xRange.min, xRange.max];
    }
    if (yRange) {
      updates['yaxis.range'] = [yRange.min, yRange.max];
    }

    if (Object.keys(updates).length > 0) {
      Plotly.relayout(this.plotId, updates);
    }
  }
}

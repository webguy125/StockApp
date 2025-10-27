/**
 * Crosshair Module
 * Manages crosshair (spike lines) on chart hover
 */

export class Crosshair {
  constructor(plotId) {
    this.plotId = plotId;
    this.enabled = true;
  }

  /**
   * Get crosshair configuration for Plotly layout
   * Returns the spike configuration for x and y axes
   */
  getConfig() {
    if (!this.enabled) {
      return {
        xaxis: { showspikes: false },
        yaxis: { showspikes: false }
      };
    }

    return {
      xaxis: {
        showspikes: true,
        spikemode: 'across',
        spikethickness: 1,
        spikecolor: '#666666',
        spikedash: 'solid'
      },
      yaxis: {
        showspikes: true,
        spikemode: 'across',
        spikethickness: 1,
        spikecolor: '#666666',
        spikedash: 'solid'
      }
    };
  }

  /**
   * Enable crosshair
   */
  enable() {
    this.enabled = true;
    this.update();
  }

  /**
   * Disable crosshair
   */
  disable() {
    this.enabled = false;
    this.update();
  }

  /**
   * Toggle crosshair on/off
   */
  toggle() {
    this.enabled = !this.enabled;
    this.update();
  }

  /**
   * Update the chart with current crosshair settings
   */
  update() {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv) return;

    const config = this.getConfig();
    Plotly.relayout(this.plotId, {
      'xaxis.showspikes': config.xaxis.showspikes,
      'yaxis.showspikes': config.yaxis.showspikes
    });
  }

  /**
   * Set crosshair color
   */
  setColor(color) {
    const plotDiv = document.getElementById(this.plotId);
    if (!plotDiv) return;

    Plotly.relayout(this.plotId, {
      'xaxis.spikecolor': color,
      'yaxis.spikecolor': color
    });
  }
}

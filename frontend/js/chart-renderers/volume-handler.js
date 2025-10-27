/**
 * Volume Handler Module
 * Manages volume display in both overlay and subgraph modes
 * DO NOT MODIFY - This handles all volume rendering logic
 */

export class VolumeHandler {
  /**
   * Create volume trace based on display mode
   * @param {Array} dates - Array of date strings
   * @param {Array} volumes - Array of volume values
   * @param {string} mode - 'overlay' or 'subgraph'
   * @returns {Object} Plotly trace object for volume
   */
  static createVolumeTrace(dates, volumes, mode = 'subgraph') {
    // ThinkorSwim style: uniform cyan/blue color
    const volumeColor = 'rgba(0, 188, 212, 0.7)';

    const baseTrace = {
      x: dates,
      y: volumes,
      type: 'bar',
      name: 'Volume',
      yaxis: 'y2',
      xaxis: 'x',
      marker: {
        color: volumeColor
      },
      hoverinfo: 'none',
      hovertemplate: '<extra></extra>'
    };

    return baseTrace;
  }

  /**
   * Create volume Y-axis configuration based on mode
   * @param {string} mode - 'overlay' or 'subgraph'
   * @param {number} mainChartBottom - Bottom boundary of main chart (for subgraph mode)
   * @returns {Object} Plotly yaxis2 configuration
   */
  static createVolumeAxis(mode = 'subgraph', mainChartBottom = 0) {
    if (mode === 'overlay') {
      // Overlay mode: volume layered on price chart
      return {
        title: '',
        gridcolor: 'transparent',
        color: '#a0a0a0',
        side: 'right',
        overlaying: 'y',
        showticklabels: false,
        tickformat: ',.0f',
        showgrid: false,
        fixedrange: false,
        layer: 'below traces'
      };
    } else {
      // Subgraph mode: separate panel below chart
      const volumeDomain = [0, Math.max(0.25, mainChartBottom - 0.03)];

      return {
        title: '',
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: volumeDomain,
        showticklabels: true,
        tickfont: { size: 9 },
        tickformat: ',.0f',
        showgrid: true,
        gridwidth: 1,
        fixedrange: false
      };
    }
  }

  /**
   * Update volume trace with new data
   * @param {HTMLElement} plotDiv - Plotly plot div
   * @param {Array} newVolumes - New volume data
   * @returns {boolean} Success status
   */
  static async updateVolume(plotDiv, newVolumes) {
    if (!plotDiv || !plotDiv.data) return false;

    // Find volume trace
    const volumeIndex = plotDiv.data.findIndex(
      trace => trace.type === 'bar' && trace.yaxis === 'y2'
    );

    if (volumeIndex === -1) {
      console.warn('No volume trace found to update');
      return false;
    }

    try {
      await Plotly.restyle(plotDiv, {
        y: [newVolumes]
      }, [volumeIndex]);
      return true;
    } catch (error) {
      console.error('Error updating volume:', error);
      return false;
    }
  }

  /**
   * Toggle volume visibility
   * @param {HTMLElement} plotDiv - Plotly plot div
   * @param {boolean} visible - Show or hide volume
   */
  static toggleVisibility(plotDiv, visible) {
    if (!plotDiv || !plotDiv.data) return;

    const volumeIndex = plotDiv.data.findIndex(
      trace => trace.type === 'bar' && trace.yaxis === 'y2'
    );

    if (volumeIndex !== -1) {
      Plotly.restyle(plotDiv, { visible: visible }, [volumeIndex]);
    }
  }
}

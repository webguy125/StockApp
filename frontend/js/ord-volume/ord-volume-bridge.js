/**
 * ORD Volume Bridge
 * Bridges the segregated ORD Volume renderer with the main chart renderer
 * Allows ORD Volume overlays to persist on the chart
 *
 * SEGREGATED IMPLEMENTATION - No shared code with existing features
 */

export class ORDVolumeBridge {
  constructor() {
    this.currentAnalysis = null;
    this.isActive = false;
    this.chartRenderer = null;
  }

  /**
   * Set the active chart renderer to draw on
   */
  setChartRenderer(renderer) {
    this.chartRenderer = renderer;
  }

  /**
   * Store analysis result for persistent rendering
   */
  setAnalysis(analysisResult) {
    this.currentAnalysis = analysisResult;
    this.isActive = true;
  }

  /**
   * Clear the analysis
   */
  clearAnalysis() {
    this.currentAnalysis = null;
    this.isActive = false;
  }

  /**
   * Draw ORD Volume overlays on the chart
   * Called after main chart renders
   */
  drawOverlays(ctx, chartState) {
    if (!this.isActive || !this.currentAnalysis) {
      return;
    }

    // Draw trendlines
    this._drawTrendlines(ctx, chartState);

    // Draw labels
    this._drawLabels(ctx, chartState);
  }

  /**
   * Draw trendlines
   * @private
   */
  _drawTrendlines(ctx, chartState) {
    if (!this.currentAnalysis.trendlines) return;

    ctx.save();

    for (const line of this.currentAnalysis.trendlines) {
      // Convert indices to screen coordinates
      const x1 = this._indexToX(line.x1, chartState);
      const y1 = this._priceToY(line.y1, chartState);
      const x2 = this._indexToX(line.x2, chartState);
      const y2 = this._priceToY(line.y2, chartState);

      // Skip if off-screen
      if (!this._isOnScreen(x1, y1, x2, y2, ctx.canvas)) {
        continue;
      }

      // Draw line
      ctx.strokeStyle = line.color || '#2196f3';
      ctx.lineWidth = line.lineWidth || 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Draw label near line start
      if (line.label) {
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px Arial';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
        ctx.shadowBlur = 4;
        const textX = x1 + 5;
        const textY = y1 - 5;
        ctx.fillText(line.label, textX, textY);
        ctx.shadowBlur = 0;
      }
    }

    ctx.restore();
  }

  /**
   * Draw labels (volume metrics)
   * @private
   */
  _drawLabels(ctx, chartState) {
    if (!this.currentAnalysis.labels) return;

    ctx.save();

    for (const label of this.currentAnalysis.labels) {
      // Convert to screen coordinates
      const x = this._indexToX(label.x, chartState);
      const y = this._priceToY(label.y, chartState);

      // Skip if off-screen
      if (x < 0 || x > ctx.canvas.width || y < 0 || y > ctx.canvas.height) {
        continue;
      }

      // Set font
      const fontWeight = label.fontWeight || 'normal';
      const fontSize = label.fontSize || 12;
      ctx.font = `${fontWeight} ${fontSize}px Arial`;

      // Measure text
      const metrics = ctx.measureText(label.text);
      const textWidth = metrics.width;
      const textHeight = fontSize;

      // Draw background
      const padding = 6;
      const bgX = x - padding;
      const bgY = y - textHeight - padding;
      const bgWidth = textWidth + padding * 2;
      const bgHeight = textHeight + padding * 2;

      ctx.fillStyle = label.backgroundColor || 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(bgX, bgY, bgWidth, bgHeight);

      // Draw border
      ctx.strokeStyle = label.color || '#ffffff';
      ctx.lineWidth = 1;
      ctx.strokeRect(bgX, bgY, bgWidth, bgHeight);

      // Draw text
      ctx.fillStyle = label.color || '#ffffff';
      ctx.fillText(label.text, x, y);
    }

    ctx.restore();
  }

  /**
   * Convert candle index to screen X coordinate
   * @private
   */
  _indexToX(index, chartState) {
    if (this.chartRenderer && this.chartRenderer.indexToX) {
      return this.chartRenderer.indexToX(index);
    }
    // Fallback: simple linear mapping
    return index * 10;
  }

  /**
   * Convert price to screen Y coordinate
   * @private
   */
  _priceToY(price, chartState) {
    if (this.chartRenderer && this.chartRenderer.priceToY) {
      return this.chartRenderer.priceToY(price);
    }
    // Fallback: simple linear mapping
    return 500 - (price / 100);
  }

  /**
   * Check if line is visible on screen
   * @private
   */
  _isOnScreen(x1, y1, x2, y2, canvas) {
    const margin = 50;
    const minX = Math.min(x1, x2);
    const maxX = Math.max(x1, x2);
    const minY = Math.min(y1, y2);
    const maxY = Math.max(y1, y2);

    return maxX >= -margin &&
           minX <= canvas.width + margin &&
           maxY >= -margin &&
           minY <= canvas.height + margin;
  }

  /**
   * Get current analysis data
   */
  getAnalysis() {
    return this.currentAnalysis;
  }

  /**
   * Check if ORD Volume is active
   */
  isActiveAnalysis() {
    return this.isActive;
  }
}

// Create global singleton instance
if (!window.ordVolumeBridge) {
  window.ordVolumeBridge = new ORDVolumeBridge();
}

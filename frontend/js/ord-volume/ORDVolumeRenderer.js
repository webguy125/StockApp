/**
 * ORD Volume Renderer
 * Completely segregated rendering logic for ORD Volume overlays
 * Handles canvas drawing, user interactions, and label dragging
 *
 * NO SHARED CODE - Standalone canvas implementation
 */

export class ORDVolumeRenderer {
  constructor(canvas, chartState) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.chartState = chartState; // Contains chart dimensions, scale functions, etc.

    // Drawing state
    this.isDrawMode = false;
    this.drawnLines = [];
    this.currentLine = null;
    this.startPoint = null;

    // Analysis overlays
    this.trendlines = [];
    this.labels = [];
    this.analysisResult = null;

    // Interaction state
    this.selectedLabel = null;
    this.isDragging = false;
    this.dragOffset = { x: 0, y: 0 };

    // Bind methods
    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp = this._onMouseUp.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);
  }

  /**
   * Enable draw mode for manual trendline drawing
   */
  enableDrawMode() {
    this.isDrawMode = true;
    this.drawnLines = [];
    this.currentLine = null;
    this.startPoint = null;

    // Add event listeners
    this.canvas.addEventListener('mousedown', this._onMouseDown);
    this.canvas.addEventListener('mousemove', this._onMouseMove);
    this.canvas.addEventListener('mouseup', this._onMouseUp);
    document.addEventListener('keydown', this._onKeyDown);

    this.canvas.style.cursor = 'crosshair';
  }

  /**
   * Clear draw mode
   */
  clearDrawingMode() {
    this.isDrawMode = false;
    this.drawnLines = [];
    this.currentLine = null;
    this.startPoint = null;

    // Remove event listeners
    this.canvas.removeEventListener('mousedown', this._onMouseDown);
    this.canvas.removeEventListener('mousemove', this._onMouseMove);
    this.canvas.removeEventListener('mouseup', this._onMouseUp);
    document.removeEventListener('keydown', this._onKeyDown);

    this.canvas.style.cursor = 'default';
  }

  /**
   * Handle mouse down - start drawing line or select label
   * @private
   */
  _onMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert to chart coordinates
    const chartX = this._screenToChartX(x);
    const chartY = this._screenToChartY(y);

    if (this.isDrawMode) {
      // Start new line
      this.startPoint = { x: chartX, y: chartY };
      this.currentLine = [chartX, chartY, chartX, chartY];
    } else {
      // Check if clicking on a label for dragging
      const label = this._findLabelAtPoint(x, y);
      if (label) {
        this.selectedLabel = label;
        this.isDragging = true;
        this.dragOffset = {
          x: x - this._chartToScreenX(label.x),
          y: y - this._chartToScreenY(label.y)
        };
      }
    }
  }

  /**
   * Handle mouse move - update current line or drag label
   * @private
   */
  _onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const chartX = this._screenToChartX(x);
    const chartY = this._screenToChartY(y);

    if (this.isDrawMode && this.currentLine && this.startPoint) {
      // Update current line end point
      this.currentLine[2] = chartX;
      this.currentLine[3] = chartY;
      this._redraw();
    } else if (this.isDragging && this.selectedLabel) {
      // Drag label
      const newX = x - this.dragOffset.x;
      const newY = y - this.dragOffset.y;
      this.selectedLabel.x = this._screenToChartX(newX);
      this.selectedLabel.y = this._screenToChartY(newY);
      this._redraw();
    }
  }

  /**
   * Handle mouse up - finish line or label drag
   * @private
   */
  _onMouseUp(e) {
    if (this.isDrawMode && this.currentLine && this.startPoint) {
      // Finish line
      this.drawnLines.push([...this.currentLine]);
      this.currentLine = null;
      this.startPoint = null;
      this._redraw();
    } else if (this.isDragging) {
      // Finish dragging
      this.isDragging = false;
      this.selectedLabel = null;
    }
  }

  /**
   * Handle key down - ESC to cancel current line
   * @private
   */
  _onKeyDown(e) {
    if (e.key === 'Escape' && this.isDrawMode) {
      this.currentLine = null;
      this.startPoint = null;
      this._redraw();
    }
  }

  /**
   * Get drawn lines (for analysis)
   * @returns {Array} Array of [x1, y1, x2, y2] lines
   */
  getDrawnLines() {
    return this.drawnLines;
  }

  /**
   * Render analysis result on chart
   * @param {Object} analysisResult - Result from ORDVolumeAnalysis
   * @param {String} symbol - Current symbol
   */
  renderAnalysis(analysisResult, symbol) {
    this.analysisResult = analysisResult;
    this.trendlines = analysisResult.trendlines;
    this.labels = analysisResult.labels;

    // Clear draw mode
    this.clearDrawingMode();

    // Enable label dragging
    this._enableLabelDragging();

    // Redraw
    this._redraw();
  }

  /**
   * Enable label dragging interaction
   * @private
   */
  _enableLabelDragging() {
    this.canvas.addEventListener('mousedown', this._onMouseDown);
    this.canvas.addEventListener('mousemove', this._onMouseMove);
    this.canvas.addEventListener('mouseup', this._onMouseUp);
  }

  /**
   * Clear all ORD Volume overlays
   */
  clearOverlays() {
    this.trendlines = [];
    this.labels = [];
    this.analysisResult = null;
    this.drawnLines = [];
    this._redraw();
  }

  /**
   * Redraw all ORD Volume elements
   * @private
   */
  _redraw() {
    // Note: In a real integration, this would trigger the main chart redraw
    // For now, we'll draw on top of existing canvas
    this._drawTrendlines();
    this._drawLabels();
    this._drawCurrentLine();
    this._drawDrawnLines();
  }

  /**
   * Draw trendlines
   * @private
   */
  _drawTrendlines() {
    if (!this.trendlines || this.trendlines.length === 0) return;

    this.ctx.save();

    for (const line of this.trendlines) {
      const x1 = this._chartToScreenX(line.x1);
      const y1 = this._chartToScreenY(line.y1);
      const x2 = this._chartToScreenX(line.x2);
      const y2 = this._chartToScreenY(line.y2);

      this.ctx.strokeStyle = line.color || '#2196f3';
      this.ctx.lineWidth = line.lineWidth || 2;
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();

      // Draw label near line
      if (line.label) {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '12px Arial';
        const textX = x1 + 5;
        const textY = y1 - 5;
        this.ctx.fillText(line.label, textX, textY);
      }
    }

    this.ctx.restore();
  }

  /**
   * Draw labels (volume metrics, strength)
   * @private
   */
  _drawLabels() {
    if (!this.labels || this.labels.length === 0) return;

    this.ctx.save();

    for (const label of this.labels) {
      const x = this._chartToScreenX(label.x);
      const y = this._chartToScreenY(label.y);

      // Measure text
      this.ctx.font = `${label.fontWeight || 'normal'} ${label.fontSize || 12}px Arial`;
      const metrics = this.ctx.measureText(label.text);
      const textWidth = metrics.width;
      const textHeight = label.fontSize || 12;

      // Draw background
      const padding = 6;
      const bgX = x - padding;
      const bgY = y - textHeight - padding;
      const bgWidth = textWidth + padding * 2;
      const bgHeight = textHeight + padding * 2;

      this.ctx.fillStyle = label.backgroundColor || 'rgba(0, 0, 0, 0.7)';
      this.ctx.fillRect(bgX, bgY, bgWidth, bgHeight);

      // Draw border if selected
      if (this.selectedLabel === label) {
        this.ctx.strokeStyle = '#2196f3';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(bgX, bgY, bgWidth, bgHeight);
      }

      // Draw text
      this.ctx.fillStyle = label.color || '#ffffff';
      this.ctx.fillText(label.text, x, y);
    }

    this.ctx.restore();
  }

  /**
   * Draw current line being drawn
   * @private
   */
  _drawCurrentLine() {
    if (!this.currentLine || !this.isDrawMode) return;

    this.ctx.save();

    const [x1, y1, x2, y2] = this.currentLine;
    const sx1 = this._chartToScreenX(x1);
    const sy1 = this._chartToScreenY(y1);
    const sx2 = this._chartToScreenX(x2);
    const sy2 = this._chartToScreenY(y2);

    this.ctx.strokeStyle = '#ffd600'; // Yellow for current line
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([5, 5]); // Dashed line
    this.ctx.beginPath();
    this.ctx.moveTo(sx1, sy1);
    this.ctx.lineTo(sx2, sy2);
    this.ctx.stroke();

    this.ctx.restore();
  }

  /**
   * Draw already drawn lines
   * @private
   */
  _drawDrawnLines() {
    if (!this.drawnLines || this.drawnLines.length === 0) return;

    this.ctx.save();

    for (const line of this.drawnLines) {
      const [x1, y1, x2, y2] = line;
      const sx1 = this._chartToScreenX(x1);
      const sy1 = this._chartToScreenY(y1);
      const sx2 = this._chartToScreenX(x2);
      const sy2 = this._chartToScreenY(y2);

      this.ctx.strokeStyle = '#00c853'; // Green for drawn lines
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.moveTo(sx1, sy1);
      this.ctx.lineTo(sx2, sy2);
      this.ctx.stroke();
    }

    this.ctx.restore();
  }

  /**
   * Find label at screen point
   * @private
   * @param {Number} screenX
   * @param {Number} screenY
   * @returns {Object|null} Label object or null
   */
  _findLabelAtPoint(screenX, screenY) {
    for (const label of this.labels) {
      const x = this._chartToScreenX(label.x);
      const y = this._chartToScreenY(label.y);

      // Simple box hit test (approximate)
      const textWidth = this.ctx.measureText(label.text).width;
      const textHeight = label.fontSize || 12;
      const padding = 6;

      const minX = x - padding;
      const maxX = x + textWidth + padding;
      const minY = y - textHeight - padding;
      const maxY = y + padding;

      if (screenX >= minX && screenX <= maxX && screenY >= minY && screenY <= maxY) {
        return label;
      }
    }

    return null;
  }

  /**
   * Convert screen X to chart index
   * @private
   * @param {Number} screenX
   * @returns {Number} Chart index
   */
  _screenToChartX(screenX) {
    // Simple linear mapping (customize based on actual chart state)
    if (this.chartState && this.chartState.xToIndex) {
      return this.chartState.xToIndex(screenX);
    }
    // Fallback: assume direct mapping
    return screenX;
  }

  /**
   * Convert screen Y to chart price
   * @private
   * @param {Number} screenY
   * @returns {Number} Chart price
   */
  _screenToChartY(screenY) {
    // Simple linear mapping (customize based on actual chart state)
    if (this.chartState && this.chartState.yToPrice) {
      return this.chartState.yToPrice(screenY);
    }
    // Fallback: assume direct mapping
    return screenY;
  }

  /**
   * Convert chart index to screen X
   * @private
   * @param {Number} chartX
   * @returns {Number} Screen X
   */
  _chartToScreenX(chartX) {
    if (this.chartState && this.chartState.indexToX) {
      return this.chartState.indexToX(chartX);
    }
    // Fallback
    return chartX;
  }

  /**
   * Convert chart price to screen Y
   * @private
   * @param {Number} chartY
   * @returns {Number} Screen Y
   */
  _chartToScreenY(chartY) {
    if (this.chartState && this.chartState.priceToY) {
      return this.chartState.priceToY(chartY);
    }
    // Fallback
    return chartY;
  }

  /**
   * Update chart state (for coordinate conversion)
   * @param {Object} chartState - Updated chart state
   */
  updateChartState(chartState) {
    this.chartState = chartState;
  }
}

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

    // Callback for line count updates
    this.onLineDrawn = null;

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
    console.log('[ORD Renderer] enableDrawMode() called');
    this.isDrawMode = true;
    this.drawnLines = [];
    this.currentLine = null;
    this.startPoint = null;

    // Add event listeners with CAPTURE phase to intercept before chart handlers
    this.canvas.addEventListener('mousedown', this._onMouseDown, true);
    this.canvas.addEventListener('mousemove', this._onMouseMove, true);
    this.canvas.addEventListener('mouseup', this._onMouseUp, true);
    document.addEventListener('keydown', this._onKeyDown);

    this.canvas.style.cursor = 'crosshair';
    console.log('[ORD Renderer] Draw mode enabled, cursor set to crosshair, canvas:', this.canvas);
  }

  /**
   * Clear draw mode
   */
  clearDrawingMode() {
    this.isDrawMode = false;
    this.drawnLines = [];
    this.currentLine = null;
    this.startPoint = null;

    // Remove event listeners (with same capture flag as when added)
    this.canvas.removeEventListener('mousedown', this._onMouseDown, true);
    this.canvas.removeEventListener('mousemove', this._onMouseMove, true);
    this.canvas.removeEventListener('mouseup', this._onMouseUp, true);
    document.removeEventListener('keydown', this._onKeyDown);

    this.canvas.style.cursor = 'default';
  }

  /**
   * Handle mouse down - start drawing line or select label
   * @private
   */
  _onMouseDown(e) {
    console.log('[ORD Renderer] Mouse down, isDrawMode:', this.isDrawMode);

    if (this.isDrawMode) {
      // CRITICAL: Stop the event from reaching the chart's pan/zoom handlers
      e.stopPropagation();
      e.preventDefault();

      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Convert to chart coordinates
      const chartX = this._screenToChartX(x);
      const chartY = this._screenToChartY(y);

      // Start new line
      this.startPoint = { x: chartX, y: chartY };
      this.currentLine = [chartX, chartY, chartX, chartY];
      console.log('[ORD Renderer] Started new line at', chartX, chartY);
    } else {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

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
    if (this.isDrawMode && this.currentLine && this.startPoint) {
      // CRITICAL: Prevent chart panning while drawing
      e.stopPropagation();
      e.preventDefault();

      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const chartX = this._screenToChartX(x);
      const chartY = this._screenToChartY(y);

      // Update current line end point
      this.currentLine[2] = chartX;
      this.currentLine[3] = chartY;

      // Update preview with current line
      this._updateDrawModePreview(true);

      this._redraw();
    } else if (this.isDragging && this.selectedLabel) {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

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
      // CRITICAL: Prevent chart from handling this event
      e.stopPropagation();
      e.preventDefault();

      const [x1, y1, x2, y2] = this.currentLine;

      // Only save line if it has some length (not a point click)
      const dx = Math.abs(x2 - x1);
      const dy = Math.abs(y2 - y1);
      const minLength = 5; // Minimum 5 candles or 5 price units

      console.log('[ORD Renderer] Line length:', {dx, dy, minLength});

      if (dx >= minLength || dy >= minLength) {
        // Finish line
        this.drawnLines.push([...this.currentLine]);
        console.log('[ORD Renderer] Line saved:', this.currentLine);

        // IMPORTANT: Immediately create a "preview" analysis for the bridge
        this._updateDrawModePreview();

        // Notify callback that a line was drawn
        if (this.onLineDrawn) {
          this.onLineDrawn(this.drawnLines.length);
        }
      } else {
        console.log('[ORD Renderer] Line too short, discarded');
      }

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
   * Update draw mode preview - convert drawn lines to analysis format
   * This makes the bridge draw them the same way as auto mode
   * @private
   * @param {Boolean} includeCurrentLine - Whether to include the line being drawn
   */
  _updateDrawModePreview(includeCurrentLine = false) {
    if (!window.ordVolumeBridge) {
      return;
    }

    // Convert completed lines to the format the bridge expects (same as auto mode)
    const trendlines = this.drawnLines.map((line, i) => {
      const [x1, y1, x2, y2] = line;
      return {
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        label: `Line ${i + 1}`,
        color: '#10B981', // Professional emerald green for completed lines
        lineWidth: 3,
        shadowColor: 'rgba(0, 0, 0, 0.3)',
        shadowBlur: 2
      };
    });

    // Add current line being drawn (amber dashed)
    if (includeCurrentLine && this.currentLine) {
      const [x1, y1, x2, y2] = this.currentLine;
      trendlines.push({
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        label: 'Drawing...',
        color: '#F59E0B', // Professional amber for current line
        lineWidth: 2,
        dash: [8, 4] // Dashed line with longer segments
      });
    }

    // Create a preview analysis in the same format as auto mode
    const previewAnalysis = {
      mode: 'draw-preview',
      trendlines: trendlines,
      labels: [] // No labels during drawing
    };

    // Store in bridge (same as auto mode does)
    window.ordVolumeBridge.setAnalysis(previewAnalysis);

    console.log('[ORD Renderer] Updated draw preview with', trendlines.length, 'lines (includesCurrent:', includeCurrentLine, ')');
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
   * Get current drawing state (includes in-progress line)
   * @returns {Object} Drawing state for rendering
   */
  getDrawingState() {
    return {
      drawnLines: this.drawnLines,
      currentLine: this.currentLine,
      isDrawMode: this.isDrawMode
    };
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
    // Trigger main chart redraw to show our overlays
    if (this.chartState && this.chartState.draw) {
      this.chartState.draw();
    } else if (window.tosApp && window.tosApp.activeChartType) {
      // Fallback: trigger redraw on active chart
      if (window.tosApp.activeChartType === 'timeframe') {
        const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
        if (currentTimeframe && currentTimeframe.renderer && currentTimeframe.renderer.draw) {
          currentTimeframe.renderer.draw();
        }
      } else if (window.tosApp.activeChartType === 'tick') {
        const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
        if (currentTickChart && currentTickChart.renderer && currentTickChart.renderer.draw) {
          currentTickChart.renderer.draw();
        }
      }
    }
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

    this.ctx.strokeStyle = '#F59E0B'; // Professional amber for current line
    this.ctx.lineWidth = 2;
    this.ctx.lineCap = 'round';
    this.ctx.setLineDash([8, 4]); // Dashed line with longer segments
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

      // Professional emerald green with subtle shadow
      this.ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
      this.ctx.shadowBlur = 2;
      this.ctx.shadowOffsetY = 1;
      this.ctx.strokeStyle = '#10B981'; // Professional emerald green
      this.ctx.lineWidth = 3;
      this.ctx.lineCap = 'round';
      this.ctx.lineJoin = 'round';
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
      const result = this.chartState.xToIndex(screenX);
      console.log(`[ORD Renderer] screenX ${screenX} → index ${result}`);
      return result;
    }
    // Fallback: assume direct mapping
    console.warn('[ORD Renderer] No xToIndex function, using fallback');
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
      const result = this.chartState.yToPrice(screenY);
      console.log(`[ORD Renderer] screenY ${screenY} → price ${result}`);
      return result;
    }
    // Fallback: assume direct mapping
    console.warn('[ORD Renderer] No yToPrice function, using fallback');
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

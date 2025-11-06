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
    this._hasLoggedDrawing = false;
    this.ordVolumeRenderer = null; // Reference to ORD Volume renderer for draw mode
  }

  /**
   * Set the active chart renderer to draw on
   */
  setChartRenderer(renderer) {
    this.chartRenderer = renderer;
  }

  /**
   * Set the ORD Volume renderer (for draw mode)
   */
  setORDVolumeRenderer(renderer) {
    this.ordVolumeRenderer = renderer;
    console.log('[ORD Bridge] Renderer set:', !!renderer, 'isDrawMode:', renderer?.isDrawMode);
  }

  /**
   * Store analysis result for persistent rendering
   */
  setAnalysis(analysisResult) {
    this.currentAnalysis = analysisResult;
    this.isActive = true;
    this._hasLoggedDrawing = false; // Reset flag for new analysis
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
    // Draw manual drawing lines (if in draw mode) - PRIORITY CHECK
    if (this.ordVolumeRenderer) {
      const drawingState = this.ordVolumeRenderer.getDrawingState();

      console.log('[ORD Bridge] drawOverlays called, drawingState:', {
        hasRenderer: !!this.ordVolumeRenderer,
        isDrawMode: drawingState.isDrawMode,
        drawnLines: drawingState.drawnLines?.length || 0,
        hasCurrent: !!drawingState.currentLine
      });

      if (drawingState.isDrawMode) {
        this._drawManualLines(ctx, chartState, drawingState);
        return; // Don't draw analysis overlays while drawing
      }
    }

    // Draw analysis overlays
    if (!this.isActive || !this.currentAnalysis) {
      return;
    }

    // Draw trendlines
    this._drawTrendlines(ctx, chartState);

    // Draw labels
    this._drawLabels(ctx, chartState);
  }

  /**
   * Draw manual drawing lines
   * @private
   */
  _drawManualLines(ctx, chartState, drawingState) {
    console.log('[ORD Bridge] Drawing manual lines:', {
      drawnCount: drawingState.drawnLines.length,
      hasCurrent: !!drawingState.currentLine,
      canvasSize: `${ctx.canvas.width}x${ctx.canvas.height}`
    });

    ctx.save();

    // Draw completed lines
    for (let i = 0; i < drawingState.drawnLines.length; i++) {
      const line = drawingState.drawnLines[i];
      const [x1, y1, x2, y2] = line;

      console.log(`[ORD Bridge] Line ${i} raw:`, {x1, y1, x2, y2});

      const sx1 = this._indexToX(x1, chartState);
      const sy1 = this._priceToY(y1, chartState);
      const sx2 = this._indexToX(x2, chartState);
      const sy2 = this._priceToY(y2, chartState);

      console.log(`[ORD Bridge] Line ${i} screen:`, {sx1, sy1, sx2, sy2});

      ctx.strokeStyle = '#00c853'; // Green for completed lines
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();

      console.log(`[ORD Bridge] ✅ Line ${i} drawn`);
    }

    // Draw current line being drawn (if any)
    if (drawingState.currentLine) {
      const [x1, y1, x2, y2] = drawingState.currentLine;
      const sx1 = this._indexToX(x1, chartState);
      const sy1 = this._priceToY(y1, chartState);
      const sx2 = this._indexToX(x2, chartState);
      const sy2 = this._priceToY(y2, chartState);

      console.log('[ORD Bridge] Current line:', {x1, y1, x2, y2}, '→', {sx1, sy1, sx2, sy2});

      ctx.strokeStyle = '#ffd600'; // Yellow for current line
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]); // Dashed
      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();
      ctx.setLineDash([]); // Reset dash
    }

    ctx.restore();
  }

  /**
   * Draw trendlines
   * @private
   */
  _drawTrendlines(ctx, chartState) {
    if (!this.currentAnalysis.trendlines) return;

    ctx.save();

    // Only log once when first activated
    const shouldLog = !this._hasLoggedDrawing;
    if (shouldLog) {
      console.log('[ORD Bridge] First draw - trendlines:', this.currentAnalysis.trendlines.length);
    }

    for (const line of this.currentAnalysis.trendlines) {
      // Convert indices to screen coordinates
      const x1 = this._indexToX(line.x1, chartState);
      const y1 = this._priceToY(line.y1, chartState);
      const x2 = this._indexToX(line.x2, chartState);
      const y2 = this._priceToY(line.y2, chartState);

      // Log first time only
      if (shouldLog) {
        console.log(`[ORD Bridge] Line: candle ${line.x1}-${line.x2} → screen ${x1.toFixed(0)},${y1.toFixed(0)} to ${x2.toFixed(0)},${y2.toFixed(0)} | canvas ${ctx.canvas.width}x${ctx.canvas.height}`);
      }

      // Skip if off-screen
      if (!this._isOnScreen(x1, y1, x2, y2, ctx.canvas)) {
        if (shouldLog) {
          console.log(`[ORD Bridge] ↑ Line OFF-SCREEN, skipped`);
        }
        continue;
      }

      // Draw line with specified color and style
      ctx.strokeStyle = line.color || '#00FF00';
      ctx.lineWidth = line.lineWidth || 4;

      // Set dash pattern if specified
      if (line.dash) {
        ctx.setLineDash(line.dash);
      }

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Reset dash
      if (line.dash) {
        ctx.setLineDash([]);
      }

      if (shouldLog) {
        console.log(`[ORD Bridge] ✅ Line drawn successfully`);
      }

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

    // Mark as logged after first draw
    if (shouldLog) {
      this._hasLoggedDrawing = true;
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
    // Active if there's an analysis OR if we're in draw mode
    if (this.isActive) {
      console.log('[ORD Bridge] isActiveAnalysis: true (has analysis)');
      return true;
    }

    // Also active if renderer is in draw mode
    if (this.ordVolumeRenderer) {
      const drawingState = this.ordVolumeRenderer.getDrawingState();
      if (drawingState.isDrawMode) {
        console.log('[ORD Bridge] isActiveAnalysis: true (draw mode active)');
        return true;
      }
    }

    return false;
  }
}

// Create global singleton instance
if (!window.ordVolumeBridge) {
  window.ordVolumeBridge = new ORDVolumeBridge();
}

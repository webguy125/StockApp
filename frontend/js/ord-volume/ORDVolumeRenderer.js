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

    // Line selection state (two-step: select then move)
    this.selectedLineIndex = null; // Which line is currently selected

    // Line endpoint dragging state
    this.isDraggingEndpoint = false;
    this.selectedEndpoint = null; // 'start' or 'end'

    // Whole line dragging state
    this.isDraggingLine = false;
    this.dragStartPoint = null; // Where the drag started

    // Right-click state tracking
    this.rightClickStartPos = null; // Track if right-click moved (pan vs menu)

    // Callback for line count updates
    this.onLineDrawn = null;

    // Bind methods
    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp = this._onMouseUp.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onContextMenu = this._onContextMenu.bind(this);
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
    this.canvas.addEventListener('contextmenu', this._onContextMenu, true);
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
    this.canvas.removeEventListener('contextmenu', this._onContextMenu, true);
    document.removeEventListener('keydown', this._onKeyDown);

    this.canvas.style.cursor = 'default';
  }

  /**
   * Handle mouse down - two-step system: select line, then move it
   * @private
   */
  _onMouseDown(e) {
    console.log('[ORD Renderer] Mouse down, isDrawMode:', this.isDrawMode, 'button:', e.button);

    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Track right-click position for pan vs menu detection
    if (e.button === 2) {
      this.rightClickStartPos = { x, y };
      // Don't preventDefault - let the chart handle right-click panning
      return;
    }

    if (this.isDrawMode) {
      // STEP 1: Check if clicking on endpoints of SELECTED line (only selected line can be moved)
      if (this.selectedLineIndex !== null) {
        const endpoint = this._findEndpointAtPoint(x, y);
        if (endpoint && endpoint.lineIndex === this.selectedLineIndex) {
          // CRITICAL: Stop the event from reaching the chart's pan/zoom handlers
          e.stopPropagation();
          e.preventDefault();

          // Start dragging endpoint of selected line
          this.isDraggingEndpoint = true;
          this.selectedEndpoint = endpoint.endpoint;
          console.log(`[ORD Renderer] Started dragging ${endpoint.endpoint} of selected line ${this.selectedLineIndex + 1}`);
          return;
        }

        // Check if clicking on middle of SELECTED line (to drag entire line)
        const clickedLineIndex = this._findLineAtPoint(x, y);
        if (clickedLineIndex === this.selectedLineIndex) {
          // CRITICAL: Stop the event from reaching the chart's pan/zoom handlers
          e.stopPropagation();
          e.preventDefault();

          // Start dragging entire selected line
          this.isDraggingLine = true;
          this.dragStartPoint = {
            x: this._screenToChartX(x),
            y: this._screenToChartY(y)
          };
          console.log(`[ORD Renderer] Started dragging entire selected line ${this.selectedLineIndex + 1}`);
          return;
        }
      }

      // STEP 2: Check if clicking on MIDDLE of ANY line (to SELECT it) - excludes endpoints
      const lineIndex = this._findLineMiddleAtPoint(x, y);
      if (lineIndex !== -1) {
        // CRITICAL: Stop the event from reaching the chart's pan/zoom handlers
        e.stopPropagation();
        e.preventDefault();

        // Select this line
        this.selectedLineIndex = lineIndex;
        console.log(`[ORD Renderer] Selected line ${lineIndex + 1} (clicked middle)`);
        this._redraw(); // Redraw to show selection highlight
        return;
      }

      // STEP 3: Clicking on empty space - deselect and start new line
      if (!this.currentLine) {
        // CRITICAL: Stop the event from reaching the chart's pan/zoom handlers
        e.stopPropagation();
        e.preventDefault();

        // Deselect any selected line
        this.selectedLineIndex = null;

        // Convert to chart coordinates
        const chartX = this._screenToChartX(x);
        const chartY = this._screenToChartY(y);

        // Start new line
        this.startPoint = { x: chartX, y: chartY };
        this.currentLine = [chartX, chartY, chartX, chartY];
        console.log('[ORD Renderer] Started new line at', chartX, chartY);
      }
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
   * Handle mouse move - update current line, drag endpoint, or drag label
   * @private
   */
  _onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (this.isDraggingLine && this.selectedLineIndex !== null && this.dragStartPoint) {
      // CRITICAL: Prevent chart panning while dragging line
      e.stopPropagation();
      e.preventDefault();

      // Change cursor to grabbing during drag
      this.canvas.style.cursor = 'grabbing';

      const chartX = this._screenToChartX(x);
      const chartY = this._screenToChartY(y);

      // Calculate how far we've moved from the start
      const dx = chartX - this.dragStartPoint.x;
      const dy = chartY - this.dragStartPoint.y;

      // Update both endpoints by the same delta (move entire line)
      const line = this.drawnLines[this.selectedLineIndex];
      const [origX1, origY1, origX2, origY2] = line;

      // Store original line on first move
      if (!this.originalLine) {
        this.originalLine = [...line];
      }

      // Move both endpoints
      line[0] = this.originalLine[0] + dx;
      line[1] = this.originalLine[1] + dy;
      line[2] = this.originalLine[2] + dx;
      line[3] = this.originalLine[3] + dy;

      // Clear analysis while dragging (will recalculate on mouse up)
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.clearAnalysis();
      }

      // Update preview with modified line
      this._updateDrawModePreview();
      this._redraw();

    } else if (this.isDraggingEndpoint && this.selectedLineIndex !== null) {
      // CRITICAL: Prevent chart panning while dragging endpoint
      e.stopPropagation();
      e.preventDefault();

      const chartX = this._screenToChartX(x);
      const chartY = this._screenToChartY(y);

      // Update the appropriate endpoint
      const line = this.drawnLines[this.selectedLineIndex];
      if (this.selectedEndpoint === 'start') {
        line[0] = chartX;
        line[1] = chartY;
      } else {
        line[2] = chartX;
        line[3] = chartY;
      }

      // Clear analysis while dragging (will recalculate on mouse up)
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.clearAnalysis();
      }

      // Update preview with modified line
      this._updateDrawModePreview();
      this._redraw();

    } else if (this.isDrawMode && this.currentLine && this.startPoint) {
      // CRITICAL: Prevent chart panning while drawing
      e.stopPropagation();
      e.preventDefault();

      const chartX = this._screenToChartX(x);
      const chartY = this._screenToChartY(y);

      // Update current line end point
      this.currentLine[2] = chartX;
      this.currentLine[3] = chartY;

      // Update preview with current line
      this._updateDrawModePreview(true);

      this._redraw();
    } else if (this.isDragging && this.selectedLabel) {
      // Drag label
      const newX = x - this.dragOffset.x;
      const newY = y - this.dragOffset.y;
      this.selectedLabel.x = this._screenToChartX(newX);
      this.selectedLabel.y = this._screenToChartY(newY);
      this._redraw();
    } else if (this.isDrawMode && !this.isDraggingEndpoint && !this.isDraggingLine && !this.currentLine) {
      // Change cursor when hovering (only when not actively drawing or dragging)
      // Only show move/grab cursors for SELECTED line
      if (this.selectedLineIndex !== null) {
        const endpoint = this._findEndpointAtPoint(x, y);
        if (endpoint && endpoint.lineIndex === this.selectedLineIndex) {
          this.canvas.style.cursor = 'grab'; // Endpoints of SELECTED line get grab cursor
          return;
        }

        const lineIndex = this._findLineAtPoint(x, y);
        if (lineIndex === this.selectedLineIndex) {
          this.canvas.style.cursor = 'move'; // Middle of SELECTED line gets move cursor
          return;
        }
      }

      // For all other lines or empty space, show pointer or crosshair
      // Use _findLineMiddleAtPoint to check if hovering over a selectable middle area
      const lineMiddleIndex = this._findLineMiddleAtPoint(x, y);
      if (lineMiddleIndex !== -1) {
        this.canvas.style.cursor = 'pointer'; // Line middle (selectable area) gets pointer
      } else {
        this.canvas.style.cursor = 'crosshair'; // Empty space or endpoints get crosshair
      }
    }
  }

  /**
   * Handle mouse up - finish line, endpoint drag, or label drag
   * @private
   */
  _onMouseUp(e) {
    if (this.isDraggingLine) {
      // Finish dragging entire line
      console.log(`[ORD Renderer] Finished dragging line ${this.selectedLineIndex + 1}`);

      this.isDraggingLine = false;
      this.dragStartPoint = null;
      this.originalLine = null; // Clear the stored original line

      // Sort lines after moving (in case it moved past other lines)
      // Note: _sortLinesByTime() updates selectedLineIndex automatically
      this._sortLinesByTime();

      // Re-run analysis with modified lines if we have 3+
      if (this.drawnLines.length >= 3 && this.onLineDrawn) {
        console.log('[ORD Renderer] Re-running analysis after line drag...');
        this.onLineDrawn(this.drawnLines.length);
      } else {
        // Just update preview if fewer than 3 lines
        this._updateDrawModePreview();
        this._redraw();
      }

      // Don't clear selectedLineIndex - keep the line selected after moving

    } else if (this.isDraggingEndpoint) {
      // Finish dragging endpoint
      console.log(`[ORD Renderer] Finished dragging endpoint of line ${this.selectedLineIndex + 1}`);

      this.isDraggingEndpoint = false;
      this.selectedEndpoint = null;

      // Sort lines after moving endpoint (in case it changed chronological order)
      // Note: _sortLinesByTime() updates selectedLineIndex automatically
      this._sortLinesByTime();

      // Re-run analysis with modified lines if we have 3+
      if (this.drawnLines.length >= 3 && this.onLineDrawn) {
        console.log('[ORD Renderer] Re-running analysis after endpoint drag...');
        this.onLineDrawn(this.drawnLines.length);
      } else {
        // Just update preview if fewer than 3 lines
        this._updateDrawModePreview();
        this._redraw();
      }

      // Don't clear selectedLineIndex - keep the line selected after moving

    } else if (this.isDrawMode && this.currentLine && this.startPoint) {
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

        // Sort lines chronologically
        this._sortLinesByTime();

        // Notify callback that a line was drawn (runs analysis if >= 3 lines)
        if (this.onLineDrawn) {
          this.onLineDrawn(this.drawnLines.length);
        } else {
          // If no callback, at least update preview
          this._updateDrawModePreview();
        }
      } else {
        console.log('[ORD Renderer] Line too short, discarded');
      }

      this.currentLine = null;
      this.startPoint = null;
      this._redraw();
    } else if (this.isDragging) {
      // Finish dragging label
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
    window.ordVolumeBridge.setAnalysis(previewAnalysis, null); // No signals in draw mode

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
   * Handle right-click context menu
   * Only show menu if right-click didn't move (wasn't a pan)
   * @private
   */
  _onContextMenu(e) {
    e.preventDefault();
    e.stopPropagation();

    // If right-click moved (panning), don't show menu
    if (this.rightClickStartPos) {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const dx = Math.abs(x - this.rightClickStartPos.x);
      const dy = Math.abs(y - this.rightClickStartPos.y);

      // If moved more than 5 pixels, it was a pan, not a menu request
      if (dx > 5 || dy > 5) {
        this.rightClickStartPos = null;
        return;
      }
    }

    this.rightClickStartPos = null;

    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (this.isDrawMode) {
      // Draw mode: Find clicked line and show delete option
      const clickedLineIndex = this._findLineAtPoint(x, y);

      if (clickedLineIndex !== -1) {
        // Show context menu for deleting this specific line
        this._showContextMenu(e.clientX, e.clientY, [
          {
            label: `Delete Line ${clickedLineIndex + 1}`,
            action: () => this._deleteLineAtIndex(clickedLineIndex)
          },
          {
            label: 'Delete All Lines',
            action: () => this._deleteAllLines()
          }
        ]);
      } else {
        // Clicked on empty space - show delete all option
        this._showContextMenu(e.clientX, e.clientY, [
          {
            label: 'Delete All Lines',
            action: () => this._deleteAllLines()
          }
        ]);
      }
    } else {
      // Auto mode: Show delete all option
      if (window.ordVolumeBridge && window.ordVolumeBridge.currentAnalysis) {
        this._showContextMenu(e.clientX, e.clientY, [
          {
            label: 'Delete All ORD Lines',
            action: () => this._deleteAllORDLines()
          }
        ]);
      }
    }
  }

  /**
   * Sort drawn lines by chronological order (left to right on chart)
   * This ensures line numbers are sequential from left to right
   * @private
   */
  _sortLinesByTime() {
    // Store selected line before sorting
    let selectedLine = null;
    if (this.selectedLineIndex !== null) {
      selectedLine = this.drawnLines[this.selectedLineIndex];
    }

    // Sort lines by leftmost X position
    this.drawnLines.sort((a, b) => {
      const startA = Math.min(a[0], a[2]); // Leftmost X of line A
      const startB = Math.min(b[0], b[2]); // Leftmost X of line B
      return startA - startB;
    });

    // Update selected index to point to the same line after sorting
    if (selectedLine) {
      this.selectedLineIndex = this.drawnLines.findIndex(line =>
        line[0] === selectedLine[0] &&
        line[1] === selectedLine[1] &&
        line[2] === selectedLine[2] &&
        line[3] === selectedLine[3]
      );
    }

    console.log('[ORD Renderer] Lines sorted chronologically (left to right)');
  }

  /**
   * Get drawn lines (for analysis)
   * @returns {Array} Array of [x1, y1, x2, y2] lines
   */
  getDrawnLines() {
    return this.drawnLines;
  }

  /**
   * Get current drawing state (includes in-progress line and selected line)
   * @returns {Object} Drawing state for rendering
   */
  getDrawingState() {
    return {
      drawnLines: this.drawnLines,
      currentLine: this.currentLine,
      isDrawMode: this.isDrawMode,
      selectedLineIndex: this.selectedLineIndex // Which line is selected
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
      const rawIndex = this.chartState.xToIndex(screenX);
      // Round to nearest candle index for snapping
      const snappedIndex = Math.round(rawIndex);
      // Reduced logging - only log when significantly different
      if (Math.abs(rawIndex - snappedIndex) > 0.1) {
        console.log(`[ORD Renderer] Snapped: ${rawIndex.toFixed(2)} → ${snappedIndex}`);
      }
      return snappedIndex;
    }
    // Fallback: assume direct mapping
    console.warn('[ORD Renderer] No xToIndex function, using fallback');
    return Math.round(screenX);
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
      // Price doesn't need logging every time
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

  /**
   * Find which line endpoint is at the clicked point (ONLY for selected line)
   * Returns {lineIndex, endpoint} or null
   * @private
   */
  _findEndpointAtPoint(screenX, screenY) {
    const threshold = 15; // pixels - larger for easier endpoint selection

    for (let i = 0; i < this.drawnLines.length; i++) {
      const [x1, y1, x2, y2] = this.drawnLines[i];

      // Convert chart coords to screen coords
      const sx1 = this._chartToScreenX(x1);
      const sy1 = this._chartToScreenY(y1);
      const sx2 = this._chartToScreenX(x2);
      const sy2 = this._chartToScreenY(y2);

      // Check distance to start point
      const distToStart = Math.sqrt((screenX - sx1) ** 2 + (screenY - sy1) ** 2);
      if (distToStart < threshold) {
        return { lineIndex: i, endpoint: 'start' };
      }

      // Check distance to end point
      const distToEnd = Math.sqrt((screenX - sx2) ** 2 + (screenY - sy2) ** 2);
      if (distToEnd < threshold) {
        return { lineIndex: i, endpoint: 'end' };
      }
    }

    return null;
  }

  /**
   * Find which line's MIDDLE is at the clicked point (excludes endpoints)
   * This is used for SELECTION only - returns line index or -1
   * @private
   */
  _findLineMiddleAtPoint(screenX, screenY) {
    const lineThreshold = 10; // pixels - distance to line for selection
    const endpointExclusionRadius = 20; // pixels - don't select if near endpoints

    for (let i = 0; i < this.drawnLines.length; i++) {
      const [x1, y1, x2, y2] = this.drawnLines[i];

      // Convert chart coords to screen coords
      const sx1 = this._chartToScreenX(x1);
      const sy1 = this._chartToScreenY(y1);
      const sx2 = this._chartToScreenX(x2);
      const sy2 = this._chartToScreenY(y2);

      // Check if near endpoints - if so, DON'T select
      const distToStart = Math.sqrt((screenX - sx1) ** 2 + (screenY - sy1) ** 2);
      const distToEnd = Math.sqrt((screenX - sx2) ** 2 + (screenY - sy2) ** 2);

      if (distToStart < endpointExclusionRadius || distToEnd < endpointExclusionRadius) {
        continue; // Skip this line - too close to endpoints
      }

      // Calculate distance from point to line segment
      const dist = this._distanceToLineSegment(screenX, screenY, sx1, sy1, sx2, sy2);

      if (dist < lineThreshold) {
        return i; // Found a line middle click
      }
    }

    return -1; // No line middle found
  }

  /**
   * Find which line is at the clicked point
   * @private
   */
  _findLineAtPoint(screenX, screenY) {
    const threshold = 10; // pixels

    for (let i = 0; i < this.drawnLines.length; i++) {
      const [x1, y1, x2, y2] = this.drawnLines[i];

      // Convert chart coords to screen coords
      const sx1 = this._chartToScreenX(x1);
      const sy1 = this._chartToScreenY(y1);
      const sx2 = this._chartToScreenX(x2);
      const sy2 = this._chartToScreenY(y2);

      // Calculate distance from point to line segment
      const dist = this._distanceToLineSegment(screenX, screenY, sx1, sy1, sx2, sy2);

      if (dist < threshold) {
        return i;
      }
    }

    return -1;
  }

  /**
   * Calculate distance from point to line segment
   * @private
   */
  _distanceToLineSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lengthSquared = dx * dx + dy * dy;

    if (lengthSquared === 0) {
      // Line is a point
      return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
    }

    // Project point onto line, clamped to segment
    let t = ((px - x1) * dx + (py - y1) * dy) / lengthSquared;
    t = Math.max(0, Math.min(1, t));

    const projX = x1 + t * dx;
    const projY = y1 + t * dy;

    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
  }

  /**
   * Show context menu at position
   * @private
   */
  _showContextMenu(x, y, menuItems) {
    // Remove any existing context menu
    const existingMenu = document.getElementById('ord-context-menu');
    if (existingMenu) {
      existingMenu.remove();
    }

    // Create context menu
    const menu = document.createElement('div');
    menu.id = 'ord-context-menu';
    menu.style.cssText = `
      position: fixed;
      left: ${x}px;
      top: ${y}px;
      background: #2a2a2a;
      border: 1px solid #444;
      border-radius: 4px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      z-index: 100000;
      min-width: 180px;
      padding: 4px 0;
    `;

    menuItems.forEach(item => {
      const menuItem = document.createElement('div');
      menuItem.textContent = item.label;
      menuItem.style.cssText = `
        padding: 8px 16px;
        color: #fff;
        cursor: pointer;
        font-size: 13px;
        transition: background 0.2s;
      `;
      menuItem.onmouseover = () => menuItem.style.background = '#444';
      menuItem.onmouseout = () => menuItem.style.background = 'transparent';
      menuItem.onclick = () => {
        item.action();
        menu.remove();
      };
      menu.appendChild(menuItem);
    });

    document.body.appendChild(menu);

    // Close menu when clicking anywhere else
    const closeMenu = (e) => {
      if (!menu.contains(e.target)) {
        menu.remove();
        document.removeEventListener('click', closeMenu);
      }
    };
    setTimeout(() => document.addEventListener('click', closeMenu), 100);
  }

  /**
   * Delete line at specific index
   * @private
   */
  _deleteLineAtIndex(index) {
    console.log(`[ORD Renderer] Deleting line ${index + 1}`);

    // CRITICAL: Remove from context menu first to prevent interaction
    const existingMenu = document.getElementById('ord-context-menu');
    if (existingMenu) {
      existingMenu.remove();
    }

    this.drawnLines.splice(index, 1);

    // Deselect if we deleted the selected line
    if (this.selectedLineIndex === index) {
      this.selectedLineIndex = null;
    } else if (this.selectedLineIndex > index) {
      // Adjust selected index if a line before it was deleted
      this.selectedLineIndex--;
    }

    // Clear analysis overlays
    if (window.ordVolumeBridge) {
      window.ordVolumeBridge.clearAnalysis();
      console.log('[ORD Renderer] Analysis cleared after deletion');
    }

    // Update preview with remaining lines (lightweight - just converts to display format)
    if (this.drawnLines.length > 0) {
      this._updateDrawModePreview();
    }

    // Redraw chart
    this._redraw();

    console.log(`[ORD Renderer] Line deleted, ${this.drawnLines.length} lines remaining`);

    // Re-run analysis if we still have 3+ lines
    if (this.drawnLines.length >= 3 && this.onLineDrawn) {
      console.log('[ORD Renderer] Re-running analysis with remaining lines...');
      this.onLineDrawn(this.drawnLines.length);
    }
  }

  /**
   * Delete all drawn lines
   * @private
   */
  _deleteAllLines() {
    console.log('[ORD Renderer] Deleting all drawn lines');

    // CRITICAL: Remove context menu first
    const existingMenu = document.getElementById('ord-context-menu');
    if (existingMenu) {
      existingMenu.remove();
    }

    this.drawnLines = [];
    this.selectedLineIndex = null; // Deselect

    // Clear analysis when all lines deleted
    if (window.ordVolumeBridge) {
      window.ordVolumeBridge.clearAnalysis();
      console.log('[ORD Renderer] Analysis cleared (all lines deleted)');
    }

    // Redraw chart (no lines to show)
    this._redraw();

    console.log('[ORD Renderer] All lines deleted');
  }

  /**
   * Delete all ORD lines (auto mode)
   * @private
   */
  _deleteAllORDLines() {
    console.log('[ORD Renderer] Deleting all ORD lines (auto mode)');
    if (window.ordVolumeBridge) {
      window.ordVolumeBridge.clearAnalysis();

      // Trigger redraw
      if (this.chartState && this.chartState.draw) {
        this.chartState.draw();
      }
    }
  }
}

/**
 * ORD Volume Bridge
 * Bridges the segregated ORD Volume renderer with the main chart renderer
 * Allows ORD Volume overlays to persist on the chart
 *
 * SEGREGATED IMPLEMENTATION - No shared code with existing features
 */

export class ORDVolumeBridge {
  constructor() {
    this.currentAnalysis = null; // DEPRECATED: Use analysisStore instead
    this.analysisStore = new Map(); // Store analysis per chart: 'timeframe:1d' or 'tick:50'
    this.isActive = false;
    this.chartRenderer = null;
    this._hasLoggedDrawing = false;
    this.ordVolumeRenderer = null; // Reference to ORD Volume renderer for draw mode
    this.selectedWaves = []; // Track selected waves for custom comparison
    this.customComparison = null; // Store custom comparison data
    this.showTradeSignals = true; // Toggle for trade signal visibility
    this.candles = null; // Reference to candle data for signal positioning
    this.infoPanelElement = null; // HTML overlay for info panel

    // Create info panel HTML element
    this._createInfoPanelElement();
  }

  /**
   * Get unique chart key for current context
   * @private
   */
  _getChartKey() {
    if (!window.tosApp) return 'default';

    if (window.tosApp.activeChartType === 'timeframe') {
      const timeframeId = window.tosApp.currentTimeframeId || 'unknown';
      return `timeframe:${timeframeId}`;
    } else if (window.tosApp.activeChartType === 'tick') {
      const tickId = window.tosApp.currentTickChartId || 'unknown';
      return `tick:${tickId}`;
    }

    return 'default';
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
   * Store analysis result for persistent rendering (per chart)
   * @param {Object} analysisResult - Analysis result
   * @param {Array} candles - Candle data
   * @param {String} symbol - Optional symbol (defaults to current)
   * @param {String} timeframeId - Optional timeframe ID (defaults to current)
   */
  setAnalysis(analysisResult, candles, symbol = null, timeframeId = null) {
    const chartKey = this._getChartKey();

    // Get current symbol and timeframe if not provided
    if (!symbol && window.tosApp) {
      symbol = window.tosApp.currentSymbol || null;
    }
    if (!timeframeId && window.tosApp) {
      timeframeId = window.tosApp.currentTimeframeId || window.tosApp.currentTickChartId || null;
    }

    // Store in per-chart map with metadata
    this.analysisStore.set(chartKey, {
      analysis: analysisResult,
      candles: candles,
      timestamp: Date.now(),
      candleCount: candles ? candles.length : null,
      symbol: symbol,
      timeframeId: timeframeId
    });

    // Also update legacy properties for current chart
    this.currentAnalysis = analysisResult;
    this.candles = candles;
    this.isActive = true;
    this._hasLoggedDrawing = false;

    console.log(`[ORD Bridge] Analysis stored for ${chartKey} (${symbol} @ ${timeframeId}, ${candles ? candles.length : 0} candles)`);
  }

  /**
   * Get analysis for current chart
   * @returns {Object|null} Analysis result or null
   */
  getAnalysis() {
    const chartKey = this._getChartKey();
    console.log(`[ORD Bridge] 🔑 Getting analysis for key: "${chartKey}"`);
    console.log(`[ORD Bridge] 📦 Store contents:`, Array.from(this.analysisStore.keys()));

    const stored = this.analysisStore.get(chartKey);

    if (stored) {
      // Update current properties
      this.currentAnalysis = stored.analysis;
      this.candles = stored.candles;
      this.isActive = true;
      console.log(`[ORD Bridge] ✅ Found and loaded analysis for "${chartKey}", isActive=${this.isActive}`);
      return stored.analysis;
    }

    console.log(`[ORD Bridge] ❌ No analysis found for "${chartKey}"`);
    return null;
  }

  /**
   * Get analysis with full metadata (timestamp, candle count)
   * @returns {Object|null} {analysis, timestamp, candleCount, symbol, timeframeId} or null
   */
  getAnalysisWithMetadata() {
    const chartKey = this._getChartKey();
    const stored = this.analysisStore.get(chartKey);

    if (stored) {
      return {
        analysis: stored.analysis,
        timestamp: stored.timestamp,
        candleCount: stored.candleCount || null,
        symbol: stored.symbol || null,
        timeframeId: stored.timeframeId || null
      };
    }

    return null;
  }

  /**
   * Check if saved analysis is stale (more than 3 candles old)
   * @param {Number} savedCandleCount - Candle count when analysis was saved
   * @param {Number} currentCandleCount - Current candle count
   * @returns {Boolean} True if stale (difference > 3)
   */
  isAnalysisStale(savedCandleCount, currentCandleCount) {
    if (!savedCandleCount || !currentCandleCount) {
      return false; // Can't determine staleness
    }

    const difference = Math.abs(currentCandleCount - savedCandleCount);
    return difference > 3;
  }

  /**
   * Clear current chart display without deleting from storage
   * (Useful when switching charts)
   */
  clearCurrentChartDisplay() {
    this.currentAnalysis = null;
    this.candles = null;
    this.isActive = false;

    // Hide info panel
    if (this.infoPanelElement) {
      this.infoPanelElement.style.display = 'none';
    }

    console.log('[ORD Bridge] Cleared current chart display (storage preserved)');
  }

  /**
   * Clear the analysis for current chart
   */
  clearAnalysis() {
    const chartKey = this._getChartKey();
    this.analysisStore.delete(chartKey);
    this.currentAnalysis = null;
    this.isActive = false;

    // Hide info panel
    if (this.infoPanelElement) {
      this.infoPanelElement.style.display = 'none';
    }

    console.log(`[ORD Bridge] Analysis cleared for ${chartKey}`);
  }

  /**
   * Clear ALL analyses (when switching symbols)
   */
  clearAllAnalyses() {
    this.analysisStore.clear();
    this.currentAnalysis = null;
    this.isActive = false;

    // Hide info panel
    if (this.infoPanelElement) {
      this.infoPanelElement.style.display = 'none';
    }

    console.log('[ORD Bridge] All analyses cleared (symbol change)');
  }

  /**
   * Draw ORD Volume overlays on the chart
   * Called after main chart renders
   */
  drawOverlays(ctx, chartState) {
    // Draw manual drawing lines (if in draw mode) - PRIORITY CHECK
    if (this.ordVolumeRenderer) {
      const drawingState = this.ordVolumeRenderer.getDrawingState();

      // Debug logging disabled - was causing excessive console spam
      // console.log('[ORD Bridge] drawOverlays called, drawingState:', {
      //   hasRenderer: !!this.ordVolumeRenderer,
      //   isDrawMode: drawingState.isDrawMode,
      //   drawnLines: drawingState.drawnLines?.length || 0,
      //   hasCurrent: !!drawingState.currentLine
      // });

      if (drawingState.isDrawMode) {
        this._drawManualLines(ctx, chartState, drawingState);
        // CHANGED: Don't return early - draw analysis overlays too if available
        // This allows live calculations to show while drawing
      }
    }

    // Auto-load analysis for current chart if not already loaded
    if (!this.currentAnalysis) {
      this.getAnalysis(); // This will load from store if exists
    }

    // Draw analysis overlays
    if (!this.isActive || !this.currentAnalysis) {
      return;
    }

    // Draw trendlines
    this._drawTrendlines(ctx, chartState);

    // Draw labels
    this._drawLabels(ctx, chartState);

    // Draw trade signals (if enabled)
    if (this.showTradeSignals) {
      this._drawTradeSignals(ctx, chartState);
    }

    // Draw ORD Volume info panel (always show to help users interpret results)
    this._drawInfoPanel(ctx, chartState);
  }

  /**
   * Draw manual drawing lines
   * @private
   */
  _drawManualLines(ctx, chartState, drawingState) {
    console.log('[ORD Bridge] Drawing manual lines:', {
      drawnCount: drawingState.drawnLines.length,
      hasCurrent: !!drawingState.currentLine,
      selectedLineIndex: drawingState.selectedLineIndex,
      canvasSize: `${ctx.canvas.width}x${ctx.canvas.height}`
    });

    ctx.save();

    // Draw completed lines with professional styling
    for (let i = 0; i < drawingState.drawnLines.length; i++) {
      const line = drawingState.drawnLines[i];
      const [x1, y1, x2, y2] = line;
      const isSelected = (i === drawingState.selectedLineIndex);

      console.log(`[ORD Bridge] Line ${i} raw:`, {x1, y1, x2, y2, isSelected});

      const sx1 = this._indexToX(x1, chartState);
      const sy1 = this._priceToY(y1, chartState);
      const sx2 = this._indexToX(x2, chartState);
      const sy2 = this._priceToY(y2, chartState);

      console.log(`[ORD Bridge] Line ${i} screen:`, {sx1, sy1, sx2, sy2});

      // Highlight selected line with different color and thicker line
      if (isSelected) {
        ctx.shadowColor = 'rgba(255, 200, 0, 0.5)';
        ctx.shadowBlur = 8;
        ctx.shadowOffsetY = 0;
        ctx.strokeStyle = '#FCD34D'; // Yellow for selected line
        ctx.lineWidth = 4;
      } else {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
        ctx.shadowBlur = 2;
        ctx.shadowOffsetY = 1;
        ctx.strokeStyle = '#10B981'; // Emerald green for normal lines
        ctx.lineWidth = 3;
      }

      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();

      // Draw endpoint circles for selected line
      if (isSelected) {
        ctx.fillStyle = '#FCD34D';
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.arc(sx1, sy1, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(sx2, sy2, 6, 0, Math.PI * 2);
        ctx.fill();
      }

      // Reset shadow
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

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

      // Professional amber for current drawing
      ctx.strokeStyle = '#F59E0B'; // Amber for current line
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.setLineDash([8, 4]); // Dashed with longer segments
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

    // Debug logging disabled - no longer needed
    // const shouldLog = !this._hasLoggedDrawing;
    // if (shouldLog) {
    //   console.log('[ORD Bridge] First draw - trendlines:', this.currentAnalysis.trendlines.length);
    // }

    const trendlines = this.currentAnalysis.trendlines;
    const lastTrendlineIndex = trendlines.length - 1;

    for (let i = 0; i < trendlines.length; i++) {
      const line = trendlines[i];
      const isLastLine = (i === lastTrendlineIndex);

      // For the LAST trendline, make it follow the current candle's high/low
      let adjustedY2 = line.y2;

      if (isLastLine && this.candles && this.candles.length > 0) {
        const currentCandle = this.candles[this.candles.length - 1];
        const isUptrend = line.y2 > line.y1;

        // Uptrend → follow HIGH, Downtrend → follow LOW
        if (isUptrend) {
          adjustedY2 = currentCandle.High || currentCandle.high;
        } else {
          adjustedY2 = currentCandle.Low || currentCandle.low;
        }
      }

      // Convert indices to screen coordinates
      const x1 = this._indexToX(line.x1, chartState);
      const y1 = this._priceToY(line.y1, chartState);
      const x2 = this._indexToX(line.x2, chartState);
      const y2 = this._priceToY(adjustedY2, chartState);

      // Debug logging disabled
      // if (shouldLog) {
      //   console.log(`[ORD Bridge] Line: candle ${line.x1}-${line.x2} → screen ${x1.toFixed(0)},${y1.toFixed(0)} to ${x2.toFixed(0)},${y2.toFixed(0)} | canvas ${ctx.canvas.width}x${ctx.canvas.height}`);
      // }

      // Skip if off-screen
      if (!this._isOnScreen(x1, y1, x2, y2, ctx.canvas)) {
        // if (shouldLog) {
        //   console.log(`[ORD Bridge] ↑ Line OFF-SCREEN, skipped`);
        // }
        continue;
      }

      // Add shadow for depth
      if (line.shadowColor && line.shadowBlur) {
        ctx.shadowColor = line.shadowColor;
        ctx.shadowBlur = line.shadowBlur;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 1;
      }

      // Draw line with specified color and style
      ctx.strokeStyle = line.color || '#3B82F6';
      ctx.lineWidth = line.lineWidth || 3;
      ctx.lineCap = 'round'; // Smooth line ends
      ctx.lineJoin = 'round'; // Smooth line joins

      // Set dash pattern if specified
      if (line.dash) {
        ctx.setLineDash(line.dash);
      }

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Reset dash and shadow
      if (line.dash) {
        ctx.setLineDash([]);
      }
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;

      // Debug logging disabled
      // if (shouldLog) {
      //   console.log(`[ORD Bridge] ✅ Line drawn successfully`);
      // }
    }

    // Debug logging disabled
    // Mark as logged after first draw
    // if (shouldLog) {
    //   this._hasLoggedDrawing = true;
    // }

    ctx.restore();
  }

  /**
   * Draw labels (volume metrics)
   * @private
   */
  _drawLabels(ctx, chartState) {
    if (!this.currentAnalysis.labels) return;

    ctx.save();

    // Get last trendline info for live tracking
    const trendlines = this.currentAnalysis.trendlines || [];
    const lastTrendline = trendlines[trendlines.length - 1];
    let lastTrendlineAdjustedY2 = null;

    if (lastTrendline && this.candles && this.candles.length > 0) {
      const currentCandle = this.candles[this.candles.length - 1];
      const isUptrend = lastTrendline.y2 > lastTrendline.y1;

      // Calculate adjusted endpoint (same logic as trendline drawing)
      if (isUptrend) {
        lastTrendlineAdjustedY2 = currentCandle.High || currentCandle.high;
      } else {
        lastTrendlineAdjustedY2 = currentCandle.Low || currentCandle.low;
      }
    }

    for (let i = 0; i < this.currentAnalysis.labels.length; i++) {
      const label = this.currentAnalysis.labels[i];

      // Adjust Y coordinate for the last wave's percentage label to follow the trendline
      let adjustedLabelY = label.y;

      if (label.isPercentageLabel && lastTrendlineAdjustedY2 !== null) {
        // Check if this label is for the last trendline (wave)
        const isLastWaveLabel = (i === this.currentAnalysis.labels.length - 1 ||
                                  label.waveIndex === (trendlines.length - 1));

        if (isLastWaveLabel) {
          // Adjust label Y to match the adjusted trendline endpoint
          adjustedLabelY = lastTrendlineAdjustedY2;
        }
      }

      // Convert to screen coordinates
      let x = this._indexToX(label.x, chartState);
      let y = this._priceToY(adjustedLabelY, chartState);

      // Apply fixed pixel offset for percentage and wave labels
      if (label.pixelOffset !== undefined && label.isUpward !== undefined) {
        // Apply pixel offset in screen space (up = negative Y, down = positive Y)
        if (label.isUpward) {
          y -= label.pixelOffset; // Move up (above the connection point)
        } else {
          y += label.pixelOffset; // Move down (below the connection point)
        }
      }

      // Skip if off-screen
      if (x < 0 || x > ctx.canvas.width || y < 0 || y > ctx.canvas.height) {
        continue;
      }

      // Check if this is a percentage label and we should override with custom comparison
      let labelText = label.text;
      let labelColor = label.color || '#FFFFFF';

      if (label.isPercentageLabel && this.customComparison) {
        // Check if this label should show the custom comparison
        if (label.waveIndex === this.customComparison.wave2Index) {
          labelText = `${this.customComparison.percentage}%`;
          labelColor = this.customComparison.color;
        }
      }

      // Set font
      const fontWeight = label.fontWeight || 'normal';
      const fontSize = label.fontSize || 12;
      ctx.font = `${fontWeight} ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif`;

      // Calculate angle from line endpoints in screen coordinates
      let angle = 0;
      if (label.lineX1 !== undefined && label.lineY1 !== undefined &&
          label.lineX2 !== undefined && label.lineY2 !== undefined) {
        // Convert line endpoints to screen coordinates
        const sx1 = this._indexToX(label.lineX1, chartState);
        const sy1 = this._priceToY(label.lineY1, chartState);
        const sx2 = this._indexToX(label.lineX2, chartState);
        const sy2 = this._priceToY(label.lineY2, chartState);

        // Calculate angle in screen space
        angle = Math.atan2(sy2 - sy1, sx2 - sx1);

        // Keep text readable (left-to-right)
        if (angle > Math.PI / 2) {
          angle -= Math.PI;
        } else if (angle < -Math.PI / 2) {
          angle += Math.PI;
        }
      }

      // Save context for rotation
      ctx.save();

      // Translate to label position and rotate
      ctx.translate(x, y);
      ctx.rotate(angle);

      // Draw text with shadow for readability (rotated to match line)
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
      ctx.shadowBlur = 3;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;
      ctx.fillStyle = labelColor;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, 0, 0);

      // Restore rotation context
      ctx.restore();
    }

    ctx.restore();
  }

  /**
   * Draw trade signals (Professional ORD Volume signals with full details)
   * @private
   */
  _drawTradeSignals(ctx, chartState) {
    if (!this.currentAnalysis.tradeSignals || this.currentAnalysis.tradeSignals.length === 0) return;

    // Need candles to position signals
    if (!this.candles || this.candles.length === 0) {
      console.log('[ORD Bridge] Cannot draw signals: candles not available');
      return;
    }

    ctx.save();

    console.log('[ORD Bridge] Drawing', this.currentAnalysis.tradeSignals.length, 'professional trade signals');

    for (const signal of this.currentAnalysis.tradeSignals) {
      // Get entry price position (signals now contain full signal objects from ORDVolumeSignals)
      const entryX = this.candles.length - 1; // Current bar
      const entryY = signal.entry;

      // Convert to screen coordinates
      const entryScreenX = this._indexToX(entryX, chartState);
      const entryScreenY = this._priceToY(entryY, chartState);

      // Skip if off-screen
      if (entryScreenX < 0 || entryScreenX > ctx.canvas.width) {
        continue;
      }

      console.log(`[ORD Bridge] Signal ${signal.signal_id}:`, {
        direction: signal.direction,
        entry: signal.entry,
        stop: signal.stop,
        targets: [signal.target_1, signal.target_2, signal.target_3],
        confluence: signal.confluence_score,
        probability: signal.probability
      });

      // Draw signal based on direction
      if (signal.direction === 'LONG') {
        this._drawProfessionalLongSignal(ctx, chartState, signal, entryScreenX, entryScreenY);
      } else if (signal.direction === 'SHORT') {
        this._drawProfessionalShortSignal(ctx, chartState, signal, entryScreenX, entryScreenY);
      }
    }

    ctx.restore();
  }

  /**
   * Draw professional LONG signal with entry, stop, targets, and details
   * @private
   */
  _drawProfessionalLongSignal(ctx, chartState, signal, entryX, entryY) {
    ctx.save();

    // Draw horizontal price levels
    this._drawPriceLevel(ctx, chartState, signal.stop, '#EF4444', 'STOP', true); // Red for stop
    this._drawPriceLevel(ctx, chartState, signal.entry, '#10B981', 'ENTRY', false); // Green for entry
    this._drawPriceLevel(ctx, chartState, signal.target_1, '#3B82F6', 'T1', false); // Blue for targets
    this._drawPriceLevel(ctx, chartState, signal.target_2, '#3B82F6', 'T2', false);
    this._drawPriceLevel(ctx, chartState, signal.target_3, '#3B82F6', 'T3', false);

    // Draw upward arrow at entry point
    const arrowSize = 24;
    const yOffset = 50; // Position below entry

    ctx.beginPath();
    ctx.moveTo(entryX, entryY + yOffset); // Arrow tip
    ctx.lineTo(entryX - arrowSize / 2, entryY + yOffset + arrowSize); // Left point
    ctx.lineTo(entryX - arrowSize / 4, entryY + yOffset + arrowSize); // Left of shaft
    ctx.lineTo(entryX - arrowSize / 4, entryY + yOffset + arrowSize * 1.5); // Shaft left
    ctx.lineTo(entryX + arrowSize / 4, entryY + yOffset + arrowSize * 1.5); // Shaft right
    ctx.lineTo(entryX + arrowSize / 4, entryY + yOffset + arrowSize); // Right of shaft
    ctx.lineTo(entryX + arrowSize / 2, entryY + yOffset + arrowSize); // Right point
    ctx.closePath();

    // Fill with green
    ctx.fillStyle = '#10B981';
    ctx.fill();

    // Outline
    ctx.strokeStyle = '#059669';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw info panel
    this._drawSignalInfoPanel(ctx, signal, entryX + 60, entryY);

    ctx.restore();
  }

  /**
   * Draw professional SHORT signal with entry, stop, targets, and details
   * @private
   */
  _drawProfessionalShortSignal(ctx, chartState, signal, entryX, entryY) {
    ctx.save();

    // Draw horizontal price levels
    this._drawPriceLevel(ctx, chartState, signal.stop, '#EF4444', 'STOP', true); // Red for stop
    this._drawPriceLevel(ctx, chartState, signal.entry, '#EF4444', 'ENTRY', false); // Red for entry (short)
    this._drawPriceLevel(ctx, chartState, signal.target_1, '#3B82F6', 'T1', false); // Blue for targets
    this._drawPriceLevel(ctx, chartState, signal.target_2, '#3B82F6', 'T2', false);
    this._drawPriceLevel(ctx, chartState, signal.target_3, '#3B82F6', 'T3', false);

    // Draw downward arrow at entry point
    const arrowSize = 24;
    const yOffset = -50; // Position above entry

    ctx.beginPath();
    ctx.moveTo(entryX, entryY + yOffset); // Arrow tip
    ctx.lineTo(entryX - arrowSize / 2, entryY + yOffset - arrowSize); // Left point
    ctx.lineTo(entryX - arrowSize / 4, entryY + yOffset - arrowSize); // Left of shaft
    ctx.lineTo(entryX - arrowSize / 4, entryY + yOffset - arrowSize * 1.5); // Shaft left
    ctx.lineTo(entryX + arrowSize / 4, entryY + yOffset - arrowSize * 1.5); // Shaft right
    ctx.lineTo(entryX + arrowSize / 4, entryY + yOffset - arrowSize); // Right of shaft
    ctx.lineTo(entryX + arrowSize / 2, entryY + yOffset - arrowSize); // Right point
    ctx.closePath();

    // Fill with red
    ctx.fillStyle = '#EF4444';
    ctx.fill();

    // Outline
    ctx.strokeStyle = '#DC2626';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw info panel
    this._drawSignalInfoPanel(ctx, signal, entryX + 60, entryY);

    ctx.restore();
  }

  /**
   * Draw horizontal price level line
   * @private
   */
  _drawPriceLevel(ctx, chartState, price, color, label, isDashed) {
    const y = this._priceToY(price, chartState);
    const canvasWidth = ctx.canvas.width;

    ctx.save();

    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    if (isDashed) {
      ctx.setLineDash([8, 4]);
    }

    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasWidth, y);
    ctx.stroke();

    if (isDashed) {
      ctx.setLineDash([]);
    }

    // Draw label on right side
    ctx.font = 'bold 11px Arial';
    ctx.fillStyle = color;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
    ctx.shadowBlur = 3;
    ctx.fillText(`${label} $${price.toFixed(2)}`, canvasWidth - 10, y);

    ctx.restore();
  }

  /**
   * Draw signal information panel
   * @private
   */
  _drawSignalInfoPanel(ctx, signal, x, y) {
    ctx.save();

    // Panel dimensions
    const panelWidth = 280;
    const panelHeight = 180;
    const padding = 12;
    const lineHeight = 16;

    // Draw panel background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    this._roundRect(ctx, x, y - panelHeight / 2, panelWidth, panelHeight, 8);
    ctx.fill();

    // Draw panel border
    ctx.strokeStyle = signal.direction === 'LONG' ? '#10B981' : '#EF4444';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw content
    let currentY = y - panelHeight / 2 + padding + 12;

    // Title
    ctx.font = 'bold 14px Arial';
    ctx.fillStyle = signal.direction === 'LONG' ? '#10B981' : '#EF4444';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`${signal.direction} SIGNAL`, x + padding, currentY);
    currentY += lineHeight + 4;

    // Confluence and probability
    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(`Confluence: ${signal.confluence_score}/5`, x + padding, currentY);
    ctx.fillText(`Probability: ${signal.probability}%`, x + padding + 140, currentY);
    currentY += lineHeight + 4;

    // Risk/Reward
    ctx.font = '11px Arial';
    ctx.fillStyle = '#A0A0A0';
    ctx.fillText(`Risk/Reward: ${signal.rr}`, x + padding, currentY);
    currentY += lineHeight + 6;

    // ORD triggers
    ctx.font = 'bold 11px Arial';
    ctx.fillStyle = '#FFD700';
    ctx.fillText('ORD Triggers:', x + padding, currentY);
    currentY += lineHeight;

    ctx.font = '10px Arial';
    ctx.fillStyle = '#FFFFFF';
    for (let i = 0; i < Math.min(2, signal.ord_triggers.length); i++) {
      const trigger = signal.ord_triggers[i];
      const truncated = trigger.length > 35 ? trigger.substring(0, 32) + '...' : trigger;
      ctx.fillText(`• ${truncated}`, x + padding + 8, currentY);
      currentY += lineHeight - 2;
    }

    if (signal.ord_triggers.length > 2) {
      ctx.fillStyle = '#A0A0A0';
      ctx.fillText(`  +${signal.ord_triggers.length - 2} more...`, x + padding + 8, currentY);
      currentY += lineHeight;
    } else {
      currentY += 4;
    }

    // Elliott Wave triggers (if any)
    if (signal.ew_triggers && signal.ew_triggers.length > 0) {
      currentY += 2;
      ctx.font = 'bold 11px Arial';
      ctx.fillStyle = '#87CEEB';
      ctx.fillText('Elliott Wave:', x + padding, currentY);
      currentY += lineHeight;

      ctx.font = '10px Arial';
      ctx.fillStyle = '#FFFFFF';
      const ewTruncated = signal.ew_triggers[0].length > 35 ? signal.ew_triggers[0].substring(0, 32) + '...' : signal.ew_triggers[0];
      ctx.fillText(`• ${ewTruncated}`, x + padding + 8, currentY);
    }

    // ORD Quote at bottom
    currentY = y + panelHeight / 2 - padding - 12;
    ctx.font = 'italic 10px Arial';
    ctx.fillStyle = '#FFD700';
    ctx.textAlign = 'left';
    const quote = signal.ord_quote.length > 40 ? signal.ord_quote.substring(0, 37) + '...' : signal.ord_quote;
    ctx.fillText(`"${quote}"`, x + padding, currentY);

    ctx.restore();
  }

  /**
   * Draw rounded rectangle path
   * @private
   */
  _roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
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
   * Check if ORD Volume is active
   */
  isActiveAnalysis() {
    // Active if there's an analysis OR if we're in draw mode
    if (this.isActive) {
      // console.log('[ORD Bridge] isActiveAnalysis: true (has analysis)');
      return true;
    }

    // Also active if renderer is in draw mode
    if (this.ordVolumeRenderer) {
      const drawingState = this.ordVolumeRenderer.getDrawingState();
      if (drawingState.isDrawMode) {
        // console.log('[ORD Bridge] isActiveAnalysis: true (draw mode active)');
        return true;
      }
    }

    // console.log('[ORD Bridge] isActiveAnalysis: false (no analysis, not in draw mode)');
    return false;
  }

  /**
   * Handle clicks on percentage labels for custom wave comparison
   * @param {MouseEvent} event - Click event
   * @param {Object} chartState - Chart state with coordinate conversion functions
   */
  handleClick(event, chartState) {
    if (!this.currentAnalysis || !this.currentAnalysis.waveData) {
      return;
    }

    const canvas = event.target;
    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Check if click is on any volume label (on the trendlines)
    let clickedWaveIndex = null;

    // Search through all labels to find volume labels
    if (this.currentAnalysis.labels) {
      for (const label of this.currentAnalysis.labels) {
        if (!label.isVolumeLabel) continue;

        // Convert label position to screen coordinates
        const screenX = this._indexToX(label.x, chartState);
        const screenY = this._priceToY(label.y, chartState);

        // Check if click is near this volume label (within 40 pixels for easier clicking)
        const distance = Math.sqrt(Math.pow(clickX - screenX, 2) + Math.pow(clickY - screenY, 2));

        if (distance < 40) {
          clickedWaveIndex = label.waveIndex;
          break;
        }
      }
    }

    if (clickedWaveIndex !== null) {
      // Add to selected waves
      this.selectedWaves.push(clickedWaveIndex);

      if (this.selectedWaves.length === 2) {
        // Calculate comparison between the two selected waves
        const wave1 = this.currentAnalysis.waveData[this.selectedWaves[0]];
        const wave2 = this.currentAnalysis.waveData[this.selectedWaves[1]];

        const percentage = ((wave2.avgVolume / wave1.avgVolume) * 100).toFixed(1);
        const ratio = wave2.avgVolume / wave1.avgVolume;

        let strengthColor;
        if (ratio >= 1.10) strengthColor = '#00FF00';
        else if (ratio >= 0.92) strengthColor = '#FFFF00';
        else strengthColor = '#FF0000';

        this.customComparison = {
          wave1Index: this.selectedWaves[0],
          wave2Index: this.selectedWaves[1],
          percentage: percentage,
          color: strengthColor
        };

        console.log(`[ORD Bridge] Custom comparison: Wave ${this.selectedWaves[0]} to Wave ${this.selectedWaves[1]} = ${percentage}%`);

        // Reset selection for next comparison
        this.selectedWaves = [];

        // Trigger redraw
        if (window.tosApp && window.tosApp.activeChartType) {
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
    } else {
      // Click anywhere else - reset to default
      if (this.customComparison !== null) {
        console.log('[ORD Bridge] Resetting to default comparison');
        this.customComparison = null;
        this.selectedWaves = [];

        // Trigger redraw
        if (window.tosApp && window.tosApp.activeChartType) {
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
    }
  }

  /**
   * Create HTML info panel element (fixed position overlay)
   * @private
   */
  _createInfoPanelElement() {
    // Create panel container
    const panel = document.createElement('div');
    panel.id = 'ord-volume-info-panel';
    panel.style.cssText = `
      position: fixed;
      top: 80px;
      right: 20px;
      width: 420px;
      max-height: 500px;
      background: rgba(0, 0, 0, 0.9);
      border: 2px solid #2196F3;
      border-radius: 8px;
      color: white;
      font-family: Arial, sans-serif;
      font-size: 12px;
      z-index: 1000;
      display: none;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
      user-select: none;
      transition: all 0.3s ease;
    `;

    // Create draggable header with modern design
    const header = document.createElement('div');
    header.style.cssText = `
      cursor: move;
      padding: 0;
      margin: 0 0 10px 0;
      background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
      border-bottom: 3px solid #1565C0;
      border-radius: 6px 6px 0 0;
      box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
      position: relative;
      overflow: hidden;
    `;

    // Add subtle pattern overlay
    header.innerHTML = `
      <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,0.03) 10px, rgba(255,255,255,0.03) 20px); pointer-events: none;"></div>
      <div style="position: relative; padding: 12px 15px; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="background: rgba(255,255,255,0.2); padding: 6px 10px; border-radius: 4px; font-size: 18px;">📊</div>
          <div>
            <h3 style="margin: 0; color: #FFFFFF; font-size: 16px; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
              ORD Volume Analysis
            </h3>
            <div style="font-size: 10px; color: rgba(255,255,255,0.8); margin-top: 2px;">Professional Trading Insights</div>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <button id="ord-pin-button" style="background: rgba(0,0,0,0.2); padding: 4px 10px; border-radius: 4px; font-size: 11px; color: rgba(255,255,255,0.9); display: flex; align-items: center; gap: 5px; border: 1px solid rgba(255,255,255,0.2); cursor: pointer; transition: all 0.2s;" title="Pin below Quick Order">
            <span id="ord-pin-icon">📍</span>
            <span id="ord-pin-text">Pin</span>
          </button>
          <div style="background: rgba(0,0,0,0.2); padding: 4px 10px; border-radius: 4px; font-size: 11px; color: rgba(255,255,255,0.9); display: flex; align-items: center; gap: 5px;">
            <span>📌</span>
            <span>Drag</span>
          </div>
        </div>
      </div>
    `;

    // Create content container (scrollable)
    const content = document.createElement('div');
    content.id = 'ord-volume-info-content';
    content.style.cssText = `
      max-height: 340px;
      overflow-y: auto;
      padding: 5px 15px 15px 15px;
    `;

    panel.appendChild(header);
    panel.appendChild(content);
    document.body.appendChild(panel);

    // Make panel draggable
    this._makePanelDraggable(panel, header);

    // Add pin functionality
    this._addPinFunctionality(panel);

    this.infoPanelElement = panel;
    this.isPinned = false;

    // Auto-pin on creation (simulate pin button click after a delay to ensure DOM is ready)
    setTimeout(() => {
      const pinButton = document.getElementById('ord-pin-button');
      if (pinButton) {
        pinButton.click();
        console.log('[ORD Bridge] Auto-pinned panel on creation');
      }
    }, 250);
  }

  /**
   * Make panel draggable by header
   * @private
   */
  _makePanelDraggable(panel, header) {
    let isDragging = false;
    let currentX;
    let currentY;
    let initialX;
    let initialY;
    let rafId = null;

    const mouseDownHandler = (e) => {
      // Don't allow dragging if pinned
      if (this.isPinned) {
        return;
      }

      isDragging = true;

      // Calculate initial offset
      const rect = panel.getBoundingClientRect();
      initialX = e.clientX - rect.left;
      initialY = e.clientY - rect.top;

      // Disable transitions for smooth dragging
      panel.style.transition = 'none';
      header.style.cursor = 'grabbing';

      e.preventDefault();
    };

    const mouseMoveHandler = (e) => {
      if (!isDragging || this.isPinned) return;

      e.preventDefault();

      // Cancel any pending animation frame
      if (rafId) {
        cancelAnimationFrame(rafId);
      }

      // Use requestAnimationFrame for smooth 60fps updates
      rafId = requestAnimationFrame(() => {
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;

        // Keep panel within viewport bounds
        const maxX = window.innerWidth - panel.offsetWidth;
        const maxY = window.innerHeight - panel.offsetHeight;

        currentX = Math.max(0, Math.min(currentX, maxX));
        currentY = Math.max(0, Math.min(currentY, maxY));

        // Use transform for better performance (GPU accelerated)
        panel.style.left = currentX + 'px';
        panel.style.top = currentY + 'px';
        panel.style.right = 'auto'; // Remove right positioning when dragging
      });
    };

    const mouseUpHandler = () => {
      if (isDragging) {
        isDragging = false;

        // Re-enable transitions
        panel.style.transition = 'all 0.3s ease';

        if (!this.isPinned) {
          header.style.cursor = 'move';
        }

        // Cancel any pending animation frame
        if (rafId) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
      }
    };

    header.addEventListener('mousedown', mouseDownHandler);
    document.addEventListener('mousemove', mouseMoveHandler);
    document.addEventListener('mouseup', mouseUpHandler);

    // Store handlers for cleanup if needed
    this._dragHandlers = { mouseDownHandler, mouseMoveHandler, mouseUpHandler };
  }

  /**
   * Add pin functionality to position panel below Quick Order
   * @private
   */
  _addPinFunctionality(panel) {
    const pinButton = document.getElementById('ord-pin-button');
    const pinIcon = document.getElementById('ord-pin-icon');
    const pinText = document.getElementById('ord-pin-text');

    if (!pinButton) return;

    pinButton.addEventListener('click', (e) => {
      e.stopPropagation(); // Prevent drag from starting

      this.isPinned = !this.isPinned;

      if (this.isPinned) {
        // Pin below Account Summary (Margin Usage section) on right side of screen
        const accountSummary = document.querySelector('.tos-account-summary');
        const quickOrderElement = document.querySelector('.tos-active-trader');
        const header = panel.querySelector('div'); // Get header element

        // If we can find Account Summary, position below it. Otherwise use Quick Order or fallback
        if (accountSummary) {
          const summaryRect = accountSummary.getBoundingClientRect();
          const quickOrderRect = quickOrderElement ? quickOrderElement.getBoundingClientRect() : null;

          const summaryBottom = summaryRect.bottom;
          const quickOrderRight = quickOrderRect ? window.innerWidth - quickOrderRect.right : 20;
          const quickOrderWidth = quickOrderRect ? quickOrderRect.width : 380;

          console.log(`[ORD Bridge] 📍 PIN: Account Summary (Margin Usage) found at bottom=${summaryBottom}px`);
          console.log(`[ORD Bridge] 📍 PIN: Quick Order width=${quickOrderWidth}px, right=${quickOrderRight}px`);

          panel.style.left = 'auto';
          panel.style.right = quickOrderRight + 'px';
          panel.style.top = (summaryBottom + 5) + 'px'; // 5px gap below Account Summary (minimal gap)
          panel.style.width = quickOrderWidth + 'px'; // Match Quick Order width

          // Calculate available height (subtract 20px for bottom margin instead of 30px for more space)
          const availableHeight = window.innerHeight - summaryBottom - 20;
          panel.style.maxHeight = availableHeight + 'px';

          // Update content div to use more of the available space (subtract ~80px for header)
          const contentDiv = document.getElementById('ord-volume-info-content');
          if (contentDiv) {
            contentDiv.style.maxHeight = (availableHeight - 80) + 'px';
          }

          console.log(`[ORD Bridge] ✅ PIN: Panel positioned at top=${panel.style.top}, right=${panel.style.right}, width=${panel.style.width}, maxHeight=${panel.style.maxHeight}`);
        } else {
          // Fallback: position on right side below assumed Account Summary position
          console.warn('[ORD Bridge] Account Summary element not found, using fallback position');
          panel.style.left = 'auto';
          panel.style.right = '20px';
          panel.style.top = '700px'; // Below typical Account Summary height (more space)
          panel.style.width = '380px';
          panel.style.maxHeight = (window.innerHeight - 720) + 'px';
        }

        // Change cursor to default (not draggable when pinned)
        if (header) {
          header.style.cursor = 'default';
        }

        pinIcon.textContent = '📌';
        pinText.textContent = 'Unpin';
        pinButton.style.background = 'rgba(33, 150, 243, 0.4)';
        pinButton.style.borderColor = 'rgba(33, 150, 243, 0.6)';
      } else {
        // Unpin - return to floating position (right side)
        const header = panel.querySelector('div'); // Get header element

        panel.style.left = 'auto';
        panel.style.right = '20px';
        panel.style.top = '80px';
        panel.style.width = '420px';
        panel.style.maxHeight = '500px';

        // Reset content div max-height to default
        const contentDiv = document.getElementById('ord-volume-info-content');
        if (contentDiv) {
          contentDiv.style.maxHeight = '340px';
        }

        // Restore draggable cursor
        if (header) {
          header.style.cursor = 'move';
        }

        pinIcon.textContent = '📍';
        pinText.textContent = 'Pin';
        pinButton.style.background = 'rgba(0,0,0,0.2)';
        pinButton.style.borderColor = 'rgba(255,255,255,0.2)';
      }
    });

    // Hover effect
    pinButton.addEventListener('mouseenter', () => {
      if (!this.isPinned) {
        pinButton.style.background = 'rgba(33, 150, 243, 0.3)';
        pinButton.style.borderColor = 'rgba(33, 150, 243, 0.5)';
      }
    });

    pinButton.addEventListener('mouseleave', () => {
      if (!this.isPinned) {
        pinButton.style.background = 'rgba(0,0,0,0.2)';
        pinButton.style.borderColor = 'rgba(255,255,255,0.2)';
      }
    });
  }

  /**
   * Update and show/hide the info panel
   * @private
   */
  _drawInfoPanel(ctx, chartState) {
    if (!this.infoPanelElement) {
      this._createInfoPanelElement();
    }

    if (!this.currentAnalysis) {
      this.infoPanelElement.style.display = 'none';
      return;
    }

    const analysis = this.currentAnalysis;

    // Get wave data - could be 'waveData' or 'waves' depending on source
    const waveData = analysis.waveData || analysis.waves || [];

    // Calculate percentage changes from wave to wave
    const percentages = [];
    for (let i = 1; i < waveData.length; i++) {
      const currentWave = waveData[i];
      const previousWave = waveData[i - 1];
      if (currentWave.avgVolume && previousWave.avgVolume) {
        const percentage = (currentWave.avgVolume / previousWave.avgVolume) * 100;
        percentages.push(percentage);
      }
    }

    const hasStrongVolume = percentages.some(p => p >= 110);
    const hasWeakVolume = percentages.some(p => p < 90);
    const strongCount = percentages.filter(p => p >= 110).length;
    const weakCount = percentages.filter(p => p < 90).length;
    const neutralCount = percentages.filter(p => p >= 90 && p < 110).length;

    // Build HTML content with modern, appealing design
    let html = `
      <!-- Analysis Results Section -->
      <div style="background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(33, 150, 243, 0.05)); border-left: 4px solid #2196F3; border-radius: 6px; padding: 12px 15px; margin-bottom: 15px;">
        <h4 style="color: #2196F3; font-size: 14px; margin: 0 0 12px 0; font-weight: 600; display: flex; align-items: center;">
          <span style="font-size: 18px; margin-right: 8px;">📊</span> Your Analysis
        </h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px; margin-bottom: 10px;">
          <div style="background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px;">
            <div style="color: #AAA; font-size: 11px; margin-bottom: 3px;">Waves Analyzed</div>
            <div style="color: #FFF; font-size: 16px; font-weight: bold;">${waveData.length}</div>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px;">
            <div style="color: #AAA; font-size: 11px; margin-bottom: 3px;">Overall Strength</div>
            <div style="color: ${analysis.strength === 'Strong' ? '#00FF00' : analysis.strength === 'Weak' ? '#FF6666' : '#FFFF00'}; font-size: 16px; font-weight: bold;">${analysis.strength || 'N/A'}</div>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; font-size: 12px;">
          <div style="background: rgba(0,255,0,0.1); padding: 8px; border-radius: 4px; border: 1px solid rgba(0,255,0,0.3);">
            <div style="color: #AAA; font-size: 11px; margin-bottom: 3px;">Strong (≥110%)</div>
            <div style="color: #00FF00; font-size: 16px; font-weight: bold;">${strongCount}</div>
          </div>
          <div style="background: rgba(255,255,0,0.1); padding: 8px; border-radius: 4px; border: 1px solid rgba(255,255,0,0.3);">
            <div style="color: #AAA; font-size: 11px; margin-bottom: 3px;">Neutral (90-110%)</div>
            <div style="color: #FFFF00; font-size: 16px; font-weight: bold;">${neutralCount}</div>
          </div>
          <div style="background: rgba(255,102,102,0.1); padding: 8px; border-radius: 4px; border: 1px solid rgba(255,102,102,0.3);">
            <div style="color: #AAA; font-size: 11px; margin-bottom: 3px;">Weak (&lt;90%)</div>
            <div style="color: #FF6666; font-size: 16px; font-weight: bold;">${weakCount}</div>
          </div>
        </div>
      </div>

      <!-- Volume Interpretation Guide -->
      <div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 215, 0, 0.02)); border-left: 4px solid #FFD700; border-radius: 6px; padding: 12px 15px; margin-bottom: 15px;">
        <h4 style="color: #FFD700; font-size: 14px; margin: 0 0 10px 0; font-weight: 600; display: flex; align-items: center;">
          <span style="font-size: 18px; margin-right: 8px;">📈</span> Volume Interpretation
        </h4>
        <div style="line-height: 1.8; font-size: 12px;">
          <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="background: #00FF00; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
            <span style="color: #00FF00; font-weight: 600;">≥110%</span>
            <span style="color: #CCC; margin-left: 8px;">Strong momentum (bullish if uptrend)</span>
          </div>
          <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="background: #FFFF00; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
            <span style="color: #FFFF00; font-weight: 600;">90-110%</span>
            <span style="color: #CCC; margin-left: 8px;">Normal/healthy continuation</span>
          </div>
          <div style="display: flex; align-items: center;">
            <span style="background: #FF6666; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
            <span style="color: #FF6666; font-weight: 600;">&lt;90%</span>
            <span style="color: #CCC; margin-left: 8px;">Weakening (potential reversal)</span>
          </div>
        </div>
      </div>

      <!-- Trading Recommendation -->
      <div style="background: ${hasStrongVolume ? 'linear-gradient(135deg, rgba(0,255,0,0.15), rgba(0,255,0,0.05))' : hasWeakVolume ? 'linear-gradient(135deg, rgba(255,102,102,0.15), rgba(255,102,102,0.05))' : 'linear-gradient(135deg, rgba(255,255,0,0.1), rgba(255,255,0,0.02))'}; border-left: 4px solid ${hasStrongVolume ? '#00FF00' : hasWeakVolume ? '#FF6666' : '#FFFF00'}; border-radius: 6px; padding: 12px 15px; margin-bottom: 15px;">
        <h4 style="color: ${hasStrongVolume ? '#00FF00' : hasWeakVolume ? '#FF6666' : '#FFFF00'}; font-size: 14px; margin: 0 0 10px 0; font-weight: 600; display: flex; align-items: center;">
          <span style="font-size: 18px; margin-right: 8px;">💡</span> Trading Signal
        </h4>
    `;

    if (hasStrongVolume && waveData.length >= 3) {
      html += `
        <div style="font-size: 13px; color: #00FF00; font-weight: 600; margin-bottom: 6px;">✓ Strong Volume Detected</div>
        <div style="font-size: 12px; color: #CCC; line-height: 1.6;">Momentum is present. Look for continuation in the current trend direction. Consider entries on pullbacks with volume confirmation.</div>
      `;
    } else if (hasWeakVolume) {
      html += `
        <div style="font-size: 13px; color: #FF6666; font-weight: 600; margin-bottom: 6px;">⚠ Weakening Volume</div>
        <div style="font-size: 12px; color: #CCC; line-height: 1.6;">Volume is declining, suggesting potential exhaustion. Wait for confirmation before entering new positions. Consider reducing position size.</div>
      `;
    } else {
      html += `
        <div style="font-size: 13px; color: #FFFF00; font-weight: 600; margin-bottom: 6px;">○ Normal Volume Levels</div>
        <div style="font-size: 12px; color: #CCC; line-height: 1.6;">Volume is in neutral territory. Monitor for changes in the volume pattern that could signal the next directional move.</div>
      `;
    }

    html += `
      </div>

      <!-- Important Disclaimer -->
      <div style="background: rgba(255,102,102,0.1); border: 1px solid rgba(255,102,102,0.3); border-radius: 6px; padding: 12px 15px; margin-bottom: 10px;">
        <h4 style="color: #FF6666; font-size: 13px; margin: 0 0 8px 0; font-weight: 600; display: flex; align-items: center;">
          <span style="font-size: 16px; margin-right: 6px;">⚠</span> Risk Disclaimer
        </h4>
        <div style="font-size: 11px; color: #CCC; line-height: 1.7;">
          <div style="margin-bottom: 6px; color: #FFF;">ORD Volume is <strong>ONE</strong> indicator among many.</div>
          <div style="color: #AAA;">Always combine with:</div>
          <div style="margin-top: 4px; padding-left: 12px;">
            <div>▸ Price action & support/resistance levels</div>
            <div>▸ Trend indicators (moving averages)</div>
            <div>▸ Momentum oscillators (RSI, MACD)</div>
            <div>▸ Market context, fundamentals & news</div>
          </div>
        </div>
      </div>
    `;

    const signals = analysis.tradeSignals || analysis.signals || [];
    if (signals.length === 0) {
      html += `
      <div style="background: rgba(100, 150, 255, 0.1); border: 1px solid rgba(100, 150, 255, 0.3); border-radius: 6px; padding: 12px 15px;">
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
          <span style="font-size: 18px; margin-right: 8px;">ℹ️</span>
          <span style="color: #6496FF; font-weight: 600; font-size: 13px;">No Automated Signals</span>
        </div>
        <div style="color: #CCC; font-size: 11px; line-height: 1.7;">
          <div style="margin-bottom: 4px;">Automated signals require high confluence (4+ factors).</div>
          <div style="color: #AAA;">👉 Use the volume percentages on the chart to make informed manual trading decisions.</div>
        </div>
      </div>
      `;
    }

    // Update content container (not the whole panel, to preserve header)
    const contentContainer = document.getElementById('ord-volume-info-content');
    if (contentContainer) {
      contentContainer.innerHTML = html;
    }

    this.infoPanelElement.style.display = 'block';
  }
}

// Create global singleton instance
if (!window.ordVolumeBridge) {
  window.ordVolumeBridge = new ORDVolumeBridge();
}

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
    this.selectedWaves = []; // Track selected waves for custom comparison
    this.customComparison = null; // Store custom comparison data
    this.showTradeSignals = true; // Toggle for trade signal visibility
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

    // Draw trade signals (if enabled)
    if (this.showTradeSignals) {
      this._drawTradeSignals(ctx, chartState);
    }
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

    // Draw completed lines with professional styling
    for (let i = 0; i < drawingState.drawnLines.length; i++) {
      const line = drawingState.drawnLines[i];
      const [x1, y1, x2, y2] = line;

      console.log(`[ORD Bridge] Line ${i} raw:`, {x1, y1, x2, y2});

      const sx1 = this._indexToX(x1, chartState);
      const sy1 = this._priceToY(y1, chartState);
      const sx2 = this._indexToX(x2, chartState);
      const sy2 = this._priceToY(y2, chartState);

      console.log(`[ORD Bridge] Line ${i} screen:`, {sx1, sy1, sx2, sy2});

      // Professional green with subtle shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
      ctx.shadowBlur = 2;
      ctx.shadowOffsetY = 1;
      ctx.strokeStyle = '#10B981'; // Emerald green for completed lines
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();

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

      if (shouldLog) {
        console.log(`[ORD Bridge] ✅ Line drawn successfully`);
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

    for (let i = 0; i < this.currentAnalysis.labels.length; i++) {
      const label = this.currentAnalysis.labels[i];

      // Convert to screen coordinates
      let x = this._indexToX(label.x, chartState);
      let y = this._priceToY(label.y, chartState);

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
   * Draw trade signals (BUY/SELL indicators)
   * @private
   */
  _drawTradeSignals(ctx, chartState) {
    if (!this.currentAnalysis.tradeSignals) return;

    ctx.save();

    for (const signal of this.currentAnalysis.tradeSignals) {
      // Convert to screen coordinates
      const x = this._indexToX(signal.x, chartState);
      const y = this._priceToY(signal.y, chartState);

      // Skip if off-screen
      if (x < 0 || x > ctx.canvas.width || y < 0 || y > ctx.canvas.height) {
        continue;
      }

      // Draw arrow and label
      if (signal.type === 'BUY') {
        // BUY signal - green arrow pointing up
        this._drawBuySignal(ctx, x, y);
      } else if (signal.type === 'SELL') {
        // SELL signal - red arrow pointing down
        this._drawSellSignal(ctx, x, y);
      }
    }

    ctx.restore();
  }

  /**
   * Draw BUY signal arrow
   * @private
   */
  _drawBuySignal(ctx, x, y) {
    const arrowSize = 20;
    const yOffset = 40; // Position below the point

    ctx.save();

    // Draw upward arrow
    ctx.beginPath();
    ctx.moveTo(x, y + yOffset); // Arrow tip
    ctx.lineTo(x - arrowSize / 2, y + yOffset + arrowSize); // Left point
    ctx.lineTo(x - arrowSize / 4, y + yOffset + arrowSize); // Left of shaft
    ctx.lineTo(x - arrowSize / 4, y + yOffset + arrowSize * 1.5); // Shaft left
    ctx.lineTo(x + arrowSize / 4, y + yOffset + arrowSize * 1.5); // Shaft right
    ctx.lineTo(x + arrowSize / 4, y + yOffset + arrowSize); // Right of shaft
    ctx.lineTo(x + arrowSize / 2, y + yOffset + arrowSize); // Right point
    ctx.closePath();

    // Fill with green
    ctx.fillStyle = '#00FF00';
    ctx.fill();

    // Outline
    ctx.strokeStyle = '#008800';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Add "BUY" text
    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
    ctx.shadowBlur = 3;
    ctx.fillText('BUY', x, y + yOffset + arrowSize * 1.5 + 5);

    ctx.restore();
  }

  /**
   * Draw SELL signal arrow
   * @private
   */
  _drawSellSignal(ctx, x, y) {
    const arrowSize = 20;
    const yOffset = -40; // Position above the point

    ctx.save();

    // Draw downward arrow
    ctx.beginPath();
    ctx.moveTo(x, y + yOffset); // Arrow tip
    ctx.lineTo(x - arrowSize / 2, y + yOffset - arrowSize); // Left point
    ctx.lineTo(x - arrowSize / 4, y + yOffset - arrowSize); // Left of shaft
    ctx.lineTo(x - arrowSize / 4, y + yOffset - arrowSize * 1.5); // Shaft left
    ctx.lineTo(x + arrowSize / 4, y + yOffset - arrowSize * 1.5); // Shaft right
    ctx.lineTo(x + arrowSize / 4, y + yOffset - arrowSize); // Right of shaft
    ctx.lineTo(x + arrowSize / 2, y + yOffset - arrowSize); // Right point
    ctx.closePath();

    // Fill with red
    ctx.fillStyle = '#FF0000';
    ctx.fill();

    // Outline
    ctx.strokeStyle = '#880000';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Add "SELL" text
    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
    ctx.shadowBlur = 3;
    ctx.fillText('SELL', x, y + yOffset - arrowSize * 1.5 - 5);

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
}

// Create global singleton instance
if (!window.ordVolumeBridge) {
  window.ordVolumeBridge = new ORDVolumeBridge();
}

/**
 * Canvas-Based Candlestick Chart Renderer
 * Custom renderer using HTML5 Canvas - NO PLOTLY
 * Full control over rendering, dates, and live updates
 * Updated: 2025-10-24 - TradingView-style unified price scale (candles can appear in volume area)
 */

import { SelectionManager } from './SelectionManager.js?v=20251030-fix';

export class CanvasRenderer {
  constructor(timeframeInterval = '1d') {
    this.containerId = 'tos-plot';
    this.canvas = null;
    this.ctx = null;
    this.data = [];
    this.symbol = '';
    this.timeframeInterval = timeframeInterval; // Store timeframe for adaptive gridlines

    // Chart dimensions
    this.width = 0;
    this.height = 0;
    this.chartHeight = 0; // Full height for unified price scale (TradingView style)
    this.volumeHeight = 80; // Fixed 80px for volume bars at bottom
    this.margin = { top: 60, right: 80, bottom: 60, left: 60 };

    // View state (for pan/zoom)
    this.startIndex = 0;
    this.endIndex = 0;
    this.candleWidth = 8;
    this.candleSpacing = 2;

    // Price range
    this.minPrice = 0;
    this.maxPrice = 0;
    this.maxVolume = 0;

    // Mouse state
    this.mouseX = -1;
    this.mouseY = -1;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.dragStartIndex = 0;
    this.dragStartMinPrice = 0;
    this.dragStartMaxPrice = 0;
    this.hoveredIndex = -1;

    // Axis expand/compress state
    this.isDraggingYAxis = false; // Y-axis expand/compress
    this.isDraggingXAxis = false; // X-axis expand/compress
    this.yAxisDragStartY = 0;
    this.yAxisDragStartRange = 0;
    this.yAxisDragStartCenter = 0;
    this.xAxisDragStartX = 0;
    this.xAxisDragStartVisibleCandles = 0;

    // Live price tracking
    this.livePrice = null;

    // Auto-scroll behavior for tick charts
    this.autoScrollEnabled = true; // Enable auto-scroll to latest candle by default
    this.hasRendered = false; // Track if we've rendered at least once

    // Drawing tools storage
    this.drawings = []; // Store all completed drawings
    this.previewDrawing = null; // Current drawing being previewed
    this.eraserHoveredDrawing = null; // Drawing hovered by eraser tool

    // Selection manager for editing drawings
    this.selectionManager = null; // Initialized after canvas is created

    // Colors
    this.colors = {
      background: '#1a1a1a',
      grid: '#404040',
      text: '#a0a0a0',
      textBright: '#e0e0e0',
      crosshair: '#666666',
      bullCandle: '#00c851',
      bearCandle: '#ff4444',
      volume: 'rgba(0, 188, 212, 0.7)',
      livePriceLine: '#00bcd4',
      highLine: '#ff9800',
      lowLine: '#9c27b0'
    };
  }

  /**
   * Initialize and render the chart
   */
  async render(data, symbol) {
    console.log(`🎨 Canvas: Rendering ${data.length} candles for ${symbol} (autoScroll=${this.autoScrollEnabled}, hasRendered=${this.hasRendered}, canvas=${!!this.canvas})`);

    if (data.length > 0) {
      console.log(`📅 Canvas rendering: ${data[0].Date} to ${data[data.length - 1].Date} (${data.length} candles)`);
    }

    // Detect if data was shifted (oldest candle removed) - BEFORE updating this.data
    let dataWasShifted = false;
    if (this.data.length > 0 && data.length > 0) {
      // If the first candle's date changed, data was shifted
      if (this.data[0].Date !== data[0].Date) {
        dataWasShifted = true;
        console.log(`🔄 Canvas: Data shifted detected (old first: ${this.data[0].Date}, new first: ${data[0].Date})`);
      }
    }

    this.data = data;
    this.symbol = symbol;

    if (data.length === 0) {
      console.error('❌ No data to render');
      return false;
    }

    // Create canvas only if it doesn't exist yet (first render)
    // On subsequent renders, reuse the existing canvas to preserve pan/zoom state
    if (!this.canvas || !this.ctx) {
      console.log(`🆕 Creating canvas (first render or canvas was destroyed)`);
      this.createCanvas();
    }

    if (!this.canvas || !this.ctx) {
      console.error('❌ Failed to create canvas');
      return false;
    }

    // Set initial view to show the most recent 100 candles (only if auto-scroll enabled)
    const candlesToShow = 100;

    // Check current view position and update auto-scroll state accordingly
    // This catches cases where user panned but render() is called before mouse detection
    if (this.hasRendered && this.endIndex !== undefined) {
      const distanceFromLiveEdge = (data.length - 1) - this.endIndex;
      console.log(`📏 Distance from live edge: ${distanceFromLiveEdge.toFixed(2)} candles (endIndex=${this.endIndex}, dataLength=${data.length})`);

      // Negative distance means endIndex is beyond data bounds (user panned left but data changed)
      // Treat this as being far from live edge
      if ((distanceFromLiveEdge < 0 || distanceFromLiveEdge > 1) && this.autoScrollEnabled) {
        this.autoScrollEnabled = false;
        console.log('⏸️ Auto-scroll DISABLED - view is not at live edge');
      } else if (distanceFromLiveEdge >= 0 && distanceFromLiveEdge <= 0.5 && !this.autoScrollEnabled) {
        this.autoScrollEnabled = true;
        console.log('▶️ Auto-scroll ENABLED - view returned to live edge');
      }
    }

    // Only reset view if auto-scroll is enabled OR this is the very first render
    if (this.autoScrollEnabled || !this.hasRendered) {
      this.endIndex = data.length - 1;
      this.startIndex = this.endIndex - candlesToShow;
      console.log(`📊 Auto-scroll enabled: Showing latest candles ${this.startIndex} to ${this.endIndex}`);
      this.hasRendered = true; // Mark as rendered
    } else {
      // Auto-scroll is paused - maintain the user's view
      // console.log(`⏸️ Auto-scroll paused: Current view ${this.startIndex} to ${this.endIndex}`);

      // If data was shifted, compensate by shifting indices left to keep same view
      if (dataWasShifted) {
        this.startIndex = Math.max(0, this.startIndex - 1);
        this.endIndex = Math.max(candlesToShow - 1, this.endIndex - 1);
        // console.log(`⬅️ Compensating for data shift: adjusted view to ${this.startIndex} - ${this.endIndex}`);
      }

      // console.log(`⏸️ Final view: ${this.startIndex} to ${this.endIndex}`);
    }

    console.log(`📊 Showing candles ${this.startIndex} to ${this.endIndex}`);

    // Calculate price range
    this.calculateRanges();

    console.log(`💰 Price range: ${this.minPrice.toFixed(2)} - ${this.maxPrice.toFixed(2)}`);

    // Draw chart
    this.draw();

    // Setup mouse events (only once)
    if (!this.eventsSetup) {
      this.setupEvents();
      this.eventsSetup = true;
    }

    // Load saved drawings for this symbol (only on first render or symbol change)
    if (!this.drawingsLoaded || this.lastLoadedSymbol !== symbol) {
      this.lastLoadedSymbol = symbol;
      this.drawingsLoaded = true;
      await this.loadDrawings();
    }

    // console.log('✅ Canvas chart rendered successfully');
    return true;
  }

  // ==================== DRAWING PERSISTENCE METHODS ====================

  /**
   * Load saved drawings for current symbol from backend
   */
  async loadDrawings() {
    if (!this.symbol) {
      console.warn('⚠️ No symbol set, skipping drawing load');
      return;
    }

    try {
      const response = await fetch(`/drawings/${this.symbol}`);
      if (response.ok) {
        const drawings = await response.json();

        // Clean up legacy drawings missing chart coordinates
        const validDrawings = this.cleanupInvalidDrawings(drawings);
        this.drawings = validDrawings;

        // If cleanup removed any drawings, update the backend file
        if (drawings.length !== validDrawings.length) {
          console.log(`🧹 Cleaning up backend file (removed ${drawings.length - validDrawings.length} invalid drawings)`);
          await this.saveAllDrawings();
        }

        console.log(`✅ Loaded ${this.drawings.length} drawings for ${this.symbol}`);
        this.draw(); // Redraw with loaded drawings
      }
    } catch (error) {
      console.error('❌ Error loading drawings:', error);
    }
  }

  /**
   * Clean up drawings that are missing required chart coordinates
   * These are legacy drawings from before coordinate conversion was implemented
   */
  cleanupInvalidDrawings(drawings) {
    const validDrawings = drawings.filter(drawing => {
      // Check if drawing has required coordinates based on its type
      if (drawing.action.includes('fibonacci-extension')) {
        return drawing.point1Index !== undefined && drawing.point1Price !== undefined &&
               drawing.point2Index !== undefined && drawing.point2Price !== undefined &&
               drawing.point3Index !== undefined && drawing.point3Price !== undefined;
      } else if (drawing.action.includes('fibonacci-time-zones')) {
        return drawing.chartIndex !== undefined;
      } else if (drawing.action.includes('fibonacci-') || drawing.action.includes('gann-')) {
        return drawing.startIndex !== undefined && drawing.startPrice !== undefined &&
               drawing.endIndex !== undefined && drawing.endPrice !== undefined;
      } else if (drawing.action === 'place-dot' || drawing.action === 'place-arrow') {
        return drawing.chartIndex !== undefined && drawing.chartPrice !== undefined;
      } else if (drawing.action.includes('horizontal-line')) {
        return drawing.price !== undefined;
      } else if (drawing.action.includes('vertical-line')) {
        return drawing.chartIndex !== undefined;
      } else if (drawing.chartPoints) {
        // Pattern tools and polygon
        return drawing.chartPoints.length > 0 &&
               drawing.chartPoints[0].chartIndex !== undefined &&
               drawing.chartPoints[0].chartPrice !== undefined;
      } else if (drawing.centerIndex !== undefined) {
        // Circle
        return drawing.centerPrice !== undefined && drawing.radiusInPriceUnits !== undefined;
      } else if (drawing.startIndex !== undefined) {
        // Rectangle, Ellipse, and other standard shapes/lines
        return drawing.startPrice !== undefined &&
               drawing.endIndex !== undefined &&
               drawing.endPrice !== undefined;
      }

      // If we don't recognize the format, keep it (to be safe)
      return true;
    });

    const removedCount = drawings.length - validDrawings.length;
    if (removedCount > 0) {
      console.warn(`🧹 Cleaned up ${removedCount} invalid drawings missing chart coordinates`);
    }

    return validDrawings;
  }

  /**
   * Save a drawing to backend
   */
  async saveDrawing(drawing) {
    if (!this.symbol) {
      console.warn('⚠️ No symbol set, skipping drawing save');
      return;
    }

    try {
      const response = await fetch('/save_drawing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: this.symbol,
          drawing: drawing
        })
      });

      if (response.ok) {
        console.log(`💾 Saved drawing: ${drawing.id} (${drawing.action})`);
      } else {
        console.error('❌ Failed to save drawing:', await response.text());
      }
    } catch (error) {
      console.error('❌ Error saving drawing:', error);
    }
  }

  /**
   * Delete a drawing from backend
   */
  async deleteDrawing(drawingId) {
    if (!this.symbol) {
      console.warn('⚠️ No symbol set, skipping drawing delete');
      return;
    }

    try {
      const response = await fetch('/delete_drawing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: this.symbol,
          drawing_id: drawingId
        })
      });

      if (response.ok) {
        console.log(`🗑️ Deleted drawing from backend: ${drawingId}`);
      } else {
        console.error('❌ Failed to delete drawing:', await response.text());
      }
    } catch (error) {
      console.error('❌ Error deleting drawing:', error);
    }
  }

  /**
   * Save all drawings to backend (replaces existing file)
   * Used for cleanup operations
   */
  async saveAllDrawings() {
    if (!this.symbol) {
      console.warn('⚠️ No symbol set, skipping save all drawings');
      return;
    }

    try {
      const response = await fetch('/save_all_drawings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: this.symbol,
          drawings: this.drawings
        })
      });

      if (response.ok) {
        console.log(`💾 Saved all ${this.drawings.length} drawings for ${this.symbol}`);
      } else {
        console.error('❌ Failed to save all drawings:', await response.text());
      }
    } catch (error) {
      console.error('❌ Error saving all drawings:', error);
    }
  }

  /**
   * Create canvas element
   */
  createCanvas() {
    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error('Container not found:', this.containerId);
      return;
    }

    console.log(`📦 Container found:`, container.getBoundingClientRect());

    // Clear container
    container.innerHTML = '';

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.canvas.style.backgroundColor = '#1a1a1a';
    this.canvas.style.cursor = 'default'; // Let tools manage cursor
    container.appendChild(this.canvas);

    // Set canvas size
    this.resize();

    // Get context
    this.ctx = this.canvas.getContext('2d');

    // Reset events flag since we created a new canvas
    // This ensures setupEvents() will run again for the new canvas element
    this.eventsSetup = false;
  }

  /**
   * Resize canvas to fit container
   */
  resize() {
    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error('❌ Container not found during resize');
      return;
    }

    const rect = container.getBoundingClientRect();

    this.width = Math.floor(rect.width);
    this.height = Math.floor(rect.height);

    console.log(`📐 Canvas size: ${this.width}x${this.height}`);

    // Ensure minimum size
    if (this.width < 100 || this.height < 100) {
      console.warn('⚠️ Canvas too small, waiting for proper size');
      setTimeout(() => {
        this.resize();
        if (this.data && this.data.length > 0) {
          this.calculateVolumeRange(); // Preserve price pan position
          this.draw();
        }
      }, 100);
      return;
    }

    // Set actual canvas size (for retina displays)
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;

    // Canvas style size (CSS pixels)
    this.canvas.style.width = this.width + 'px';
    this.canvas.style.height = this.height + 'px';

    // Get new context after resize
    this.ctx = this.canvas.getContext('2d');

    // Scale context for retina
    this.ctx.scale(dpr, dpr);

    // Calculate chart areas - UNIFIED: Full height for price scale (TradingView style)
    this.chartHeight = this.height - this.margin.top - this.margin.bottom; // Full height for candles
    this.volumeHeight = 80; // Fixed 80px at bottom for volume bars

    console.log(`📊 Chart area: ${this.chartHeight.toFixed(0)}px (unified price scale), Volume bar height: ${this.volumeHeight}px`);
  }

  /**
   * Calculate price and volume ranges for visible data (full recalculation)
   * Use this only on initial load or when loading new data
   */
  calculateRanges() {
    // Clamp and floor indices to valid data range for calculations (fractional indices support)
    const clampedStart = Math.max(0, Math.floor(this.startIndex));
    const clampedEnd = Math.min(this.data.length - 1, Math.ceil(this.endIndex));

    const visibleData = this.data.slice(clampedStart, clampedEnd + 1);

    if (visibleData.length === 0) return;

    this.minPrice = Math.min(...visibleData.map(d => d.Low));
    this.maxPrice = Math.max(...visibleData.map(d => d.High));

    // Add padding and round dynamically based on price range
    const priceRange = this.maxPrice - this.minPrice;

    // Use less padding for intraday timeframes to make small candle movements more visible
    let paddingPercent;
    if (this.timeframeInterval === '1m' || this.timeframeInterval === '5m') {
      paddingPercent = 0.03; // 3% padding for minute charts (tighter view)
    } else if (this.timeframeInterval === '15m' || this.timeframeInterval === '30m' || this.timeframeInterval === '1h') {
      paddingPercent = 0.05; // 5% padding for hour charts
    } else {
      paddingPercent = 0.15; // 15% padding for daily/weekly/monthly charts
    }

    const padding = priceRange * paddingPercent;

    // Determine rounding interval based on the actual price range (adaptive)
    // Smaller ranges = finer increments, larger ranges = coarser increments
    let roundingInterval;
    if (priceRange < 10) {
      roundingInterval = 1;       // Micro range: round to $1 (for very zoomed intraday)
    } else if (priceRange < 50) {
      roundingInterval = 5;       // Tiny range: round to $5
    } else if (priceRange < 100) {
      roundingInterval = 10;      // Very tight range: round to $10
    } else if (priceRange < 500) {
      roundingInterval = 50;      // Tight range: round to $50
    } else if (priceRange < 1000) {
      roundingInterval = 100;     // Small range: round to $100
    } else if (priceRange < 5000) {
      roundingInterval = 500;     // Medium range: round to $500
    } else if (priceRange < 10000) {
      roundingInterval = 1000;    // Large range: round to $1,000
    } else if (priceRange < 50000) {
      roundingInterval = 5000;    // Very large range: round to $5,000
    } else {
      roundingInterval = 10000;   // Huge range: round to $10,000
    }

    // Round down min, then subtract one more interval for extra space
    this.minPrice = Math.floor((this.minPrice - padding) / roundingInterval) * roundingInterval - roundingInterval;

    // Round up max, then add one more interval for extra space
    this.maxPrice = Math.ceil((this.maxPrice + padding) / roundingInterval) * roundingInterval + roundingInterval;

    // Also update volume range
    this.calculateVolumeRange();
  }

  /**
   * Calculate only the volume range (preserves manually panned price ranges)
   * Use this during zoom/resize to avoid resetting vertical pan position
   */
  calculateVolumeRange() {
    // Clamp and floor indices to valid data range for calculations (fractional indices support)
    const clampedStart = Math.max(0, Math.floor(this.startIndex));
    const clampedEnd = Math.min(this.data.length - 1, Math.ceil(this.endIndex));

    const visibleData = this.data.slice(clampedStart, clampedEnd + 1);

    if (visibleData.length === 0) return;

    // Exclude last candle from maxVolume (it's incomplete/growing)
    // This prevents today's small volume from being dwarfed by historical volumes
    const volumesForScale = visibleData.length > 1
      ? visibleData.slice(0, -1).map(d => d.Volume || 0)  // All except last
      : visibleData.map(d => d.Volume || 0);  // If only 1 candle, use it

    this.maxVolume = Math.max(...volumesForScale);
  }

  /**
   * Detect if a horizontal line at a given price is acting as support or resistance
   * Returns 'support' (green), 'resistance' (red), or 'neutral' (orange)
   *
   * Algorithm:
   * - Look at candles near the line (within a price tolerance)
   * - Count bounces: candles that approach the line but don't cross significantly
   * - Determine direction: if more candles are above the line, it's support; if below, it's resistance
   *
   * @param {number} price - The price level of the horizontal line
   * @param {number} lookbackBars - Number of bars to analyze (default 50)
   * @returns {string} 'support', 'resistance', or 'neutral'
   */
  detectSupportResistance(price, lookbackBars = 50) {
    if (!this.data || this.data.length === 0) return 'neutral';

    // Define price tolerance (e.g., 0.5% of the price for detecting touches)
    const tolerance = price * 0.005; // 0.5% tolerance

    // Get visible or recent candles
    const startIdx = Math.max(0, Math.floor(this.endIndex) - lookbackBars);
    const endIdx = Math.min(this.data.length - 1, Math.ceil(this.endIndex));
    const candlesToAnalyze = this.data.slice(startIdx, endIdx + 1);

    let candlesAbove = 0;  // Candles mostly above the line
    let candlesBelow = 0;  // Candles mostly below the line
    let touches = 0;       // Candles that touch/bounce off the line

    for (const candle of candlesToAnalyze) {
      const high = candle.High;
      const low = candle.Low;
      const close = candle.Close;

      // Check if candle touches the line (within tolerance)
      const touchesLine = (low <= price + tolerance && high >= price - tolerance);

      if (touchesLine) {
        touches++;

        // Determine if it's a bounce from above (resistance) or below (support)
        if (close < price) {
          // Candle closed below line - likely tested resistance
          candlesBelow++;
        } else {
          // Candle closed above line - likely tested support
          candlesAbove++;
        }
      } else {
        // Candle doesn't touch - just count which side it's on
        if (low > price + tolerance) {
          candlesAbove++;
        } else if (high < price - tolerance) {
          candlesBelow++;
        }
      }
    }

    // Decision logic:
    // If more candles are above the line, it's acting as support (holding price up)
    // If more candles are below the line, it's acting as resistance (holding price down)
    const ratio = candlesAbove / (candlesAbove + candlesBelow || 1);

    if (ratio > 0.6) {
      return 'support';      // Green - most candles above, line supports price
    } else if (ratio < 0.4) {
      return 'resistance';   // Red - most candles below, line resists price
    } else {
      return 'neutral';      // Orange - unclear or equal distribution
    }
  }

  /**
   * Main draw function
   */
  draw() {
    if (!this.ctx) {
      console.error('❌ No context to draw');
      return;
    }

    // Clear canvas with background
    this.ctx.fillStyle = this.colors.background;
    this.ctx.fillRect(0, 0, this.width, this.height);

    // Draw title
    this.drawTitle();

    // Draw grid
    this.drawGrid();

    // Draw candles
    this.drawCandles();

    // Draw volume
    this.drawVolume();

    // Draw axes
    this.drawAxes();

    // Draw price markers (Ask, Bid, High, Low, Current)
    this.drawPriceMarkers();

    // Draw live price lines (current, high, low)
    this.drawLivePriceLines();

    // Draw all completed drawings
    this.drawAllDrawings();

    // Draw selection highlight (on top of drawings, below preview)
    if (this.selectionManager) {
      this.selectionManager.drawSelection();
    }

    // Draw preview of current drawing
    if (this.previewDrawing) {
      this.drawPreview(this.previewDrawing);
    }

    // Draw crosshair (when not dragging for cleaner pan experience)
    if (this.mouseX >= 0 && this.mouseY >= 0 && !this.isDragging) {
      this.drawCrosshair();
    }

    // Draw hover info
    if (this.hoveredIndex >= 0) {
      this.drawHoverInfo();
    }
  }

  /**
   * Calculate price levels with dynamic increment to create square gridlines
   * Price gridlines should match the vertical gridline spacing for square grids
   */
  calculatePriceLevels() {
    // Calculate vertical gridline spacing in pixels
    const chartWidth = this.width - this.margin.left - this.margin.right;
    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles;
    const gridInterval = 2; // Vertical gridline every 2 candles
    const verticalGridSpacing = totalWidth * gridInterval; // Pixels between vertical gridlines

    // Calculate how much price range corresponds to the same pixel distance
    const priceRange = this.maxPrice - this.minPrice;
    const pixelsPerPrice = this.chartHeight / priceRange;
    const targetPriceIncrement = verticalGridSpacing / pixelsPerPrice;

    // Round to a nice increment (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, etc.)
    const niceIncrements = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000];
    let increment = niceIncrements[0];

    for (let i = 0; i < niceIncrements.length; i++) {
      if (niceIncrements[i] >= targetPriceIncrement) {
        increment = niceIncrements[i];
        break;
      }
      // If we're past the last increment, use the last one
      if (i === niceIncrements.length - 1) {
        increment = niceIncrements[i];
      }
    }

    // Calculate first price level (round down to nearest increment)
    const firstLevel = Math.floor(this.minPrice / increment) * increment;

    // Generate all price levels (extend beyond visible range)
    const levels = [];
    let currentLevel = firstLevel;

    // Extend beyond maxPrice for top padding
    while (currentLevel <= this.maxPrice + increment) {
      levels.push(currentLevel);
      currentLevel += increment;
    }

    return levels;
  }

  /**
   * Draw grid lines (unified canvas, TradingView style)
   */
  drawGrid() {
    const ctx = this.ctx;
    const chartTop = this.margin.top;
    const chartBottom = this.height - this.margin.bottom; // Full height (unified)
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;

    // Horizontal grid lines (price levels) - at nice round increments
    const priceLevels = this.calculatePriceLevels();
    for (const price of priceLevels) {
      const y = this.priceToY(price);
      ctx.beginPath();
      ctx.moveTo(chartLeft, y);
      ctx.lineTo(chartRight, y);
      ctx.stroke();
    }

    // Vertical grid lines (anchored to candles, extends beyond data like TradingView)
    const chartWidth = chartRight - chartLeft;
    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles;

    // Calculate gridline interval (every Nth candle gets a gridline)
    const gridInterval = 2; // Gridline every other candle

    // Find the first candle index that should have a gridline
    const firstGridIndex = Math.ceil(this.startIndex / gridInterval) * gridInterval;

    // Calculate how many gridlines we need to extend beyond the data to fill the chart
    const candlesInView = this.endIndex - this.startIndex;
    const extraGridlines = Math.ceil(candlesInView / gridInterval) + 10; // Extra gridlines beyond data

    // Draw gridlines anchored to candles, extending beyond visible data
    for (let i = 0; i <= extraGridlines; i++) {
      const dataIndex = firstGridIndex + (i * gridInterval);

      // Calculate center X position (even if candle doesn't exist)
      const relativeIndex = dataIndex - this.startIndex;
      const centerX = chartLeft + (relativeIndex * totalWidth) + (totalWidth / 2);

      // Stop drawing past the right edge
      if (centerX > chartRight) break;

      // Only draw if in visible x range
      if (centerX >= chartLeft) {
        ctx.beginPath();
        ctx.moveTo(centerX, chartTop);
        ctx.lineTo(centerX, chartBottom);
        ctx.stroke();
      }
    }
  }

  /**
   * Draw candlesticks
   */
  drawCandles() {
    const ctx = this.ctx;
    const chartTop = this.margin.top;
    const chartLeft = this.margin.left;
    const chartWidth = this.width - this.margin.left - this.margin.right;

    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles; // Distribute across full width
    const candleWidth = Math.max(2, totalWidth * 0.7); // 70% of space for candle, 30% for gap
    const spacing = totalWidth - candleWidth;

    // Use integer loop bounds for fractional index support
    const loopStart = Math.floor(this.startIndex);
    const loopEnd = Math.ceil(this.endIndex);

    for (let i = loopStart; i <= loopEnd; i++) {
      // Skip indices outside the data bounds
      if (i < 0 || i >= this.data.length) continue;

      const candle = this.data[i];
      const x = chartLeft + ((i - this.startIndex) * totalWidth) + spacing / 2;

      const open = this.priceToY(candle.Open);
      const close = this.priceToY(candle.Close);
      const high = this.priceToY(candle.High);
      const low = this.priceToY(candle.Low);

      const isBull = candle.Close >= candle.Open;
      const color = isBull ? this.colors.bullCandle : this.colors.bearCandle;

      // Draw wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, high);
      ctx.lineTo(x + candleWidth / 2, low);
      ctx.stroke();

      // Draw body
      ctx.fillStyle = color;
      const bodyTop = Math.min(open, close);
      const bodyHeight = Math.abs(close - open);
      ctx.fillRect(x, bodyTop, candleWidth, Math.max(1, bodyHeight));
    }
  }

  /**
   * Draw volume bars (fixed position at bottom, like TradingView)
   */
  drawVolume() {
    const ctx = this.ctx;
    // Volume bars in fixed 80px area at bottom of canvas
    const volumeTop = this.height - this.margin.bottom - this.volumeHeight;
    const chartLeft = this.margin.left;
    const chartWidth = this.width - this.margin.left - this.margin.right;

    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles;
    const barWidth = Math.max(2, totalWidth * 0.7);
    const spacing = totalWidth - barWidth;

    ctx.fillStyle = this.colors.volume;

    // Use integer loop bounds for fractional index support
    const loopStart = Math.floor(this.startIndex);
    const loopEnd = Math.ceil(this.endIndex);

    for (let i = loopStart; i <= loopEnd; i++) {
      // Skip indices outside the data bounds
      if (i < 0 || i >= this.data.length) continue;

      const candle = this.data[i];
      const x = chartLeft + ((i - this.startIndex) * totalWidth) + spacing / 2;
      const volume = candle.Volume || 0;
      const barHeight = (volume / this.maxVolume) * this.volumeHeight;

      ctx.fillRect(x, volumeTop + this.volumeHeight - barHeight, barWidth, barHeight);
    }
  }

  /**
   * Draw axes labels
   */
  drawAxes() {
    const ctx = this.ctx;
    const chartTop = this.margin.top;
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;
    const chartWidth = this.width - this.margin.left - this.margin.right;

    ctx.fillStyle = this.colors.text;
    ctx.font = '11px Arial';
    ctx.textAlign = 'right';

    // Price labels (right side) - at nice round increments
    const priceLevels = this.calculatePriceLevels();
    for (const price of priceLevels) {
      const y = this.priceToY(price);
      // Format price with comma separators for better readability
      const formattedPrice = price.toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      });
      ctx.fillText(formattedPrice, chartRight + 50, y + 4);
    }

    // Date labels (below volume bars) - TradingView adaptive style
    // Volume bars end at: this.height - this.margin.bottom
    const volumeBottom = this.height - this.margin.bottom;
    const labelY = volumeBottom + 18; // Position labels 18px below volume bars

    ctx.textAlign = 'center';
    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles;

    // Larger, brighter font for better visibility
    ctx.font = '11px Arial';
    ctx.fillStyle = '#e0e0e0';

    // Adaptive labeling based on zoom level (like TradingView)
    const loopStart = Math.floor(this.startIndex);
    const loopEnd = Math.ceil(this.endIndex);

    let lastMonth = null;
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    // Determine day labeling density based on zoom level
    let dayInterval;
    if (visibleCandles > 250) {
      dayInterval = 0; // No day labels, only months
    } else if (visibleCandles > 150) {
      dayInterval = 10; // Show every 10th day
    } else if (visibleCandles > 100) {
      dayInterval = 5; // Show every 5th day
    } else if (visibleCandles > 50) {
      dayInterval = 2; // Show every other day
    } else {
      dayInterval = 1; // Show all days
    }

    for (let i = loopStart; i <= loopEnd; i++) {
      if (i < 0 || i >= this.data.length) continue;

      const dateStr = this.data[i].Date;
      const x = chartLeft + ((i - this.startIndex) * totalWidth) + (totalWidth / 2);

      // Parse date (YYYY-MM-DD format)
      const parts = dateStr.split('-');
      if (parts.length !== 3) continue;

      const year = parseInt(parts[0]);
      const month = parseInt(parts[1]);
      const day = parseInt(parts[2]);

      // Check if this is the start of a new month
      const isNewMonth = (lastMonth === null || lastMonth !== month);

      if (isNewMonth) {
        // Draw month label
        const monthLabel = monthNames[month - 1];
        ctx.font = 'bold 11px Arial';

        // Draw background
        const textWidth = ctx.measureText(monthLabel).width;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x - textWidth / 2 - 3, labelY - 12, textWidth + 6, 16);

        // Draw month text
        ctx.fillStyle = '#00bfff'; // Blue for month labels
        ctx.fillText(monthLabel, x, labelY);

        lastMonth = month;
      } else if (dayInterval > 0 && day % dayInterval === 0) {
        // Draw day number (if zoom level allows)
        ctx.font = '11px Arial';

        const dayLabel = day.toString();

        // Draw background
        const textWidth = ctx.measureText(dayLabel).width;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x - textWidth / 2 - 2, labelY - 12, textWidth + 4, 16);

        // Draw day text
        ctx.fillStyle = '#a0a0a0'; // Dimmer for day numbers
        ctx.fillText(dayLabel, x, labelY);
      }
    }
  }

  /**
   * Draw title
   */
  drawTitle() {
    const ctx = this.ctx;
    ctx.fillStyle = this.colors.textBright;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`${this.symbol} - Daily`, this.width / 2, 30);
  }

  /**
   * Get formatted date/time string for a candle
   * Format depends on timeframe interval
   */
  getFormattedDateTime(dateStr) {
    if (!dateStr) return '';

    try {
      const date = new Date(dateStr);

      // Different formats based on interval
      if (this.timeframeInterval === '1m' || this.timeframeInterval === '5m' ||
          this.timeframeInterval === '15m' || this.timeframeInterval === '30m') {
        // Intraday: show time
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
      } else if (this.timeframeInterval === '1h') {
        // Hour: show date and hour
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        const hours = date.getHours().toString().padStart(2, '0');
        return `${month}/${day} ${hours}:00`;
      } else {
        // Daily/Weekly/Monthly: show date
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        const year = date.getFullYear();
        return `${month}/${day}/${year}`;
      }
    } catch (e) {
      return dateStr;
    }
  }

  /**
   * Draw crosshair (TradingView style)
   */
  drawCrosshair() {
    const ctx = this.ctx;
    const chartTop = this.margin.top;
    const chartBottom = this.height - this.margin.bottom; // Full height (unified canvas)
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    // Save context state
    ctx.save();

    // Make crosshair highly visible - bright color
    ctx.strokeStyle = '#aaaaaa'; // Very bright gray
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Vertical line (full height)
    ctx.beginPath();
    ctx.moveTo(this.mouseX, chartTop);
    ctx.lineTo(this.mouseX, chartBottom);
    ctx.stroke();

    // Horizontal line (full width)
    ctx.beginPath();
    ctx.moveTo(chartLeft, this.mouseY);
    ctx.lineTo(chartRight, this.mouseY);
    ctx.stroke();

    // Restore context state
    ctx.restore();

    // Draw price label on crosshair (right side)
    const price = this.yToPrice(this.mouseY);
    if (price >= this.minPrice && price <= this.maxPrice) {
      const priceText = price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

      // Background box
      ctx.fillStyle = '#333333';
      const textWidth = ctx.measureText(priceText).width;
      ctx.fillRect(chartRight + 5, this.mouseY - 10, 75, 20);

      // Price text
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(priceText, chartRight + 10, this.mouseY + 4);
    }

    // Draw date/time label at bottom of crosshair (TradingView style)
    if (this.hoveredIndex >= 0 && this.hoveredIndex < this.data.length) {
      const dataIndex = Math.floor(this.hoveredIndex);
      if (dataIndex >= 0 && dataIndex < this.data.length) {
        const candle = this.data[dataIndex];
        const dateTimeText = this.getFormattedDateTime(candle.Date);

        // Measure text for background box
        ctx.font = 'bold 11px Arial';
        const textWidth = ctx.measureText(dateTimeText).width;
        const boxWidth = textWidth + 12;
        const boxHeight = 20;

        // Position at bottom of chart, centered on crosshair
        const boxX = this.mouseX - (boxWidth / 2);
        const boxY = chartBottom + 5;

        // Background box
        ctx.fillStyle = '#333333';
        ctx.fillRect(boxX, boxY, boxWidth, boxHeight);

        // Border
        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 1;
        ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

        // Date/time text
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(dateTimeText, this.mouseX, boxY + 14);
      }
    }
  }

  /**
   * Convert Y coordinate to price
   */
  yToPrice(y) {
    const chartTop = this.margin.top;
    const relativeY = y - chartTop;
    const normalized = 1 - (relativeY / this.chartHeight); // Flip because Y increases downward
    return this.minPrice + (normalized * (this.maxPrice - this.minPrice));
  }

  /**
   * Draw hover info box
   */
  drawHoverInfo() {
    if (this.hoveredIndex < 0 || this.hoveredIndex >= this.data.length) return;

    // Floor the index for array access (may be fractional from smooth panning)
    const dataIndex = Math.floor(this.hoveredIndex);
    if (dataIndex < 0 || dataIndex >= this.data.length) return;

    const candle = this.data[dataIndex];

    // Update external OHLCV data bar if it exists
    const dataBar = document.getElementById('tos-chart-data-bar');
    if (dataBar) {
      const isLatest = dataIndex === this.data.length - 1;
      const prefix = isLatest ? '<span style="color: #00bfff; margin-right: 10px;">LATEST:</span>' : '';

      const change = candle.Close - candle.Open;
      const changePercent = ((change / candle.Open) * 100).toFixed(2);
      const changeColor = change >= 0 ? '#00c851' : '#ff4444';

      const volumeStr = candle.Volume
        ? `<span style="color: #a0a0a0;">Vol: <span style="color: #e0e0e0;">${candle.Volume.toLocaleString()}</span></span>`
        : '';

      dataBar.innerHTML = `
        ${prefix}
        <span style="color: #a0a0a0;">Date: <span style="color: #e0e0e0;">${candle.Date}</span></span>
        <span style="color: #a0a0a0;">O: <span style="color: #e0e0e0;">${candle.Open.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">H: <span style="color: #e0e0e0;">${candle.High.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">L: <span style="color: #e0e0e0;">${candle.Low.toFixed(2)}</span></span>
        <span style="color: #a0a0a0;">C: <span style="color: #e0e0e0;">${candle.Close.toFixed(2)}</span></span>
        ${volumeStr}
        <span style="color: ${changeColor};">${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent}%)</span>
      `;
    }
  }

  /**
   * Draw TradingView-style price markers (Ask, Bid, High, Low, Current)
   */
  drawPriceMarkers() {
    const ctx = this.ctx;
    const chartTop = this.margin.top;
    const chartRight = this.width - this.margin.right;

    // Get the last candle (most recent)
    const lastCandle = this.data[this.data.length - 1];
    if (!lastCandle) return;

    // Use live price if available, otherwise use last candle close
    const currentPrice = this.livePrice || lastCandle.Close;
    const currentY = this.priceToY(currentPrice);

    // Draw horizontal dashed line at current price (extends across entire chart)
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(this.margin.left, currentY);
    ctx.lineTo(chartRight, currentY);
    ctx.stroke();
    ctx.setLineDash([]); // Reset to solid lines

    // Calculate visible range high and low
    const clampedStart = Math.max(0, this.startIndex);
    const clampedEnd = Math.min(this.data.length - 1, this.endIndex);
    const visibleData = this.data.slice(clampedStart, clampedEnd + 1);

    const visibleHigh = Math.max(...visibleData.map(d => d.High));
    const visibleLow = Math.min(...visibleData.map(d => d.Low));

    const highY = this.priceToY(visibleHigh);
    const lowY = this.priceToY(visibleLow);

    // Helper function to draw price marker box
    const drawPriceBox = (y, price, label, bgColor, textColor = '#ffffff') => {
      const priceText = price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      const padding = 6;
      const boxHeight = 24;
      const boxWidth = 85;

      // Draw background box
      ctx.fillStyle = bgColor;
      ctx.fillRect(chartRight + 5, y - boxHeight / 2, boxWidth, boxHeight);

      // Draw label text (smaller, top line)
      ctx.fillStyle = textColor;
      ctx.font = 'bold 9px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(label, chartRight + 10, y - 4);

      // Draw price text (larger, bottom line)
      ctx.font = 'bold 11px Arial';
      ctx.fillText(priceText, chartRight + 10, y + 10);
    };

    // Draw High marker (blue box at top)
    drawPriceBox(highY, visibleHigh, 'High', '#1976d2');

    // Draw Low marker (blue box at bottom)
    drawPriceBox(lowY, visibleLow, 'Low', '#1976d2');

    // Draw current price marker (green if up, red if down)
    const isUp = currentPrice >= lastCandle.Open;
    const currentBgColor = isUp ? '#00c851' : '#ff4444';
    drawPriceBox(currentY, currentPrice, 'Last', currentBgColor);

    // Optional: Draw simulated Bid/Ask based on last price with small spread
    // For crypto, typical spread is ~0.01-0.05%
    const spread = currentPrice * 0.0002; // 0.02% spread
    const askPrice = currentPrice + spread;
    const bidPrice = currentPrice - spread;
    const askY = this.priceToY(askPrice);
    const bidY = this.priceToY(bidPrice);

    // Only draw if they're visible and not too close to other markers
    const minDistance = 30; // minimum pixels between markers
    if (Math.abs(askY - highY) > minDistance && Math.abs(askY - currentY) > minDistance) {
      drawPriceBox(askY, askPrice, 'Ask', '#d32f2f');
    }
    if (Math.abs(bidY - lowY) > minDistance && Math.abs(bidY - currentY) > minDistance) {
      drawPriceBox(bidY, bidPrice, 'Bid', '#1565c0');
    }
  }

  /**
   * Draw live price lines (current price, chart high, chart low)
   */
  drawLivePriceLines() {
    if (this.data.length === 0) return;

    const ctx = this.ctx;
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    // Calculate visible range high and low (same as in drawPriceMarkers)
    const clampedStart = Math.max(0, this.startIndex);
    const clampedEnd = Math.min(this.data.length - 1, this.endIndex);
    const visibleData = this.data.slice(clampedStart, clampedEnd + 1);

    const visibleHigh = Math.max(...visibleData.map(d => d.High));
    const visibleLow = Math.min(...visibleData.map(d => d.Low));

    // Draw chart high line (orange) - ALWAYS
    const highY = this.priceToY(visibleHigh);
    ctx.strokeStyle = this.colors.highLine;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(chartLeft, highY);
    ctx.lineTo(chartRight, highY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw chart low line (purple) - ALWAYS
    const lowY = this.priceToY(visibleLow);
    ctx.strokeStyle = this.colors.lowLine;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(chartLeft, lowY);
    ctx.lineTo(chartRight, lowY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw live price line (cyan/teal) - ONLY if we have live price
    if (this.livePrice) {
      const livePriceY = this.priceToY(this.livePrice);
      ctx.strokeStyle = this.colors.livePriceLine;
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 4]);
      ctx.beginPath();
      ctx.moveTo(chartLeft, livePriceY);
      ctx.lineTo(chartRight, livePriceY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw label for live price
      ctx.fillStyle = this.colors.livePriceLine;
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(`LIVE: $${this.livePrice.toFixed(2)}`, chartLeft + 10, livePriceY - 5);
    }
  }

  /**
   * Convert price to Y coordinate
   */
  priceToY(price) {
    const chartTop = this.margin.top;
    const normalized = (price - this.minPrice) / (this.maxPrice - this.minPrice);
    return chartTop + this.chartHeight - (normalized * this.chartHeight);
  }

  /**
   * Convert X coordinate to data index
   */
  xToIndex(x) {
    const chartLeft = this.margin.left;
    const chartWidth = this.width - this.margin.left - this.margin.right;
    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles; // Must match drawing logic
    const spacing = totalWidth - (totalWidth * 0.7); // Match candle spacing
    const offset = spacing / 2 + (totalWidth * 0.7) / 2; // Center offset (matches indexToX)

    const relativeX = x - chartLeft - offset; // Account for centering offset
    const candleIndex = Math.floor(relativeX / totalWidth);
    const index = this.startIndex + candleIndex;

    return Math.max(this.startIndex, Math.min(this.endIndex, index));
  }

  /**
   * Convert data index to X coordinate
   */
  indexToX(index) {
    const chartLeft = this.margin.left;
    const chartWidth = this.width - this.margin.left - this.margin.right;
    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles;
    const spacing = totalWidth - (totalWidth * 0.7); // Match candle spacing

    return chartLeft + ((index - this.startIndex) * totalWidth) + spacing / 2 + (totalWidth * 0.7) / 2;
  }

  /**
   * Setup mouse events
   */
  setupEvents() {
    // Initialize selection manager for drawing editing
    if (!this.selectionManager) {
      this.selectionManager = new SelectionManager(this);
      console.log('✅ SelectionManager initialized');
    }

    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
    this.canvas.addEventListener('mouseleave', (e) => this.onMouseLeave(e));
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e));

    // Listen for drawing tool actions from tool panel
    this.canvas.addEventListener('tool-action', (e) => this.onToolAction(e));

    // Global mouseup to catch release outside canvas
    window.addEventListener('mouseup', (e) => this.onMouseUp(e));

    window.addEventListener('resize', () => {
      this.resize();
      this.calculateVolumeRange(); // Only update volume, preserve price pan
      this.draw();
    });
  }

  /**
   * Mouse move handler
   */
  onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouseX = e.clientX - rect.left;
    this.mouseY = e.clientY - rect.top;

    // Let SelectionManager handle first (for drawing move/resize)
    if (this.selectionManager && this.selectionManager.onMouseMove(e, this.mouseX, this.mouseY)) {
      return; // SelectionManager handled the event
    }

    // Y-axis expand/compress
    if (this.isDraggingYAxis) {
      const dy = this.mouseY - this.yAxisDragStartY;
      // Drag up = negative dy = expand (increase range)
      // Drag down = positive dy = compress (decrease range)
      const scaleFactor = 1 - (dy / 200); // 200px drag = 2x change
      const newRange = this.yAxisDragStartRange * scaleFactor;

      // Apply new range while keeping center fixed
      this.minPrice = this.yAxisDragStartCenter - (newRange / 2);
      this.maxPrice = this.yAxisDragStartCenter + (newRange / 2);

      console.log(`📏 Y-axis: dy=${dy.toFixed(0)}px, scale=${scaleFactor.toFixed(2)}x, range=${newRange.toFixed(0)}`);
    }
    // X-axis expand/compress
    else if (this.isDraggingXAxis) {
      const dx = this.mouseX - this.xAxisDragStartX;
      // Drag left = negative dx = expand (show more candles)
      // Drag right = positive dx = compress (show fewer candles)
      const scaleFactor = 1 + (dx / 200); // 200px drag = 2x change
      let newVisibleCandles = Math.floor(this.xAxisDragStartVisibleCandles * scaleFactor);

      // Minimum 10 visible candles
      newVisibleCandles = Math.max(10, Math.min(this.data.length, newVisibleCandles));

      // Keep center roughly the same
      const center = (this.startIndex + this.endIndex) / 2;
      this.startIndex = Math.max(0, Math.floor(center - newVisibleCandles / 2));
      this.endIndex = Math.min(this.data.length - 1, this.startIndex + newVisibleCandles);

      // Update volume range for new visible candles
      this.calculateVolumeRange();

      console.log(`📏 X-axis: dx=${dx.toFixed(0)}px, scale=${scaleFactor.toFixed(2)}x, candles=${newVisibleCandles}`);
    }
    // Pan if dragging
    else if (this.isDragging) {
      const dx = this.mouseX - this.dragStartX;
      const dy = this.mouseY - this.dragStartY;

      // HORIZONTAL PANNING (left/right) - smooth fractional movement
      const chartWidth = this.width - this.margin.left - this.margin.right;
      const visibleCandles = this.endIndex - this.startIndex + 1;
      const totalWidth = chartWidth / visibleCandles;
      const indexDelta = -(dx / totalWidth); // Remove Math.floor for smooth movement
      const newStartIndex = this.dragStartIndex + indexDelta;

      this.startIndex = newStartIndex;
      this.endIndex = this.startIndex + visibleCandles - 1;

      // Detect if user has panned away from live edge (disable auto-scroll)
      // Allow a small tolerance (1 candle) to account for fractional positioning
      const distanceFromLiveEdge = (this.data.length - 1) - this.endIndex;

      if (distanceFromLiveEdge > 1) {
        // User has panned away from live data
        if (this.autoScrollEnabled) {
          this.autoScrollEnabled = false;
          console.log('⏸️ Auto-scroll PAUSED - user panned away from live edge');
        }
      } else if (distanceFromLiveEdge <= 0.5) {
        // User has panned back to live edge
        if (!this.autoScrollEnabled) {
          this.autoScrollEnabled = true;
          console.log('▶️ Auto-scroll RESUMED - user returned to live edge');
        }
      }

      // VERTICAL PANNING (up/down) - TradingView style (unrestricted)
      // Calculate price range shift based on vertical drag
      const priceRange = this.dragStartMaxPrice - this.dragStartMinPrice;
      const pixelsPerPrice = this.chartHeight / priceRange;
      const priceShift = dy / pixelsPerPrice; // Drag down = see higher prices, drag up = see lower prices

      // Apply the shift to both min and max to pan the view (completely unrestricted like TradingView)
      this.minPrice = this.dragStartMinPrice + priceShift;
      this.maxPrice = this.dragStartMaxPrice + priceShift;

      if (Math.abs(indexDelta) > 0.1 || Math.abs(dy) > 1) {
        console.log(`🖱️ Panning: dx=${dx.toFixed(0)}px (${indexDelta.toFixed(2)} candles), dy=${dy.toFixed(0)}px (${priceShift.toFixed(0)} price shift)`);
      }
    } else {
      // Update hovered candle
      this.hoveredIndex = this.xToIndex(this.mouseX);

      // Update cursor based on active tool (when not dragging)
      const activeTool = window.toolRegistry?.getActiveTool();

      // Special handling for eraser tool - highlight drawing under cursor
      if (activeTool && activeTool.id === 'eraser-cursor' && this.selectionManager) {
        const hoveredDrawing = this.selectionManager.findDrawingAtPoint(this.mouseX, this.mouseY);
        if (hoveredDrawing) {
          this.canvas.style.cursor = 'pointer'; // Show clickable cursor
          // Store for highlight rendering
          this.eraserHoveredDrawing = hoveredDrawing;
        } else {
          this.canvas.style.cursor = activeTool.cursorStyle || 'default';
          this.eraserHoveredDrawing = null;
        }
      }
      // Special handling for dot/arrow cursor - show preview
      else if (activeTool && (activeTool.id === 'dot-cursor' || activeTool.id === 'arrow-cursor')) {
        this.eraserHoveredDrawing = null;

        // Get preview from tool's onMouseMove
        const result = activeTool.onMouseMove({ clientX: this.mouseX + this.canvas.getBoundingClientRect().left, clientY: this.mouseY + this.canvas.getBoundingClientRect().top }, {});
        if (result) {
          // Convert to chart coordinates
          const chartX = this.xToIndex(this.mouseX);
          const chartY = this.yToPrice(this.mouseY);
          result.chartIndex = chartX;
          result.chartPrice = chartY;

          // Auto-detect arrow direction for preview
          if (result.action === 'preview-arrow' && this.data.length > 0) {
            const nearestIndex = Math.round(chartX);
            if (nearestIndex >= 0 && nearestIndex < this.data.length) {
              const candle = this.data[nearestIndex];
              const candleHigh = candle.High;
              const candleLow = candle.Low;

              // If arrow is below the candle, point up (bullish)
              // If arrow is above the candle, point down (bearish)
              if (chartY < candleLow) {
                result.direction = 'up';
              } else if (chartY > candleHigh) {
                result.direction = 'down';
              } else {
                // If inside the candle body, use proximity to decide
                const candleMid = (candleHigh + candleLow) / 2;
                result.direction = chartY < candleMid ? 'up' : 'down';
              }
            }
          }

          // Store as preview
          this.previewDrawing = result;
        }

        this.canvas.style.cursor = activeTool.cursorStyle;
      }
      else {
        this.eraserHoveredDrawing = null;
        this.previewDrawing = null; // Clear preview when switching tools
        if (activeTool && activeTool.cursorStyle) {
          this.canvas.style.cursor = activeTool.cursorStyle;
        } else {
          this.canvas.style.cursor = 'default';
        }
      }
    }

    this.draw();
  }

  /**
   * Mouse down handler
   */
  onMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Check if eraser tool is active
    const activeTool = window.toolRegistry?.getActiveTool();
    if (activeTool && activeTool.id === 'eraser-cursor' && this.selectionManager) {
      // Eraser mode: delete drawing on click
      const deletedDrawing = this.selectionManager.findDrawingAtPoint(mouseX, mouseY);
      if (deletedDrawing) {
        const index = this.drawings.indexOf(deletedDrawing);
        if (index > -1) {
          this.drawings.splice(index, 1);
          console.log(`🗑️ Eraser deleted drawing at index ${index} (${this.drawings.length} remaining)`);

          // Delete from backend
          this.deleteDrawing(deletedDrawing.id);

          this.draw();
          return; // Handled by eraser
        }
      }
    }

    // Check if horizontal or vertical line tool is active
    if (activeTool && (activeTool.id === 'horizontal-line' || activeTool.id === 'vertical-line')) {
      const result = activeTool.onClick(e, {});
      if (result) {
        // Convert to chart coordinates
        if (result.action === 'place-horizontal-line') {
          result.price = this.yToPrice(mouseY);
          delete result.y; // Remove screen coordinate

          // Smart support/resistance detection
          const srType = this.detectSupportResistance(result.price);
          if (srType === 'support') {
            result.lineColor = '#00c851'; // Green for support
          } else if (srType === 'resistance') {
            result.lineColor = '#ff4444'; // Red for resistance
          } else {
            result.lineColor = '#ff9800'; // Orange for neutral
          }
          console.log(`✅ Horizontal line detected as: ${srType}`);
        } else if (result.action === 'place-vertical-line') {
          result.chartIndex = this.xToIndex(mouseX);
          delete result.x; // Remove screen coordinate
        }

        // Store as a completed drawing
        this.drawings.push(result);
        console.log(`✅ Placed ${activeTool.id} at chart coords`);

        // Save to backend
        this.saveDrawing(result);

        this.draw();
        return; // Handled
      }
    }

    // Check if dot or arrow cursor is active (place markers on click)
    if (activeTool && (activeTool.id === 'dot-cursor' || activeTool.id === 'arrow-cursor')) {
      // Get the result from the tool's onClick handler
      const result = activeTool.onClick(e, {});
      if (result) {
        // Convert to chart coordinates
        const chartX = this.xToIndex(mouseX);
        const chartY = this.yToPrice(mouseY);

        // Add chart coordinates to the action
        result.chartIndex = chartX;
        result.chartPrice = chartY;

        // Auto-detect arrow direction based on position relative to candles
        if (result.action === 'place-arrow' && this.data.length > 0) {
          const nearestIndex = Math.round(chartX);
          if (nearestIndex >= 0 && nearestIndex < this.data.length) {
            const candle = this.data[nearestIndex];
            const candleHigh = candle.High;
            const candleLow = candle.Low;

            // If arrow is below the candle, point up (bullish)
            // If arrow is above the candle, point down (bearish)
            if (chartY < candleLow) {
              result.direction = 'up';
            } else if (chartY > candleHigh) {
              result.direction = 'down';
            } else {
              // If inside the candle body, use proximity to decide
              const candleMid = (candleHigh + candleLow) / 2;
              result.direction = chartY < candleMid ? 'up' : 'down';
            }
          }
        }

        // Store as a completed drawing
        this.drawings.push(result);
        console.log(`✅ Placed ${activeTool.id} at chart coords [${chartX}, $${chartY?.toFixed(2)}]`, result.direction ? `direction: ${result.direction}` : '');

        // Save to backend
        this.saveDrawing(result);

        this.draw();
        return; // Handled
      }
    }

    // Let SelectionManager handle first (for drawing selection/editing)
    if (this.selectionManager && this.selectionManager.onMouseDown(e, mouseX, mouseY)) {
      console.log('✅ SelectionManager handled mouseDown');
      return; // SelectionManager handled the event
    }
    console.log('⏭️ SelectionManager did not handle mouseDown, continuing to pan/zoom logic');

    // Check if any drawing tool is active (other than default cursor modes)
    // If so, prevent panning - the tool is still being drawn
    if (activeTool && activeTool.id !== 'default-cursor' && activeTool.id !== 'eraser-cursor') {
      console.log(`🚫 Drawing tool '${activeTool.id}' is active - panning disabled`);
      return; // Don't allow panning while drawing
    }

    // Check if clicking on Y-axis (price axis on right)
    const chartRight = this.width - this.margin.right;
    const chartTop = this.margin.top;
    const chartBottom = this.height - this.margin.bottom;

    // Check if clicking on X-axis (time axis on bottom)
    const chartLeft = this.margin.left;

    if (mouseX >= chartRight && mouseY >= chartTop && mouseY <= chartBottom) {
      // Clicked on Y-axis (price axis)
      this.isDraggingYAxis = true;
      this.yAxisDragStartY = mouseY;
      this.yAxisDragStartRange = this.maxPrice - this.minPrice;
      this.yAxisDragStartCenter = (this.maxPrice + this.minPrice) / 2;
      this.canvas.style.cursor = 'ns-resize';
      console.log(`📏 Y-axis expand/compress started`);
    } else if (mouseY >= chartBottom && mouseX >= chartLeft && mouseX <= chartRight) {
      // Clicked on X-axis (time axis)
      this.isDraggingXAxis = true;
      this.xAxisDragStartX = mouseX;
      this.xAxisDragStartVisibleCandles = this.endIndex - this.startIndex;
      this.canvas.style.cursor = 'ew-resize';
      console.log(`📏 X-axis expand/compress started`);
    } else {
      // Normal chart panning
      this.isDragging = true;
      this.dragStartX = mouseX;
      this.dragStartY = mouseY;
      this.dragStartIndex = this.startIndex;
      this.dragStartMinPrice = this.minPrice;
      this.dragStartMaxPrice = this.maxPrice;
      this.canvas.style.cursor = 'grabbing';
      console.log(`🖱️ Pan started at index ${this.startIndex}, price range: ${this.minPrice.toFixed(0)} - ${this.maxPrice.toFixed(0)}`);
    }

    // Prevent text selection while dragging
    e.preventDefault();
  }

  /**
   * Mouse up handler
   */
  onMouseUp(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Let SelectionManager handle first (for finishing drawing move/resize)
    if (this.selectionManager && this.selectionManager.onMouseUp(e, mouseX, mouseY)) {
      return; // SelectionManager handled the event
    }

    this.isDragging = false;
    this.isDraggingYAxis = false;
    this.isDraggingXAxis = false;
    // Restore cursor based on active tool
    const activeTool = window.toolRegistry?.getActiveTool();
    if (activeTool && activeTool.cursorStyle) {
      this.canvas.style.cursor = activeTool.cursorStyle;
    } else {
      this.canvas.style.cursor = 'default';
    }
  }

  /**
   * Mouse leave handler
   */
  onMouseLeave(e) {
    this.mouseX = -1;
    this.mouseY = -1;
    this.hoveredIndex = -1;
    this.isDragging = false;
    this.isDraggingYAxis = false;
    this.isDraggingXAxis = false;
    // Restore cursor based on active tool
    const activeTool = window.toolRegistry?.getActiveTool();
    if (activeTool && activeTool.cursorStyle) {
      this.canvas.style.cursor = activeTool.cursorStyle;
    } else {
      this.canvas.style.cursor = 'default';
    }
    this.draw();
  }

  /**
   * Handle drawing tool actions from tool panel
   */
  onToolAction(e) {
    const action = e.detail;

    if (!action || !action.action) {
      console.warn('⚠️ Invalid tool action:', action);
      return;
    }

    // Ignore "start" actions, "hover" actions, "cancel" actions, and cursor actions first
    const ignoreActions = [
      'start-trend-line', 'start-ray-line', 'start-extended-line',
      'start-gann-box', 'start-gann-fan', 'start-gann-square', 'start-gann-angles',
      'start-parallel-channel', 'start-parallel-channel-line1',
      'start-fibonacci-retracement', 'start-fibonacci-extension-p1', 'start-fibonacci-extension-p2',
      'start-fibonacci-fan', 'start-fibonacci-arcs', 'start-fibonacci-spiral',
      'start-rectangle', 'start-circle', 'start-ellipse',
      'start-callout',
      'hover', 'cancel-drawing', 'check-selection'
    ];

    // Silently ignore these actions (don't process or save them)
    if (ignoreActions.includes(action.action)) {
      return;
    }

    // Check if this is a preview action (log less verbosely)
    const isPreviewAction = action.action.startsWith('preview-');

    if (!isPreviewAction) {
      console.log('🎨 Tool action received:', action.action, action);
    } else if (action.action === 'preview-double-top-bottom') {
      console.log('👁️ Preview:', action.action, 'points:', action.points?.length, 'step:', action.step);
    }

    // Check if this is a preview action or a completed drawing
    const isPreview = action.action.startsWith('preview-') ||
                      action.action.includes('-point-') ||
                      action.action === 'update-polygon';

    // Convert screen coordinates to chart coordinates for line-based drawings
    let convertedAction = action;
    const needsConversion = action.action.includes('trend-line') ||
        action.action.includes('ray-line') ||
        action.action.includes('extended-line') ||
        action.action.includes('parallel-channel') ||
        action.action.includes('fibonacci-') ||
        action.action.includes('gann-') ||
        action.action.includes('head-and-shoulders') ||
        action.action.includes('triangle') ||
        action.action.includes('wedge') ||
        action.action.includes('double-top-bottom') ||
        action.action.includes('rectangle') ||
        action.action.includes('circle') ||
        action.action.includes('ellipse') ||
        action.action.includes('polygon');

    if (needsConversion) {
      convertedAction = this.convertToChartCoordinates(action);
    }

    if (isPreview) {
      // Update preview drawing
      this.previewDrawing = convertedAction;
    } else {
      // Completed drawing - add to drawings array
      this.drawings.push(convertedAction);
      this.previewDrawing = null; // Clear preview
      console.log(`✅ Drawing added: ${convertedAction.action} (total: ${this.drawings.length})`, convertedAction);

      // Save to backend
      this.saveDrawing(convertedAction);
    }

    // Redraw chart with new drawing/preview
    this.draw();
  }

  /**
   * Convert screen coordinates to chart coordinates (index and price)
   */
  convertToChartCoordinates(action) {
    const converted = { ...action };

    // Handle parallel channel structure (line1Start, line1End, parallelY)
    if (action.action && action.action.includes('parallel-channel')) {
      if (action.line1Start) {
        converted.startIndex = this.xToIndex(action.line1Start.x);
        converted.startPrice = this.yToPrice(action.line1Start.y);
        delete converted.line1Start;
      }
      if (action.line1End) {
        converted.endIndex = this.xToIndex(action.line1End.x);
        converted.endPrice = this.yToPrice(action.line1End.y);
        delete converted.line1End;
      }
      if (action.parallelY !== undefined) {
        converted.parallelPrice = this.yToPrice(action.parallelY);
        delete converted.parallelY;
        delete converted.parallelX;
      }
    } else if (action.action && action.action.includes('fibonacci-extension')) {
      // Handle Fibonacci Extension 3-point structure (point1, point2, point3)
      if (action.point1) {
        converted.point1Index = this.xToIndex(action.point1.x);
        converted.point1Price = this.yToPrice(action.point1.y);
        delete converted.point1;
      }
      if (action.point2) {
        converted.point2Index = this.xToIndex(action.point2.x);
        converted.point2Price = this.yToPrice(action.point2.y);
        delete converted.point2;
      }
      if (action.point3) {
        converted.point3Index = this.xToIndex(action.point3.x);
        converted.point3Price = this.yToPrice(action.point3.y);
        delete converted.point3;
      }
    } else if (action.points && Array.isArray(action.points) && (
      action.action.includes('head-and-shoulders') ||
      action.action.includes('triangle') ||
      action.action.includes('wedge') ||
      action.action.includes('double-top-bottom') ||
      action.action.includes('polygon')
    )) {
      // Handle pattern tools and polygon with points array
      converted.chartPoints = action.points.map(point => {
        const rawIndex = this.xToIndex(point.x);
        const rawPrice = this.yToPrice(point.y);

        // Snap to nearest candle high/low for pattern tools (not polygon)
        if (action.action.includes('head-and-shoulders') ||
            action.action.includes('triangle') ||
            action.action.includes('wedge') ||
            action.action.includes('double-top-bottom')) {
          const snappedIndex = Math.max(0, Math.min(Math.round(rawIndex), this.data.length - 1));
          const candle = this.data[snappedIndex];

          if (candle) {
            // Snap to closest of high or low
            const distToHigh = Math.abs(rawPrice - candle.High);
            const distToLow = Math.abs(rawPrice - candle.Low);
            const snappedPrice = distToHigh < distToLow ? candle.High : candle.Low;

            return {
              chartIndex: snappedIndex,
              chartPrice: snappedPrice
            };
          }
        }

        // No snapping for polygon or if no candle data
        return {
          chartIndex: rawIndex,
          chartPrice: rawPrice
        };
      });
      // Keep currentX/currentY for preview rendering (they show mouse position)
      // Only delete the points array to avoid confusion
      delete converted.points;
    } else if (action.action && action.action.includes('circle') && action.centerX !== undefined && action.centerY !== undefined) {
      // Handle circle with centerX/centerY/radius
      converted.centerIndex = this.xToIndex(action.centerX);
      converted.centerPrice = this.yToPrice(action.centerY);
      // Radius needs special handling - convert using pixel distance
      const radiusInCanvasUnits = action.radius;
      // Store radius in chart units (approximate using price scale)
      const priceRange = this.maxPrice - this.minPrice;
      const canvasHeight = this.canvas.height;
      converted.radiusInPriceUnits = (radiusInCanvasUnits / canvasHeight) * priceRange;
      // Remove screen coordinates
      delete converted.centerX;
      delete converted.centerY;
      delete converted.radius;
    } else {
      // Handle standard line structure (startX/Y, endX/Y)
      // Convert start point
      if (action.startX !== undefined && action.startY !== undefined) {
        converted.startIndex = this.xToIndex(action.startX);
        converted.startPrice = this.yToPrice(action.startY);
        // Remove screen coordinates to avoid confusion
        delete converted.startX;
        delete converted.startY;
      }

      // Convert end point
      if (action.endX !== undefined && action.endY !== undefined) {
        converted.endIndex = this.xToIndex(action.endX);
        converted.endPrice = this.yToPrice(action.endY);
        // Remove screen coordinates to avoid confusion
        delete converted.endX;
        delete converted.endY;
      }

      // Handle single point tools (like Fibonacci Time Zones) - x, y only
      if (action.x !== undefined && action.y !== undefined && !action.startX && !action.endX) {
        converted.chartIndex = this.xToIndex(action.x);
        converted.chartPrice = this.yToPrice(action.y);
        // Remove screen coordinates to avoid confusion
        delete converted.x;
        delete converted.y;
      }

    }

    return converted;
  }

  /**
   * Wheel handler (zoom) - preserves vertical pan position
   */
  onWheel(e) {
    e.preventDefault();

    const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
    const visibleCandles = this.endIndex - this.startIndex;
    const newVisibleCandles = Math.floor(visibleCandles * zoomFactor);

    // Keep center roughly the same
    const center = (this.startIndex + this.endIndex) / 2;
    this.startIndex = Math.max(0, Math.floor(center - newVisibleCandles / 2));
    this.endIndex = Math.min(this.data.length - 1, this.startIndex + newVisibleCandles);

    // Ensure minimum visible candles (adaptive to data size)
    // For charts with few candles, allow showing all of them
    const minVisibleCandles = Math.min(10, Math.max(2, this.data.length));
    if (this.endIndex - this.startIndex < minVisibleCandles) {
      this.endIndex = Math.min(this.data.length - 1, this.startIndex + minVisibleCandles);
    }

    // If we still can't meet minimum, show all available data
    if (this.endIndex - this.startIndex < 2 && this.data.length > 0) {
      this.startIndex = 0;
      this.endIndex = this.data.length - 1;
    }

    // Only update volume range, preserve manually panned price ranges
    this.calculateVolumeRange();
    this.draw();
  }

  /**
   * Update live price for the last candle
   * @param {number} price - Current price
   * @param {number} volume - Current 24h volume (optional)
   */
  updateLivePrice(price, volume = null) {
    // console.log(`🖼️ CanvasRenderer.updateLivePrice called: price=${price}, volume=${volume}, data.length=${this.data.length}`);

    if (this.data.length === 0) {
      // console.log('  ⚠️ No data loaded, skipping update');
      return false;
    }

    const lastIndex = this.data.length - 1;
    const lastCandle = this.data[lastIndex];
    // console.log(`  📊 Updating last candle [${lastIndex}]: old Close=${lastCandle.Close}, new Close=${price}`);

    // Set live price for the live price line
    this.livePrice = price;

    // Update last candle price
    lastCandle.Close = price;
    lastCandle.High = Math.max(lastCandle.High, price);
    lastCandle.Low = Math.min(lastCandle.Low, price);

    // Update volume if provided
    if (volume !== null) {
      lastCandle.Volume = volume;
    }

    // Only update volume range, preserve price pan position
    this.calculateVolumeRange();

    // Redraw
    this.draw();
    // console.log('  ✅ Chart redrawn with new live price');

    return true;
  }

  /**
   * Draw all completed drawings
   */
  drawAllDrawings() {
    if (!this.ctx || this.drawings.length === 0) return;

    // Get the currently selected drawing from SelectionManager
    const selectedDrawing = this.selectionManager?.getSelectedDrawing();

    this.drawings.forEach(drawing => {
      // Highlight drawing if selected (turns yellow)
      const isSelected = selectedDrawing === drawing;

      // Highlight drawing if hovered by eraser tool
      const isEraserHovered = this.eraserHoveredDrawing === drawing;

      // Priority: selected > eraser hovered
      let highlightColor = null;
      if (isSelected) {
        highlightColor = '#ffeb3b'; // Yellow for selected
      } else if (isEraserHovered) {
        // Use yellow for dots/arrows (so we don't confuse with red arrows)
        // Use red for trend lines (original behavior)
        if (drawing.action === 'place-dot' || drawing.action === 'place-arrow') {
          highlightColor = '#ffeb3b'; // Yellow
        } else {
          highlightColor = '#ff4444'; // Red
        }
      }

      this.drawSingleDrawing(drawing, highlightColor);
    });
  }

  /**
   * Draw preview of current drawing
   */
  drawPreview(drawing) {
    if (!this.ctx || !drawing) return;

    // Draw with lower opacity for preview
    const originalAlpha = this.ctx.globalAlpha;
    this.ctx.globalAlpha = 0.5;
    this.drawSingleDrawing(drawing);
    this.ctx.globalAlpha = originalAlpha;
  }

  /**
   * Draw a single drawing (used for both completed and preview)
   * @param {Object} drawing - Drawing object to render
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawSingleDrawing(drawing, overrideColor = null) {
    if (!drawing || !drawing.action) return;

    const ctx = this.ctx;

    // Route to specific drawing method based on action
    if (drawing.action.includes('trend-line')) {
      this.drawTrendLine(drawing, overrideColor);
    } else if (drawing.action === 'place-dot' || drawing.action === 'preview-dot') {
      this.drawDot(drawing, overrideColor);
    } else if (drawing.action === 'place-arrow' || drawing.action === 'preview-arrow') {
      this.drawArrowMarker(drawing, overrideColor);
    } else if (drawing.action.includes('horizontal-line')) {
      this.drawHorizontalLine(drawing, overrideColor);
    } else if (drawing.action.includes('vertical-line')) {
      this.drawVerticalLine(drawing, overrideColor);
    } else if (drawing.action.includes('ray-line')) {
      this.drawRayLine(drawing, overrideColor);
    } else if (drawing.action.includes('extended-line')) {
      this.drawExtendedLine(drawing, overrideColor);
    } else if (drawing.action.includes('parallel-channel')) {
      this.drawParallelChannel(drawing, overrideColor);
    } else if (drawing.action.includes('fibonacci-retracement')) {
      this.drawFibonacciRetracement(drawing, overrideColor);
    } else if (drawing.action === 'preview-fibonacci-extension-p1') {
      // Draw first line preview for Fibonacci Extension
      this.drawSimpleLine(drawing, overrideColor || '#9c27b0');
    } else if (drawing.action.includes('fibonacci-extension')) {
      this.drawFibonacciExtension(drawing, overrideColor);
    } else if (drawing.action.includes('fibonacci-fan')) {
      this.drawFibonacciFan(drawing, overrideColor);
    } else if (drawing.action.includes('fibonacci-arcs')) {
      this.drawFibonacciArcs(drawing, overrideColor);
    } else if (drawing.action.includes('fibonacci-time-zones')) {
      this.drawFibonacciTimeZones(drawing, overrideColor);
    } else if (drawing.action.includes('fibonacci-spiral')) {
      this.drawFibonacciSpiral(drawing, overrideColor);
    } else if (drawing.action.includes('gann-fan')) {
      this.drawGannFan(drawing, overrideColor);
    } else if (drawing.action.includes('gann-box')) {
      this.drawGannBox(drawing, overrideColor);
    } else if (drawing.action.includes('gann-square')) {
      this.drawGannSquare(drawing, overrideColor);
    } else if (drawing.action.includes('gann-angles')) {
      this.drawGannAngles(drawing, overrideColor);
    } else if (drawing.action.includes('head-and-shoulders')) {
      this.drawHeadAndShoulders(drawing, overrideColor);
    } else if (drawing.action.includes('triangle')) {
      this.drawTriangle(drawing, overrideColor);
    } else if (drawing.action.includes('wedge')) {
      this.drawWedge(drawing, overrideColor);
    } else if (drawing.action.includes('double-top-bottom')) {
      this.drawDoubleTopBottom(drawing, overrideColor);
    } else if (drawing.action.includes('rectangle')) {
      this.drawRectangle(drawing, overrideColor);
    } else if (drawing.action.includes('circle')) {
      this.drawCircle(drawing, overrideColor);
    } else if (drawing.action.includes('ellipse')) {
      this.drawEllipse(drawing, overrideColor);
    } else if (drawing.action.includes('polygon')) {
      this.drawPolygon(drawing, overrideColor);
    } else if (drawing.action.includes('text-label')) {
      this.drawTextLabel(drawing);
    } else if (drawing.action.includes('callout')) {
      this.drawCallout(drawing);
    } else if (drawing.action.includes('note')) {
      this.drawNote(drawing);
    } else if (drawing.action.includes('price-label')) {
      this.drawPriceLabel(drawing);
    }
  }

  // ==================== CURSOR MARKER METHODS ====================

  /**
   * Draw a dot marker
   * @param {Object} drawing - Drawing object with dot properties
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawDot(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { color, size, chartIndex, chartPrice } = drawing;

    // Convert chart coordinates to screen coordinates
    const x = this.indexToX(chartIndex);
    const y = this.priceToY(chartPrice);

    ctx.fillStyle = overrideColor || color || '#00bfff';
    ctx.beginPath();
    ctx.arc(x, y, size || 6, 0, Math.PI * 2);
    ctx.fill();
  }

  /**
   * Draw an arrow marker
   * @param {Object} drawing - Drawing object with arrow properties
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawArrowMarker(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { color, size, direction, chartIndex, chartPrice } = drawing;

    // Convert chart coordinates to screen coordinates
    const x = this.indexToX(chartIndex);
    const y = this.priceToY(chartPrice);

    const arrowSize = size || 20;
    const halfSize = arrowSize / 2;

    // Determine arrow color
    let arrowColor = overrideColor; // Override takes precedence (for eraser highlight)
    if (!arrowColor) {
      // Use custom color if provided, otherwise auto-color based on direction
      arrowColor = color || (direction === 'down' ? '#ff4444' : '#00c851');
    }

    ctx.fillStyle = arrowColor;
    ctx.beginPath();

    // Draw arrow based on direction
    switch (direction) {
      case 'up':
        ctx.moveTo(x, y - halfSize);
        ctx.lineTo(x + halfSize * 0.6, y + halfSize);
        ctx.lineTo(x - halfSize * 0.6, y + halfSize);
        break;
      case 'down':
        ctx.moveTo(x, y + halfSize);
        ctx.lineTo(x + halfSize * 0.6, y - halfSize);
        ctx.lineTo(x - halfSize * 0.6, y - halfSize);
        break;
      case 'left':
        ctx.moveTo(x - halfSize, y);
        ctx.lineTo(x + halfSize, y + halfSize * 0.6);
        ctx.lineTo(x + halfSize, y - halfSize * 0.6);
        break;
      case 'right':
        ctx.moveTo(x + halfSize, y);
        ctx.lineTo(x - halfSize, y + halfSize * 0.6);
        ctx.lineTo(x - halfSize, y - halfSize * 0.6);
        break;
      default:
        // Default to up arrow
        ctx.moveTo(x, y - halfSize);
        ctx.lineTo(x + halfSize * 0.6, y + halfSize);
        ctx.lineTo(x - halfSize * 0.6, y + halfSize);
    }

    ctx.closePath();
    ctx.fill();
  }

  // ==================== TREND LINE DRAWING METHODS ====================

  /**
   * Draw a trend line
   * @param {Object} drawing - Drawing object
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawTrendLine(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { lineColor, lineWidth, style } = drawing;

    // Convert chart coordinates to screen coordinates
    let startX, startY, endX, endY;

    if (drawing.startIndex !== undefined && drawing.startPrice !== undefined) {
      // Use chart coordinates (anchored to data)
      startX = this.indexToX(drawing.startIndex);
      startY = this.priceToY(drawing.startPrice);
      endX = this.indexToX(drawing.endIndex);
      endY = this.priceToY(drawing.endPrice);

      console.log('📍 Drawing trend line with chart coords:', {
        chart: `[${drawing.startIndex}, $${drawing.startPrice}] → [${drawing.endIndex}, $${drawing.endPrice}]`,
        screen: `(${startX}, ${startY}) → (${endX}, ${endY})`
      });
    } else {
      // Fallback to screen coordinates (legacy)
      startX = drawing.startX;
      startY = drawing.startY;
      endX = drawing.endX;
      endY = drawing.endY;

      console.log('📍 Drawing trend line with screen coords:', {
        screen: `(${startX}, ${startY}) → (${endX}, ${endY})`
      });
    }

    if (startX === undefined || startY === undefined || endX === undefined || endY === undefined) {
      console.error('❌ Invalid coordinates for trend line:', drawing);
      return;
    }

    // Use override color if provided (for eraser highlight), otherwise use drawing's color
    ctx.strokeStyle = overrideColor || lineColor || '#2196f3';
    ctx.lineWidth = lineWidth || 2;

    // Set line style
    if (style === 'dashed') {
      ctx.setLineDash([10, 5]);
    } else if (style === 'dotted') {
      ctx.setLineDash([2, 3]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw arrows at both ends pointing in opposite directions
    const arrowSize = 12;
    const angle = Math.atan2(endY - startY, endX - startX);

    // Arrow at start point (pointing backward/left)
    this.drawArrow(ctx, startX, startY, angle + Math.PI, arrowSize, lineColor || '#2196f3');

    // Arrow at end point (pointing forward/right)
    this.drawArrow(ctx, endX, endY, angle, arrowSize, lineColor || '#2196f3');
  }

  /**
   * Draw an arrow at a specific point
   */
  drawArrow(ctx, x, y, angle, size, color) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.translate(x, y);
    ctx.rotate(angle);

    // Draw arrow triangle
    ctx.beginPath();
    ctx.moveTo(0, 0); // Arrow tip at the endpoint
    ctx.lineTo(-size, -size / 2); // Top wing
    ctx.lineTo(-size, size / 2); // Bottom wing
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  /**
   * Draw a simple line from (x1, y1) to (x2, y2)
   * Used for preview lines before full tool rendering
   * @param {Object} drawing - Drawing object with x1, y1, x2, y2
   * @param {string} color - Line color
   */
  drawSimpleLine(drawing, color = '#2196f3') {
    const ctx = this.ctx;
    const { x1, y1, x2, y2 } = drawing;

    if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
      console.error('❌ Invalid coordinates for simple line:', drawing);
      return;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  /**
   * Draw horizontal line
   * @param {Object} drawing - Drawing object with price coordinate
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawHorizontalLine(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { price, lineColor, lineWidth, style } = drawing;

    // Convert price to screen Y coordinate
    const y = this.priceToY(price);

    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    ctx.strokeStyle = overrideColor || lineColor || '#ff9800';
    ctx.lineWidth = lineWidth || 2;

    if (style === 'dashed') {
      ctx.setLineDash([10, 5]);
    } else if (style === 'dotted') {
      ctx.setLineDash([2, 3]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(chartLeft, y);
    ctx.lineTo(chartRight, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Draw vertical line
   * @param {Object} drawing - Drawing object with chartIndex coordinate
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawVerticalLine(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { chartIndex, lineColor, lineWidth, style } = drawing;

    // Convert chartIndex to screen X coordinate
    const x = this.indexToX(chartIndex);

    const chartTop = this.margin.top;
    const chartBottom = this.height - this.margin.bottom;

    ctx.strokeStyle = overrideColor || lineColor || '#9c27b0';
    ctx.lineWidth = lineWidth || 2;

    if (style === 'dashed') {
      ctx.setLineDash([10, 5]);
    } else if (style === 'dotted') {
      ctx.setLineDash([2, 3]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(x, chartTop);
    ctx.lineTo(x, chartBottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Draw ray line (extends infinitely in one direction)
   * @param {Object} drawing - Drawing object with chart coordinates
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawRayLine(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, lineColor, lineWidth, style } = drawing;

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    // Calculate direction and extend to edge
    const dx = endX - startX;
    const dy = endY - startY;
    const chartRight = this.width - this.margin.right;
    const chartLeft = this.margin.left;

    // Extend line to edge of chart in the direction of the ray
    const extendX = dx > 0 ? chartRight : chartLeft;
    const slope = dy / dx;
    const extendY = startY + slope * (extendX - startX);

    ctx.strokeStyle = overrideColor || lineColor || '#4caf50';
    ctx.lineWidth = lineWidth || 2;

    if (style === 'dashed') {
      ctx.setLineDash([10, 5]);
    } else if (style === 'dotted') {
      ctx.setLineDash([2, 3]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(extendX, extendY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Draw extended line (extends infinitely in both directions)
   * @param {Object} drawing - Drawing object with chart coordinates
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawExtendedLine(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, lineColor, lineWidth, style } = drawing;

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    // Calculate slope
    const dx = endX - startX;
    const dy = endY - startY;
    const slope = dy / dx;

    // Extend to both edges
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    const y1 = startY + slope * (chartLeft - startX);
    const y2 = startY + slope * (chartRight - startX);

    ctx.strokeStyle = overrideColor || lineColor || '#f44336';
    ctx.lineWidth = lineWidth || 2;

    if (style === 'dashed') {
      ctx.setLineDash([10, 5]);
    } else if (style === 'dotted') {
      ctx.setLineDash([2, 3]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(chartLeft, y1);
    ctx.lineTo(chartRight, y2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Draw parallel channel
   * @param {Object} drawing - Drawing object with chart coordinates
   * @param {string|null} overrideColor - Optional color override (for eraser highlight)
   */
  drawParallelChannel(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, parallelPrice, lineColor, lineWidth, fillOpacity } = drawing;

    // Convert chart coordinates to screen coordinates for line 1
    const line1StartX = this.indexToX(startIndex);
    const line1StartY = this.priceToY(startPrice);
    const line1EndX = this.indexToX(endIndex);
    const line1EndY = this.priceToY(endPrice);

    // Calculate parallel line offset (in screen coordinates)
    const parallelY = this.priceToY(parallelPrice);
    const offset = parallelY - line1StartY;

    // Draw first line
    ctx.strokeStyle = overrideColor || lineColor || '#00bcd4';
    ctx.lineWidth = lineWidth || 2;
    ctx.beginPath();
    ctx.moveTo(line1StartX, line1StartY);
    ctx.lineTo(line1EndX, line1EndY);
    ctx.stroke();

    // Draw second line (parallel)
    ctx.beginPath();
    ctx.moveTo(line1StartX, line1StartY + offset);
    ctx.lineTo(line1EndX, line1EndY + offset);
    ctx.stroke();

    // Fill channel
    if (!overrideColor && fillOpacity && fillOpacity > 0) {
      ctx.fillStyle = `${lineColor || '#00bcd4'}${Math.floor(fillOpacity * 255).toString(16).padStart(2, '0')}`;
      ctx.beginPath();
      ctx.moveTo(line1StartX, line1StartY);
      ctx.lineTo(line1EndX, line1EndY);
      ctx.lineTo(line1EndX, line1EndY + offset);
      ctx.lineTo(line1StartX, line1StartY + offset);
      ctx.closePath();
      ctx.fill();
    }
  }

  // ==================== FIBONACCI DRAWING METHODS ====================

  /**
   * Draw Fibonacci retracement
   */
  drawFibonacciRetracement(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, levels, levelColors, lineColor, lineWidth, showLabels } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Fibonacci Retracement missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    const priceRange = endPrice - startPrice;

    // Draw main line from start to end
    ctx.strokeStyle = overrideColor || lineColor || '#2196f3';
    ctx.lineWidth = overrideColor ? 3 : 1;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Get chart boundaries
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    // Draw each Fibonacci level as horizontal lines across the chart
    levels.forEach(level => {
      const levelPrice = startPrice + priceRange * level;
      const levelY = this.priceToY(levelPrice);
      const color = overrideColor || levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = overrideColor ? 3 : (lineWidth || 1);
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(chartLeft, levelY);
      ctx.lineTo(chartRight, levelY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw label on the right side
      if (showLabels && !overrideColor) {
        ctx.fillStyle = color;
        ctx.font = '11px Arial';
        const labelText = `${(level * 100).toFixed(1)}% (${levelPrice.toFixed(2)})`;
        ctx.fillText(labelText, chartRight + 5, levelY + 4);
      }
    });
  }

  /**
   * Draw Fibonacci extension
   */
  drawFibonacciExtension(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { point1Index, point1Price, point2Index, point2Price, point3Index, point3Price,
            levels, levelColors, lineColor, lineWidth, showLabels } = drawing;

    // Safety check for required properties
    if (point1Index === undefined || point1Price === undefined ||
        point2Index === undefined || point2Price === undefined ||
        point3Index === undefined || point3Price === undefined) {
      console.error('❌ Fibonacci Extension missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const p1X = this.indexToX(point1Index);
    const p1Y = this.priceToY(point1Price);
    const p2X = this.indexToX(point2Index);
    const p2Y = this.priceToY(point2Price);
    const p3X = this.indexToX(point3Index);
    const p3Y = this.priceToY(point3Price);

    // Calculate swing range in price
    const swingPrice = point2Price - point1Price;

    // Draw connecting lines (P1 → P2 → P3)
    ctx.strokeStyle = overrideColor || lineColor || '#9c27b0';
    ctx.lineWidth = overrideColor ? 3 : 1;
    ctx.beginPath();
    ctx.moveTo(p1X, p1Y);
    ctx.lineTo(p2X, p2Y);
    ctx.lineTo(p3X, p3Y);
    ctx.stroke();

    // Get chart boundaries
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    // Draw extension levels as horizontal lines from P3
    levels.forEach(level => {
      const levelPrice = point3Price + swingPrice * level;
      const levelY = this.priceToY(levelPrice);
      const color = overrideColor || levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = overrideColor ? 3 : (lineWidth || 1);
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(p3X, levelY);
      ctx.lineTo(chartRight, levelY);
      ctx.stroke();
      ctx.setLineDash([]);

      if (showLabels && !overrideColor) {
        ctx.fillStyle = color;
        ctx.font = '11px Arial';
        const labelText = `${(level * 100).toFixed(1)}% (${levelPrice.toFixed(2)})`;
        ctx.fillText(labelText, chartRight + 5, levelY + 4);
      }
    });
  }

  /**
   * Draw Fibonacci fan
   */
  drawFibonacciFan(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, levels, levelColors, lineColor, lineWidth } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Fibonacci Fan missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    const priceRange = endPrice - startPrice;
    const chartRight = this.width - this.margin.right;

    // Draw main line
    ctx.strokeStyle = overrideColor || lineColor || '#00bcd4';
    ctx.lineWidth = overrideColor ? 3 : 1;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Draw fan lines from start point to various levels at end X position
    levels.forEach(level => {
      const fanPrice = startPrice + priceRange * level;
      const fanY = this.priceToY(fanPrice);
      const color = overrideColor || levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = overrideColor ? 3 : (lineWidth || 1);
      ctx.beginPath();
      ctx.moveTo(startX, startY);

      // Extend the fan line to the right edge of the chart
      const slope = (fanY - startY) / (endX - startX);
      const extendedY = startY + slope * (chartRight - startX);
      ctx.lineTo(chartRight, extendedY);
      ctx.stroke();
    });
  }

  /**
   * Draw Fibonacci arcs
   */
  drawFibonacciArcs(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, levels, levelColors, lineColor, lineWidth } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Fibonacci Arcs missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    const dx = endX - startX;
    const dy = endY - startY;
    const radius = Math.sqrt(dx * dx + dy * dy);

    // Draw each arc centered at the start point
    levels.forEach(level => {
      const arcRadius = radius * level;
      const color = overrideColor || levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = overrideColor ? 3 : (lineWidth || 1);
      ctx.beginPath();
      ctx.arc(startX, startY, arcRadius, 0, Math.PI * 2);
      ctx.stroke();
    });
  }

  /**
   * Draw Fibonacci time zones
   */
  drawFibonacciTimeZones(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { chartIndex, sequence, lineColor, lineWidth } = drawing;

    // Safety check for required properties
    if (chartIndex === undefined) {
      console.error('❌ Fibonacci Time Zones missing required coordinates:', drawing);
      return;
    }

    const chartTop = this.margin.top;
    const chartBottom = this.height - this.margin.bottom;

    // Draw vertical lines at Fibonacci sequence intervals from the start index
    sequence.forEach((fib, index) => {
      const fibIndex = chartIndex + fib;

      // Only draw if within visible range
      if (fibIndex >= this.startIndex && fibIndex <= this.endIndex) {
        const lineX = this.indexToX(fibIndex);

        ctx.strokeStyle = overrideColor || lineColor || '#673ab7';
        ctx.lineWidth = overrideColor ? 3 : (lineWidth || 1);
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(lineX, chartTop);
        ctx.lineTo(lineX, chartBottom);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw label at the top
        ctx.fillStyle = lineColor || '#673ab7';
        ctx.font = '10px Arial';
        ctx.fillText(fib.toString(), lineX + 3, chartTop + 12);
      }
    });
  }

  /**
   * Draw Fibonacci spiral
   */
  drawFibonacciSpiral(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, lineColor, lineWidth, showSquares } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Fibonacci Spiral missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    // Simplified spiral - draw golden ratio rectangles
    ctx.strokeStyle = overrideColor || lineColor || '#e91e63';
    ctx.lineWidth = overrideColor ? 3 : (lineWidth || 2);

    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);
    const goldenRatio = 1.618;

    // Draw main rectangle
    ctx.strokeRect(Math.min(startX, endX), Math.min(startY, endY), width, height);

    // Draw spiral curve (simplified)
    ctx.beginPath();
    ctx.arc(startX, startY, width / 2, 0, Math.PI / 2);
    ctx.stroke();
  }

  // ==================== GANN DRAWING METHODS ====================

  /**
   * Draw Gann fan
   */
  drawGannFan(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, angles, lineColor, lineWidth, showLabels } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Gann Fan missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    const chartRight = this.width - this.margin.right;
    const trendDirection = endY < startY ? -1 : 1;

    Object.entries(angles).forEach(([name, ratio]) => {
      const dx = chartRight - startX;
      const dy = dx * ratio * trendDirection;

      ctx.strokeStyle = overrideColor || lineColor || '#ff9800';
      ctx.lineWidth = lineWidth || 1;
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(startX + dx, startY + dy);
      ctx.stroke();
    });
  }

  /**
   * Draw Gann box
   */
  drawGannBox(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, lineColor, lineWidth, showDiagonals, showQuarters } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Gann Box missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    const width = endX - startX;
    const height = endY - startY;

    // Draw box outline
    ctx.strokeStyle = overrideColor || lineColor || '#00bcd4';
    ctx.lineWidth = lineWidth || 1;
    ctx.strokeRect(startX, startY, width, height);

    // Draw quarter divisions
    if (showQuarters) {
      ctx.beginPath();
      ctx.moveTo(startX + width / 2, startY);
      ctx.lineTo(startX + width / 2, endY);
      ctx.moveTo(startX, startY + height / 2);
      ctx.lineTo(endX, startY + height / 2);
      ctx.stroke();
    }

    // Draw diagonals
    if (showDiagonals) {
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.moveTo(endX, startY);
      ctx.lineTo(startX, endY);
      ctx.stroke();
    }
  }

  /**
   * Draw Gann square
   */
  drawGannSquare(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, lineColor, lineWidth, divisions, showLabels } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Gann Square missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    const width = endX - startX;
    const height = endY - startY;
    const size = Math.min(Math.abs(width), Math.abs(height));

    // Draw square
    ctx.strokeStyle = overrideColor || lineColor || '#4caf50';
    ctx.lineWidth = lineWidth || 1;
    ctx.strokeRect(startX, startY, size, size);

    // Draw grid divisions
    const cellSize = size / divisions;
    for (let i = 1; i < divisions; i++) {
      // Vertical lines
      ctx.beginPath();
      ctx.moveTo(startX + i * cellSize, startY);
      ctx.lineTo(startX + i * cellSize, startY + size);
      ctx.stroke();

      // Horizontal lines
      ctx.beginPath();
      ctx.moveTo(startX, startY + i * cellSize);
      ctx.lineTo(startX + size, startY + i * cellSize);
      ctx.stroke();
    }
  }

  /**
   * Draw Gann angles
   */
  drawGannAngles(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startIndex, startPrice, endIndex, endPrice, angleType, lineColor, lineWidth, extendBoth } = drawing;

    // Safety check for required properties
    if (startIndex === undefined || startPrice === undefined || endIndex === undefined || endPrice === undefined) {
      console.error('❌ Gann Angles missing required coordinates:', drawing);
      return;
    }

    // Convert chart coordinates to screen coordinates
    const startX = this.indexToX(startIndex);
    const startY = this.priceToY(startPrice);
    const endX = this.indexToX(endIndex);
    const endY = this.priceToY(endPrice);

    // Calculate angle based on type
    const angleRatios = {
      '1x8': 1/8, '1x4': 1/4, '1x3': 1/3, '1x2': 1/2,
      '1x1': 1, '2x1': 2, '3x1': 3, '4x1': 4, '8x1': 8
    };

    const ratio = angleRatios[angleType] || 1;
    const direction = endY < startY ? -1 : 1;
    const chartRight = this.width - this.margin.right;
    const dx = chartRight - startX;
    const dy = dx * ratio * direction;

    ctx.strokeStyle = overrideColor || lineColor || '#f44336';
    ctx.lineWidth = lineWidth || 2;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(startX + dx, startY + dy);
    ctx.stroke();
  }

  // ==================== PATTERN DRAWING METHODS ====================

  /**
   * Draw head and shoulders pattern
   */
  drawHeadAndShoulders(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { points, chartPoints, lineColor, lineWidth, showLabels, currentX, currentY } = drawing;

    // Build screen points array (for preview or completed)
    let screenPoints;
    if (chartPoints && chartPoints.length >= 5) {
      // Completed drawing - convert chart coordinates to screen coordinates
      screenPoints = chartPoints.map(p => ({
        x: this.indexToX(p.chartIndex),
        y: this.priceToY(p.chartPrice)
      }));
    } else if (chartPoints && chartPoints.length > 0) {
      // Preview with chart coordinates - convert to screen and add current mouse position
      screenPoints = chartPoints.map(p => ({
        x: this.indexToX(p.chartIndex),
        y: this.priceToY(p.chartPrice)
      }));
      if (currentX !== undefined && currentY !== undefined && chartPoints.length < 5) {
        // Convert currentX/currentY (raw canvas coords) to chart coords and back to screen coords for consistency
        const currentIndex = this.xToIndex(currentX);
        const currentPrice = this.yToPrice(currentY);
        screenPoints.push({
          x: this.indexToX(currentIndex),
          y: this.priceToY(currentPrice)
        });
      }
    } else if (points && points.length > 0) {
      // Preview with screen coordinates - use directly, add current mouse position if available
      screenPoints = [...points];
      if (currentX !== undefined && currentY !== undefined && points.length < 5) {
        screenPoints.push({ x: currentX, y: currentY });
      }
    } else {
      // Not enough data to draw - silently return
      return;
    }

    if (!screenPoints || screenPoints.length < 2) return;

    ctx.strokeStyle = overrideColor || lineColor || '#ff5722';
    ctx.lineWidth = lineWidth || 2;

    // Connect the pattern points
    ctx.beginPath();
    ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
    for (let i = 1; i < screenPoints.length; i++) {
      ctx.lineTo(screenPoints[i].x, screenPoints[i].y);
    }
    ctx.stroke();

    // Draw neckline (only if we have all 5 points)
    if (screenPoints.length >= 5) {
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
      ctx.lineTo(screenPoints[4].x, screenPoints[4].y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Labels (only if we have enough points and labels are enabled)
    if (showLabels) {
      ctx.fillStyle = lineColor;
      ctx.font = '11px Arial';
      if (screenPoints.length >= 1) ctx.fillText('LS', screenPoints[0].x, screenPoints[0].y - 5);
      if (screenPoints.length >= 3) ctx.fillText('H', screenPoints[2].x, screenPoints[2].y - 5);
      if (screenPoints.length >= 5) ctx.fillText('RS', screenPoints[4].x, screenPoints[4].y - 5);
    }
  }

  /**
   * Draw triangle pattern
   */
  drawTriangle(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { points, chartPoints, lineColor, lineWidth, fillOpacity, currentX, currentY } = drawing;

    // Build screen points array (for preview or completed)
    let screenPoints;
    if (chartPoints && chartPoints.length >= 4) {
      // Completed drawing - convert chart coordinates to screen coordinates
      screenPoints = chartPoints.map(p => ({
        x: this.indexToX(p.chartIndex),
        y: this.priceToY(p.chartPrice)
      }));
    } else if (chartPoints && chartPoints.length > 0) {
      // Preview with chart coordinates - convert to screen and add current mouse position
      screenPoints = chartPoints.map(p => ({
        x: this.indexToX(p.chartIndex),
        y: this.priceToY(p.chartPrice)
      }));
      if (currentX !== undefined && currentY !== undefined && chartPoints.length < 4) {
        // Convert currentX/currentY (raw canvas coords) to chart coords and back to screen coords for consistency
        const currentIndex = this.xToIndex(currentX);
        const currentPrice = this.yToPrice(currentY);
        screenPoints.push({
          x: this.indexToX(currentIndex),
          y: this.priceToY(currentPrice)
        });
      }
    } else if (points && points.length > 0) {
      // Preview with screen coordinates - use directly, add current mouse position if available
      screenPoints = [...points];
      if (currentX !== undefined && currentY !== undefined && points.length < 4) {
        screenPoints.push({ x: currentX, y: currentY });
      }
    } else {
      // Not enough data to draw - silently return
      return;
    }

    if (!screenPoints || screenPoints.length < 2) return;

    ctx.strokeStyle = overrideColor || lineColor || '#9c27b0';
    ctx.lineWidth = lineWidth || 2;

    // Draw lines based on how many points we have
    if (screenPoints.length === 2) {
      // Step 1: Only have point 0 and cursor - draw preview of first leg
      ctx.beginPath();
      ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
      ctx.lineTo(screenPoints[1].x, screenPoints[1].y);
      ctx.stroke();
    } else if (screenPoints.length === 3) {
      // Step 2: Have points 0, 1, and cursor - draw completed first leg + preview second leg
      ctx.beginPath();
      ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
      ctx.lineTo(screenPoints[1].x, screenPoints[1].y);
      ctx.stroke();

      // Preview second leg from point 1 to cursor
      ctx.beginPath();
      ctx.moveTo(screenPoints[1].x, screenPoints[1].y);
      ctx.lineTo(screenPoints[2].x, screenPoints[2].y);
      ctx.stroke();
    } else if (screenPoints.length >= 4) {
      // Step 3+: Have points 0, 1, 2, and possibly cursor/point 3
      // Draw first trend line (points 0 to 2)
      ctx.beginPath();
      ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
      ctx.lineTo(screenPoints[2].x, screenPoints[2].y);
      ctx.stroke();

      // Draw second trend line (points 1 to 3)
      ctx.beginPath();
      ctx.moveTo(screenPoints[1].x, screenPoints[1].y);
      ctx.lineTo(screenPoints[3].x, screenPoints[3].y);
      ctx.stroke();
    }

    // Fill (only if we have all 4 points)
    if (fillOpacity && fillOpacity > 0 && screenPoints.length >= 4) {
      ctx.fillStyle = `${lineColor}${Math.floor(fillOpacity * 255).toString(16)}`;
      ctx.beginPath();
      ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
      ctx.lineTo(screenPoints[2].x, screenPoints[2].y);
      ctx.lineTo(screenPoints[3].x, screenPoints[3].y);
      ctx.lineTo(screenPoints[1].x, screenPoints[1].y);
      ctx.closePath();
      ctx.fill();
    }
  }

  /**
   * Draw wedge pattern
   */
  drawWedge(drawing, overrideColor = null) {
    this.drawTriangle(drawing, overrideColor); // Same rendering as triangle
  }

  /**
   * Draw double top/bottom pattern
   */
  drawDoubleTopBottom(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { points, chartPoints, lineColor, lineWidth, showLabels, patternType, currentX, currentY } = drawing;

    // Build screen points array (for preview or completed)
    let screenPoints;
    if (chartPoints && chartPoints.length >= 3) {
      // Completed drawing - convert chart coordinates to screen coordinates
      screenPoints = chartPoints.map(p => ({
        x: this.indexToX(p.chartIndex),
        y: this.priceToY(p.chartPrice)
      }));
    } else if (chartPoints && chartPoints.length > 0) {
      // Preview with chart coordinates - convert to screen and add current mouse position
      screenPoints = chartPoints.map(p => ({
        x: this.indexToX(p.chartIndex),
        y: this.priceToY(p.chartPrice)
      }));
      if (currentX !== undefined && currentY !== undefined && chartPoints.length < 3) {
        // Convert currentX/currentY (raw canvas coords) to chart coords and back to screen coords for consistency
        const currentIndex = this.xToIndex(currentX);
        const currentPrice = this.yToPrice(currentY);
        screenPoints.push({
          x: this.indexToX(currentIndex),
          y: this.priceToY(currentPrice)
        });
      }
    } else if (points && points.length > 0) {
      // Preview with screen coordinates - use directly, add current mouse position if available
      screenPoints = [...points];
      if (currentX !== undefined && currentY !== undefined && points.length < 3) {
        screenPoints.push({ x: currentX, y: currentY });
      }
    } else {
      // Not enough data to draw - silently return
      return;
    }

    if (!screenPoints || screenPoints.length < 2) {
      console.log('⚠️ Not enough points to draw:', {
        screenPointsLength: screenPoints?.length,
        hasCurrentX: currentX !== undefined,
        hasCurrentY: currentY !== undefined,
        pointsLength: points?.length
      });
      return;
    }

    ctx.strokeStyle = overrideColor || lineColor || '#f44336';
    ctx.lineWidth = lineWidth || 2;

    // Connect points
    ctx.beginPath();
    ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
    for (let i = 1; i < screenPoints.length; i++) {
      ctx.lineTo(screenPoints[i].x, screenPoints[i].y);
    }
    ctx.stroke();

    // Draw neckline (only if we have all 3 points)
    if (screenPoints.length >= 3) {
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(screenPoints[0].x, screenPoints[1].y);
      ctx.lineTo(screenPoints[2].x, screenPoints[1].y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Labels (only if we have all points)
    if (showLabels && screenPoints.length >= 3) {
      ctx.fillStyle = lineColor;
      ctx.font = '11px Arial';
      const label = patternType === 'top' ? 'Double Top' : 'Double Bottom';
      ctx.fillText(label, screenPoints[1].x, screenPoints[1].y - 10);
    }
  }

  // ==================== SHAPE DRAWING METHODS ====================

  /**
   * Draw rectangle
   */
  drawRectangle(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, startIndex, startPrice, endIndex, endPrice, lineColor, lineWidth, fillColor, filled } = drawing;

    // Get screen coordinates
    let screenStartX, screenStartY, screenEndX, screenEndY;

    if (startIndex !== undefined && startPrice !== undefined && endIndex !== undefined && endPrice !== undefined) {
      // Convert from chart coordinates to screen coordinates
      screenStartX = this.indexToX(startIndex);
      screenStartY = this.priceToY(startPrice);
      screenEndX = this.indexToX(endIndex);
      screenEndY = this.priceToY(endPrice);
    } else {
      // Use screen coordinates directly (for preview)
      screenStartX = startX;
      screenStartY = startY;
      screenEndX = endX;
      screenEndY = endY;
    }

    const width = screenEndX - screenStartX;
    const height = screenEndY - screenStartY;

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fillRect(screenStartX, screenStartY, width, height);
    }

    ctx.strokeStyle = overrideColor || lineColor || '#2196f3';
    ctx.lineWidth = lineWidth || 2;
    ctx.strokeRect(screenStartX, screenStartY, width, height);
  }

  /**
   * Draw circle
   */
  drawCircle(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { centerX, centerY, radius, centerIndex, centerPrice, radiusInPriceUnits, lineColor, lineWidth, fillColor, filled } = drawing;

    // Get screen coordinates
    let screenCenterX, screenCenterY, screenRadius;

    if (centerIndex !== undefined && centerPrice !== undefined && radiusInPriceUnits !== undefined) {
      // Convert from chart coordinates to screen coordinates
      screenCenterX = this.indexToX(centerIndex);
      screenCenterY = this.priceToY(centerPrice);
      // Convert radius from price units to screen pixels
      const priceRange = this.maxPrice - this.minPrice;
      const canvasHeight = this.canvas.height;
      screenRadius = (radiusInPriceUnits / priceRange) * canvasHeight;
    } else {
      // Use screen coordinates directly (for preview)
      screenCenterX = centerX;
      screenCenterY = centerY;
      screenRadius = radius;
    }

    ctx.beginPath();
    ctx.arc(screenCenterX, screenCenterY, screenRadius, 0, Math.PI * 2);

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    ctx.strokeStyle = overrideColor || lineColor || '#00c853';
    ctx.lineWidth = lineWidth || 2;
    ctx.stroke();
  }

  /**
   * Draw ellipse
   */
  drawEllipse(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, startIndex, startPrice, endIndex, endPrice, lineColor, lineWidth, fillColor, filled } = drawing;

    // Get screen coordinates
    let screenStartX, screenStartY, screenEndX, screenEndY;

    if (startIndex !== undefined && startPrice !== undefined && endIndex !== undefined && endPrice !== undefined) {
      // Convert from chart coordinates to screen coordinates
      screenStartX = this.indexToX(startIndex);
      screenStartY = this.priceToY(startPrice);
      screenEndX = this.indexToX(endIndex);
      screenEndY = this.priceToY(endPrice);
    } else {
      // Use screen coordinates directly (for preview)
      screenStartX = startX;
      screenStartY = startY;
      screenEndX = endX;
      screenEndY = endY;
    }

    const centerX = (screenStartX + screenEndX) / 2;
    const centerY = (screenStartY + screenEndY) / 2;
    const radiusX = Math.abs(screenEndX - screenStartX) / 2;
    const radiusY = Math.abs(screenEndY - screenStartY) / 2;

    ctx.beginPath();
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    ctx.strokeStyle = overrideColor || lineColor || '#ff9800';
    ctx.lineWidth = lineWidth || 2;
    ctx.stroke();
  }

  /**
   * Draw polygon
   */
  drawPolygon(drawing, overrideColor = null) {
    const ctx = this.ctx;
    const { points, chartPoints, lineColor, lineWidth, fillColor, filled } = drawing;

    // Get screen points
    let screenPoints;
    if (chartPoints && chartPoints.length >= 3) {
      // Convert from chart coordinates to screen coordinates
      screenPoints = chartPoints.map(p => ({
        x: this.indexToX(p.chartIndex),
        y: this.priceToY(p.chartPrice)
      }));
    } else if (points && points.length >= 3) {
      // Use screen coordinates directly (for preview or old drawings)
      screenPoints = points;
    } else {
      return; // Not enough points to draw
    }

    ctx.beginPath();
    ctx.moveTo(screenPoints[0].x, screenPoints[0].y);
    screenPoints.forEach(point => {
      ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    ctx.strokeStyle = overrideColor || lineColor || '#9c27b0';
    ctx.lineWidth = lineWidth || 2;
    ctx.stroke();
  }

  // ==================== ANNOTATION DRAWING METHODS ====================

  /**
   * Draw text label
   */
  drawTextLabel(drawing) {
    const ctx = this.ctx;
    const { x, y, text, fontSize, fontFamily, textColor, backgroundColor, showBackground } = drawing;

    ctx.font = `${fontSize || 14}px ${fontFamily || 'Arial'}`;

    if (showBackground && backgroundColor) {
      const metrics = ctx.measureText(text);
      const padding = 4;
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(x - padding, y - fontSize - padding, metrics.width + padding * 2, fontSize + padding * 2);
    }

    ctx.fillStyle = textColor || '#ffffff';
    ctx.fillText(text || 'Text', x, y);
  }

  /**
   * Draw callout
   */
  drawCallout(drawing) {
    const ctx = this.ctx;
    const { pointerX, pointerY, textX, textY, text, textColor, backgroundColor, borderColor, fontSize } = drawing;

    // Draw arrow from pointer to text box
    ctx.strokeStyle = borderColor || '#2196f3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pointerX, pointerY);
    ctx.lineTo(textX, textY);
    ctx.stroke();

    // Draw text box
    ctx.font = `${fontSize || 14}px Arial`;
    const metrics = ctx.measureText(text || 'Callout');
    const padding = 8;
    const boxWidth = metrics.width + padding * 2;
    const boxHeight = fontSize + padding * 2;

    ctx.fillStyle = backgroundColor || 'rgba(33, 150, 243, 0.9)';
    ctx.fillRect(textX - boxWidth / 2, textY - boxHeight / 2, boxWidth, boxHeight);

    ctx.strokeStyle = borderColor || '#2196f3';
    ctx.strokeRect(textX - boxWidth / 2, textY - boxHeight / 2, boxWidth, boxHeight);

    ctx.fillStyle = textColor || '#ffffff';
    ctx.fillText(text || 'Callout', textX - metrics.width / 2, textY + fontSize / 3);
  }

  /**
   * Draw note
   */
  drawNote(drawing) {
    const ctx = this.ctx;
    const { x, y, text, noteColor, textColor, fontSize, width, height } = drawing;

    // Draw note background
    ctx.fillStyle = noteColor || '#ffeb3b';
    ctx.fillRect(x, y, width || 150, height || 100);

    ctx.strokeStyle = '#daa520';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width || 150, height || 100);

    // Draw text
    ctx.fillStyle = textColor || '#000000';
    ctx.font = `${fontSize || 12}px Arial`;
    ctx.fillText(text || 'Note', x + 10, y + 20);
  }

  /**
   * Draw price label
   */
  drawPriceLabel(drawing) {
    const ctx = this.ctx;
    const { x, y, labelColor, textColor, fontSize, showLine } = drawing;
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    // Draw horizontal line
    if (showLine) {
      ctx.strokeStyle = labelColor || '#00c853';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(chartLeft, y);
      ctx.lineTo(chartRight, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw price label
    const price = this.yToPrice(y);
    const priceText = price.toFixed(2);

    ctx.fillStyle = labelColor || '#00c853';
    ctx.fillRect(chartRight - 60, y - 12, 55, 24);

    ctx.fillStyle = textColor || '#ffffff';
    ctx.font = `${fontSize || 12}px Arial`;
    ctx.fillText(priceText, chartRight - 55, y + 4);
  }

  /**
   * Clean up
   */
  destroy() {
    if (this.canvas) {
      this.canvas.remove();
      this.canvas = null;
      this.ctx = null;
    }
    // Reset flags so new chart instance starts fresh
    this.autoScrollEnabled = true;
    this.hasRendered = false;
  }

  /**
   * Get dummy interactions object for compatibility
   */
  getInteractions() {
    return {
      crosshair: { enabled: true },
      panZoom: { enabled: true },
      keyboardControls: { enabled: true }
    };
  }
}

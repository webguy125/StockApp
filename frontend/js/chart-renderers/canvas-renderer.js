/**
 * Canvas-Based Candlestick Chart Renderer
 * Custom renderer using HTML5 Canvas - NO PLOTLY
 * Full control over rendering, dates, and live updates
 * Updated: 2025-10-24 - TradingView-style unified price scale (candles can appear in volume area)
 */

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

    // console.log('✅ Canvas chart rendered successfully');
    return true;
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
    this.canvas.style.cursor = 'grab'; // Indicate chart is draggable
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

    const relativeX = x - chartLeft;
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
    this.isDragging = false;
    this.isDraggingYAxis = false;
    this.isDraggingXAxis = false;
    this.canvas.style.cursor = 'grab';
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
    this.canvas.style.cursor = 'grab';
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

    console.log('🎨 Tool action received:', action.action, action);

    // Ignore "start" actions - they're just the first click
    const ignoreActions = ['start-trend-line', 'start-ray-line', 'start-extended-line'];
    if (ignoreActions.includes(action.action)) {
      console.log('⏭️ Ignoring start action');
      return;
    }

    // Check if this is a preview action or a completed drawing
    const isPreview = action.action.startsWith('preview-') ||
                      action.action.includes('-point-') ||
                      action.action === 'update-polygon';

    // Convert screen coordinates to chart coordinates for trend lines
    let convertedAction = action;
    if (action.action.includes('trend-line')) {
      convertedAction = this.convertToChartCoordinates(action);
      console.log('🔄 Action after conversion:', convertedAction);
    }

    if (isPreview) {
      // Update preview drawing
      this.previewDrawing = convertedAction;
      console.log('👁️ Preview updated:', convertedAction.action);
    } else {
      // Completed drawing - add to drawings array
      this.drawings.push(convertedAction);
      this.previewDrawing = null; // Clear preview
      console.log(`✅ Drawing added: ${convertedAction.action} (total: ${this.drawings.length})`, convertedAction);
    }

    // Redraw chart with new drawing/preview
    this.draw();
  }

  /**
   * Convert screen coordinates to chart coordinates (index and price)
   */
  convertToChartCoordinates(action) {
    const converted = { ...action };

    // Convert start point
    if (action.startX !== undefined && action.startY !== undefined) {
      converted.startIndex = this.xToIndex(action.startX);
      converted.startPrice = this.yToPrice(action.startY);
    }

    // Convert end point
    if (action.endX !== undefined && action.endY !== undefined) {
      converted.endIndex = this.xToIndex(action.endX);
      converted.endPrice = this.yToPrice(action.endY);
    }

    console.log('🔄 Converted coordinates:', {
      screen: `(${action.startX}, ${action.startY}) → (${action.endX}, ${action.endY})`,
      chart: `[${converted.startIndex}, $${converted.startPrice?.toFixed(2)}] → [${converted.endIndex}, $${converted.endPrice?.toFixed(2)}]`
    });

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

    this.drawings.forEach(drawing => {
      this.drawSingleDrawing(drawing);
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
   */
  drawSingleDrawing(drawing) {
    if (!drawing || !drawing.action) return;

    const ctx = this.ctx;

    // Route to specific drawing method based on action
    if (drawing.action.includes('trend-line')) {
      this.drawTrendLine(drawing);
    } else if (drawing.action.includes('horizontal-line')) {
      this.drawHorizontalLine(drawing);
    } else if (drawing.action.includes('vertical-line')) {
      this.drawVerticalLine(drawing);
    } else if (drawing.action.includes('ray-line')) {
      this.drawRayLine(drawing);
    } else if (drawing.action.includes('extended-line')) {
      this.drawExtendedLine(drawing);
    } else if (drawing.action.includes('parallel-channel')) {
      this.drawParallelChannel(drawing);
    } else if (drawing.action.includes('fibonacci-retracement')) {
      this.drawFibonacciRetracement(drawing);
    } else if (drawing.action.includes('fibonacci-extension')) {
      this.drawFibonacciExtension(drawing);
    } else if (drawing.action.includes('fibonacci-fan')) {
      this.drawFibonacciFan(drawing);
    } else if (drawing.action.includes('fibonacci-arcs')) {
      this.drawFibonacciArcs(drawing);
    } else if (drawing.action.includes('fibonacci-time-zones')) {
      this.drawFibonacciTimeZones(drawing);
    } else if (drawing.action.includes('fibonacci-spiral')) {
      this.drawFibonacciSpiral(drawing);
    } else if (drawing.action.includes('gann-fan')) {
      this.drawGannFan(drawing);
    } else if (drawing.action.includes('gann-box')) {
      this.drawGannBox(drawing);
    } else if (drawing.action.includes('gann-square')) {
      this.drawGannSquare(drawing);
    } else if (drawing.action.includes('gann-angles')) {
      this.drawGannAngles(drawing);
    } else if (drawing.action.includes('head-and-shoulders')) {
      this.drawHeadAndShoulders(drawing);
    } else if (drawing.action.includes('triangle')) {
      this.drawTriangle(drawing);
    } else if (drawing.action.includes('wedge')) {
      this.drawWedge(drawing);
    } else if (drawing.action.includes('double-top-bottom')) {
      this.drawDoubleTopBottom(drawing);
    } else if (drawing.action.includes('rectangle')) {
      this.drawRectangle(drawing);
    } else if (drawing.action.includes('circle')) {
      this.drawCircle(drawing);
    } else if (drawing.action.includes('ellipse')) {
      this.drawEllipse(drawing);
    } else if (drawing.action.includes('polygon')) {
      this.drawPolygon(drawing);
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

  // ==================== TREND LINE DRAWING METHODS ====================

  /**
   * Draw a trend line
   */
  drawTrendLine(drawing) {
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

    ctx.strokeStyle = lineColor || '#2196f3';
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
  }

  /**
   * Draw horizontal line
   */
  drawHorizontalLine(drawing) {
    const ctx = this.ctx;
    const { y, lineColor, lineWidth, style } = drawing;
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    ctx.strokeStyle = lineColor || '#ff9800';
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
   */
  drawVerticalLine(drawing) {
    const ctx = this.ctx;
    const { x, lineColor, lineWidth, style } = drawing;
    const chartTop = this.margin.top;
    const chartBottom = this.height - this.margin.bottom;

    ctx.strokeStyle = lineColor || '#9c27b0';
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
   */
  drawRayLine(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, lineColor, lineWidth, style } = drawing;

    // Calculate direction and extend to edge
    const dx = endX - startX;
    const dy = endY - startY;
    const chartRight = this.width - this.margin.right;
    const chartBottom = this.height - this.margin.bottom;

    // Extend line to edge of chart
    const extendX = dx > 0 ? chartRight : this.margin.left;
    const slope = dy / dx;
    const extendY = startY + slope * (extendX - startX);

    ctx.strokeStyle = lineColor || '#4caf50';
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
   */
  drawExtendedLine(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, lineColor, lineWidth, style } = drawing;

    // Calculate slope
    const dx = endX - startX;
    const dy = endY - startY;
    const slope = dy / dx;

    // Extend to both edges
    const chartLeft = this.margin.left;
    const chartRight = this.width - this.margin.right;

    const y1 = startY + slope * (chartLeft - startX);
    const y2 = startY + slope * (chartRight - startX);

    ctx.strokeStyle = lineColor || '#f44336';
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
   */
  drawParallelChannel(drawing) {
    const ctx = this.ctx;
    const { line1Start, line1End, parallelY, lineColor, lineWidth, fillOpacity } = drawing;

    // Draw first line
    ctx.strokeStyle = lineColor || '#00bcd4';
    ctx.lineWidth = lineWidth || 2;
    ctx.beginPath();
    ctx.moveTo(line1Start.x, line1Start.y);
    ctx.lineTo(line1End.x, line1End.y);
    ctx.stroke();

    // Calculate parallel line offset
    const offset = parallelY - line1Start.y;

    // Draw second line (parallel)
    ctx.beginPath();
    ctx.moveTo(line1Start.x, line1Start.y + offset);
    ctx.lineTo(line1End.x, line1End.y + offset);
    ctx.stroke();

    // Fill channel
    if (fillOpacity && fillOpacity > 0) {
      ctx.fillStyle = `${lineColor}${Math.floor(fillOpacity * 255).toString(16)}`;
      ctx.beginPath();
      ctx.moveTo(line1Start.x, line1Start.y);
      ctx.lineTo(line1End.x, line1End.y);
      ctx.lineTo(line1End.x, line1End.y + offset);
      ctx.lineTo(line1Start.x, line1Start.y + offset);
      ctx.closePath();
      ctx.fill();
    }
  }

  // ==================== FIBONACCI DRAWING METHODS ====================

  /**
   * Draw Fibonacci retracement
   */
  drawFibonacciRetracement(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, levels, levelColors, lineColor, lineWidth, showLabels } = drawing;

    const dy = endY - startY;

    // Draw main line
    ctx.strokeStyle = lineColor || '#2196f3';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Draw each Fibonacci level
    levels.forEach(level => {
      const y = startY + dy * level;
      const color = levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth || 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(startX, y);
      ctx.lineTo(endX, y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw label
      if (showLabels) {
        ctx.fillStyle = color;
        ctx.font = '11px Arial';
        ctx.fillText(`${(level * 100).toFixed(1)}%`, endX + 5, y + 4);
      }
    });
  }

  /**
   * Draw Fibonacci extension
   */
  drawFibonacciExtension(drawing) {
    const ctx = this.ctx;
    const { point1, point2, point3, levels, levelColors, lineColor, lineWidth, showLabels } = drawing;

    // Calculate swing range
    const swingHeight = point2.y - point1.y;

    // Draw connecting lines
    ctx.strokeStyle = lineColor || '#9c27b0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(point1.x, point1.y);
    ctx.lineTo(point2.x, point2.y);
    ctx.lineTo(point3.x, point3.y);
    ctx.stroke();

    // Draw extension levels
    levels.forEach(level => {
      const y = point3.y + swingHeight * level;
      const color = levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth || 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(point3.x, y);
      ctx.lineTo(point3.x + 100, y);
      ctx.stroke();
      ctx.setLineDash([]);

      if (showLabels) {
        ctx.fillStyle = color;
        ctx.font = '11px Arial';
        ctx.fillText(`${(level * 100).toFixed(1)}%`, point3.x + 105, y + 4);
      }
    });
  }

  /**
   * Draw Fibonacci fan
   */
  drawFibonacciFan(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, levels, levelColors, lineColor, lineWidth } = drawing;

    const dx = endX - startX;
    const dy = endY - startY;

    // Draw main line
    ctx.strokeStyle = lineColor || '#00bcd4';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Draw fan lines
    levels.forEach(level => {
      const fanY = startY + dy * level;
      const color = levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth || 1;
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, fanY);
      ctx.stroke();
    });
  }

  /**
   * Draw Fibonacci arcs
   */
  drawFibonacciArcs(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, levels, levelColors, lineColor, lineWidth } = drawing;

    const dx = endX - startX;
    const dy = endY - startY;
    const radius = Math.sqrt(dx * dx + dy * dy);

    // Draw each arc
    levels.forEach(level => {
      const arcRadius = radius * level;
      const color = levelColors[level] || lineColor;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth || 1;
      ctx.beginPath();
      ctx.arc(startX, startY, arcRadius, 0, Math.PI * 2);
      ctx.stroke();
    });
  }

  /**
   * Draw Fibonacci time zones
   */
  drawFibonacciTimeZones(drawing) {
    const ctx = this.ctx;
    const { x, sequence, lineColor, lineWidth } = drawing;
    const chartTop = this.margin.top;
    const chartBottom = this.height - this.margin.bottom;

    // Assume each candle represents 1 time unit
    const candleWidth = (this.width - this.margin.left - this.margin.right) / (this.endIndex - this.startIndex + 1);

    sequence.forEach((fib, index) => {
      const lineX = x + (fib * candleWidth);

      if (lineX >= this.margin.left && lineX <= this.width - this.margin.right) {
        ctx.strokeStyle = lineColor || '#673ab7';
        ctx.lineWidth = lineWidth || 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(lineX, chartTop);
        ctx.lineTo(lineX, chartBottom);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  }

  /**
   * Draw Fibonacci spiral
   */
  drawFibonacciSpiral(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, lineColor, lineWidth, showSquares } = drawing;

    // Simplified spiral - draw golden ratio rectangles
    ctx.strokeStyle = lineColor || '#e91e63';
    ctx.lineWidth = lineWidth || 2;

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
  drawGannFan(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, angles, lineColor, lineWidth } = drawing;

    const chartRight = this.width - this.margin.right;
    const trendDirection = endY < startY ? -1 : 1;

    Object.entries(angles).forEach(([name, ratio]) => {
      const dx = chartRight - startX;
      const dy = dx * ratio * trendDirection;

      ctx.strokeStyle = lineColor || '#ff9800';
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
  drawGannBox(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, lineColor, lineWidth, showDiagonals, showQuarters } = drawing;

    const width = endX - startX;
    const height = endY - startY;

    // Draw box outline
    ctx.strokeStyle = lineColor || '#00bcd4';
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
  drawGannSquare(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, lineColor, lineWidth, divisions } = drawing;

    const width = endX - startX;
    const height = endY - startY;
    const size = Math.min(width, height);

    // Draw square
    ctx.strokeStyle = lineColor || '#4caf50';
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
  drawGannAngles(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, angleType, lineColor, lineWidth, extendBoth } = drawing;

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

    ctx.strokeStyle = lineColor || '#f44336';
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
  drawHeadAndShoulders(drawing) {
    const ctx = this.ctx;
    const { points, lineColor, lineWidth, showLabels } = drawing;

    if (!points || points.length < 5) return;

    ctx.strokeStyle = lineColor || '#ff5722';
    ctx.lineWidth = lineWidth || 2;

    // Connect the pattern points
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(point => {
      ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();

    // Draw neckline
    if (points.length >= 5) {
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[4].x, points[4].y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Labels
    if (showLabels && points.length >= 5) {
      ctx.fillStyle = lineColor;
      ctx.font = '11px Arial';
      ctx.fillText('LS', points[0].x, points[0].y - 5);
      ctx.fillText('H', points[2].x, points[2].y - 5);
      ctx.fillText('RS', points[4].x, points[4].y - 5);
    }
  }

  /**
   * Draw triangle pattern
   */
  drawTriangle(drawing) {
    const ctx = this.ctx;
    const { points, lineColor, lineWidth, fillOpacity } = drawing;

    if (!points || points.length < 4) return;

    ctx.strokeStyle = lineColor || '#9c27b0';
    ctx.lineWidth = lineWidth || 2;

    // Draw two trend lines
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[2].x, points[2].y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(points[1].x, points[1].y);
    ctx.lineTo(points[3].x, points[3].y);
    ctx.stroke();

    // Fill
    if (fillOpacity && fillOpacity > 0) {
      ctx.fillStyle = `${lineColor}${Math.floor(fillOpacity * 255).toString(16)}`;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[2].x, points[2].y);
      ctx.lineTo(points[3].x, points[3].y);
      ctx.lineTo(points[1].x, points[1].y);
      ctx.closePath();
      ctx.fill();
    }
  }

  /**
   * Draw wedge pattern
   */
  drawWedge(drawing) {
    this.drawTriangle(drawing); // Same rendering as triangle
  }

  /**
   * Draw double top/bottom pattern
   */
  drawDoubleTopBottom(drawing) {
    const ctx = this.ctx;
    const { points, lineColor, lineWidth, showLabels, patternType } = drawing;

    if (!points || points.length < 3) return;

    ctx.strokeStyle = lineColor || '#f44336';
    ctx.lineWidth = lineWidth || 2;

    // Connect points
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(point => {
      ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();

    // Draw neckline
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[1].y);
    ctx.lineTo(points[2].x, points[1].y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    if (showLabels) {
      ctx.fillStyle = lineColor;
      ctx.font = '11px Arial';
      const label = patternType === 'top' ? 'Double Top' : 'Double Bottom';
      ctx.fillText(label, points[1].x, points[1].y - 10);
    }
  }

  // ==================== SHAPE DRAWING METHODS ====================

  /**
   * Draw rectangle
   */
  drawRectangle(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, lineColor, lineWidth, fillColor, filled } = drawing;

    const width = endX - startX;
    const height = endY - startY;

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fillRect(startX, startY, width, height);
    }

    ctx.strokeStyle = lineColor || '#2196f3';
    ctx.lineWidth = lineWidth || 2;
    ctx.strokeRect(startX, startY, width, height);
  }

  /**
   * Draw circle
   */
  drawCircle(drawing) {
    const ctx = this.ctx;
    const { centerX, centerY, radius, lineColor, lineWidth, fillColor, filled } = drawing;

    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    ctx.strokeStyle = lineColor || '#00c853';
    ctx.lineWidth = lineWidth || 2;
    ctx.stroke();
  }

  /**
   * Draw ellipse
   */
  drawEllipse(drawing) {
    const ctx = this.ctx;
    const { startX, startY, endX, endY, lineColor, lineWidth, fillColor, filled } = drawing;

    const centerX = (startX + endX) / 2;
    const centerY = (startY + endY) / 2;
    const radiusX = Math.abs(endX - startX) / 2;
    const radiusY = Math.abs(endY - startY) / 2;

    ctx.beginPath();
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    ctx.strokeStyle = lineColor || '#ff9800';
    ctx.lineWidth = lineWidth || 2;
    ctx.stroke();
  }

  /**
   * Draw polygon
   */
  drawPolygon(drawing) {
    const ctx = this.ctx;
    const { points, lineColor, lineWidth, fillColor, filled } = drawing;

    if (!points || points.length < 3) return;

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(point => {
      ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();

    if (filled && fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    ctx.strokeStyle = lineColor || '#9c27b0';
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

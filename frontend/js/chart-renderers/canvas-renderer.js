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

    // Live price tracking
    this.livePrice = null;

    // Auto-scroll behavior for tick charts
    this.autoScrollEnabled = true; // Enable auto-scroll to latest candle by default
    this.hasRendered = false; // Track if we've rendered at least once

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
   * Calculate price levels with dynamic increment based on timeframe
   */
  calculatePriceLevels() {
    // Determine gridline increment based on timeframe
    // Shorter timeframes = finer gridlines, longer timeframes = coarser gridlines
    let increment;

    if (this.timeframeInterval === '1m' || this.timeframeInterval === '5m') {
      increment = 100;    // Very fine gridlines for minute charts
    } else if (this.timeframeInterval === '15m' || this.timeframeInterval === '30m' || this.timeframeInterval === '1h') {
      increment = 500;    // Fine gridlines for hour charts
    } else if (this.timeframeInterval === '1d') {
      increment = 1000;   // Standard gridlines for daily (current working setting)
    } else if (this.timeframeInterval === '1wk' || this.timeframeInterval === '1w') {
      increment = 5000;   // Wide gridlines for weekly
    } else if (this.timeframeInterval === '1mo') {
      increment = 10000;  // Very wide gridlines for monthly
    } else if (this.timeframeInterval === '3mo') {
      increment = 20000;  // Extra wide gridlines for 3-month candles
    } else if (this.timeframeInterval === '6mo') {
      increment = 30000;  // Very wide gridlines for 6-month candles
    } else {
      increment = 1000;   // Default fallback
    }

    // Calculate first price level (round down to nearest increment, then go one more below for padding)
    const firstLevel = Math.floor(this.minPrice / increment) * increment - increment;

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

    // Date labels (below volume bars) - IMPROVED VISIBILITY
    const volumeBottom = this.margin.top + this.chartHeight + 10 + this.volumeHeight;

    ctx.textAlign = 'center';
    const visibleCandles = this.endIndex - this.startIndex + 1;
    const totalWidth = chartWidth / visibleCandles;

    // Show dates at regular intervals
    const maxLabels = Math.min(15, visibleCandles);
    const labelInterval = Math.max(1, Math.floor(visibleCandles / maxLabels));

    // Larger, brighter font for better visibility
    ctx.font = '13px Arial';
    ctx.fillStyle = '#e0e0e0'; // Brighter text color

    for (let i = 0; i < visibleCandles; i += labelInterval) {
      const dataIndex = Math.floor(this.startIndex + i); // Floor for array access
      if (dataIndex >= 0 && dataIndex < this.data.length) {
        const dateStr = this.data[dataIndex].Date;
        const x = chartLeft + (i * totalWidth) + (totalWidth / 2);

        // Parse date string directly (YYYY-MM-DD format)
        // Don't use Date constructor to avoid timezone issues
        const parts = dateStr.split('-');
        const month = parseInt(parts[1]);
        const day = parseInt(parts[2]);
        const displayDate = `${month}/${day}`;

        // Draw background rectangle for better readability
        const textWidth = ctx.measureText(displayDate).width;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'; // Semi-transparent background
        ctx.fillRect(x - textWidth / 2 - 3, volumeBottom + 8, textWidth + 6, 18);

        // Draw the date text
        ctx.fillStyle = '#e0e0e0'; // Bright text
        ctx.fillText(displayDate, x, volumeBottom + 22);
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

    // Draw price label on crosshair
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
   * Setup mouse events
   */
  setupEvents() {
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
    this.canvas.addEventListener('mouseleave', (e) => this.onMouseLeave(e));
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e));

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

    // Pan if dragging
    if (this.isDragging) {
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
    this.isDragging = true;
    const rect = this.canvas.getBoundingClientRect();
    this.dragStartX = e.clientX - rect.left;
    this.dragStartY = e.clientY - rect.top;
    this.dragStartIndex = this.startIndex;
    this.dragStartMinPrice = this.minPrice;
    this.dragStartMaxPrice = this.maxPrice;
    this.canvas.style.cursor = 'grabbing';

    console.log(`🖱️ Pan started at index ${this.startIndex}, price range: ${this.minPrice.toFixed(0)} - ${this.maxPrice.toFixed(0)}`);

    // Prevent text selection while dragging
    e.preventDefault();
  }

  /**
   * Mouse up handler
   */
  onMouseUp(e) {
    this.isDragging = false;
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
    this.canvas.style.cursor = 'grab';
    this.draw();
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

    // Ensure minimum visible candles
    if (this.endIndex - this.startIndex < 10) {
      this.endIndex = Math.min(this.data.length - 1, this.startIndex + 10);
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

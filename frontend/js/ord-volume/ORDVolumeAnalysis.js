/**
 * ORDVolumeAnalysis Module
 * Completely segregated implementation for Ord Volume analysis
 * Supports both draw and auto modes for trendline analysis
 *
 * NO SHARED CODE - All functionality self-contained
 */

export class ORDVolumeAnalysis {
  constructor(candles) {
    if (!Array.isArray(candles) || candles.length === 0) {
      throw new Error('Invalid candles: must be a non-empty array of OHLCV objects');
    }

    // Validate candle structure
    const requiredKeys = ['open', 'high', 'low', 'close', 'volume'];
    if (!requiredKeys.every(key => key in candles[0])) {
      throw new Error('Invalid candle structure: must include open, high, low, close, volume');
    }

    this.candles = candles;
    this.mode = null;
    this.waveLines = [];
    this.waveData = [];
    this.retestStrength = null;
    this.retestColor = null;
  }

  /**
   * Analyze in Draw Mode
   * User provides trendlines manually
   * @param {Array} waveLines - Array of [x1, y1, x2, y2] coordinates
   * @returns {Object} Analysis results with overlays
   */
  analyzeDrawMode(waveLines) {
    if (!Array.isArray(waveLines) || waveLines.length < 3) {
      throw new Error('Draw mode requires at least 3 trendlines (Initial, Correction, Retest)');
    }

    // If more than 7 lines, use only the first 7
    if (waveLines.length > 7) {
      console.warn(`[ORD Volume] ${waveLines.length} lines drawn, using first 7 only`);
      waveLines = waveLines.slice(0, 7);
    }

    // Validate each line
    for (let i = 0; i < waveLines.length; i++) {
      const line = waveLines[i];
      if (!Array.isArray(line) || line.length !== 4) {
        throw new Error(`Invalid line at index ${i}: must be [x1, y1, x2, y2]`);
      }
    }

    this.mode = 'draw';
    this.waveLines = waveLines;

    // Calculate volume for each wave
    this.waveData = this._calculateWaveVolumes(waveLines);

    // Classify retest strength (compare 3rd wave to 1st wave)
    if (this.waveData.length >= 3) {
      this._classifyRetestStrength(this.waveData[0].avgVolume, this.waveData[2].avgVolume);
    }

    // Generate overlay data
    return this._generateOverlays();
  }

  /**
   * Analyze in Auto Mode
   * Automatically detect trendlines using price structure
   * @param {Number} lineCount - Number of lines to generate (3-7)
   * @returns {Object} Analysis results with overlays
   */
  analyzeAutoMode(lineCount = 3) {
    if (!Number.isInteger(lineCount) || lineCount < 3 || lineCount > 7) {
      throw new Error('lineCount must be an integer between 3 and 7');
    }

    this.mode = 'auto';

    // Detect swing points using fractal detection
    const swingPoints = this._detectSwingPoints();

    if (swingPoints.length < lineCount + 1) {
      throw new Error(`Not enough swing points detected. Need ${lineCount + 1}, found ${swingPoints.length}`);
    }

    // Generate trendlines from swing points (zigzag)
    this.waveLines = this._generateTrendlines(swingPoints, lineCount);

    // Calculate volume for each wave
    this.waveData = this._calculateWaveVolumes(this.waveLines);

    // Classify retest strength
    if (this.waveData.length >= 3) {
      this._classifyRetestStrength(this.waveData[0].avgVolume, this.waveData[2].avgVolume);
    }

    // Generate overlay data
    return this._generateOverlays();
  }

  /**
   * Detect swing points using 2-period fractal detection
   * @private
   * @returns {Array} Array of swing point objects {index, price, type}
   */
  _detectSwingPoints() {
    const swings = [];
    const lookback = 2; // 2-period fractals

    for (let i = lookback; i < this.candles.length - lookback; i++) {
      const candle = this.candles[i];

      // Check if this is a swing high (fractal high)
      let isSwingHigh = true;
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && this.candles[j].high >= candle.high) {
          isSwingHigh = false;
          break;
        }
      }

      if (isSwingHigh) {
        swings.push({
          index: i,
          price: candle.high,
          type: 'high'
        });
        continue;
      }

      // Check if this is a swing low (fractal low)
      let isSwingLow = true;
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && this.candles[j].low <= candle.low) {
          isSwingLow = false;
          break;
        }
      }

      if (isSwingLow) {
        swings.push({
          index: i,
          price: candle.low,
          type: 'low'
        });
      }
    }

    // Filter to create proper zigzag (alternating highs and lows)
    return this._createZigzag(swings);
  }

  /**
   * Create zigzag pattern from swing points (alternating highs/lows)
   * @private
   * @param {Array} swings - Raw swing points
   * @returns {Array} Filtered zigzag points
   */
  _createZigzag(swings) {
    if (swings.length === 0) return [];

    const zigzag = [swings[0]];
    let lastType = swings[0].type;

    for (let i = 1; i < swings.length; i++) {
      const current = swings[i];

      // If same type as last, keep the more extreme one
      if (current.type === lastType) {
        const prev = zigzag[zigzag.length - 1];
        if (current.type === 'high' && current.price > prev.price) {
          zigzag[zigzag.length - 1] = current;
        } else if (current.type === 'low' && current.price < prev.price) {
          zigzag[zigzag.length - 1] = current;
        }
      } else {
        // Different type, add to zigzag
        zigzag.push(current);
        lastType = current.type;
      }
    }

    return zigzag;
  }

  /**
   * Generate trendlines from swing points
   * @private
   * @param {Array} swingPoints - Zigzag swing points
   * @param {Number} lineCount - Number of lines to generate
   * @returns {Array} Array of [x1, y1, x2, y2] lines
   */
  _generateTrendlines(swingPoints, lineCount) {
    const lines = [];
    const rightmostCandleIndex = this.candles.length - 1;

    // PREDICTIVE MODE: Always grab the LAST N legs from the end of the data
    // For 3 lines, we need the last 3-4 swing points
    const numPointsNeeded = lineCount;
    const startIndex = Math.max(0, swingPoints.length - numPointsNeeded);
    const recentSwings = swingPoints.slice(startIndex);

    console.log('[ORD Volume] Total swing points:', swingPoints.length);
    console.log('[ORD Volume] Rightmost candle index:', rightmostCandleIndex);
    console.log('[ORD Volume] Taking last', numPointsNeeded, 'swing points starting at index', startIndex);
    console.log('[ORD Volume] Recent swings:', recentSwings.map(s => `${s.type} at index ${s.index}`));

    // IMPORTANT: Build lines in REVERSE order (from right to left)
    // This ensures the first line (Retest) starts at the rightmost candle

    // Line 1 (Retest): From last swing point TO rightmost candle
    if (recentSwings.length >= 1) {
      const lastSwing = recentSwings[recentSwings.length - 1];
      const price1 = this._snapPriceToCandle(lastSwing.index, lastSwing.price);
      const price2 = this._snapPriceToCandle(rightmostCandleIndex, this.candles[rightmostCandleIndex].close);

      lines.push([
        lastSwing.index,
        price1,
        rightmostCandleIndex,
        price2
      ]);

      console.log(`[ORD Volume] Line 0 (Retest): candle ${lastSwing.index} @ ${price1} → candle ${rightmostCandleIndex} @ ${price2}`);
    }

    // Remaining lines: Connect consecutive swing points (working backward)
    for (let i = recentSwings.length - 2; i >= 0 && lines.length < lineCount; i--) {
      const point1 = recentSwings[i];
      const point2 = recentSwings[i + 1];

      const price1 = this._snapPriceToCandle(point1.index, point1.price);
      const price2 = this._snapPriceToCandle(point2.index, point2.price);

      lines.push([
        point1.index,
        price1,
        point2.index,
        price2
      ]);

      console.log(`[ORD Volume] Line ${lines.length - 1}: candle ${point1.index} @ ${price1} → candle ${point2.index} @ ${price2}`);
    }

    // IMPORTANT: Reverse the lines array so they're in correct order (Initial, Correction, Retest)
    lines.reverse();

    console.log('[ORD Volume] Generated', lines.length, 'trendlines');
    console.log('[ORD Volume] Last line ends at candle index:', lines[lines.length - 1][2]);

    return lines;
  }

  /**
   * Snap a price to the nearest OHLC value of a candle
   * Ensures trendlines anchor to actual price data, not floating coordinates
   * @private
   * @param {Number} candleIndex - Index of the candle
   * @param {Number} targetPrice - The price to snap
   * @returns {Number} The closest OHLC value
   */
  _snapPriceToCandle(candleIndex, targetPrice) {
    if (candleIndex < 0 || candleIndex >= this.candles.length) {
      return targetPrice; // Out of bounds
    }

    const candle = this.candles[candleIndex];
    const ohlc = [candle.open, candle.high, candle.low, candle.close];

    // Find the OHLC value closest to the target price
    let closestPrice = ohlc[0];
    let minDistance = Math.abs(targetPrice - closestPrice);

    for (const price of ohlc) {
      const distance = Math.abs(targetPrice - price);
      if (distance < minDistance) {
        minDistance = distance;
        closestPrice = price;
      }
    }

    return closestPrice;
  }

  /**
   * Calculate average volume for each wave
   * @private
   * @param {Array} waveLines - Array of [x1, y1, x2, y2] lines
   * @returns {Array} Wave data with volume metrics
   */
  _calculateWaveVolumes(waveLines) {
    const waveData = [];

    for (let i = 0; i < waveLines.length; i++) {
      const line = waveLines[i];
      const [x1, y1, x2, y2] = line;

      // Get index range for this wave
      const startIdx = Math.ceil(Math.min(x1, x2));
      const endIdx = Math.floor(Math.max(x1, x2));

      // Validate indices
      if (startIdx < 0 || endIdx >= this.candles.length || startIdx > endIdx) {
        throw new Error(`Invalid line indices at wave ${i}: start=${startIdx}, end=${endIdx}`);
      }

      // Calculate average volume
      let totalVolume = 0;
      let candleCount = 0;

      for (let j = startIdx; j <= endIdx; j++) {
        totalVolume += this.candles[j].volume;
        candleCount++;
      }

      const avgVolume = candleCount > 0 ? totalVolume / candleCount : 0;

      // Determine wave label
      const labels = ['Initial', 'Correction', 'Retest', 'Wave 4', 'Wave 5', 'Wave 6', 'Wave 7'];
      const label = labels[i] || `Wave ${i + 1}`;

      waveData.push({
        index: i,
        label: label,
        line: line,
        startIdx: startIdx,
        endIdx: endIdx,
        candleCount: candleCount,
        totalVolume: totalVolume,
        avgVolume: avgVolume
      });
    }

    return waveData;
  }

  /**
   * Classify retest strength by comparing retest volume to initial volume
   * @private
   * @param {Number} initialVolume - Average volume of initial wave
   * @param {Number} retestVolume - Average volume of retest wave
   */
  _classifyRetestStrength(initialVolume, retestVolume) {
    if (initialVolume === 0) {
      this.retestStrength = 'Weak';
      this.retestColor = 'red';
      return;
    }

    const ratio = retestVolume / initialVolume;

    if (ratio >= 1.10) {
      this.retestStrength = 'Strong';
      this.retestColor = 'green';
    } else if (ratio >= 0.92) {
      this.retestStrength = 'Neutral';
      this.retestColor = 'yellow';
    } else {
      this.retestStrength = 'Weak';
      this.retestColor = 'red';
    }
  }

  /**
   * Generate overlay data for chart rendering
   * @private
   * @returns {Object} Complete overlay data
   */
  _generateOverlays() {
    const trendlines = [];
    const labels = [];

    // Generate trendlines with labels
    for (let i = 0; i < this.waveData.length; i++) {
      const wave = this.waveData[i];
      const [x1, y1, x2, y2] = wave.line;

      // Trendline object
      trendlines.push({
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        label: wave.label,
        color: '#2196f3', // Blue line
        lineWidth: 2
      });

      // Calculate label position (above/below midpoint with offset)
      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2;

      // Offset based on price range to avoid overlap
      const priceRange = Math.abs(y2 - y1);
      const offset = priceRange * 0.1; // 10% offset
      const labelY = y1 < y2 ? midY + offset : midY - offset;

      // Volume label
      labels.push({
        x: midX,
        y: labelY,
        text: `AvgVol${wave.label}: ${this._formatVolume(wave.avgVolume)}`,
        color: '#ffffff',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        fontSize: 12,
        draggable: true
      });
    }

    // Add retest strength label (positioned at retest wave if available)
    if (this.retestStrength && this.waveData.length >= 3) {
      const retestWave = this.waveData[2];
      const [x1, y1, x2, y2] = retestWave.line;
      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2;
      const priceRange = Math.abs(y2 - y1);
      const offset = priceRange * 0.2; // Larger offset for strength label
      const labelY = y1 < y2 ? midY + offset : midY - offset;

      labels.push({
        x: midX,
        y: labelY,
        text: `RetestStrength: ${this.retestStrength}`,
        color: this._getColorHex(this.retestColor),
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        fontSize: 14,
        fontWeight: 'bold',
        draggable: true
      });
    }

    // Return complete analysis result
    return {
      mode: this.mode,
      trendlines: trendlines,
      labels: labels,
      strength: this.retestStrength,
      color: this.retestColor,
      waveData: this.waveData,
      lines: this.waveLines // For persistence in draw mode
    };
  }

  /**
   * Format volume number for display
   * @private
   * @param {Number} volume
   * @returns {String} Formatted volume string
   */
  _formatVolume(volume) {
    if (volume >= 1e9) {
      return (volume / 1e9).toFixed(2) + 'B';
    } else if (volume >= 1e6) {
      return (volume / 1e6).toFixed(2) + 'M';
    } else if (volume >= 1e3) {
      return (volume / 1e3).toFixed(2) + 'K';
    } else {
      return volume.toFixed(2);
    }
  }

  /**
   * Convert color name to hex
   * @private
   * @param {String} colorName
   * @returns {String} Hex color code
   */
  _getColorHex(colorName) {
    const colorMap = {
      'green': '#00c853',
      'yellow': '#ffd600',
      'red': '#ff1744'
    };
    return colorMap[colorName] || '#ffffff';
  }
}

/**
 * EXAMPLE USAGE:
 *
 * // Example candle data (OHLCV)
 * const candles = [
 *   {open: 100, high: 105, low: 98, close: 103, volume: 1000000},
 *   {open: 103, high: 108, low: 102, close: 106, volume: 1200000},
 *   // ... more candles
 * ];
 *
 * // Create analyzer instance
 * const analyzer = new ORDVolumeAnalysis(candles);
 *
 * // DRAW MODE: User-drawn trendlines
 * const userLines = [
 *   [0, 100, 10, 110],   // Initial wave
 *   [10, 110, 20, 105],  // Correction wave
 *   [20, 105, 30, 115]   // Retest wave
 * ];
 * const drawResult = analyzer.analyzeDrawMode(userLines);
 * console.log(drawResult);
 * // Returns: {mode, trendlines, labels, strength, color, waveData, lines}
 *
 * // AUTO MODE: Automatic detection
 * const analyzer2 = new ORDVolumeAnalysis(candles);
 * const autoResult = analyzer2.analyzeAutoMode(3); // 3 trendlines
 * console.log(autoResult);
 * // Returns: {mode, trendlines, labels, strength, color, waveData, lines}
 *
 * // Access specific data
 * console.log('Retest Strength:', autoResult.strength); // 'Strong', 'Neutral', or 'Weak'
 * console.log('Color:', autoResult.color); // 'green', 'yellow', or 'red'
 * console.log('Trendlines:', autoResult.trendlines); // Array of line objects for rendering
 * console.log('Labels:', autoResult.labels); // Array of text overlay objects
 */

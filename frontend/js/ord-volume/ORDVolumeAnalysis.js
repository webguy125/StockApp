/**
 * ORDVolumeAnalysis Module
 * Completely segregated implementation for Ord Volume analysis
 * Supports both draw and auto modes for trendline analysis
 *
 * NO SHARED CODE - All functionality self-contained
 */

import { ORDVolumeIndicators } from './ORDVolumeIndicators.js';
import { ORDVolumeSignals } from './ORDVolumeSignals.js';

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
    this.sensitivity = 'high'; // Default: 'normal' (lookback 2) or 'high' (lookback 1)
  }

  /**
   * Set sensitivity for swing point detection
   * @param {String} sensitivity - 'normal' or 'high'
   */
  setSensitivity(sensitivity) {
    if (sensitivity !== 'normal' && sensitivity !== 'high') {
      console.warn(`[ORD Volume] Invalid sensitivity: ${sensitivity}, using 'high'`);
      this.sensitivity = 'high';
    } else {
      this.sensitivity = sensitivity;
    }
    console.log(`[ORD Volume] Sensitivity set to: ${this.sensitivity} (lookback ${this.sensitivity === 'high' ? 1 : 2})`);
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

    // If more than 100 lines, use only the first 100
    if (waveLines.length > 100) {
      console.warn(`[ORD Volume] ${waveLines.length} lines drawn, using first 100 only`);
      waveLines = waveLines.slice(0, 100);
    }

    // Validate each line
    for (let i = 0; i < waveLines.length; i++) {
      const line = waveLines[i];
      if (!Array.isArray(line) || line.length !== 4) {
        throw new Error(`Invalid line at index ${i}: must be [x1, y1, x2, y2]`);
      }
    }

    // CRITICAL: Sort lines by starting X position (chronological order)
    // This ensures lines are numbered 1, 2, 3... from left to right on chart
    waveLines = waveLines.slice().sort((a, b) => {
      const startA = Math.min(a[0], a[2]); // Leftmost X of line A
      const startB = Math.min(b[0], b[2]); // Leftmost X of line B
      return startA - startB;
    });

    console.log('[ORD Volume] Lines sorted by chronological order (left to right)');

    this.mode = 'draw';
    this.waveLines = waveLines;

    // Calculate volume for each wave
    this.waveData = this._calculateWaveVolumes(waveLines);

    // Apply Elliott Wave labeling
    this._applyElliottWaveLabeling(this.waveData);

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
   * @param {Number} lineCount - Number of lines to generate (3-100)
   * @returns {Object} Analysis results with overlays
   */
  analyzeAutoMode(lineCount = 3) {
    if (!Number.isInteger(lineCount) || lineCount < 3 || lineCount > 100) {
      throw new Error('lineCount must be an integer between 3 and 100');
    }

    // CRITICAL: Check bar count BEFORE any calculations
    const MAX_BARS = 4500; // Balanced limit - Daily works (4077 bars), but 15m/5m/1m blocked
    if (this.candles.length > MAX_BARS) {
      throw new Error(`Dataset too large. Maximum ${MAX_BARS} bars supported, got ${this.candles.length} bars. Use Daily or Weekly timeframe.`);
    }

    this.mode = 'auto';

    // Detect swing points using fractal detection
    const swingPoints = this._detectSwingPoints();

    // GRACEFUL FALLBACK: Use whatever swing points are available
    const minRequired = lineCount + 1;
    if (swingPoints.length < minRequired) {
      const maxPossible = Math.max(3, swingPoints.length - 1);
      console.warn(`[ORD Volume] Requested ${lineCount} lines but only ${swingPoints.length} swing points detected. Using ${maxPossible} lines instead.`);

      if (swingPoints.length < 4) {
        throw new Error(`Insufficient swing points for analysis. Found ${swingPoints.length}, minimum 4 required. Try a longer timeframe or use Daily/Weekly data.`);
      }

      // Adjust lineCount to maximum available
      lineCount = maxPossible;
    }

    // Generate trendlines from swing points (zigzag)
    this.waveLines = this._generateTrendlines(swingPoints, lineCount);

    // Calculate volume for each wave
    this.waveData = this._calculateWaveVolumes(this.waveLines);

    // Apply Elliott Wave labeling
    this._applyElliottWaveLabeling(this.waveData);

    // Classify retest strength
    if (this.waveData.length >= 3) {
      this._classifyRetestStrength(this.waveData[0].avgVolume, this.waveData[2].avgVolume);
    }

    // Generate overlay data
    return this._generateOverlays();
  }

  /**
   * Detect swing points using adaptive fractal detection
   * @private
   * @returns {Array} Array of swing point objects {index, price, type}
   */
  _detectSwingPoints() {
    const swings = [];
    // Use sensitivity setting: 'high' = lookback 1 (more swings), 'normal' = lookback 2 (fewer swings)
    const lookback = this.sensitivity === 'high' ? 1 : 2;

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

    console.log(`[ORD Volume] Detected ${swings.length} raw swing points (lookback=${lookback}, sensitivity=${this.sensitivity})`);

    // Filter to create proper zigzag (alternating highs and lows)
    const zigzag = this._createZigzag(swings);

    console.log(`[ORD Volume] After zigzag filter: ${zigzag.length} swing points available for trendline generation`);

    return zigzag;
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

      waveData.push({
        index: i,
        label: null, // Will be assigned by Elliott Wave labeling
        elliottLabel: null, // Elliott Wave notation (1-5, A-C)
        isImpulse: null, // true for 1-5, false for A-C
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
   * Apply Elliott Wave labeling to wave data
   * Finds valid Wave 1 patterns and labels complete 8-wave cycles
   * Waves between cycles are left as neutral (unlabeled)
   * @private
   * @param {Array} waveData - Array of wave objects with price/volume data
   */
  _applyElliottWaveLabeling(waveData) {
    if (waveData.length < 3) return; // Need at least 3 waves

    const impulseLabels = ['1', '2', '3', '4', '5'];
    const correctiveLabels = ['A', 'B', 'C'];

    let i = 0;

    while (i < waveData.length) {
      // Find next valid Wave 1 starting from current position
      let wave1Index = this._findFirstWave1FromIndex(waveData, i);

      if (wave1Index === -1) {
        // No more Wave 1 patterns found, rest are neutral
        console.log(`[Elliott Wave] No more Wave 1 patterns found after index ${i}`);
        break;
      }

      console.log(`[Elliott Wave] Found Wave 1 at index ${wave1Index}`);

      // Label this complete 8-wave cycle (1-5, A-C)
      for (let cyclePos = 0; cyclePos < 8 && (wave1Index + cyclePos) < waveData.length; cyclePos++) {
        const wave = waveData[wave1Index + cyclePos];

        if (cyclePos < 5) {
          // Impulse waves (1-5)
          wave.elliottLabel = impulseLabels[cyclePos];
          wave.label = impulseLabels[cyclePos];
          wave.isImpulse = true;
        } else {
          // Corrective waves (A-C)
          wave.elliottLabel = correctiveLabels[cyclePos - 5];
          wave.label = correctiveLabels[cyclePos - 5];
          wave.isImpulse = false;
        }
      }

      // Move past this cycle and look for next Wave 1
      // Start searching after Wave C (8 waves from current Wave 1)
      i = wave1Index + 8;
    }

    console.log('[Elliott Wave] Labeling complete. Unlabeled waves are neutral.');
  }

  /**
   * Find the first valid Wave 1 in the wave data starting from a given index
   * Wave 1 criteria: First significant impulse after a corrective low with higher high + higher low
   * @private
   * @param {Array} waveData - Array of wave objects
   * @param {Number} startIndex - Index to start searching from
   * @returns {Number} Index of first Wave 1, or -1 if not found
   */
  _findFirstWave1FromIndex(waveData, startIndex = 0) {
    // Look for first upward impulse that creates a higher high
    for (let i = Math.max(1, startIndex); i < waveData.length - 1; i++) {
      const prevWave = waveData[i - 1];
      const currentWave = waveData[i];
      const nextWave = waveData[i + 1];

      const [px1, py1, px2, py2] = prevWave.line;
      const [cx1, cy1, cx2, cy2] = currentWave.line;
      const [nx1, ny1, nx2, ny2] = nextWave.line;

      // Check if current wave is an upward impulse (higher high than previous)
      const isUpward = cy2 > cy1;
      const higherHigh = cy2 > py2;

      // Check if next wave (Wave 2) retraces but doesn't go below Wave 1 start
      const wave2Retraces = ny2 < cy2;
      const wave2DoesntExceed100Percent = ny2 > cy1;

      if (isUpward && higherHigh && wave2Retraces && wave2DoesntExceed100Percent) {
        console.log(`[Elliott Wave] Found potential Wave 1 at index ${i}`);
        return i;
      }
    }

    // If no perfect Wave 1 found, start at first upward wave
    for (let i = 0; i < waveData.length; i++) {
      const [x1, y1, x2, y2] = waveData[i].line;
      if (y2 > y1) {
        console.log(`[Elliott Wave] Using first upward wave at index ${i} as Wave 1`);
        return i;
      }
    }

    return -1; // No valid Wave 1 found
  }

  /**
   * Generate overlay data for chart rendering
   * @private
   * @returns {Object} Complete overlay data
   */
  _generateOverlays() {
    const trendlines = [];
    const labels = [];

    // Elliott Wave color palette
    const impulseColor = '#FFD700';  // Gold for impulse waves (1-5)
    const correctiveColor = '#87CEEB'; // Sky blue for corrective waves (A-C)

    // Generate trendlines with labels
    for (let i = 0; i < this.waveData.length; i++) {
      const wave = this.waveData[i];
      const [x1, y1, x2, y2] = wave.line;

      // Get color based on Elliott Wave type (impulse vs corrective)
      // If no label (before Wave 1), use gray
      let waveColor = '#888888'; // Default gray for unlabeled waves
      if (wave.isImpulse === true) {
        waveColor = impulseColor; // Gold for 1-5
      } else if (wave.isImpulse === false) {
        waveColor = correctiveColor; // Blue for A-C
      }

      // Trendline object with enhanced styling
      trendlines.push({
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        label: wave.label,
        color: waveColor,
        lineWidth: 3, // Thicker professional line
        shadowColor: 'rgba(0, 0, 0, 0.3)',
        shadowBlur: 2
      });

      // Simple, intuitive label positioning
      // Position at center of line, check local candles, place above or below with minimal offset
      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2;
      const midIdx = Math.floor(midX);

      // Check small local area (3 candles around center)
      let maxHighLocal = -Infinity;
      let minLowLocal = Infinity;

      for (let offset = -1; offset <= 1; offset++) {
        const idx = midIdx + offset;
        if (idx >= 0 && idx < this.candles.length && this.candles[idx]) {
          maxHighLocal = Math.max(maxHighLocal, this.candles[idx].high);
          minLowLocal = Math.min(minLowLocal, this.candles[idx].low);
        }
      }

      let labelY = midY;
      if (maxHighLocal !== -Infinity && minLowLocal !== Infinity) {
        // Simple decision: which side has more space?
        const spaceAbove = Math.abs(midY - maxHighLocal);
        const spaceBelow = Math.abs(midY - minLowLocal);

        // Use small, consistent offset to stay close to line
        const localRange = maxHighLocal - minLowLocal;
        const smallOffset = localRange * 0.25; // Just 25% to stay close

        // Position on the side with more space
        if (spaceAbove > spaceBelow) {
          labelY = maxHighLocal - smallOffset;
        } else {
          labelY = minLowLocal + smallOffset;
        }
      }

      const bestX = midX;
      const bestY = labelY;

      // Volume label with rotation matching the line angle (no background)
      // Positioned at center, avoiding candles
      labels.push({
        x: bestX,
        y: bestY,
        lineX1: x1, // Store line endpoints for angle calculation
        lineY1: y1,
        lineX2: x2,
        lineY2: y2,
        text: `${this._formatVolume(wave.avgVolume)}`,
        color: waveColor, // Use wave color for text
        fontSize: 13,
        fontWeight: 'bold',
        draggable: true,
        isVolumeLabel: true, // Marker for click detection
        waveIndex: i // Track which wave this label belongs to
      });
    }

    // Add percentage comparison labels and wave labels at the END of each wave
    // At the connection point where a wave ends
    for (let i = 1; i < this.waveData.length; i++) {
      const endingWave = this.waveData[i]; // This wave is ENDING at this connection point
      const waveBeforeEnding = this.waveData[i - 1]; // Wave before the ending wave

      // The connection point is where the ending wave ends
      // Ending wave: [ex1, ey1, ex2, ey2]
      // Connection point is at (ex2, ey2)
      const [ex1, ey1, ex2, ey2] = endingWave.line;

      // Calculate percentage difference (ending wave compared to wave before it)
      const percentage = ((endingWave.avgVolume / waveBeforeEnding.avgVolume) * 100).toFixed(1);

      // Determine color based on strength classification
      let strengthColor;
      const ratio = endingWave.avgVolume / waveBeforeEnding.avgVolume;
      if (ratio >= 1.10) {
        strengthColor = this._getColorHex('green'); // Strong
      } else if (ratio >= 0.92) {
        strengthColor = this._getColorHex('yellow'); // Neutral
      } else {
        strengthColor = this._getColorHex('red'); // Weak
      }

      // Determine direction of the ending wave
      // If ending wave is going UP (ey2 > ey1), put label ABOVE
      // If ending wave is going DOWN (ey2 < ey1), put label BELOW
      const isEndingWaveUp = ey2 > ey1;

      // Use fixed pixel offset (calculated from 35 lines baseline)
      // Store connection point and direction info - bridge will apply pixel offset
      // Percentage label at outer position (30 pixels from connection point)
      labels.push({
        x: ex2, // X position at the END of the ending wave
        y: ey2, // Y position at END of ending wave (bridge will apply pixel offset)
        text: `${percentage}%`,
        color: strengthColor, // Color-coded based on strength
        fontSize: 10, // Smaller text
        fontWeight: 'bold',
        angle: 0, // Horizontal
        draggable: true,
        isPercentageLabel: true, // Marker for custom comparison override
        waveIndex: i, // Track which wave this label belongs to
        isUpward: isEndingWaveUp, // Direction for pixel offset
        pixelOffset: 30 // Fixed pixel offset from connection point
      });

      // Elliott Wave label for the ENDING wave (between percentage and pivot point)
      if (endingWave.label) {
        labels.push({
          x: ex2,
          y: ey2, // Y position at END of ending wave (bridge will apply pixel offset)
          text: endingWave.label, // "1", "2", "3", "4", "5", "A", "B", "C"
          color: '#FFFFFF', // White for visibility
          fontSize: 14,
          fontWeight: 'bold',
          angle: 0,
          draggable: true,
          isWaveLabel: true, // Marker for wave labels
          isUpward: isEndingWaveUp, // Direction for pixel offset
          pixelOffset: 15 // Halfway between connection and percentage label
        });
      }
    }

    // Calculate ORD Volume indicators (Timothy Ord's proprietary methodology)
    const ordIndicators = new ORDVolumeIndicators(this.candles);
    const ordAnalysis = ordIndicators.calculate();

    // Generate trade signals with confluence scoring
    const signalGenerator = new ORDVolumeSignals(this.candles, ordIndicators, this.waveData);
    const tradeSignals = signalGenerator.generateSignals();

    // Return complete analysis result
    return {
      mode: this.mode,
      trendlines: trendlines,
      labels: labels,
      strength: this.retestStrength,
      color: this.retestColor,
      waveData: this.waveData,
      lines: this.waveLines, // For persistence in draw mode
      tradeSignals: tradeSignals, // Professional ORD Volume signals
      ordVolumeState: ordAnalysis.currentState // Current ORD Volume state
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
      'green': '#10B981',  // Professional emerald green
      'yellow': '#F59E0B', // Professional amber
      'red': '#EF4444'     // Professional red
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

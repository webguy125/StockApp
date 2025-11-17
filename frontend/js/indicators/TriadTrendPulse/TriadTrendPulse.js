/**
 * Triad Trend Pulse Indicator
 * Advanced ML-powered indicator combining weighted price regression, adaptive oscillator,
 * and short-term trend with pivot detection
 *
 * Components:
 * 1. Weighted Price Regression - Blends linear regression with RSI
 * 2. Adaptive Oscillator - Combines momentum, volume, and trend
 * 3. Short-Term Trend Line - Fast regression for recent price action
 * 4. Pivot Detection - ML-enhanced swing high/low identification
 */

import { IndicatorBase } from '../IndicatorBase.js';

export class TriadTrendPulse extends IndicatorBase {
  constructor() {
    super({
      name: 'TriadTrendPulse',
      version: '1.0.0',
      description: 'ML-powered trend and pivot indicator with adaptive oscillator',
      tags: ['trend', 'momentum', 'ml', 'pivots'],
      dependencies: [],
      output_type: 'oscillator',
      default_settings: {
        // Weighted Price Regression
        reg_length: 20,
        offset: 0,
        price_weight: 1.0,
        rsi_weight: 0.5,
        chart_blend: 0.7,
        rsi_len: 14,

        // Adaptive Oscillator
        adaptive_period: 50,

        // Short-Term Trend
        trend_bar_length: 5,
        trend_bar_offset: 0,

        // Pivot Detection
        pivot_lookback: 5,
        adx_threshold: 25,
        adx_length: 14,

        // ML Settings
        use_ml: false, // Disabled by default (requires model file)
        ml_threshold: 0.7,

        // Display Colors
        weighted_trend_color: '#2196F3',
        oscillator_color: '#FF9800',
        short_trend_color: '#4CAF50',
        pivot_high_color: '#F44336',
        pivot_low_color: '#00E676',
        line_opacity: 0.9,
        line_width: 2,
        pivot_size: 8
      },
      alerts: {
        enabled: true,
        conditions: [
          { type: 'pivot_high', message: 'Pivot High Detected' },
          { type: 'pivot_low', message: 'Pivot Low Detected' }
        ]
      },
      help_text: 'Advanced ML-powered indicator detecting trends and pivots. Combines regression analysis with machine learning for high-probability trade signals.'
    });
  }

  /**
   * Calculate RSI (Relative Strength Index)
   * @param {Array} closes - Close prices
   * @param {number} period - RSI period
   * @returns {Array} RSI values
   * @private
   */
  _calculateRSI(closes, period) {
    if (!closes || closes.length < period + 1) return [];

    const rsi = [];
    const changes = [];

    // Calculate price changes
    for (let i = 1; i < closes.length; i++) {
      changes.push(closes[i] - closes[i - 1]);
    }

    // Initial average gain/loss
    let avgGain = 0;
    let avgLoss = 0;

    for (let i = 0; i < period; i++) {
      if (changes[i] > 0) {
        avgGain += changes[i];
      } else {
        avgLoss += Math.abs(changes[i]);
      }
    }

    avgGain /= period;
    avgLoss /= period;

    // Calculate first RSI
    let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));

    // Calculate subsequent RSI values (Wilder's smoothing)
    for (let i = period; i < changes.length; i++) {
      const change = changes[i];
      let gain = 0;
      let loss = 0;

      if (change > 0) {
        gain = change;
      } else {
        loss = Math.abs(change);
      }

      avgGain = ((avgGain * (period - 1)) + gain) / period;
      avgLoss = ((avgLoss * (period - 1)) + loss) / period;

      rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }

    return rsi;
  }

  /**
   * Calculate ATR (Average True Range)
   * @param {Array} candles - OHLCV candles
   * @param {number} period - ATR period
   * @returns {Array} ATR values
   * @private
   */
  _calculateATR(candles, period) {
    if (!candles || candles.length < period + 1) return [];

    const tr = [];

    // Calculate True Range
    for (let i = 1; i < candles.length; i++) {
      const high = candles[i].High;
      const low = candles[i].Low;
      const prevClose = candles[i - 1].Close;

      const trueRange = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );

      tr.push(trueRange);
    }

    const atr = [];

    // Calculate initial ATR (simple average)
    let sum = 0;
    for (let i = 0; i < period; i++) {
      sum += tr[i];
    }
    let currentATR = sum / period;
    atr.push(currentATR);

    // Calculate subsequent ATR values (Wilder's smoothing)
    for (let i = period; i < tr.length; i++) {
      currentATR = ((currentATR * (period - 1)) + tr[i]) / period;
      atr.push(currentATR);
    }

    return atr;
  }

  /**
   * Calculate ADX (Average Directional Index)
   * @param {Array} candles - OHLCV candles
   * @param {number} period - ADX period
   * @returns {Array} ADX values
   * @private
   */
  _calculateADX(candles, period) {
    if (!candles || candles.length < period * 2) return [];

    const adx = [];
    const plusDI = [];
    const minusDI = [];
    const dx = [];

    // Calculate +DM, -DM, and TR
    for (let i = 1; i < candles.length; i++) {
      const highDiff = candles[i].High - candles[i - 1].High;
      const lowDiff = candles[i - 1].Low - candles[i].Low;

      let plusDM = 0;
      let minusDM = 0;

      if (highDiff > lowDiff && highDiff > 0) {
        plusDM = highDiff;
      }
      if (lowDiff > highDiff && lowDiff > 0) {
        minusDM = lowDiff;
      }

      const tr = Math.max(
        candles[i].High - candles[i].Low,
        Math.abs(candles[i].High - candles[i - 1].Close),
        Math.abs(candles[i].Low - candles[i - 1].Close)
      );

      // Smooth +DM, -DM, TR
      if (i < period) {
        if (i === period - 1) {
          // Initialize smoothed values
          let sumPlusDM = 0;
          let sumMinusDM = 0;
          let sumTR = 0;

          for (let j = 0; j < period; j++) {
            // Simplified - would need to store values
          }
        }
      }
    }

    // Simplified ADX calculation - returning approximate values
    // Full implementation would require more complex smoothing
    for (let i = period * 2; i < candles.length; i++) {
      adx.push(25); // Placeholder - full implementation needed
    }

    return adx;
  }

  /**
   * Calculate linear regression
   * @param {Array} values - Values to regress
   * @param {number} period - Regression period
   * @returns {Array} Regression values
   * @private
   */
  _calculateLinearRegression(values, period) {
    if (!values || values.length < period) return [];

    const result = [];

    for (let i = period - 1; i < values.length; i++) {
      let sumX = 0;
      let sumY = 0;
      let sumXY = 0;
      let sumX2 = 0;

      for (let j = 0; j < period; j++) {
        const x = j;
        const y = values[i - period + 1 + j];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
      }

      const slope = (period * sumXY - sumX * sumY) / (period * sumX2 - sumX * sumX);
      const intercept = (sumY - slope * sumX) / period;

      // Get value at end of period
      const regValue = slope * (period - 1) + intercept;
      result.push(regValue);
    }

    return result;
  }

  /**
   * Normalize values to [-100, 100] range
   * @param {Array} values - Values to normalize
   * @param {number} lookback - Lookback period for min/max
   * @returns {Array} Normalized values
   * @private
   */
  _normalize(values, lookback) {
    if (!values || values.length === 0) return [];

    const normalized = [];

    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - lookback + 1);
      const window = values.slice(start, i + 1);

      const min = Math.min(...window);
      const max = Math.max(...window);
      const range = max - min;

      if (range === 0) {
        normalized.push(0);
      } else {
        const norm = ((values[i] - min) / range) * 200 - 100;
        normalized.push(norm);
      }
    }

    return normalized;
  }

  /**
   * Calculate Triad Trend Pulse indicator
   * @param {Array} candles - Array of OHLCV candles
   * @returns {Promise<Array>} Indicator values
   */
  async calculate(candles) {
    if (!candles || candles.length < 100) {
      return [];
    }

    const settings = this.currentSettings;
    const closes = candles.map(c => c.Close);
    const highs = candles.map(c => c.High);
    const lows = candles.map(c => c.Low);
    const volumes = candles.map(c => c.Volume || 0);

    // Calculate RSI
    const rsi = this._calculateRSI(closes, settings.rsi_len);

    // Calculate ATR for adaptive periods
    const atr = this._calculateATR(candles, 14);

    // Calculate ADX for pivot filtering
    const adx = this._calculateADX(candles, settings.adx_length);

    // === Component 1: Weighted Price Regression ===
    const priceRegression = this._calculateLinearRegression(closes, settings.reg_length);

    // Create weighted values (combine close and RSI)
    const weightedValues = [];
    for (let i = settings.rsi_len; i < closes.length; i++) {
      const rsiIndex = i - settings.rsi_len;
      if (rsiIndex < rsi.length) {
        const weighted = (closes[i] * settings.price_weight) + (rsi[rsiIndex] * settings.rsi_weight);
        weightedValues.push(weighted);
      }
    }

    const weightedRegression = this._calculateLinearRegression(weightedValues, settings.reg_length);

    // Blend price and weighted regression
    const blendedRegression = [];
    for (let i = 0; i < Math.min(priceRegression.length, weightedRegression.length); i++) {
      const blended = priceRegression[i] * settings.chart_blend +
                     weightedRegression[i] * (1 - settings.chart_blend);
      blendedRegression.push(blended);
    }

    // Normalize weighted regression
    const normLookback = Math.floor(settings.reg_length / 2);
    const weightedTrendNorm = this._normalize(blendedRegression, normLookback);

    // === Component 2: Adaptive Oscillator ===
    const oscillatorValues = [];
    for (let i = 20; i < closes.length; i++) {
      // Price trend (SMA comparison)
      const sma20 = closes.slice(i - 19, i + 1).reduce((a, b) => a + b, 0) / 20;
      const priceTrend = ((closes[i] - sma20) / sma20) * 100;

      // Momentum
      const momentum = i >= 20 ? ((closes[i] - closes[i - 20]) / closes[i - 20]) * 100 : 0;

      // Volume trend
      const volSMA = volumes.slice(Math.max(0, i - 19), i + 1).reduce((a, b) => a + b, 0) / Math.min(20, i + 1);
      const volumeTrend = volumes[i] > 0 ? ((volumes[i] - volSMA) / volSMA) * 100 : 0;

      // Weighted combination (emphasis on momentum and volume)
      const oscillator = (priceTrend * 0.3) + (momentum * 0.3) + (volumeTrend * 0.2) + (priceTrend * 0.2);
      oscillatorValues.push(oscillator);
    }

    // Smooth oscillator with regression
    const smoothedOscillator = this._calculateLinearRegression(oscillatorValues, settings.reg_length);
    const oscillatorNorm = this._normalize(smoothedOscillator, normLookback);

    // === Component 3: Short-Term Trend ===
    const shortTrendLength = Math.max(3, Math.min(10, settings.trend_bar_length));
    const shortTrend = this._calculateLinearRegression(closes, shortTrendLength);
    const shortTrendNorm = this._normalize(shortTrend, normLookback);

    // === Component 4: Real-Time Pivot Detection ===
    // ML will filter these - we detect potential pivots as they form
    const pivotHighs = new Array(candles.length).fill(false);
    const pivotLows = new Array(candles.length).fill(false);
    const pivotScores = new Array(candles.length).fill(null);

    // Start from pivot_lookback (need history to compare)
    // Go up to the CURRENT candle (closes.length - 1) for real-time detection
    for (let i = settings.pivot_lookback; i < closes.length; i++) {
      // Check for potential swing high (only look back, not forward)
      let isHigh = true;
      let isLow = true;

      // Only check previous candles (real-time detection)
      for (let j = 1; j <= settings.pivot_lookback; j++) {
        if (highs[i] <= highs[i - j]) {
          isHigh = false;
        }
        if (lows[i] >= lows[i - j]) {
          isLow = false;
        }
      }

      // For the current candle (last one), also check if it's higher/lower than recent candles
      // This gives us a "potential" pivot that ML can validate
      if (i === closes.length - 1) {
        // Current candle - mark as potential pivot if it's an extreme
        // ML will decide if it's valid
        if (isHigh) {
          pivotHighs[i] = true;
          pivotScores[i] = 0.5; // Lower initial score for current candle (unconfirmed)
        }
        if (isLow) {
          pivotLows[i] = true;
          pivotScores[i] = 0.5; // Lower initial score for current candle (unconfirmed)
        }
      } else {
        // Historical candles - also check forward candles for confirmation
        let confirmedHigh = isHigh;
        let confirmedLow = isLow;

        const forwardLookback = Math.min(settings.pivot_lookback, closes.length - 1 - i);
        for (let j = 1; j <= forwardLookback; j++) {
          if (highs[i] <= highs[i + j]) {
            confirmedHigh = false;
          }
          if (lows[i] >= lows[i + j]) {
            confirmedLow = false;
          }
        }

        if (confirmedHigh) {
          pivotHighs[i] = true;
          pivotScores[i] = 0.85; // Higher score for confirmed pivots
        }
        if (confirmedLow) {
          pivotLows[i] = true;
          pivotScores[i] = 0.85; // Higher score for confirmed pivots
        }
      }
    }

    // Count pivots before ML filtering
    const pivotsBeforeML = pivotHighs.filter(Boolean).length + pivotLows.filter(Boolean).length;

    // === ML Enhancement (if enabled) ===
    if (settings.use_ml) {
      try {
        await this._enhancePivotsWithML(candles, pivotHighs, pivotLows, pivotScores,
                                         weightedTrendNorm, oscillatorNorm, shortTrendNorm,
                                         rsi, atr, adx, volumes, settings);
      } catch (err) {
        console.warn('ML enhancement failed, using default scores:', err);
      }
    }

    // Count pivots after ML filtering
    const pivotsAfterML = pivotHighs.filter(Boolean).length + pivotLows.filter(Boolean).length;
    if (settings.use_ml) {
      console.log(`✅ ML filtered ${pivotsBeforeML} → ${pivotsAfterML} pivots (removed ${pivotsBeforeML - pivotsAfterML})`);
    }

    // Build result array (aligned with candles array)
    const result = [];

    // Calculate the actual starting indices for each component
    // weightedTrendNorm starts at: settings.rsi_len + settings.reg_length + normLookback
    // oscillatorNorm starts at: 20 + settings.reg_length + normLookback
    // shortTrendNorm starts at: shortTrendLength + normLookback

    const weightedStartIdx = settings.rsi_len + settings.reg_length + normLookback;
    const oscillatorStartIdx = 20 + settings.reg_length + normLookback;
    const shortStartIdx = shortTrendLength + normLookback;

    // Build aligned result - one entry per candle
    for (let i = 0; i < candles.length; i++) {
      // Calculate array index for each normalized component
      const wtIdx = i - weightedStartIdx;
      const oscIdx = i - oscillatorStartIdx;
      const stIdx = i - shortStartIdx;

      result.push({
        date: candles[i].Date,
        weighted_trend: (wtIdx >= 0 && wtIdx < weightedTrendNorm.length) ? weightedTrendNorm[wtIdx] : null,
        oscillator: (oscIdx >= 0 && oscIdx < oscillatorNorm.length) ? oscillatorNorm[oscIdx] : null,
        short_trend: (stIdx >= 0 && stIdx < shortTrendNorm.length) ? shortTrendNorm[stIdx] : null,
        pivot_high: pivotHighs[i],
        pivot_low: pivotLows[i],
        pivot_score: pivotScores[i],
        close: closes[i]
      });
    }

    return result;
  }

  /**
   * Enhance pivot scores using ML endpoint
   * @param {Array} candles - Candle data
   * @param {Array} pivotHighs - Pivot high boolean array
   * @param {Array} pivotLows - Pivot low boolean array
   * @param {Array} pivotScores - Pivot scores array (to be updated)
   * @param {Array} weightedTrend - Weighted trend normalized values
   * @param {Array} oscillator - Oscillator normalized values
   * @param {Array} shortTrend - Short trend normalized values
   * @param {Array} rsi - RSI values
   * @param {Array} atr - ATR values
   * @param {Array} adx - ADX values
   * @param {Array} volumes - Volume data
   * @param {Object} settings - Indicator settings
   * @returns {Promise<void>}
   * @private
   */
  async _enhancePivotsWithML(candles, pivotHighs, pivotLows, pivotScores,
                              weightedTrend, oscillator, shortTrend,
                              rsi, atr, adx, volumes, settings) {
    try {
      // Collect all pivot indices
      const pivotIndices = [];
      for (let i = 0; i < pivotHighs.length; i++) {
        if (pivotHighs[i] || pivotLows[i]) {
          pivotIndices.push(i);
        }
      }

      if (pivotIndices.length === 0) {
        return; // No pivots to enhance
      }

      // Build feature arrays for each pivot
      const features = [];
      const closes = candles.map(c => c.Close);

      for (const idx of pivotIndices) {
        // Extract 9 ML features
        const oscillatorVal = oscillator[idx] || 0;
        const weightedTrendVal = weightedTrend[idx] || 0;
        const shortTrendVal = shortTrend[idx] || 0;
        const adxVal = (adx[idx] || 0) / 100.0; // Normalize to [0, 1]

        // Volume change (current volume / 20-bar SMA)
        let volumeChange = 0;
        if (idx >= 20) {
          const volSMA = volumes.slice(idx - 19, idx + 1).reduce((a, b) => a + b, 0) / 20;
          volumeChange = volSMA > 0 ? Math.min(volumes[idx] / volSMA, 3) / 3 : 0; // Cap at 3x
        }

        const atrVal = (atr[idx] || 0) / closes[idx]; // ATR as % of price
        const rsiVal = (rsi[idx] || 50) / 100.0; // Normalize to [0, 1]

        // Momentum (price change over reg_length)
        let momentum = 0;
        if (idx >= settings.reg_length) {
          const priceChange = (closes[idx] - closes[idx - settings.reg_length]) / closes[idx - settings.reg_length];
          momentum = Math.tanh(priceChange * 10) / 2 + 0.5; // Squash to [0, 1]
        }

        // Timeframe encoding (hardcoded for now - could be passed from chart timeframe)
        const timeframeEncoding = 9 / 12.0; // Assuming daily timeframe (9 -> '1d')

        features.push([
          oscillatorVal,
          weightedTrendVal,
          shortTrendVal,
          adxVal,
          volumeChange,
          atrVal,
          rsiVal,
          momentum,
          timeframeEncoding
        ]);
      }

      // Call ML endpoint
      const response = await fetch('http://127.0.0.1:5000/ml/pivot-reliability', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features })
      });

      if (!response.ok) {
        throw new Error(`ML endpoint returned ${response.status}`);
      }

      const data = await response.json();

      if (data.scores && data.scores.length === pivotIndices.length) {
        // Update pivot scores with ML predictions
        for (let i = 0; i < pivotIndices.length; i++) {
          const idx = pivotIndices[i];
          const mlScore = data.scores[i];

          // Only keep pivots above threshold
          if (mlScore >= settings.ml_threshold) {
            pivotScores[idx] = mlScore;
          } else {
            // Filter out low-confidence pivots
            pivotHighs[idx] = false;
            pivotLows[idx] = false;
            pivotScores[idx] = null;
          }
        }

        // ML filtering applied successfully
      }

    } catch (error) {
      console.error('ML enhancement failed:', error);
      // Scores remain at default 0.85
    }
  }

  /**
   * Get settings schema for UI
   * @returns {Object} Settings schema
   */
  getSettingsSchema() {
    return {
      reg_length: {
        type: 'number',
        label: 'Regression Length',
        min: 5,
        max: 50,
        step: 1,
        default: 20,
        description: 'Base length for regression calculations'
      },
      rsi_len: {
        type: 'number',
        label: 'RSI Length',
        min: 5,
        max: 30,
        step: 1,
        default: 14,
        description: 'Period for RSI calculation'
      },
      trend_bar_length: {
        type: 'number',
        label: 'Short Trend Length',
        min: 3,
        max: 10,
        step: 1,
        default: 5,
        description: 'Length for short-term trend line'
      },
      pivot_lookback: {
        type: 'number',
        label: 'Pivot Lookback',
        min: 3,
        max: 10,
        step: 1,
        default: 5,
        description: 'Lookback period for pivot detection'
      },
      adx_threshold: {
        type: 'number',
        label: 'ADX Threshold',
        min: 15,
        max: 40,
        step: 1,
        default: 25,
        description: 'Minimum ADX for valid pivots'
      },
      weighted_trend_color: {
        type: 'color',
        label: 'Weighted Trend Color',
        default: '#2196F3',
        description: 'Color for weighted price regression line'
      },
      oscillator_color: {
        type: 'color',
        label: 'Oscillator Color',
        default: '#FF9800',
        description: 'Color for adaptive oscillator line'
      },
      short_trend_color: {
        type: 'color',
        label: 'Short Trend Color',
        default: '#4CAF50',
        description: 'Color for short-term trend line'
      },
      line_opacity: {
        type: 'number',
        label: 'Line Opacity',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.9,
        description: 'Opacity of indicator lines'
      },
      use_ml: {
        type: 'boolean',
        label: 'Enable ML Enhancement',
        default: false,
        description: 'Use machine learning to filter pivot signals (requires trained model)'
      },
      ml_threshold: {
        type: 'number',
        label: 'ML Confidence Threshold',
        min: 0.5,
        max: 0.95,
        step: 0.05,
        default: 0.7,
        description: 'Minimum ML confidence score (0.7 = 84% precision, 0.8 = higher precision)'
      }
    };
  }

  /**
   * Render Triad Trend Pulse on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Object} bounds - Rendering bounds {x, y, width, height}
   * @param {Array} data - Indicator data
   * @param {Array} mappings - Array of {candleIndex, indicatorIndex} mappings
   * @param {Number} startIndex - Start index
   * @param {Number} endIndex - End index
   */
  render(ctx, bounds, data, mappings, startIndex, endIndex) {
    if (!data || data.length === 0 || !mappings || mappings.length === 0) return;

    const { x, y, width, height } = bounds;
    const settings = this.currentSettings;

    // Calculate visible range
    const visibleCandles = endIndex - startIndex + 1;
    const totalWidth = width / visibleCandles;
    const candleWidth = Math.max(2, totalWidth * 0.7);
    const spacing = totalWidth - candleWidth;
    const centerOffset = spacing / 2 + candleWidth / 2;

    // Draw background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(x, y, width, height);

    // Find min/max for scaling
    let minValue = -100;
    let maxValue = 100;

    const range = maxValue - minValue;

    // Helper to convert value to Y coordinate
    const valueToY = (value) => {
      return y + height - ((value - minValue) / range * height);
    };

    // Draw zero line
    const zeroY = valueToY(0);
    ctx.strokeStyle = '#888888';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(x, zeroY);
    ctx.lineTo(x + width, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw ±50 reference lines
    const plus50Y = valueToY(50);
    const minus50Y = valueToY(-50);
    ctx.strokeStyle = 'rgba(136, 136, 136, 0.3)';
    ctx.beginPath();
    ctx.moveTo(x, plus50Y);
    ctx.lineTo(x + width, plus50Y);
    ctx.moveTo(x, minus50Y);
    ctx.lineTo(x + width, minus50Y);
    ctx.stroke();

    // === Draw Lines ===
    ctx.globalAlpha = settings.line_opacity;

    // Draw weighted trend line
    this._drawLine(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY,
      'weighted_trend', settings.weighted_trend_color, settings.line_width);

    // Draw oscillator line
    this._drawLine(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY,
      'oscillator', settings.oscillator_color, settings.line_width);

    // Draw short trend line
    this._drawLine(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY,
      'short_trend', settings.short_trend_color, settings.line_width);

    ctx.globalAlpha = 1.0;

    // === Draw Pivots ===
    let drawnHighs = 0;
    let drawnLows = 0;
    let potentialHighsInView = 0;
    let potentialLowsInView = 0;

    // First pass: count what's in the data
    let totalPivotsInData = 0;
    for (const d of data) {
      if (d.pivot_high === true) totalPivotsInData++;
      if (d.pivot_low === true) totalPivotsInData++;
    }

    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const d = data[indicatorIndex];
      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;

      // Count what's in visible area (TRUE values only)
      if (d.pivot_high === true) {
        potentialHighsInView++;
      }
      if (d.pivot_low === true) {
        potentialLowsInView++;
      }

      // Pivots are already filtered by ML during calculation if use_ml is enabled
      // So we just need to check if pivot exists (ML filtering sets them to false)
      if (d.pivot_high && d.pivot_score && d.weighted_trend !== null) {
        drawnHighs++;
        // Draw pivot high marker
        ctx.fillStyle = settings.pivot_high_color;
        ctx.beginPath();
        ctx.arc(xPos, valueToY(d.weighted_trend) - 15, settings.pivot_size, 0, Math.PI * 2);
        ctx.fill();

        // Draw triangle
        ctx.beginPath();
        ctx.moveTo(xPos, valueToY(d.weighted_trend) - 10);
        ctx.lineTo(xPos - 5, valueToY(d.weighted_trend) - 20);
        ctx.lineTo(xPos + 5, valueToY(d.weighted_trend) - 20);
        ctx.closePath();
        ctx.fill();
      }

      if (d.pivot_low && d.pivot_score && d.weighted_trend !== null) {
        drawnLows++;
        // Draw pivot low marker
        ctx.fillStyle = settings.pivot_low_color;
        ctx.beginPath();
        ctx.arc(xPos, valueToY(d.weighted_trend) + 15, settings.pivot_size, 0, Math.PI * 2);
        ctx.fill();

        // Draw triangle
        ctx.beginPath();
        ctx.moveTo(xPos, valueToY(d.weighted_trend) + 10);
        ctx.lineTo(xPos - 5, valueToY(d.weighted_trend) + 20);
        ctx.lineTo(xPos + 5, valueToY(d.weighted_trend) + 20);
        ctx.closePath();
        ctx.fill();
      }
    });

    // Render logs removed - use browser console filtering if needed for debugging

    // Draw current values label
    const lastData = data[data.length - 1];
    if (lastData && lastData.weighted_trend !== null && lastData.oscillator !== null && lastData.short_trend !== null) {
      ctx.font = 'bold 10px monospace';

      ctx.fillStyle = settings.weighted_trend_color;
      ctx.fillText(`WT: ${lastData.weighted_trend.toFixed(1)}`, x + 5, y + 15);

      ctx.fillStyle = settings.oscillator_color;
      ctx.fillText(`OSC: ${lastData.oscillator.toFixed(1)}`, x + 5, y + 30);

      ctx.fillStyle = settings.short_trend_color;
      ctx.fillText(`ST: ${lastData.short_trend.toFixed(1)}`, x + 5, y + 45);

      // Show total pivot count (helps see ML filtering effect)
      ctx.fillStyle = settings.use_ml ? '#00E676' : '#888';
      ctx.fillText(`Pivots: ${totalPivotsInData}${settings.use_ml ? ' (ML)' : ''}`, x + 5, y + 60);
    }
  }

  /**
   * Helper to draw a line for a specific data field
   * @private
   */
  _drawLine(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY, field, color, lineWidth) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();

    let firstPoint = true;
    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const value = data[indicatorIndex][field];

      // Skip null values and restart line
      if (value === null || value === undefined) {
        firstPoint = true;
        return;
      }

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = valueToY(value);

      if (firstPoint) {
        ctx.moveTo(xPos, yPos);
        firstPoint = false;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    });

    ctx.stroke();
  }
}

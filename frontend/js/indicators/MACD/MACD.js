/**
 * MACD (Moving Average Convergence Divergence) Indicator
 * Trend-following momentum indicator showing relationship between two EMAs
 * Custom implementation - no third-party libraries
 */

import { IndicatorBase } from '../IndicatorBase.js';

export class MACD extends IndicatorBase {
  constructor() {
    super({
      name: 'MACD',
      version: '2.0.0',
      description: 'Moving Average Convergence Divergence with signal line and histogram',
      tags: ['momentum', 'trend'],
      dependencies: [],
      output_type: 'oscillator',
      default_settings: {
        macd_line_color: '#0000FF',
        signal_line_color: '#FF0000',
        histogram_color_positive: '#00FF00',
        histogram_color_negative: '#FF0000',
        line_style: 'solid',
        line_opacity: 0.9,
        histogram_opacity: 0.6,
        fast_period: 12,
        slow_period: 26,
        signal_period: 9,

        // Signal Detection
        enable_signals: true,
        signal_cooldown: 3,          // Bars between signals

        // Machine Learning Filter
        enable_ml_filter: false,     // Disabled - MACD signals show without ML filtering
        ml_confidence_threshold: 0.7,
        show_ml_score: false,

        // Signal Colors
        buy_signal_color: '#00FF00',
        sell_signal_color: '#FF0000'
      },
      alerts: {
        enabled: true,
        conditions: [
          { type: 'cross_over', field: 'macd', target: 'signal', message: 'MACD Bullish Crossover' },
          { type: 'cross_under', field: 'macd', target: 'signal', message: 'MACD Bearish Crossover' }
        ]
      },
      help_text: 'MACD shows the relationship between two EMAs. Crossovers indicate potential trend changes.'
    });
  }

  /**
   * Calculate EMA (Exponential Moving Average)
   * @param {Array} values - Price values
   * @param {number} period - EMA period
   * @returns {Array} EMA values
   * @private
   */
  _calculateEMA(values, period) {
    if (!values || values.length < period) {
      return [];
    }

    const ema = [];
    const multiplier = 2 / (period + 1);

    // First EMA is SMA
    let sum = 0;
    for (let i = 0; i < period; i++) {
      sum += values[i];
    }
    let prevEMA = sum / period;
    ema.push(prevEMA);

    // Calculate subsequent EMAs
    for (let i = period; i < values.length; i++) {
      const currentEMA = (values[i] - prevEMA) * multiplier + prevEMA;
      ema.push(currentEMA);
      prevEMA = currentEMA;
    }

    return ema;
  }

  /**
   * Calculate MACD values from OHLCV data
   * Formula:
   *   MACD Line = EMA(fast) - EMA(slow)
   *   Signal Line = EMA(MACD Line, signal_period)
   *   Histogram = MACD Line - Signal Line
   *
   * @param {Array} candles - Array of OHLCV candles
   * @returns {Array} MACD values with metadata
   */
  async calculate(candles) {
    if (!candles || candles.length < this.currentSettings.slow_period + this.currentSettings.signal_period) {
      return [];
    }

    const fastPeriod = this.currentSettings.fast_period;
    const slowPeriod = this.currentSettings.slow_period;
    const signalPeriod = this.currentSettings.signal_period;

    // Extract close prices
    const closes = candles.map(c => c.Close);

    // Calculate fast and slow EMAs
    const fastEMA = this._calculateEMA(closes, fastPeriod);
    const slowEMA = this._calculateEMA(closes, slowPeriod);

    // Calculate MACD Line (fast EMA - slow EMA)
    // Align arrays: fast EMA starts at index (fastPeriod-1), slow EMA starts at (slowPeriod-1)
    const macdLine = [];
    const startIndex = slowPeriod - 1;

    for (let i = 0; i < slowEMA.length; i++) {
      const fastIndex = i + (slowPeriod - fastPeriod);
      if (fastIndex >= 0 && fastIndex < fastEMA.length) {
        macdLine.push(fastEMA[fastIndex] - slowEMA[i]);
      }
    }

    // Calculate Signal Line (EMA of MACD Line)
    const signalLine = this._calculateEMA(macdLine, signalPeriod);

    // Calculate Histogram (MACD Line - Signal Line)
    const result = [];
    const signalStartIndex = signalPeriod - 1;
    const firstValidIndex = startIndex + signalStartIndex;

    // Add null padding for early candles to keep array aligned
    for (let i = 0; i < firstValidIndex; i++) {
      result.push({
        date: candles[i].Date,
        macd: null,
        signal: null,
        histogram: null,
        crossover_signal: null,
        ml_score: null
      });
    }

    // Add calculated values
    for (let i = 0; i < signalLine.length; i++) {
      const macdIndex = i + signalStartIndex;
      const candleIndex = startIndex + macdIndex;

      if (candleIndex < candles.length) {
        const macdValue = macdLine[macdIndex];
        const signalValue = signalLine[i];
        const histogram = macdValue - signalValue;

        result.push({
          date: candles[candleIndex].Date,
          macd: macdValue,
          signal: signalValue,
          histogram: histogram,
          crossover_signal: null,
          ml_score: null
        });
      }
    }

    // === Signal Detection ===
    if (this.currentSettings.enable_signals) {
      await this._detectSignals(result, candles, this.currentSettings);
    }

    return result;
  }

  /**
   * Detect MACD crossover signals with momentum confirmation
   * Buy: MACD crosses above Signal line WITH both rising AND in bullish zone
   * Sell: MACD crosses below Signal line WITH both falling AND in bearish zone
   * This filters out low-probability whipsaws in ranging markets
   * @private
   */
  async _detectSignals(data, candles, settings) {
    if (data.length < 2) return;

    let lastSignalIndex = -settings.signal_cooldown;

    for (let i = 1; i < data.length; i++) {
      const current = data[i];
      const prev = data[i - 1];

      // Skip if values are null
      if (current.macd === null || current.signal === null ||
          prev.macd === null || prev.signal === null) continue;

      // Check cooldown
      if (i - lastSignalIndex < settings.signal_cooldown) continue;

      let crossoverSignal = null;

      // Buy signal: MACD crosses above Signal line WITH momentum confirmation
      // Requires: 1) Crossover, 2) Both lines rising, 3) Above zero line (bullish zone)
      if (prev.macd <= prev.signal && current.macd > current.signal) {
        // Check momentum: both MACD and Signal should be rising
        const macdRising = current.macd > prev.macd;
        const signalRising = current.signal > prev.signal;

        // Check zone: crossing in bullish territory (above zero)
        const inBullishZone = current.macd > 0;

        // All conditions must be met for high-quality buy signal
        if (macdRising && signalRising && inBullishZone) {
          crossoverSignal = 'buy';
        }
      }
      // Sell signal: MACD crosses below Signal line WITH momentum confirmation
      // Requires: 1) Crossover, 2) Both lines falling, 3) Below zero line (bearish zone)
      else if (prev.macd >= prev.signal && current.macd < current.signal) {
        // Check momentum: both MACD and Signal should be falling
        const macdFalling = current.macd < prev.macd;
        const signalFalling = current.signal < prev.signal;

        // Check zone: crossing in bearish territory (below zero)
        const inBearishZone = current.macd < 0;

        // All conditions must be met for high-quality sell signal
        if (macdFalling && signalFalling && inBearishZone) {
          crossoverSignal = 'sell';
        }
      }

      if (crossoverSignal) {
        // Apply ML filter if enabled
        if (settings.enable_ml_filter) {
          const mlScore = await this._getMLScore(candles, i, current, settings);
          current.ml_score = mlScore;

          if (mlScore >= settings.ml_confidence_threshold) {
            current.crossover_signal = crossoverSignal;
            lastSignalIndex = i;
            console.log(`🎯 MACD: ${crossoverSignal.toUpperCase()} signal at ${current.date} (MACD: ${current.macd.toFixed(2)}, Signal: ${current.signal.toFixed(2)}, ML: ${(mlScore * 100).toFixed(0)}%)`);
          } else {
            console.log(`❌ MACD: ${crossoverSignal.toUpperCase()} rejected by ML at ${current.date} (ML: ${(mlScore * 100).toFixed(0)}% < ${(settings.ml_confidence_threshold * 100).toFixed(0)}%)`);
          }
        } else {
          // No ML filter, show signal
          current.crossover_signal = crossoverSignal;
          lastSignalIndex = i;
          console.log(`🎯 MACD: ${crossoverSignal.toUpperCase()} signal at ${current.date} (MACD: ${current.macd.toFixed(2)}, Signal: ${current.signal.toFixed(2)})`);
        }
      }
    }
  }

  /**
   * Get ML confidence score for a signal
   * @private
   */
  async _getMLScore(candles, index, signalData, settings) {
    try {
      // Calculate ML features
      const features = this._calculateMLFeatures(candles, index, signalData, settings);

      if (!features || features.some(f => isNaN(f))) {
        return 0.5; // Fallback score
      }

      // Get symbol from window
      const symbol = window.currentSymbol || 'UNKNOWN';

      // Call ML API
      const response = await fetch('http://127.0.0.1:5000/ml/pivot-reliability', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol,
          features: features
        })
      });

      if (!response.ok) {
        return 0.5;
      }

      const result = await response.json();
      return result.scores[0];

    } catch (error) {
      console.error('❌ MACD ML score calculation failed:', error);
      return 0.5;
    }
  }

  /**
   * Calculate ML features for signal validation
   * Returns 9 features matching the TriadTrendPulse model
   * @private
   */
  _calculateMLFeatures(candles, index, signalData, settings) {
    try {
      const closes = candles.map(c => c.Close);
      const highs = candles.map(c => c.High);
      const lows = candles.map(c => c.Low);
      const volumes = candles.map(c => c.Volume || 0);

      // Feature 1: Oscillator range (MACD histogram normalized)
      const histogramRange = signalData.histogram;
      const avgPrice = closes[index];
      const oscillatorRange = Math.tanh(histogramRange / avgPrice * 50) / 2 + 0.5;

      // Feature 2: Weighted price trend (MACD line normalized)
      const weightedPriceTrend = Math.tanh(signalData.macd / avgPrice * 50) / 2 + 0.5;

      // Feature 3: Short-term trend (5-bar)
      let shortTrend = 0.5;
      if (index >= 5) {
        const change = (closes[index] - closes[index - 5]) / closes[index - 5];
        shortTrend = Math.tanh(change * 10) / 2 + 0.5;
      }

      // Feature 4: ADX
      const adx = this._calculateADX(highs, lows, closes, index, 14);
      const adxNorm = adx / 100.0;

      // Feature 5: Volume change
      let volumeChange = 1.0;
      if (index >= 20) {
        const volumeSMA = volumes.slice(index - 19, index + 1).reduce((a, b) => a + b, 0) / 20;
        volumeChange = Math.min(volumes[index] / volumeSMA, 3) / 3;
      }

      // Feature 6: ATR (volatility)
      const atr = this._calculateATR(highs, lows, closes, index, 14);
      const atrNorm = atr / closes[index];

      // Feature 7: RSI
      const rsi = this._calculateRSI(closes, index, 14);
      const rsiNorm = rsi / 100.0;

      // Feature 8: Momentum (20-bar)
      let momentum = 0.5;
      if (index >= 20) {
        const change = (closes[index] - closes[index - 20]) / closes[index - 20];
        momentum = Math.tanh(change * 10) / 2 + 0.5;
      }

      // Feature 9: Timeframe (default daily)
      const timeframeNorm = 0.5;

      return [
        oscillatorRange,
        weightedPriceTrend,
        shortTrend,
        adxNorm,
        volumeChange,
        atrNorm,
        rsiNorm,
        momentum,
        timeframeNorm
      ];

    } catch (error) {
      console.error('❌ MACD ML feature calculation failed:', error);
      return null;
    }
  }

  /**
   * Calculate RSI at specific index
   * @private
   */
  _calculateRSI(closes, index, period) {
    if (index < period) return 50;

    const changes = [];
    for (let i = Math.max(1, index - period); i <= index; i++) {
      changes.push(closes[i] - closes[i - 1]);
    }

    let avgGain = 0, avgLoss = 0;
    for (let i = 0; i < period && i < changes.length; i++) {
      if (changes[i] > 0) avgGain += changes[i];
      else avgLoss += Math.abs(changes[i]);
    }
    avgGain /= period;
    avgLoss /= period;

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  /**
   * Calculate ATR at specific index
   * @private
   */
  _calculateATR(highs, lows, closes, index, period) {
    if (index < period) return 0;

    let tr = 0;
    for (let i = Math.max(1, index - period + 1); i <= index; i++) {
      const high = highs[i];
      const low = lows[i];
      const prevClose = closes[i - 1];
      tr += Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    }

    return tr / period;
  }

  /**
   * Calculate ADX at specific index (simplified)
   * @private
   */
  _calculateADX(highs, lows, closes, index, period) {
    if (index < period) return 0;

    let upMoves = 0, downMoves = 0;
    for (let i = Math.max(1, index - period + 1); i <= index; i++) {
      const upMove = highs[i] - highs[i - 1];
      const downMove = lows[i - 1] - lows[i];

      if (upMove > downMove && upMove > 0) upMoves += upMove;
      if (downMove > upMove && downMove > 0) downMoves += downMove;
    }

    const total = upMoves + downMoves;
    if (total === 0) return 0;

    const dx = Math.abs((upMoves - downMoves) / total) * 100;
    return Math.min(dx, 100);
  }

  /**
   * Get settings schema for UI
   * @returns {Object} Settings schema
   */
  getSettingsSchema() {
    return {
      fast_period: {
        type: 'number',
        label: 'Fast Period',
        min: 2,
        max: 100,
        step: 1,
        default: 12,
        description: 'Fast EMA period (typically 12)'
      },
      slow_period: {
        type: 'number',
        label: 'Slow Period',
        min: 2,
        max: 200,
        step: 1,
        default: 26,
        description: 'Slow EMA period (typically 26)'
      },
      signal_period: {
        type: 'number',
        label: 'Signal Period',
        min: 2,
        max: 50,
        step: 1,
        default: 9,
        description: 'Signal line EMA period (typically 9)'
      },
      macd_line_color: {
        type: 'color',
        label: 'MACD Line Color',
        default: '#0000FF',
        description: 'Color of the MACD line'
      },
      signal_line_color: {
        type: 'color',
        label: 'Signal Line Color',
        default: '#FF0000',
        description: 'Color of the signal line'
      },
      histogram_color_positive: {
        type: 'color',
        label: 'Histogram Positive Color',
        default: '#00FF00',
        description: 'Histogram color when MACD > Signal'
      },
      histogram_color_negative: {
        type: 'color',
        label: 'Histogram Negative Color',
        default: '#FF0000',
        description: 'Histogram color when MACD < Signal'
      },
      line_opacity: {
        type: 'number',
        label: 'Line Opacity',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.9,
        description: 'Opacity of MACD and Signal lines (0-1)'
      },
      histogram_opacity: {
        type: 'number',
        label: 'Histogram Opacity',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.6,
        description: 'Opacity of histogram bars (0-1)'
      },
      enable_signals: {
        type: 'boolean',
        label: 'Enable Signals',
        default: true,
        description: 'Enable buy/sell signal detection at MACD/Signal crossovers'
      },
      signal_cooldown: {
        type: 'number',
        label: 'Signal Cooldown',
        min: 1,
        max: 10,
        step: 1,
        default: 3,
        description: 'Minimum bars between signals'
      },
      enable_ml_filter: {
        type: 'boolean',
        label: 'Enable ML Filter',
        default: true,
        description: 'Use machine learning to validate signal quality'
      },
      ml_confidence_threshold: {
        type: 'number',
        label: 'ML Confidence Threshold',
        min: 0.5,
        max: 0.95,
        step: 0.05,
        default: 0.7,
        description: 'Minimum ML confidence score to display signal (0.7 = 70%)'
      },
      show_ml_score: {
        type: 'boolean',
        label: 'Show ML Score',
        default: true,
        description: 'Display ML confidence percentage on signal labels'
      },
      buy_signal_color: {
        type: 'color',
        label: 'Buy Signal Color',
        default: '#00FF00',
        description: 'Color for buy signal markers'
      },
      sell_signal_color: {
        type: 'color',
        label: 'Sell Signal Color',
        default: '#FF0000',
        description: 'Color for sell signal markers'
      }
    };
  }

  /**
   * Render MACD on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Object} bounds - Rendering bounds {x, y, width, height}
   * @param {Array} data - MACD data
   * @param {Array} mappings - Array of {candleIndex, indicatorIndex} mappings
   * @param {Number} startIndex - Start index (may be fractional for smooth panning)
   * @param {Number} endIndex - End index (may be fractional for smooth panning)
   */
  render(ctx, bounds, data, mappings, startIndex, endIndex) {
    if (!data || data.length === 0 || !mappings || mappings.length === 0) {
      return;
    }

    const { x, y, width, height } = bounds;
    const settings = this.currentSettings;

    // Calculate visible range (same as canvas drawCandles)
    const visibleCandles = endIndex - startIndex + 1;
    const totalWidth = width / visibleCandles;
    const candleWidth = Math.max(2, totalWidth * 0.7);
    const spacing = totalWidth - candleWidth;
    const centerOffset = spacing / 2 + candleWidth / 2; // Center of candle

    // Draw background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(x, y, width, height);

    // Find min/max for scaling
    let minValue = Infinity;
    let maxValue = -Infinity;

    mappings.forEach(mapping => {
      const { indicatorIndex } = mapping;
      if (indicatorIndex >= 0 && indicatorIndex < data.length) {
        const d = data[indicatorIndex];
        // Skip null values
        if (d.macd !== null && d.signal !== null && d.histogram !== null) {
          minValue = Math.min(minValue, d.macd, d.signal, d.histogram);
          maxValue = Math.max(maxValue, d.macd, d.signal, d.histogram);
        }
      }
    });

    // Add padding
    const padding = (maxValue - minValue) * 0.1;
    minValue -= padding;
    maxValue += padding;

    const range = maxValue - minValue || 1;

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

    // Draw histogram
    ctx.globalAlpha = settings.histogram_opacity;
    const barWidth = totalWidth;

    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const histogram = data[indicatorIndex].histogram;

      // Skip null values
      if (histogram === null || histogram === undefined) return;

      const barActualWidth = barWidth * 0.8;
      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset - barActualWidth / 2;
      const histY = valueToY(histogram);

      ctx.fillStyle = histogram >= 0
        ? settings.histogram_color_positive
        : settings.histogram_color_negative;

      const barHeight = Math.abs(histY - zeroY);
      const barY = histogram >= 0 ? histY : zeroY;

      ctx.fillRect(xPos, barY, barActualWidth, barHeight);
    });

    ctx.globalAlpha = 1.0;

    // Draw MACD line
    ctx.strokeStyle = settings.macd_line_color;
    ctx.globalAlpha = settings.line_opacity;
    ctx.lineWidth = 2;
    ctx.beginPath();

    let firstPoint = true;
    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const macdValue = data[indicatorIndex].macd;

      // Skip null values and restart line
      if (macdValue === null || macdValue === undefined) {
        firstPoint = true;
        return;
      }

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = valueToY(macdValue);

      if (firstPoint) {
        ctx.moveTo(xPos, yPos);
        firstPoint = false;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    });

    ctx.stroke();

    // Draw Signal line
    ctx.strokeStyle = settings.signal_line_color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    firstPoint = true;
    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const signalValue = data[indicatorIndex].signal;

      // Skip null values and restart line
      if (signalValue === null || signalValue === undefined) {
        firstPoint = true;
        return;
      }

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = valueToY(signalValue);

      if (firstPoint) {
        ctx.moveTo(xPos, yPos);
        firstPoint = false;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    });

    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Draw current values label
    const lastData = data[data.length - 1];
    if (lastData && lastData.macd !== null && lastData.signal !== null && lastData.histogram !== null) {
      ctx.font = 'bold 10px monospace';

      // MACD value
      ctx.fillStyle = settings.macd_line_color;
      ctx.fillText(`MACD: ${lastData.macd.toFixed(2)}`, x + 5, y + 15);

      // Signal value
      ctx.fillStyle = settings.signal_line_color;
      ctx.fillText(`Signal: ${lastData.signal.toFixed(2)}`, x + 5, y + 30);

      // Histogram value
      ctx.fillStyle = lastData.histogram >= 0
        ? settings.histogram_color_positive
        : settings.histogram_color_negative;
      ctx.fillText(`Hist: ${lastData.histogram.toFixed(2)}`, x + 5, y + 45);
    }

    // Draw signal markers
    if (settings.enable_signals) {
      this._drawSignalMarkers(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY, settings);
    }
  }

  /**
   * Draw signal markers (buy/sell arrows) in MACD panel
   * @private
   */
  _drawSignalMarkers(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY, settings) {
    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      const d = data[indicatorIndex];

      if (!d || !d.crossover_signal || d.macd === null) return;

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = valueToY(d.macd);

      // Draw arrow
      const arrowSize = 8;
      ctx.fillStyle = d.crossover_signal === 'buy' ? settings.buy_signal_color : settings.sell_signal_color;

      ctx.beginPath();
      if (d.crossover_signal === 'buy') {
        // Up arrow
        ctx.moveTo(xPos, yPos - arrowSize);
        ctx.lineTo(xPos - arrowSize / 2, yPos);
        ctx.lineTo(xPos + arrowSize / 2, yPos);
      } else {
        // Down arrow
        ctx.moveTo(xPos, yPos + arrowSize);
        ctx.lineTo(xPos - arrowSize / 2, yPos);
        ctx.lineTo(xPos + arrowSize / 2, yPos);
      }
      ctx.closePath();
      ctx.fill();

      // Draw ML score if enabled
      if (settings.show_ml_score && d.ml_score !== null) {
        const scoreText = `${(d.ml_score * 100).toFixed(0)}%`;
        ctx.font = '9px monospace';
        ctx.fillStyle = d.crossover_signal === 'buy' ? settings.buy_signal_color : settings.sell_signal_color;

        const textMetrics = ctx.measureText(scoreText);
        const textX = xPos - textMetrics.width / 2;
        const textY = d.crossover_signal === 'buy' ? yPos - arrowSize - 4 : yPos + arrowSize + 12;

        ctx.fillText(scoreText, textX, textY);
      }
    });
  }
}

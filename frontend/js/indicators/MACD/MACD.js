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
      version: '1.0.0',
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
        signal_period: 9
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
  calculate(candles) {
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
        histogram: null
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
          histogram: histogram
        });
      }
    }

    return result;
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
  }
}

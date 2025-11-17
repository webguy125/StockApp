/**
 * Bollinger Bands Indicator
 * Volatility bands plotted above and below a moving average
 * Custom implementation - no third-party libraries
 */

import { IndicatorBase } from '../IndicatorBase.js';

export class BollingerBands extends IndicatorBase {
  constructor() {
    super({
      name: 'BollingerBands',
      version: '1.0.0',
      description: 'Volatility bands plotted above and below a moving average',
      tags: ['volatility', 'trend'],
      dependencies: [],
      output_type: 'overlay',
      default_settings: {
        upper_band_color: '#FF6B6B',
        middle_band_color: '#4ECDC4',
        lower_band_color: '#95E1D3',
        fill_color: '#4ECDC4',
        fill_opacity: 0.1,
        line_opacity: 0.7,
        line_width: 1.5,
        lookback_period: 20,
        std_dev: 2,
        show_fill: true,
        show_middle: true
      },
      alerts: {
        enabled: false
      },
      help_text: 'Bollinger Bands measure volatility and potential breakout conditions. Price touching bands may indicate overbought/oversold.'
    });
  }

  /**
   * Calculate SMA (Simple Moving Average)
   * @param {Array} values - Price values
   * @param {number} period - SMA period
   * @returns {Array} SMA values
   * @private
   */
  _calculateSMA(values, period) {
    if (!values || values.length < period) {
      return [];
    }

    const sma = [];

    for (let i = period - 1; i < values.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += values[i - j];
      }
      sma.push(sum / period);
    }

    return sma;
  }

  /**
   * Calculate Standard Deviation
   * @param {Array} values - Price values
   * @param {number} period - Period for std dev calculation
   * @param {number} mean - Mean value (SMA)
   * @param {number} index - Current index in values array
   * @returns {number} Standard deviation
   * @private
   */
  _calculateStdDev(values, period, mean, index) {
    let sumSquaredDiff = 0;

    for (let j = 0; j < period; j++) {
      const diff = values[index - j] - mean;
      sumSquaredDiff += diff * diff;
    }

    return Math.sqrt(sumSquaredDiff / period);
  }

  /**
   * Calculate Bollinger Bands values from OHLCV data
   * Formula:
   *   Middle Band = SMA(period)
   *   Upper Band = Middle Band + (std_dev * standard deviation)
   *   Lower Band = Middle Band - (std_dev * standard deviation)
   *
   * @param {Array} candles - Array of OHLCV candles
   * @returns {Array} Bollinger Bands values with metadata
   */
  calculate(candles) {
    if (!candles || candles.length < this.currentSettings.lookback_period) {
      return [];
    }

    const period = this.currentSettings.lookback_period;
    const stdDevMultiplier = this.currentSettings.std_dev;
    const result = [];

    // Extract close prices
    const closes = candles.map(c => c.Close);

    // Calculate SMA (middle band)
    const sma = this._calculateSMA(closes, period);

    // Calculate upper and lower bands
    for (let i = 0; i < sma.length; i++) {
      const candleIndex = i + period - 1;
      const middleBand = sma[i];

      // Calculate standard deviation for this period
      const stdDev = this._calculateStdDev(closes, period, middleBand, candleIndex);

      // Calculate bands
      const upperBand = middleBand + (stdDevMultiplier * stdDev);
      const lowerBand = middleBand - (stdDevMultiplier * stdDev);

      // Calculate bandwidth (upper - lower) as % of middle
      const bandwidth = ((upperBand - lowerBand) / middleBand) * 100;

      // Calculate %B (where price is within the bands)
      // %B = (Price - Lower Band) / (Upper Band - Lower Band)
      const price = closes[candleIndex];
      const percentB = (price - lowerBand) / (upperBand - lowerBand);

      result.push({
        date: candles[candleIndex].Date,
        upper: upperBand,
        middle: middleBand,
        lower: lowerBand,
        bandwidth: bandwidth,
        percentB: percentB,
        price: price
      });
    }

    return result;
  }

  /**
   * Get settings schema for UI
   * @returns {Object} Settings schema
   */
  getSettingsSchema() {
    return {
      lookback_period: {
        type: 'number',
        label: 'Period',
        min: 2,
        max: 200,
        step: 1,
        default: 20,
        description: 'Number of periods for SMA calculation (typically 20)'
      },
      std_dev: {
        type: 'number',
        label: 'Standard Deviations',
        min: 0.5,
        max: 5,
        step: 0.5,
        default: 2,
        description: 'Number of standard deviations from middle band (typically 2)'
      },
      upper_band_color: {
        type: 'color',
        label: 'Upper Band Color',
        default: '#FF6B6B',
        description: 'Color of the upper band line'
      },
      middle_band_color: {
        type: 'color',
        label: 'Middle Band Color',
        default: '#4ECDC4',
        description: 'Color of the middle band (SMA) line'
      },
      lower_band_color: {
        type: 'color',
        label: 'Lower Band Color',
        default: '#95E1D3',
        description: 'Color of the lower band line'
      },
      fill_color: {
        type: 'color',
        label: 'Fill Color',
        default: '#4ECDC4',
        description: 'Color for filling between bands'
      },
      fill_opacity: {
        type: 'number',
        label: 'Fill Opacity',
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.1,
        description: 'Opacity of band fill (0-1)'
      },
      line_opacity: {
        type: 'number',
        label: 'Line Opacity',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.7,
        description: 'Opacity of band lines (0-1)'
      },
      line_width: {
        type: 'number',
        label: 'Line Width',
        min: 0.5,
        max: 5,
        step: 0.5,
        default: 1.5,
        description: 'Width of band lines'
      },
      show_fill: {
        type: 'boolean',
        label: 'Show Fill Between Bands',
        default: true,
        description: 'Fill area between upper and lower bands'
      },
      show_middle: {
        type: 'boolean',
        label: 'Show Middle Band',
        default: true,
        description: 'Display the middle band (SMA)'
      }
    };
  }

  /**
   * Render Bollinger Bands on canvas (overlay on price chart)
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Object} bounds - Rendering bounds {x, y, width, height}
   * @param {Array} data - Bollinger Bands data
   * @param {Array} mappings - Array of {candleIndex, indicatorIndex} mappings
   * @param {Function} priceToY - Function to convert price to Y coordinate
   * @param {Number} startIndex - Start index (may be fractional for smooth panning)
   * @param {Number} endIndex - End index (may be fractional for smooth panning)
   */
  render(ctx, bounds, data, mappings, priceToY, startIndex, endIndex) {
    if (!data || data.length === 0 || !mappings || mappings.length === 0 || !priceToY) return;

    const { x, width } = bounds;
    const settings = this.currentSettings;

    // Calculate width per candle (same as candle rendering)
    const visibleCandles = endIndex - startIndex + 1;
    const totalWidth = width / visibleCandles;
    const candleWidth = Math.max(2, totalWidth * 0.7);
    const spacing = totalWidth - candleWidth;
    const centerOffset = spacing / 2 + candleWidth / 2; // Center of candle

    // Fill between bands
    if (settings.show_fill) {
      ctx.fillStyle = settings.fill_color;
      ctx.globalAlpha = settings.fill_opacity;
      ctx.beginPath();

      // Draw upper band path
      let firstPoint = true;
      mappings.forEach((mapping) => {
        const { candleIndex, indicatorIndex } = mapping;
        if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

        const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
        const yPos = priceToY(data[indicatorIndex].upper);

        if (firstPoint) {
          ctx.moveTo(xPos, yPos);
          firstPoint = false;
        } else {
          ctx.lineTo(xPos, yPos);
        }
      });

      // Draw lower band path in reverse
      for (let i = mappings.length - 1; i >= 0; i--) {
        const { candleIndex, indicatorIndex } = mappings[i];
        if (indicatorIndex < 0 || indicatorIndex >= data.length) continue;

        const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
        const yPos = priceToY(data[indicatorIndex].lower);
        ctx.lineTo(xPos, yPos);
      }

      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }

    // Draw upper band line
    ctx.strokeStyle = settings.upper_band_color;
    ctx.globalAlpha = settings.line_opacity;
    ctx.lineWidth = settings.line_width;
    ctx.beginPath();

    let firstPoint = true;
    mappings.forEach((mapping) => {
      const { candleIndex, indicatorIndex } = mapping;
      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = priceToY(data[indicatorIndex].upper);

      if (firstPoint) {
        ctx.moveTo(xPos, yPos);
        firstPoint = false;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    });

    ctx.stroke();

    // Draw middle band line (if enabled)
    if (settings.show_middle) {
      ctx.strokeStyle = settings.middle_band_color;
      ctx.beginPath();

      firstPoint = true;
      mappings.forEach((mapping) => {
        const { candleIndex, indicatorIndex } = mapping;
        if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

        const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
        const yPos = priceToY(data[indicatorIndex].middle);

        if (firstPoint) {
          ctx.moveTo(xPos, yPos);
          firstPoint = false;
        } else {
          ctx.lineTo(xPos, yPos);
        }
      });

      ctx.stroke();
    }

    // Draw lower band line
    ctx.strokeStyle = settings.lower_band_color;
    ctx.beginPath();

    firstPoint = true;
    mappings.forEach((mapping) => {
      const { candleIndex, indicatorIndex } = mapping;
      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = priceToY(data[indicatorIndex].lower);

      if (firstPoint) {
        ctx.moveTo(xPos, yPos);
        firstPoint = false;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    });

    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Draw values label
    const lastData = data[data.length - 1];
    if (lastData) {
      ctx.font = 'bold 10px monospace';
      ctx.fillStyle = settings.upper_band_color;
      ctx.fillText(`U: ${lastData.upper.toFixed(2)}`, x + 5, priceToY(lastData.upper));

      if (settings.show_middle) {
        ctx.fillStyle = settings.middle_band_color;
        ctx.fillText(`M: ${lastData.middle.toFixed(2)}`, x + 5, priceToY(lastData.middle));
      }

      ctx.fillStyle = settings.lower_band_color;
      ctx.fillText(`L: ${lastData.lower.toFixed(2)}`, x + 5, priceToY(lastData.lower));

      // Show bandwidth
      ctx.fillStyle = '#888';
      ctx.font = '9px monospace';
      ctx.fillText(`BW: ${lastData.bandwidth.toFixed(1)}%`, x + 5, bounds.y + bounds.height - 5);
    }
  }
}

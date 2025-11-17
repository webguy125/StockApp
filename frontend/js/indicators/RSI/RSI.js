/**
 * Relative Strength Index (RSI) Indicator
 * Momentum oscillator measuring speed and change of price movements
 * Custom implementation - no third-party libraries
 */

import { IndicatorBase } from '../IndicatorBase.js';

export class RSI extends IndicatorBase {
  constructor() {
    super({
      name: 'RSI',
      version: '1.0.0',
      description: 'Momentum oscillator measuring speed and change of price movements',
      tags: ['momentum', 'oscillator'],
      dependencies: [],
      output_type: 'oscillator',
      default_settings: {
        line_color: '#00FF00',
        line_style: 'solid',
        line_opacity: 1.0,
        lookback_period: 14,
        overbought: 70,
        oversold: 30,
        show_levels: true,
        level_color: '#888888',
        level_opacity: 0.5
      },
      alerts: {
        enabled: true,
        conditions: [
          { type: 'greater_than', field: 'value', threshold: 70, message: 'RSI Overbought (>70)' },
          { type: 'less_than', field: 'value', threshold: 30, message: 'RSI Oversold (<30)' }
        ]
      },
      help_text: 'RSI identifies overbought (>70) and oversold (<30) conditions in the market. Values range from 0-100.'
    });
  }

  /**
   * Calculate RSI values from OHLCV data
   * Formula:
   *   RSI = 100 - (100 / (1 + RS))
   *   RS = Average Gain / Average Loss
   *
   * @param {Array} candles - Array of OHLCV candles
   * @returns {Array} RSI values with metadata
   */
  calculate(candles) {
    if (!candles || candles.length < this.currentSettings.lookback_period + 1) {
      return [];
    }

    const period = this.currentSettings.lookback_period;
    const result = [];

    // Add null padding for early candles to keep array aligned
    for (let i = 0; i < period; i++) {
      result.push({
        date: candles[i].Date,
        value: null,
        avgGain: null,
        avgLoss: null,
        overbought: this.currentSettings.overbought,
        oversold: this.currentSettings.oversold
      });
    }

    // Calculate price changes
    const changes = [];
    for (let i = 1; i < candles.length; i++) {
      const change = candles[i].Close - candles[i - 1].Close;
      changes.push(change);
    }

    // Initial average gain/loss (Simple Moving Average for first period)
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

    // Calculate first RSI value
    let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    let rsi = 100 - (100 / (1 + rs));

    result.push({
      date: candles[period].Date,
      value: rsi,
      avgGain: avgGain,
      avgLoss: avgLoss,
      overbought: this.currentSettings.overbought,
      oversold: this.currentSettings.oversold
    });

    // Calculate subsequent RSI values using Wilder's smoothing
    for (let i = period; i < changes.length; i++) {
      const change = changes[i];
      let gain = 0;
      let loss = 0;

      if (change > 0) {
        gain = change;
      } else {
        loss = Math.abs(change);
      }

      // Wilder's smoothing method
      avgGain = ((avgGain * (period - 1)) + gain) / period;
      avgLoss = ((avgLoss * (period - 1)) + loss) / period;

      rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi = 100 - (100 / (1 + rs));

      result.push({
        date: candles[i + 1].Date,
        value: rsi,
        avgGain: avgGain,
        avgLoss: avgLoss,
        overbought: this.currentSettings.overbought,
        oversold: this.currentSettings.oversold
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
        default: 14,
        description: 'Number of periods for RSI calculation'
      },
      overbought: {
        type: 'number',
        label: 'Overbought Level',
        min: 50,
        max: 100,
        step: 1,
        default: 70,
        description: 'Overbought threshold (typically 70)'
      },
      oversold: {
        type: 'number',
        label: 'Oversold Level',
        min: 0,
        max: 50,
        step: 1,
        default: 30,
        description: 'Oversold threshold (typically 30)'
      },
      line_color: {
        type: 'color',
        label: 'Line Color',
        default: '#00FF00',
        description: 'Color of the RSI line'
      },
      line_opacity: {
        type: 'number',
        label: 'Line Opacity',
        min: 0,
        max: 1,
        step: 0.1,
        default: 1.0,
        description: 'Opacity of the RSI line (0-1)'
      },
      show_levels: {
        type: 'boolean',
        label: 'Show Overbought/Oversold Levels',
        default: true,
        description: 'Display horizontal lines at overbought/oversold levels'
      },
      level_color: {
        type: 'color',
        label: 'Level Line Color',
        default: '#888888',
        description: 'Color of overbought/oversold level lines'
      },
      level_opacity: {
        type: 'number',
        label: 'Level Line Opacity',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.5,
        description: 'Opacity of level lines (0-1)'
      }
    };
  }

  /**
   * Render RSI on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Object} bounds - Rendering bounds {x, y, width, height}
   * @param {Array} data - RSI data
   * @param {Array} mappings - Array of {candleIndex, indicatorIndex} mappings
   * @param {Number} startIndex - Start index (may be fractional for smooth panning)
   * @param {Number} endIndex - End index (may be fractional for smooth panning)
   */
  render(ctx, bounds, data, mappings, startIndex, endIndex) {
    if (!data || data.length === 0 || !mappings || mappings.length === 0) return;

    const { x, y, width, height} = bounds;
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

    // Draw overbought/oversold levels
    if (settings.show_levels) {
      const overboughtY = y + height * (1 - settings.overbought / 100);
      const oversoldY = y + height * (1 - settings.oversold / 100);

      ctx.strokeStyle = settings.level_color;
      ctx.globalAlpha = settings.level_opacity;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);

      // Overbought line
      ctx.beginPath();
      ctx.moveTo(x, overboughtY);
      ctx.lineTo(x + width, overboughtY);
      ctx.stroke();

      // Oversold line
      ctx.beginPath();
      ctx.moveTo(x, oversoldY);
      ctx.lineTo(x + width, oversoldY);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;

      // Labels
      ctx.fillStyle = settings.level_color;
      ctx.font = '10px monospace';
      ctx.fillText(settings.overbought.toString(), x + 5, overboughtY - 2);
      ctx.fillText(settings.oversold.toString(), x + 5, oversoldY + 10);
    }

    // Draw RSI line
    ctx.strokeStyle = settings.line_color;
    ctx.globalAlpha = settings.line_opacity;
    ctx.lineWidth = 2;

    ctx.beginPath();

    let firstPoint = true;
    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;

      if (indicatorIndex < 0 || indicatorIndex >= data.length) return;

      const rsiValue = data[indicatorIndex].value;

      // Skip null values and restart line
      if (rsiValue === null || rsiValue === undefined) {
        firstPoint = true;
        return;
      }

      // Calculate X position using candleIndex for proper alignment with candle center
      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = y + height * (1 - rsiValue / 100);

      if (firstPoint) {
        ctx.moveTo(xPos, yPos);
        firstPoint = false;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    });

    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Draw current value label
    const lastData = data[data.length - 1];
    if (lastData && lastData.value !== null && lastData.value !== undefined) {
      const labelY = y + height * (1 - lastData.value / 100);
      ctx.fillStyle = settings.line_color;
      ctx.fillRect(x + width - 50, labelY - 8, 48, 16);
      ctx.fillStyle = '#000';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(lastData.value.toFixed(2), x + width - 48, labelY + 3);
    }
  }
}

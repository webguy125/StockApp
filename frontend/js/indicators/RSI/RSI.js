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
      version: '2.0.0',
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
        level_opacity: 0.5,

        // Signal Detection
        enable_signals: true,
        signal_cooldown: 3,          // Bars between signals

        // Machine Learning Filter
        enable_ml_filter: false,     // Disabled - RSI signals show without ML filtering
        ml_confidence_threshold: 0.7,
        show_ml_score: false,

        // Signal Colors
        buy_signal_color: '#00FF00',
        sell_signal_color: '#FF0000'
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
  async calculate(candles) {
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
        oversold: this.currentSettings.oversold,
        signal: null,
        ml_score: null
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
      oversold: this.currentSettings.oversold,
      signal: null,
      ml_score: null
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
        oversold: this.currentSettings.oversold,
        signal: null,
        ml_score: null
      });
    }

    // === Signal Detection ===
    if (this.currentSettings.enable_signals) {
      await this._detectSignals(result, candles, this.currentSettings);
    }

    return result;
  }

  /**
   * Detect RSI buy/sell signals
   * Buy: RSI in oversold zone AND turning up (momentum reversal)
   * Sell: RSI in overbought zone AND turning down (momentum reversal)
   * @private
   */
  async _detectSignals(data, candles, settings) {
    if (data.length < 3) return;

    let lastSignalIndex = -settings.signal_cooldown;

    for (let i = 2; i < data.length; i++) {
      const current = data[i];
      const prev = data[i - 1];
      const prev2 = data[i - 2];

      // Skip if values are null
      if (current.value === null || prev.value === null || prev2.value === null) continue;

      // Check cooldown
      if (i - lastSignalIndex < settings.signal_cooldown) continue;

      let signal = null;

      // Buy signal: RSI below oversold AND turning up (higher low forming)
      // This catches the reversal, not just the cross
      if (prev.value < settings.oversold &&
          current.value > prev.value &&
          prev.value <= prev2.value) {
        signal = 'buy';
      }
      // Sell signal: RSI above overbought AND turning down (lower high forming)
      else if (prev.value > settings.overbought &&
               current.value < prev.value &&
               prev.value >= prev2.value) {
        signal = 'sell';
      }

      if (signal) {
        // Apply ML filter if enabled
        if (settings.enable_ml_filter) {
          const mlScore = await this._getMLScore(candles, i, current, settings);
          current.ml_score = mlScore;

          if (mlScore >= settings.ml_confidence_threshold) {
            current.signal = signal;
            lastSignalIndex = i;
            console.log(`🎯 RSI: ${signal.toUpperCase()} signal at ${current.date} (RSI: ${current.value.toFixed(1)}, ML: ${(mlScore * 100).toFixed(0)}%)`);
          } else {
            console.log(`❌ RSI: ${signal.toUpperCase()} rejected by ML at ${current.date} (ML: ${(mlScore * 100).toFixed(0)}% < ${(settings.ml_confidence_threshold * 100).toFixed(0)}%)`);
          }
        } else {
          // No ML filter, show signal
          current.signal = signal;
          lastSignalIndex = i;
          console.log(`🎯 RSI: ${signal.toUpperCase()} signal at ${current.date} (RSI: ${current.value.toFixed(1)})`);
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
      console.error('❌ RSI ML score calculation failed:', error);
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

      // Feature 1: Oscillator range (RSI normalized)
      const oscillatorRange = signalData.value / 100.0;

      // Feature 2: Weighted price trend (use close vs 20-SMA)
      let weightedPriceTrend = 0.5;
      if (index >= 20) {
        const sma20 = closes.slice(index - 19, index + 1).reduce((a, b) => a + b, 0) / 20;
        const deviation = (closes[index] - sma20) / closes[index];
        weightedPriceTrend = Math.tanh(deviation * 5) / 2 + 0.5;
      }

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

      // Feature 7: RSI (already have it)
      const rsiNorm = signalData.value / 100.0;

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
      console.error('❌ RSI ML feature calculation failed:', error);
      return null;
    }
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
      },
      enable_signals: {
        type: 'boolean',
        label: 'Enable Signals',
        default: true,
        description: 'Enable buy/sell signal detection at RSI extremes'
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

    // Draw signal markers
    if (settings.enable_signals) {
      this._drawSignalMarkers(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, y, height, settings);
    }
  }

  /**
   * Draw signal markers (buy/sell arrows) in RSI panel
   * @private
   */
  _drawSignalMarkers(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, y, height, settings) {
    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      const d = data[indicatorIndex];

      if (!d || !d.signal || d.value === null) return;

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = y + height * (1 - d.value / 100);

      // Draw arrow
      const arrowSize = 8;
      ctx.fillStyle = d.signal === 'buy' ? settings.buy_signal_color : settings.sell_signal_color;

      ctx.beginPath();
      if (d.signal === 'buy') {
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
        ctx.fillStyle = d.signal === 'buy' ? settings.buy_signal_color : settings.sell_signal_color;

        const textMetrics = ctx.measureText(scoreText);
        const textX = xPos - textMetrics.width / 2;
        const textY = d.signal === 'buy' ? yPos - arrowSize - 4 : yPos + arrowSize + 12;

        ctx.fillText(scoreText, textX, textY);
      }
    });
  }
}

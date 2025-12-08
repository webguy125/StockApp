/**
 * Triad Trend Pulse Indicator
 * Normalized Adaptive Oscillator & Weighted Regression
 *
 * Converted from TradingView Pine Script
 *
 * Components:
 * 1. Weighted Regression - Blends linear regression of price with RSI
 * 2. Adaptive Oscillator - Combines momentum, volume, and macro trends
 */

import { IndicatorBase } from '../IndicatorBase.js';

export class TriadTrendPulse extends IndicatorBase {
  constructor() {
    super({
      name: 'TriadTrendPulse',
      version: '2.0.0',
      description: 'Normalized Adaptive Oscillator & Weighted Regression',
      tags: ['trend', 'momentum', 'oscillator'],
      dependencies: [],
      output_type: 'oscillator',
      default_settings: {
        // Regression Parameters
        reg_length: 20,
        offset: 0,

        // Weighted Regression Parameters
        price_weight: 1.0,
        rsi_weight: 0.5,
        chart_blend: 0.7,  // 0=combined, 1=price only
        rsi_len: 14,

        // Adaptive Oscillator
        adaptive_period: 50,

        // Signal Detection
        enable_signals: true,
        overbought_threshold: 80,    // Signal zone for sell (default 80, range: +100 to -100)
        oversold_threshold: -80,     // Signal zone for buy (default -80, range: -100 to +100)
        min_separation: 5,           // Minimum distance between lines to confirm signal
        confirmation_bars: 1,        // Number of bars signal must hold (1 = immediate)

        // Machine Learning Filter
        enable_ml_filter: true,      // Use ML to validate signals
        ml_confidence_threshold: 0.7, // Minimum ML score to show signal (0.7 = 70%)
        show_ml_score: true,         // Display ML confidence % on signal labels

        // Display Colors
        weighted_regression_color: '#9C27B0',  // Purple
        adaptive_oscillator_color: '#2196F3',  // Blue
        line_opacity: 0.9,
        line_width: 2,
        buy_signal_color: '#00FF00',   // Green
        sell_signal_color: '#FF0000'   // Red
      },
      alerts: {
        enabled: false,
        conditions: []
      },
      help_text: 'Normalized Adaptive Oscillator & Weighted Regression indicator combining price regression with RSI and adaptive market analysis.'
    });
  }

  /**
   * Normalize array to 0-1 range over a lookback period
   * @param {Array} values - Values to normalize
   * @param {number} length - Lookback period
   * @returns {Array} Normalized values
   * @private
   */
  _normalize(values, length) {
    if (!values || values.length === 0) return [];

    const result = [];

    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - length + 1);
      const window = values.slice(start, i + 1);

      const min = Math.min(...window);
      const max = Math.max(...window);
      const diff = max - min;

      result.push(diff === 0 ? 0 : (values[i] - min) / diff);
    }

    return result;
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

    // First RSI value
    let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));

    // Subsequent RSI values using smoothed averages
    for (let i = period; i < changes.length; i++) {
      const change = changes[i];

      if (change > 0) {
        avgGain = (avgGain * (period - 1) + change) / period;
        avgLoss = (avgLoss * (period - 1)) / period;
      } else {
        avgGain = (avgGain * (period - 1)) / period;
        avgLoss = (avgLoss * (period - 1) + Math.abs(change)) / period;
      }

      rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }

    return rsi;
  }

  /**
   * Calculate linear regression
   * @param {Array} values - Values to regress
   * @param {number} period - Regression period
   * @param {number} offset - Offset for projection
   * @returns {Array} Regression values
   * @private
   */
  _calculateLinearRegression(values, period, offset = 0) {
    if (!values || values.length < period) return [];

    const result = [];

    for (let i = period - 1; i < values.length; i++) {
      const slice = values.slice(i - period + 1, i + 1);

      // Linear regression calculation
      let sumX = 0;
      let sumY = 0;
      let sumXY = 0;
      let sumX2 = 0;

      for (let j = 0; j < period; j++) {
        const x = j;
        const y = slice[j];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
      }

      const n = period;
      const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
      const intercept = (sumY - slope * sumX) / n;

      // Project forward by offset
      const projectedX = period - 1 + offset;
      result.push(slope * projectedX + intercept);
    }

    return result;
  }

  /**
   * Calculate momentum (rate of change)
   * @param {Array} values - Values
   * @param {number} period - Momentum period
   * @returns {Array} Momentum values
   * @private
   */
  _calculateMomentum(values, period) {
    if (!values || values.length < period + 1) return [];

    const result = [];

    for (let i = period; i < values.length; i++) {
      result.push(values[i] - values[i - period]);
    }

    return result;
  }

  /**
   * Calculate SMA (Simple Moving Average)
   * @param {Array} values - Values
   * @param {number} period - SMA period
   * @returns {Array} SMA values
   * @private
   */
  _calculateSMA(values, period) {
    if (!values || values.length < period) return [];

    const result = [];

    for (let i = period - 1; i < values.length; i++) {
      const slice = values.slice(i - period + 1, i + 1);
      const sum = slice.reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }

    return result;
  }

  /**
   * Get ML confidence score for a signal
   * @param {Array} candles - OHLCV candle data
   * @param {number} index - Index of signal candle
   * @param {Object} signalData - Signal data point
   * @param {Object} settings - Indicator settings
   * @returns {number} ML confidence score (0.0 to 1.0)
   * @private
   */
  async _getMLScore(candles, index, signalData, settings) {
    try {
      // Calculate the 9 ML features at this candle
      const features = this._calculateMLFeatures(candles, index, signalData, settings);

      if (!features || features.some(f => isNaN(f))) {
        console.warn('⚠️ TriadTrendPulse: Invalid ML features, using fallback score 0.5');
        return 0.5;
      }

      // Get symbol from window (set by chart)
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
        console.error('❌ ML API error:', response.status);
        return 0.5; // Fallback score
      }

      const result = await response.json();
      const score = result.scores[0];

      return score;

    } catch (error) {
      console.error('❌ ML score calculation failed:', error);
      return 0.5; // Fallback score on error
    }
  }

  /**
   * Calculate ML features for a candle
   * Returns the 9 features needed by ML model
   * @private
   */
  _calculateMLFeatures(candles, index, signalData, settings) {
    try {
      const closes = candles.map(c => c.Close);
      const highs = candles.map(c => c.High);
      const lows = candles.map(c => c.Low);
      const volumes = candles.map(c => c.Volume || 0);

      // Feature 1: Oscillator range (already calculated)
      const oscillatorRange = signalData.adaptive_oscillator / 100.0; // Normalize to [0, 1]

      // Feature 2: Weighted price trend (already calculated)
      const weightedPriceTrend = signalData.weighted_regression / 100.0; // Normalize to [0, 1]

      // Feature 3: Short-term trend (5-bar price change)
      const shortTrendBars = 5;
      let shortTrend = 0;
      if (index >= shortTrendBars) {
        const change = (closes[index] - closes[index - shortTrendBars]) / closes[index - shortTrendBars];
        shortTrend = Math.tanh(change * 10) / 2 + 0.5; // Normalize to [0, 1]
      }

      // Feature 4: ADX (calculate simplified version)
      const adx = this._calculateADXValue(highs, lows, closes, index, 14);
      const adxNorm = adx / 100.0; // Normalize to [0, 1]

      // Feature 5: Volume change (current vs 20-bar SMA)
      let volumeChange = 1.0;
      if (index >= 20) {
        const volumeSMA = volumes.slice(index - 19, index + 1).reduce((a, b) => a + b, 0) / 20;
        volumeChange = Math.min(volumes[index] / volumeSMA, 3) / 3; // Cap at 3x, normalize
      }

      // Feature 6: ATR (volatility)
      const atr = this._calculateATRValue(highs, lows, closes, index, 14);
      const atrNorm = atr / closes[index]; // ATR as % of price

      // Feature 7: RSI
      const rsi = this._calculateRSIValue(closes, index, 14);
      const rsiNorm = rsi / 100.0; // Normalize to [0, 1]

      // Feature 8: Momentum (20-bar price change)
      let momentum = 0.5;
      if (index >= 20) {
        const change = (closes[index] - closes[index - 20]) / closes[index - 20];
        momentum = Math.tanh(change * 10) / 2 + 0.5; // Normalize to [0, 1]
      }

      // Feature 9: Timeframe encoding (assume daily for now, can be enhanced)
      const timeframeNorm = 0.5; // Default for daily (6/12)

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
      console.error('❌ ML feature calculation failed:', error);
      return null;
    }
  }

  /**
   * Calculate RSI value at a specific index
   * @private
   */
  _calculateRSIValue(closes, index, period) {
    if (index < period) return 50; // Default

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
   * Calculate ATR value at a specific index
   * @private
   */
  _calculateATRValue(highs, lows, closes, index, period) {
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
   * Calculate ADX value at a specific index (simplified)
   * @private
   */
  _calculateADXValue(highs, lows, closes, index, period) {
    if (index < period) return 0;

    // Simplified ADX calculation
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
   * Detect buy/sell signals based on line divergence at extremes
   * @param {Array} data - Indicator data array
   * @param {Array} candles - OHLCV candle data
   * @param {Object} settings - Indicator settings
   * @private
   */
  async _detectSignals(data, candles, settings) {
    if (data.length < settings.confirmation_bars + 1) return;

    for (let i = settings.confirmation_bars; i < data.length; i++) {
      const current = data[i];
      const wr = current.weighted_regression;
      const ao = current.adaptive_oscillator;

      // Skip if either value is null
      if (wr === null || ao === null) continue;

      // Check for overbought condition (both lines HIGH)
      const isOverbought = wr >= settings.overbought_threshold && ao >= settings.overbought_threshold;

      // Check for oversold condition (both lines LOW)
      const isOversold = wr <= settings.oversold_threshold && ao <= settings.oversold_threshold;

      // Skip if not in extreme zone
      if (!isOverbought && !isOversold) continue;

      // Calculate separation (purple moving away from blue)
      const separation = Math.abs(wr - ao);

      // Check minimum separation
      if (separation < settings.min_separation) continue;

      // Determine signal direction based on which line is leading
      let signal = null;

      if (isOverbought) {
        // At overbought: Purple crosses DOWN away from Blue = SELL
        if (wr < ao) {
          // Check confirmation: signal held for required bars
          let confirmed = true;
          for (let j = 1; j <= settings.confirmation_bars && i - j >= 0; j++) {
            const prev = data[i - j];
            if (prev.weighted_regression === null || prev.adaptive_oscillator === null) {
              confirmed = false;
              break;
            }
            // Verify separation was maintained
            if (Math.abs(prev.weighted_regression - prev.adaptive_oscillator) < settings.min_separation) {
              confirmed = false;
              break;
            }
            // Verify still overbought
            if (prev.weighted_regression < settings.overbought_threshold || prev.adaptive_oscillator < settings.overbought_threshold) {
              confirmed = false;
              break;
            }
          }

          if (confirmed) {
            signal = 'sell';
          }
        }
      }

      if (isOversold) {
        // At oversold: Purple crosses UP away from Blue = BUY
        if (wr > ao) {
          // Check confirmation: signal held for required bars
          let confirmed = true;
          for (let j = 1; j <= settings.confirmation_bars && i - j >= 0; j++) {
            const prev = data[i - j];
            if (prev.weighted_regression === null || prev.adaptive_oscillator === null) {
              confirmed = false;
              break;
            }
            // Verify separation was maintained
            if (Math.abs(prev.weighted_regression - prev.adaptive_oscillator) < settings.min_separation) {
              confirmed = false;
              break;
            }
            // Verify still oversold
            if (prev.weighted_regression > settings.oversold_threshold || prev.adaptive_oscillator > settings.oversold_threshold) {
              confirmed = false;
              break;
            }
          }

          if (confirmed) {
            signal = 'buy';
          }
        }
      }

      // Only mark signal on the confirmation bar (avoid duplicate signals)
      if (signal && (!data[i - 1] || data[i - 1].signal !== signal)) {
        // Apply ML filter if enabled
        if (settings.enable_ml_filter) {
          const mlScore = await this._getMLScore(candles, i, current, settings);
          current.ml_score = mlScore;

          if (mlScore >= settings.ml_confidence_threshold) {
            current.signal = signal;
            console.log(`🎯 TriadTrendPulse: ${signal.toUpperCase()} signal at ${current.date} (WR: ${wr.toFixed(1)}, AO: ${ao.toFixed(1)}, ML: ${(mlScore * 100).toFixed(0)}%)`);
          } else {
            console.log(`❌ TriadTrendPulse: ${signal.toUpperCase()} rejected by ML at ${current.date} (ML: ${(mlScore * 100).toFixed(0)}% < ${(settings.ml_confidence_threshold * 100).toFixed(0)}%)`);
          }
        } else {
          // No ML filter, show signal
          current.signal = signal;
          console.log(`🎯 TriadTrendPulse: ${signal.toUpperCase()} signal at ${current.date} (WR: ${wr.toFixed(1)}, AO: ${ao.toFixed(1)})`);
        }
      }
    }
  }

  /**
   * Calculate indicator values
   * @param {Array} candles - OHLCV candle data
   * @returns {Array} Calculated indicator data
   */
  async calculate(candles) {
    if (!candles || candles.length < 50) {
      console.log('⚠️ TriadTrendPulse: Not enough candles', candles?.length);
      return [];
    }

    // Use this.currentSettings (from IndicatorBase)
    const settings = this.currentSettings;

    console.log(`📊 TriadTrendPulse v${this.version}: Calculating for ${candles.length} candles`);
    console.log('📊 TriadTrendPulse: Settings:', settings);

    const closes = candles.map(c => c.Close);
    const volumes = candles.map(c => c.Volume || 0);

    // === Calculate RSI ===
    const rsiValue = this._calculateRSI(closes, settings.rsi_len);

    // === Weighted Regression Calculation ===

    // Normalize close and RSI over reg_length
    const normClose = this._normalize(closes, settings.reg_length);
    const normRSI = this._normalize(rsiValue, settings.reg_length);

    // Price regression
    const priceReg = this._calculateLinearRegression(closes, settings.reg_length, settings.offset);

    // Combined regression (price + RSI)
    const combinedValues = [];
    const rsiStartIdx = settings.rsi_len; // RSI starts after rsi_len candles

    for (let i = 0; i < normClose.length; i++) {
      const rsiIdx = i - rsiStartIdx;
      if (rsiIdx >= 0 && rsiIdx < normRSI.length) {
        const combined = settings.price_weight * normClose[i] + settings.rsi_weight * normRSI[rsiIdx];
        combinedValues.push(combined);
      } else {
        combinedValues.push(normClose[i]); // Use only price if RSI not available yet
      }
    }

    const combinedReg = this._calculateLinearRegression(combinedValues, settings.reg_length, settings.offset);

    // Blend price and combined regression
    const weightedPriceTrend = [];
    const priceRegStartIdx = settings.reg_length - 1;
    const combinedRegStartIdx = settings.reg_length - 1;

    for (let i = 0; i < closes.length; i++) {
      const priceRegIdx = i - priceRegStartIdx;
      const combinedRegIdx = i - combinedRegStartIdx;

      if (priceRegIdx >= 0 && priceRegIdx < priceReg.length &&
          combinedRegIdx >= 0 && combinedRegIdx < combinedReg.length) {
        const blended = settings.chart_blend * priceReg[priceRegIdx] +
                       (1 - settings.chart_blend) * combinedReg[combinedRegIdx];
        weightedPriceTrend.push(blended);
      } else {
        weightedPriceTrend.push(null);
      }
    }

    // Normalize weighted price trend to [-100, 100]
    const weightedPriceTrendNorm = [];
    for (let i = 0; i < weightedPriceTrend.length; i++) {
      if (weightedPriceTrend[i] === null) {
        weightedPriceTrendNorm.push(null);
        continue;
      }

      const start = Math.max(0, i - settings.reg_length + 1);
      const window = weightedPriceTrend.slice(start, i + 1).filter(v => v !== null);

      if (window.length === 0) {
        weightedPriceTrendNorm.push(null);
        continue;
      }

      const min = Math.min(...window);
      const max = Math.max(...window);
      const diff = max - min;

      if (diff === 0) {
        weightedPriceTrendNorm.push(0);
      } else {
        const normalized = ((weightedPriceTrend[i] - min) / diff) * 200 - 100;
        weightedPriceTrendNorm.push(normalized);
      }
    }

    // === Adaptive Oscillator Calculation ===

    // Momentum trend
    const momentum = this._calculateMomentum(closes, settings.reg_length);
    const momentumTrend = this._calculateLinearRegression(momentum, settings.reg_length, settings.offset);

    // Volume trend
    const volumeTrend = this._calculateLinearRegression(volumes, settings.reg_length, settings.offset);

    // Earnings impact (volume SMA)
    const earningsImpact = this._calculateSMA(volumes, settings.reg_length);

    // Macro impact (close SMA)
    const macroImpact = this._calculateSMA(closes, settings.reg_length);

    // Calculate weighted average
    const weightedAvg = [];
    const baseStartIdx = settings.reg_length - 1;
    const momentumStartIdx = settings.reg_length + (settings.reg_length - 1); // momentum + regression

    for (let i = 0; i < closes.length; i++) {
      const wptIdx = i;
      const momIdx = i - momentumStartIdx;
      const volIdx = i - baseStartIdx;
      const earnIdx = i - baseStartIdx;
      const macroIdx = i - baseStartIdx;

      if (wptIdx < weightedPriceTrend.length && weightedPriceTrend[wptIdx] !== null &&
          momIdx >= 0 && momIdx < momentumTrend.length &&
          volIdx >= 0 && volIdx < volumeTrend.length &&
          earnIdx >= 0 && earnIdx < earningsImpact.length &&
          macroIdx >= 0 && macroIdx < macroImpact.length) {

        const avg = (weightedPriceTrend[wptIdx] * 0.4) +
                    (momentumTrend[momIdx] * 0.2) +
                    (volumeTrend[volIdx] * 0.2) +
                    (earningsImpact[earnIdx] * 0.1) +
                    (macroImpact[macroIdx] * 0.1);
        weightedAvg.push(avg);
      } else {
        weightedAvg.push(null);
      }
    }

    // Dynamic oscillator
    const dynamicOscillator = this._calculateLinearRegression(
      weightedAvg.filter(v => v !== null),
      settings.reg_length,
      settings.offset
    );

    // Normalize oscillator to [-100, 100]
    const oscillatorRange = [];
    for (let i = 0; i < dynamicOscillator.length; i++) {
      const start = Math.max(0, i - settings.reg_length + 1);
      const window = dynamicOscillator.slice(start, i + 1);

      const min = Math.min(...window);
      const max = Math.max(...window);
      const diff = max - min;

      if (diff === 0) {
        oscillatorRange.push(0);
      } else {
        const normalized = ((dynamicOscillator[i] - min) / diff) * 200 - 100;
        oscillatorRange.push(normalized);
      }
    }

    // Build result array
    const result = [];
    const oscillatorStartIdx = closes.length - oscillatorRange.length;

    for (let i = 0; i < candles.length; i++) {
      const oscIdx = i - oscillatorStartIdx;

      result.push({
        date: candles[i].Date,
        weighted_regression: weightedPriceTrendNorm[i],
        adaptive_oscillator: (oscIdx >= 0 && oscIdx < oscillatorRange.length) ? oscillatorRange[oscIdx] : null,
        close: closes[i],
        signal: null,  // Will be populated by signal detection
        ml_score: null  // ML confidence score (0.0 to 1.0)
      });
    }

    // === Signal Detection ===
    if (settings.enable_signals) {
      await this._detectSignals(result, candles, settings);
    }

    // Debug: Check last few values
    const lastValues = result.slice(-3);
    console.log('📊 TriadTrendPulse: Last 3 values:', lastValues);
    console.log(`✅ TriadTrendPulse: Generated ${result.length} data points`);

    return result;
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
        description: 'Period for linear regression calculations'
      },
      offset: {
        type: 'number',
        label: 'Regression Offset',
        min: 0,
        max: 10,
        step: 1,
        default: 0,
        description: 'Forward projection offset for regression'
      },
      price_weight: {
        type: 'number',
        label: 'Price Weight',
        min: 0,
        max: 2,
        step: 0.1,
        default: 1.0,
        description: 'Weight for price in combined regression'
      },
      rsi_weight: {
        type: 'number',
        label: 'RSI Weight',
        min: 0,
        max: 2,
        step: 0.1,
        default: 0.5,
        description: 'Weight for RSI in combined regression'
      },
      chart_blend: {
        type: 'number',
        label: 'Chart Blend',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.7,
        description: 'Blend between price-only (1) and combined (0) regression'
      },
      rsi_len: {
        type: 'number',
        label: 'RSI Length',
        min: 2,
        max: 50,
        step: 1,
        default: 14,
        description: 'Period for RSI calculation'
      },
      adaptive_period: {
        type: 'number',
        label: 'Adaptive Period',
        min: 10,
        max: 100,
        step: 1,
        default: 50,
        description: 'Period for adaptive oscillator normalization'
      },
      weighted_regression_color: {
        type: 'color',
        label: 'Weighted Regression Color',
        default: '#9C27B0',
        description: 'Color for weighted regression line'
      },
      adaptive_oscillator_color: {
        type: 'color',
        label: 'Adaptive Oscillator Color',
        default: '#2196F3',
        description: 'Color for adaptive oscillator line'
      },
      line_width: {
        type: 'number',
        label: 'Line Width',
        min: 1,
        max: 5,
        step: 1,
        default: 2,
        description: 'Width of indicator lines'
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
      enable_signals: {
        type: 'boolean',
        label: 'Enable Signals',
        default: true,
        description: 'Enable buy/sell signal detection'
      },
      overbought_threshold: {
        type: 'number',
        label: 'Overbought Threshold',
        min: 0,
        max: 100,
        step: 5,
        default: 80,
        description: 'Threshold for sell signal zone (both lines must be above this)'
      },
      oversold_threshold: {
        type: 'number',
        label: 'Oversold Threshold',
        min: -100,
        max: 0,
        step: 5,
        default: -80,
        description: 'Threshold for buy signal zone (both lines must be below this)'
      },
      min_separation: {
        type: 'number',
        label: 'Minimum Separation',
        min: 0,
        max: 20,
        step: 1,
        default: 5,
        description: 'Minimum distance between lines to confirm signal'
      },
      confirmation_bars: {
        type: 'number',
        label: 'Confirmation Bars',
        min: 1,
        max: 5,
        step: 1,
        default: 1,
        description: 'Number of bars signal must hold before triggering'
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
      }
    };
  }

  /**
   * Render indicator on canvas
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Object} bounds - Rendering bounds {x, y, width, height}
   * @param {Array} data - Indicator data
   * @param {Array} mappings - Candle-to-indicator index mappings
   * @param {number} startIndex - Start candle index
   * @param {number} endIndex - End candle index
   */
  render(ctx, bounds, data, mappings, startIndex, endIndex) {
    console.log(`📊 TriadTrendPulse v${this.version}: Rendering ${data?.length} data points`);

    if (!data || data.length === 0 || !mappings || mappings.length === 0) {
      console.log('⚠️ TriadTrendPulse: No data to render');
      return;
    }

    const settings = this.currentSettings;
    const { x, y, width, height } = bounds;

    console.log(`📊 TriadTrendPulse: Rendering bounds:`, bounds);

    // Value to Y coordinate converter for [-100, 100] range
    const valueToY = (value) => {
      if (value === null || value === undefined) return null;
      // Map [-100, 100] to chart height
      const normalized = (value + 100) / 200; // Convert to [0, 1]
      return y + height - (normalized * height); // Flip Y (canvas 0 is top)
    };

    // Calculate X position helpers
    const visibleCandles = endIndex - startIndex + 1;
    const totalWidth = width / visibleCandles;
    const centerOffset = totalWidth / 2;

    // Set line style
    ctx.globalAlpha = settings.line_opacity;
    ctx.lineWidth = settings.line_width;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Draw adaptive oscillator line (blue)
    this._drawLine(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY,
      'adaptive_oscillator', settings.adaptive_oscillator_color, settings.line_width);

    // Draw weighted regression line (purple)
    this._drawLine(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY,
      'weighted_regression', settings.weighted_regression_color, settings.line_width);

    ctx.globalAlpha = 1.0;

    // Draw current values label
    const lastData = data[data.length - 1];
    if (lastData && lastData.weighted_regression !== null && lastData.adaptive_oscillator !== null) {
      ctx.font = 'bold 10px monospace';

      ctx.fillStyle = settings.weighted_regression_color;
      ctx.fillText(`WR: ${lastData.weighted_regression.toFixed(1)}`, x + 5, y + 15);

      ctx.fillStyle = settings.adaptive_oscillator_color;
      ctx.fillText(`AO: ${lastData.adaptive_oscillator.toFixed(1)}`, x + 5, y + 30);
    }

    // Draw reference lines
    ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Upper limit (+100)
    const y100 = valueToY(100);
    ctx.beginPath();
    ctx.moveTo(x, y100);
    ctx.lineTo(x + width, y100);
    ctx.stroke();

    // Lower limit (-100)
    const yMinus100 = valueToY(-100);
    ctx.beginPath();
    ctx.moveTo(x, yMinus100);
    ctx.lineTo(x + width, yMinus100);
    ctx.stroke();

    // Zero line
    const y0 = valueToY(0);
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x + width, y0);
    ctx.stroke();

    ctx.setLineDash([]);

    // Draw signal markers
    if (settings.enable_signals) {
      this._drawSignalMarkers(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY, settings);
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

    let pathStarted = false;

    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      const d = data[indicatorIndex];

      if (!d || d[field] === null || d[field] === undefined) {
        pathStarted = false;
        return;
      }

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;
      const yPos = valueToY(d[field]);

      if (yPos === null) {
        pathStarted = false;
        return;
      }

      if (!pathStarted) {
        ctx.moveTo(xPos, yPos);
        pathStarted = true;
      } else {
        ctx.lineTo(xPos, yPos);
      }
    });

    ctx.stroke();
  }

  /**
   * Draw signal markers (buy/sell arrows)
   * @private
   */
  _drawSignalMarkers(ctx, mappings, data, x, totalWidth, centerOffset, startIndex, valueToY, settings) {
    mappings.forEach(mapping => {
      const { candleIndex, indicatorIndex } = mapping;
      const d = data[indicatorIndex];

      if (!d || !d.signal) return;

      const xPos = x + ((candleIndex - startIndex) * totalWidth) + centerOffset;

      // Position signal at the weighted regression line position
      const yPos = valueToY(d.weighted_regression);
      if (yPos === null) return;

      // Draw signal marker
      if (d.signal === 'buy') {
        // Green up arrow
        ctx.fillStyle = settings.buy_signal_color;
        ctx.strokeStyle = settings.buy_signal_color;
        ctx.lineWidth = 2;

        // Arrow pointing up
        ctx.beginPath();
        ctx.moveTo(xPos, yPos + 10);  // Arrow tip
        ctx.lineTo(xPos - 5, yPos + 20);  // Left wing
        ctx.lineTo(xPos + 5, yPos + 20);  // Right wing
        ctx.closePath();
        ctx.fill();

        // Arrow shaft
        ctx.fillRect(xPos - 2, yPos + 18, 4, 8);

        // Label with optional ML score
        ctx.font = 'bold 10px Arial';
        let buyLabel = 'BUY';
        if (settings.show_ml_score && d.ml_score !== null && d.ml_score !== undefined) {
          buyLabel = `BUY ${(d.ml_score * 100).toFixed(0)}%`;
        }
        const buyLabelWidth = ctx.measureText(buyLabel).width;
        ctx.fillText(buyLabel, xPos - buyLabelWidth / 2, yPos + 38);

      } else if (d.signal === 'sell') {
        // Red down arrow
        ctx.fillStyle = settings.sell_signal_color;
        ctx.strokeStyle = settings.sell_signal_color;
        ctx.lineWidth = 2;

        // Arrow pointing down
        ctx.beginPath();
        ctx.moveTo(xPos, yPos - 10);  // Arrow tip
        ctx.lineTo(xPos - 5, yPos - 20);  // Left wing
        ctx.lineTo(xPos + 5, yPos - 20);  // Right wing
        ctx.closePath();
        ctx.fill();

        // Arrow shaft
        ctx.fillRect(xPos - 2, yPos - 26, 4, 8);

        // Label with optional ML score
        ctx.font = 'bold 10px Arial';
        let sellLabel = 'SELL';
        if (settings.show_ml_score && d.ml_score !== null && d.ml_score !== undefined) {
          sellLabel = `SELL ${(d.ml_score * 100).toFixed(0)}%`;
        }
        const sellLabelWidth = ctx.measureText(sellLabel).width;
        ctx.fillText(sellLabel, xPos - sellLabelWidth / 2, yPos - 28);
      }
    });
  }
}

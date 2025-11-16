/**
 * ORD Volume Signal Generation
 * Generates trade signals with confluence scoring
 *
 * Based on Timothy Ord's proprietary methodology:
 * - Minimum confluence score: 4
 * - Requires 3+ triggers from ORD + Elliott Wave
 * - Professional stop/target calculations
 * - Probability formula: 50 + (confluence * 9) + (rr > 5 ? 10 : 0)
 */

export class ORDVolumeSignals {
  constructor(candles, ordIndicators, waveData) {
    this.candles = candles;
    this.ordIndicators = ordIndicators;
    this.waveData = waveData;
    this.atr = ordIndicators.calculateATR(14);
  }

  /**
   * Generate trade signals with confluence scoring
   * @returns {Array} Array of trade signal objects
   */
  generateSignals() {
    const signals = [];
    const currentBar = this.candles.length - 1;
    const currentState = this.ordIndicators.indicators;

    console.log('[ORD Signals] Starting signal generation...');

    // Check for LONG signals
    const longSignal = this._checkLongSignal(currentBar, currentState);
    if (longSignal) {
      signals.push(longSignal);
    }

    // Check for SHORT signals
    const shortSignal = this._checkShortSignal(currentBar, currentState);
    if (shortSignal) {
      signals.push(shortSignal);
    }

    console.log('[ORD Signals] Generated', signals.length, 'signals');

    return signals;
  }

  /**
   * Check for LONG signal
   * Requires: confluence >= 4 AND at least 3 triggers
   * @private
   */
  _checkLongSignal(currentBar, indicators) {
    const triggers = [];
    let confluenceScore = 0;

    // ORD Volume triggers
    if (indicators.eightyPercentRule.bullish) {
      triggers.push('80% Rule bullish');
      confluenceScore++;
    }

    if (indicators.divergences.hiddenBullish) {
      triggers.push('Hidden bullish divergence');
      confluenceScore++;
    }

    if (indicators.divergences.classicBullish) {
      triggers.push('Classic bullish divergence');
      confluenceScore++;
    }

    if (indicators.climaxSpikes.greenClimax) {
      const greenVol = this.ordIndicators.histogram.green[currentBar];
      triggers.push(`Green climax +${this._formatVolume(greenVol)}`);
      confluenceScore++;
    }

    if (indicators.zeroLineCrosses.upBarsAgo <= 5) {
      triggers.push(`Zero-line cross up ${indicators.zeroLineCrosses.upBarsAgo} bars ago`);
      confluenceScore++;
    }

    // Elliott Wave triggers
    const ewTriggers = this._getElliottWaveLongTriggers(currentBar);
    triggers.push(...ewTriggers.triggers);
    confluenceScore += ewTriggers.score;

    // Volume surge check
    const avgVolume = this._averageVolume(currentBar, 21);
    if (this.candles[currentBar].volume > avgVolume * 2) {
      triggers.push('Volume surge > 2x average');
      confluenceScore++;
    }

    // Check minimum requirements
    if (confluenceScore < 4 || triggers.length < 3) {
      return null;
    }

    // Calculate entry, stop, targets
    const entry = this.candles[currentBar].close;
    const stop = this._calculateLongStop(currentBar);
    const targets = this._calculateTargets(entry, stop, 'LONG', currentBar);
    const rr = ((targets.target_3 - entry) / (entry - stop)).toFixed(1);
    const probability = this._calculateProbability(confluenceScore, parseFloat(rr));

    console.log('[ORD Signals] LONG signal generated:', {
      confluence: confluenceScore,
      triggers: triggers.length,
      probability,
      rr
    });

    return {
      signal_id: `ORD_LONG_${Date.now()}`,
      direction: 'LONG',
      entry: parseFloat(entry.toFixed(2)),
      stop: parseFloat(stop.toFixed(2)),
      target_1: parseFloat(targets.target_1.toFixed(2)),
      target_2: parseFloat(targets.target_2.toFixed(2)),
      target_3: parseFloat(targets.target_3.toFixed(2)),
      rr: `1:${rr}`,
      ord_triggers: triggers,
      ew_triggers: ewTriggers.triggers,
      confluence_score: confluenceScore,
      probability: probability,
      expires_bars: 21,
      ord_quote: this._getLongQuote(triggers)
    };
  }

  /**
   * Check for SHORT signal
   * Requires: confluence >= 4 AND at least 3 triggers
   * @private
   */
  _checkShortSignal(currentBar, indicators) {
    const triggers = [];
    let confluenceScore = 0;

    // ORD Volume triggers
    if (indicators.eightyPercentRule.bearish) {
      triggers.push('80% Rule bearish');
      confluenceScore++;
    }

    if (indicators.divergences.hiddenBearish) {
      triggers.push('Hidden bearish divergence');
      confluenceScore++;
    }

    if (indicators.divergences.classicBearish) {
      triggers.push('Classic bearish divergence');
      confluenceScore++;
    }

    if (indicators.climaxSpikes.redClimax) {
      const redVol = this.ordIndicators.histogram.red[currentBar];
      triggers.push(`Red climax -${this._formatVolume(redVol)}`);
      confluenceScore++;
    }

    if (indicators.zeroLineCrosses.downBarsAgo <= 5) {
      triggers.push(`Zero-line cross down ${indicators.zeroLineCrosses.downBarsAgo} bars ago`);
      confluenceScore++;
    }

    // Elliott Wave triggers
    const ewTriggers = this._getElliottWaveShortTriggers(currentBar);
    triggers.push(...ewTriggers.triggers);
    confluenceScore += ewTriggers.score;

    // Volume surge check
    const avgVolume = this._averageVolume(currentBar, 21);
    if (this.candles[currentBar].volume > avgVolume * 2) {
      triggers.push('Volume surge > 2x average');
      confluenceScore++;
    }

    // Check minimum requirements
    if (confluenceScore < 4 || triggers.length < 3) {
      return null;
    }

    // Calculate entry, stop, targets
    const entry = this.candles[currentBar].close;
    const stop = this._calculateShortStop(currentBar);
    const targets = this._calculateTargets(entry, stop, 'SHORT', currentBar);
    const rr = ((entry - targets.target_3) / (stop - entry)).toFixed(1);
    const probability = this._calculateProbability(confluenceScore, parseFloat(rr));

    console.log('[ORD Signals] SHORT signal generated:', {
      confluence: confluenceScore,
      triggers: triggers.length,
      probability,
      rr
    });

    return {
      signal_id: `ORD_SHORT_${Date.now()}`,
      direction: 'SHORT',
      entry: parseFloat(entry.toFixed(2)),
      stop: parseFloat(stop.toFixed(2)),
      target_1: parseFloat(targets.target_1.toFixed(2)),
      target_2: parseFloat(targets.target_2.toFixed(2)),
      target_3: parseFloat(targets.target_3.toFixed(2)),
      rr: `1:${rr}`,
      ord_triggers: triggers,
      ew_triggers: ewTriggers.triggers,
      confluence_score: confluenceScore,
      probability: probability,
      expires_bars: 21,
      ord_quote: this._getShortQuote(triggers)
    };
  }

  /**
   * Get Elliott Wave triggers for LONG signals
   * @private
   */
  _getElliottWaveLongTriggers(currentBar) {
    const triggers = [];
    let score = 0;

    // Find current wave context
    const currentWave = this._getCurrentWave(currentBar);

    if (currentWave) {
      // Wave 4 complete (preparing for Wave 5 rally)
      if (currentWave.label === '4' || currentWave.label === 'C') {
        triggers.push(`Wave ${currentWave.label} complete - rally expected`);
        score++;
      }

      // Wave 1 starting (new impulse)
      if (currentWave.label === '1') {
        triggers.push('Wave 1 impulse starting');
        score++;
      }

      // Calculate Fibonacci targets
      const fibTarget = this._getFibonacciTarget(currentWave, 'LONG');
      if (fibTarget) {
        triggers.push(`Fib 1.618 target: ${fibTarget.toFixed(2)}`);
      }
    }

    return { triggers, score };
  }

  /**
   * Get Elliott Wave triggers for SHORT signals
   * @private
   */
  _getElliottWaveShortTriggers(currentBar) {
    const triggers = [];
    let score = 0;

    const currentWave = this._getCurrentWave(currentBar);

    if (currentWave) {
      // Wave 5 exhaustion
      if (currentWave.label === '5') {
        triggers.push('Wave 5 exhaustion - correction due');
        score++;
      }

      // Wave A starting (corrective decline)
      if (currentWave.label === 'A') {
        triggers.push('Wave A correction starting');
        score++;
      }

      // Calculate Fibonacci targets
      const fibTarget = this._getFibonacciTarget(currentWave, 'SHORT');
      if (fibTarget) {
        triggers.push(`Fib 0.618 retracement: ${fibTarget.toFixed(2)}`);
      }
    }

    return { triggers, score };
  }

  /**
   * Calculate LONG stop loss
   * Most recent swing low + 0.5 * ATR(14)
   * @private
   */
  _calculateLongStop(currentBar) {
    const swingLow = this._findRecentSwingLow(currentBar, 21);
    const atr = this.atr[currentBar];
    return swingLow - (0.5 * atr);
  }

  /**
   * Calculate SHORT stop loss
   * Most recent swing high + 0.5 * ATR(14)
   * @private
   */
  _calculateShortStop(currentBar) {
    const swingHigh = this._findRecentSwingHigh(currentBar, 21);
    const atr = this.atr[currentBar];
    return swingHigh + (0.5 * atr);
  }

  /**
   * Calculate profit targets
   * Target 1: 1.0 * ATR
   * Target 2: Fib 1.618
   * Target 3: Fib 2.618
   * @private
   */
  _calculateTargets(entry, stop, direction, currentBar) {
    const atr = this.atr[currentBar];
    const risk = Math.abs(entry - stop);

    if (direction === 'LONG') {
      const target_1 = entry + (atr * 1.0);
      const target_2 = entry + (risk * 1.618);
      const target_3 = entry + (risk * 2.618);
      return { target_1, target_2, target_3 };
    } else {
      const target_1 = entry - (atr * 1.0);
      const target_2 = entry - (risk * 1.618);
      const target_3 = entry - (risk * 2.618);
      return { target_1, target_2, target_3 };
    }
  }

  /**
   * Calculate probability
   * Formula: 50 + (confluence * 9) + (rr > 5 ? 10 : 0)
   * @private
   */
  _calculateProbability(confluenceScore, rr) {
    let probability = 50 + (confluenceScore * 9);
    if (rr > 5) {
      probability += 10;
    }
    return Math.min(probability, 99); // Cap at 99%
  }

  /**
   * Get appropriate LONG quote based on triggers
   * @private
   */
  _getLongQuote(triggers) {
    if (triggers.some(t => t.includes('Green climax'))) {
      return 'Green Monster Buy — institutions are jamming the bid hard';
    }
    if (triggers.some(t => t.includes('80% Rule'))) {
      return '80% Rule Bullish — volume refused to confirm the low';
    }
    if (triggers.some(t => t.includes('Hidden'))) {
      return 'Hidden Buying Wave — textbook Ord accumulation';
    }
    if (triggers.some(t => t.includes('Zero-line cross up'))) {
      return 'Zero-line Cross Up — momentum shift to bulls';
    }
    return 'High-probability LONG setup confirmed';
  }

  /**
   * Get appropriate SHORT quote based on triggers
   * @private
   */
  _getShortQuote(triggers) {
    if (triggers.some(t => t.includes('Red climax'))) {
      return 'Red Exhaustion Sell — smart money distributing at highs';
    }
    if (triggers.some(t => t.includes('80% Rule'))) {
      return '80% Rule Bearish — distribution detected at new highs';
    }
    if (triggers.some(t => t.includes('Hidden bearish'))) {
      return 'Hidden Selling Wave — institutional distribution';
    }
    if (triggers.some(t => t.includes('Wave 5'))) {
      return 'Wave 5 Exhaustion — impulse complete, correction due';
    }
    return 'High-probability SHORT setup confirmed';
  }

  /**
   * Find recent swing low
   * @private
   */
  _findRecentSwingLow(currentBar, lookback) {
    let swingLow = Infinity;
    for (let i = Math.max(0, currentBar - lookback); i < currentBar; i++) {
      if (this.candles[i].low < swingLow) {
        swingLow = this.candles[i].low;
      }
    }
    return swingLow;
  }

  /**
   * Find recent swing high
   * @private
   */
  _findRecentSwingHigh(currentBar, lookback) {
    let swingHigh = -Infinity;
    for (let i = Math.max(0, currentBar - lookback); i < currentBar; i++) {
      if (this.candles[i].high > swingHigh) {
        swingHigh = this.candles[i].high;
      }
    }
    return swingHigh;
  }

  /**
   * Get current wave context from waveData
   * @private
   */
  _getCurrentWave(currentBar) {
    if (!this.waveData || this.waveData.length === 0) return null;

    // Find wave that contains current bar
    for (const wave of this.waveData) {
      if (wave.endIdx === currentBar || wave.startIdx === currentBar) {
        return wave;
      }
    }

    // Return most recent wave
    return this.waveData[this.waveData.length - 1];
  }

  /**
   * Calculate Fibonacci extension/retracement targets
   * @private
   */
  _getFibonacciTarget(wave, direction) {
    if (!wave) return null;

    const [x1, y1, x2, y2] = wave.line;
    const waveLength = Math.abs(y2 - y1);

    if (direction === 'LONG') {
      // 1.618 extension above current price
      return y2 + (waveLength * 1.618);
    } else {
      // 0.618 retracement below current price
      return y2 - (waveLength * 0.618);
    }
  }

  /**
   * Calculate average volume
   * @private
   */
  _averageVolume(currentBar, period) {
    let sum = 0;
    for (let i = Math.max(0, currentBar - period + 1); i <= currentBar; i++) {
      sum += this.candles[i].volume;
    }
    return sum / Math.min(period, currentBar + 1);
  }

  /**
   * Format volume for display
   * @private
   */
  _formatVolume(volume) {
    if (volume >= 1000000) {
      return (volume / 1000000).toFixed(1) + 'M';
    } else if (volume >= 1000) {
      return (volume / 1000).toFixed(1) + 'K';
    }
    return volume.toFixed(0);
  }
}

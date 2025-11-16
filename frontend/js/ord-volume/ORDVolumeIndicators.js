/**
 * ORD Volume Indicators
 * Timothy Ord's Proprietary ORD Volume Methodology
 *
 * Implements complete ORD Volume calculations:
 * - Histogram (True Range weighted green/red pressure)
 * - Cumulative buying/selling pressure (lifetime tracking)
 * - 80% Rule (52-bar lookback)
 * - Hidden & Classic divergences (34-bar, 13-bar pivots)
 * - Zero-line cross tracking
 * - Climax spike detection (2.5x 21-bar SMA)
 *
 * Based on Timothy Ord's 40-year proven methodology
 */

export class ORDVolumeIndicators {
  constructor(candles) {
    this.candles = candles;
    this.histogram = null;
    this.cumulativePressure = null;
    this.indicators = null;
  }

  /**
   * Calculate complete ORD Volume analysis
   * @returns {Object} Complete ORD Volume state
   */
  calculate() {
    const barCount = this.candles.length;
    console.log('[ORD Indicators] Starting calculation for', barCount, 'bars');

    // HARD LIMIT: Prevent browser freeze on massive datasets
    const MAX_BARS = 4500; // Balanced limit - allows Daily (~4000 bars) but blocks minute timeframes
    if (barCount > MAX_BARS) {
      throw new Error(`Dataset too large for ORD Volume analysis. Maximum ${MAX_BARS} bars supported, got ${barCount} bars. Try Daily or Weekly timeframe instead.`);
    }

    // Performance warning for large datasets
    if (barCount > 2000) {
      console.warn('[ORD Indicators] WARNING: Large dataset (' + barCount + ' bars). Calculations may take 5-15 seconds...');
    }

    // Phase 1: Calculate histogram (green/red pressure per bar)
    const startHistogram = performance.now();
    this.histogram = this._calculateHistogram();
    console.log('[ORD Indicators] Histogram calculated in', (performance.now() - startHistogram).toFixed(0), 'ms');

    // Phase 2: Calculate cumulative pressure (lifetime tracking)
    const startCumulative = performance.now();
    this.cumulativePressure = this._calculateCumulativePressure();
    console.log('[ORD Indicators] Cumulative pressure calculated in', (performance.now() - startCumulative).toFixed(0), 'ms');

    // Phase 3: Detect patterns and signals
    const startPatterns = performance.now();
    this.indicators = {
      eightyPercentRule: this._detect80PercentRule(),
      divergences: this._detectDivergences(),
      zeroLineCrosses: this._detectZeroLineCrosses(),
      climaxSpikes: this._detectClimaxSpikes()
    };
    console.log('[ORD Indicators] Pattern detection completed in', (performance.now() - startPatterns).toFixed(0), 'ms');

    // Current state
    const currentBar = this.candles.length - 1;
    const currentState = {
      current_color: this._getCurrentColor(currentBar),
      cumulative_buying_pressure: this.cumulativePressure.buying[currentBar],
      cumulative_selling_pressure: this.cumulativePressure.selling[currentBar],
      net_pressure: this.cumulativePressure.net[currentBar],
      eighty_percent_rule_bull: this.indicators.eightyPercentRule.bullish,
      eighty_percent_rule_bear: this.indicators.eightyPercentRule.bearish,
      hidden_bull_div: this.indicators.divergences.hiddenBullish,
      hidden_bear_div: this.indicators.divergences.hiddenBearish,
      classic_bull_div: this.indicators.divergences.classicBullish,
      classic_bear_div: this.indicators.divergences.classicBearish,
      zero_line_cross_up_bars_ago: this.indicators.zeroLineCrosses.upBarsAgo,
      zero_line_cross_down_bars_ago: this.indicators.zeroLineCrosses.downBarsAgo,
      green_climax_spike: this.indicators.climaxSpikes.greenClimax,
      red_climax_spike: this.indicators.climaxSpikes.redClimax
    };

    console.log('[ORD Indicators] Calculation complete:', currentState);

    return {
      histogram: this.histogram,
      cumulativePressure: this.cumulativePressure,
      indicators: this.indicators,
      currentState: currentState
    };
  }

  /**
   * Calculate ORD Volume histogram using True Range weighting
   * THE SECRET SAUCE - NOT just close > open!
   * @private
   */
  _calculateHistogram() {
    const greenPressure = [];
    const redPressure = [];
    const netPressure = [];

    for (let i = 0; i < this.candles.length; i++) {
      const candle = this.candles[i];
      const { open, high, low, close, volume } = candle;

      // True Range weighting formula (Ord's secret)
      const range = high - low;

      if (range === 0) {
        // Avoid division by zero
        greenPressure.push(0);
        redPressure.push(0);
        netPressure.push(0);
        continue;
      }

      // Green pressure: buying power within the bar
      const green = volume * ((close - low) / range);

      // Red pressure: selling power within the bar
      const red = volume * ((high - close) / range);

      // Net pressure for this bar
      const net = green - red;

      greenPressure.push(green);
      redPressure.push(red);
      netPressure.push(net);
    }

    console.log('[ORD Histogram] Calculated', greenPressure.length, 'bars');
    console.log('[ORD Histogram] Last 3 bars net pressure:', netPressure.slice(-3));

    return {
      green: greenPressure,
      red: redPressure,
      net: netPressure
    };
  }

  /**
   * Calculate cumulative buying/selling pressure
   * NEVER resets - lifetime tracking from bar 0
   * @private
   */
  _calculateCumulativePressure() {
    const cumulativeBuying = [];
    const cumulativeSelling = [];
    const cumulativeNet = [];

    let buyingSum = 0;
    let sellingSum = 0;

    for (let i = 0; i < this.histogram.green.length; i++) {
      buyingSum += this.histogram.green[i];
      sellingSum += this.histogram.red[i]; // Keep as positive, will negate in net

      cumulativeBuying.push(buyingSum);
      cumulativeSelling.push(-sellingSum); // Stored as negative
      cumulativeNet.push(buyingSum - sellingSum);
    }

    const currentBar = this.candles.length - 1;
    console.log('[ORD Cumulative] Current net pressure:', cumulativeNet[currentBar].toFixed(0));

    return {
      buying: cumulativeBuying,
      selling: cumulativeSelling,
      net: cumulativeNet
    };
  }

  /**
   * Detect 80% Rule triggers (52-bar lookback)
   * Detects institutional distribution/accumulation
   * @private
   */
  _detect80PercentRule() {
    const lookback = 52;
    let bullish = false;
    let bearish = false;

    if (this.candles.length < lookback) {
      return { bullish, bearish, trigger: null };
    }

    const currentBar = this.candles.length - 1;
    const netPressure = this.cumulativePressure.net;

    // Find 52-bar high/low and corresponding pressure
    let maxPrice = -Infinity;
    let minPrice = Infinity;
    let maxPriceBar = -1;
    let minPriceBar = -1;

    for (let i = currentBar - lookback; i < currentBar; i++) {
      if (i < 0) continue;

      const high = this.candles[i].high;
      const low = this.candles[i].low;

      if (high > maxPrice) {
        maxPrice = high;
        maxPriceBar = i;
      }
      if (low < minPrice) {
        minPrice = low;
        minPriceBar = i;
      }
    }

    // Check current bar
    const currentHigh = this.candles[currentBar].high;
    const currentLow = this.candles[currentBar].low;
    const currentPressure = netPressure[currentBar];

    // Bearish 80% Rule: New high but pressure < 80% of prior high's pressure
    if (currentHigh > maxPrice && maxPriceBar >= 0) {
      const priorPressure = netPressure[maxPriceBar];
      if (currentPressure < priorPressure * 0.8) {
        bearish = true;
        console.log('[80% Rule] BEARISH trigger: New high but weak pressure');
      }
    }

    // Bullish 80% Rule: New low but pressure > 80% of prior low's pressure
    if (currentLow < minPrice && minPriceBar >= 0) {
      const priorPressure = netPressure[minPriceBar];
      if (currentPressure > priorPressure * 0.8) {
        bullish = true;
        console.log('[80% Rule] BULLISH trigger: New low but strong pressure');
      }
    }

    return { bullish, bearish, trigger: bullish || bearish };
  }

  /**
   * Detect hidden and classic divergences (34-bar lookback, 13-bar pivots)
   * Hidden divergences are Ord's #1 signal
   * @private
   */
  _detectDivergences() {
    const lookback = 34;
    const pivotStrength = 13;

    // Find swing highs/lows in price and net pressure
    const priceSwings = this._findSwings(this.candles.map(c => c.high), this.candles.map(c => c.low), pivotStrength, lookback);
    const pressureSwings = this._findSwings(
      this.cumulativePressure.net,
      this.cumulativePressure.net,
      pivotStrength,
      lookback
    );

    let hiddenBullish = false;
    let hiddenBearish = false;
    let classicBullish = false;
    let classicBearish = false;

    // Need at least 2 swings to compare
    if (priceSwings.lows.length >= 2 && pressureSwings.lows.length >= 2) {
      const priceLow1 = priceSwings.lows[priceSwings.lows.length - 2];
      const priceLow2 = priceSwings.lows[priceSwings.lows.length - 1];
      const pressureLow1 = pressureSwings.lows[pressureSwings.lows.length - 2];
      const pressureLow2 = pressureSwings.lows[pressureSwings.lows.length - 1];

      // Hidden bullish: price lower low, pressure higher low
      if (priceLow2.value < priceLow1.value && pressureLow2.value > pressureLow1.value) {
        hiddenBullish = true;
        // Classic bullish: same + pressure in negative territory
        if (pressureLow2.value < 0) {
          classicBullish = true;
        }
        console.log('[Divergence] HIDDEN BULLISH detected');
      }
    }

    if (priceSwings.highs.length >= 2 && pressureSwings.highs.length >= 2) {
      const priceHigh1 = priceSwings.highs[priceSwings.highs.length - 2];
      const priceHigh2 = priceSwings.highs[priceSwings.highs.length - 1];
      const pressureHigh1 = pressureSwings.highs[pressureSwings.highs.length - 2];
      const pressureHigh2 = pressureSwings.highs[pressureSwings.highs.length - 1];

      // Hidden bearish: price higher high, pressure lower high
      if (priceHigh2.value > priceHigh1.value && pressureHigh2.value < pressureHigh1.value) {
        hiddenBearish = true;
        // Classic bearish: same + pressure in positive territory
        if (pressureHigh2.value > 0) {
          classicBearish = true;
        }
        console.log('[Divergence] HIDDEN BEARISH detected');
      }
    }

    return { hiddenBullish, hiddenBearish, classicBullish, classicBearish };
  }

  /**
   * Detect zero-line crosses and track bars ago
   * @private
   */
  _detectZeroLineCrosses() {
    const netPressure = this.cumulativePressure.net;
    let upBarsAgo = 999;
    let downBarsAgo = 999;

    const currentBar = this.candles.length - 1;

    // Search backwards for most recent crosses
    for (let i = currentBar; i > 0; i--) {
      const current = netPressure[i];
      const previous = netPressure[i - 1];

      // Cross up (from negative to positive)
      if (current > 0 && previous <= 0 && upBarsAgo === 999) {
        upBarsAgo = currentBar - i;
        console.log('[Zero-Line] Cross UP detected', upBarsAgo, 'bars ago');
      }

      // Cross down (from positive to negative)
      if (current < 0 && previous >= 0 && downBarsAgo === 999) {
        downBarsAgo = currentBar - i;
        console.log('[Zero-Line] Cross DOWN detected', downBarsAgo, 'bars ago');
      }

      // Stop if both found
      if (upBarsAgo !== 999 && downBarsAgo !== 999) break;
    }

    return { upBarsAgo, downBarsAgo };
  }

  /**
   * Detect climax spikes (2.5x 21-bar SMA)
   * Must occur at support (green) or resistance (red)
   * @private
   */
  _detectClimaxSpikes() {
    const smaPeriod = 21;
    const spikeThreshold = 2.5;
    const volumeThreshold = 300000; // Adjust per instrument

    const currentBar = this.candles.length - 1;

    if (currentBar < smaPeriod) {
      return { greenClimax: false, redClimax: false };
    }

    // Calculate 21-bar SMA of green and red pressure
    const greenSMA = this._sma(this.histogram.green, smaPeriod);
    const redSMA = this._sma(this.histogram.red, smaPeriod);

    const currentGreen = this.histogram.green[currentBar];
    const currentRed = this.histogram.red[currentBar];

    let greenClimax = false;
    let redClimax = false;

    // Green climax: current green > 2.5x SMA AND > threshold
    if (currentGreen > greenSMA[currentBar] * spikeThreshold && currentGreen > volumeThreshold) {
      // Check if at support (price near recent low)
      if (this._isNearSupport(currentBar)) {
        greenClimax = true;
        console.log('[Climax] GREEN spike detected:', currentGreen.toFixed(0));
      }
    }

    // Red climax: current red > 2.5x SMA AND > threshold
    if (currentRed > redSMA[currentBar] * spikeThreshold && currentRed > volumeThreshold) {
      // Check if at resistance (price near recent high)
      if (this._isNearResistance(currentBar)) {
        redClimax = true;
        console.log('[Climax] RED spike detected:', currentRed.toFixed(0));
      }
    }

    return { greenClimax, redClimax };
  }

  /**
   * Determine current ORD Volume color
   * @private
   */
  _getCurrentColor(bar) {
    const net = this.histogram.net[bar];
    if (net > 0) return 'GREEN';
    if (net < 0) return 'RED';
    return 'NEUTRAL';
  }

  /**
   * Find swing highs and lows using pivot strength
   * @private
   */
  _findSwings(highs, lows, pivotStrength, lookback) {
    const swingHighs = [];
    const swingLows = [];

    const startBar = Math.max(0, highs.length - lookback);

    for (let i = startBar + pivotStrength; i < highs.length - pivotStrength; i++) {
      // Check for swing high
      let isSwingHigh = true;
      for (let j = i - pivotStrength; j <= i + pivotStrength; j++) {
        if (j === i) continue;
        if (highs[j] >= highs[i]) {
          isSwingHigh = false;
          break;
        }
      }
      if (isSwingHigh) {
        swingHighs.push({ bar: i, value: highs[i] });
      }

      // Check for swing low
      let isSwingLow = true;
      for (let j = i - pivotStrength; j <= i + pivotStrength; j++) {
        if (j === i) continue;
        if (lows[j] <= lows[i]) {
          isSwingLow = false;
          break;
        }
      }
      if (isSwingLow) {
        swingLows.push({ bar: i, value: lows[i] });
      }
    }

    return { highs: swingHighs, lows: swingLows };
  }

  /**
   * Calculate Simple Moving Average
   * @private
   */
  _sma(data, period) {
    const sma = [];

    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        sma.push(0);
        continue;
      }

      let sum = 0;
      for (let j = i - period + 1; j <= i; j++) {
        sum += data[j];
      }
      sma.push(sum / period);
    }

    return sma;
  }

  /**
   * Check if current bar is near support (recent low)
   * @private
   */
  _isNearSupport(bar) {
    const lookback = 21;
    let minLow = Infinity;

    for (let i = Math.max(0, bar - lookback); i < bar; i++) {
      if (this.candles[i].low < minLow) {
        minLow = this.candles[i].low;
      }
    }

    const currentLow = this.candles[bar].low;
    const tolerance = (this.candles[bar].high - this.candles[bar].low) * 2; // 2 ATRs

    return currentLow <= minLow + tolerance;
  }

  /**
   * Check if current bar is near resistance (recent high)
   * @private
   */
  _isNearResistance(bar) {
    const lookback = 21;
    let maxHigh = -Infinity;

    for (let i = Math.max(0, bar - lookback); i < bar; i++) {
      if (this.candles[i].high > maxHigh) {
        maxHigh = this.candles[i].high;
      }
    }

    const currentHigh = this.candles[bar].high;
    const tolerance = (this.candles[bar].high - this.candles[bar].low) * 2; // 2 ATRs

    return currentHigh >= maxHigh - tolerance;
  }

  /**
   * Calculate Average True Range
   * @param {Number} period - ATR period
   * @returns {Array} ATR values
   */
  calculateATR(period = 14) {
    const tr = [];
    const atr = [];

    for (let i = 0; i < this.candles.length; i++) {
      const candle = this.candles[i];

      if (i === 0) {
        tr.push(candle.high - candle.low);
        atr.push(candle.high - candle.low);
        continue;
      }

      const prevClose = this.candles[i - 1].close;
      const trueRange = Math.max(
        candle.high - candle.low,
        Math.abs(candle.high - prevClose),
        Math.abs(candle.low - prevClose)
      );

      tr.push(trueRange);

      // Calculate ATR as SMA for first period bars
      if (i < period) {
        const sum = tr.slice(0, i + 1).reduce((a, b) => a + b, 0);
        atr.push(sum / (i + 1));
      } else {
        // Smoothed ATR thereafter
        const prevATR = atr[i - 1];
        const smoothedATR = (prevATR * (period - 1) + trueRange) / period;
        atr.push(smoothedATR);
      }
    }

    return atr;
  }
}

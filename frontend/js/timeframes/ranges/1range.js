/**
 * 1 Range Timeframe
 * Creates a new candle when price moves $1 from opening price
 * Range bars show pure price movement, independent of time or volume
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe1range extends BaseTimeframe {
  constructor() {
    super({
      id: '1range',
      name: '1 range',
      interval: 'range',
      period: 'live',
      category: 'ranges',
      isCustom: true
    });

    // Range aggregation state
    this.currentCandle = null;
    this.rangeSize = 1; // $1 price movement triggers new candle
    this.maxCandles = 500;
  }

  /**
   * Load custom data for range chart
   * Start with empty dataset - builds from live trades
   */
  async loadCustomData() {
    console.log(`📊 [${this.id}] Starting range aggregation ($${this.rangeSize} per candle)`);
    this.data = [];
    return this.data;
  }

  /**
   * Handle trade update - aggregate into range candles
   */
  handleTradeUpdate(trade) {
    if (!this.isActive) return;

    const price = parseFloat(trade.price);
    const size = parseFloat(trade.size);
    const timestamp = new Date(trade.time);

    // Start new candle if needed
    if (!this.currentCandle) {
      this.startNewCandle(price, timestamp);
      return;
    }

    // Update current candle
    this.currentCandle.Close = price;
    this.currentCandle.High = Math.max(this.currentCandle.High, price);
    this.currentCandle.Low = Math.min(this.currentCandle.Low, price);
    this.currentCandle.Volume += size;

    // Check if range is complete (price moved rangeSize from open)
    const priceMovement = Math.abs(price - this.currentCandle.Open);
    if (priceMovement >= this.rangeSize) {
      this.completeCandle();
      this.startNewCandle(price, timestamp);
    }

    // Update renderer
    this.renderer.updateLivePrice(price);
  }

  /**
   * Start a new candle
   */
  startNewCandle(price, timestamp) {
    this.currentCandle = {
      Date: timestamp.toISOString(),
      Open: price,
      High: price,
      Low: price,
      Close: price,
      Volume: 0
    };
  }

  /**
   * Complete current candle and add to dataset
   */
  completeCandle() {
    if (this.currentCandle) {
      this.data.push(this.currentCandle);

      // Limit dataset size
      if (this.data.length > this.maxCandles) {
        this.data.shift();
      }

      // Re-render with new candle
      this.renderer.render(this.data, this.symbol);
    }
  }
}

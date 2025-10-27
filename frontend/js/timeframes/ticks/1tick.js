/**
 * 1 Tick Timeframe
 * Creates a new candle after every 1 trade
 * Requires custom aggregation from WebSocket trade stream
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe1tick extends BaseTimeframe {
  constructor() {
    super({
      id: '1tick',
      name: '1 tick',
      interval: 'tick',
      period: 'live',
      category: 'ticks',
      isCustom: true
    });

    // Tick aggregation state
    this.currentCandle = null;
    this.ticksPerCandle = 1;
    this.tickCount = 0;
    this.maxCandles = 500; // Keep last 500 candles in memory
  }

  /**
   * Load custom data for tick chart
   * Start with empty dataset - builds from live trades
   */
  async loadCustomData() {
    console.log(`📊 [${this.id}] Starting tick aggregation (${this.ticksPerCandle} tick per candle)`);
    console.log(`📊 [${this.id}] Chart will populate from live trade stream`);
    this.data = [];

    // Create a placeholder canvas element to show "Waiting for trades..."
    const canvas = this.renderer.canvas;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#e0e0e0';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for live trades...', canvas.width / 2, canvas.height / 2);
    }

    return this.data;
  }

  /**
   * Handle trade update - aggregate into tick candles
   */
  handleTradeUpdate(trade) {
    if (!this.isActive) return;

    console.log(`🔵 [${this.id}] Trade received:`, trade.price, 'x', trade.size);

    const price = parseFloat(trade.price);
    const size = parseFloat(trade.size);
    const timestamp = new Date(trade.time);

    // Start new candle if needed
    if (!this.currentCandle || this.tickCount >= this.ticksPerCandle) {
      console.log(`🟢 [${this.id}] Starting new candle at price ${price}`);
      this.startNewCandle(price, timestamp);
    }

    // Update current candle
    this.currentCandle.Close = price;
    this.currentCandle.High = Math.max(this.currentCandle.High, price);
    this.currentCandle.Low = Math.min(this.currentCandle.Low, price);
    this.currentCandle.Volume += size;
    this.tickCount++;

    console.log(`📈 [${this.id}] Tick ${this.tickCount}/${this.ticksPerCandle}: O:${this.currentCandle.Open.toFixed(2)} H:${this.currentCandle.High.toFixed(2)} L:${this.currentCandle.Low.toFixed(2)} C:${this.currentCandle.Close.toFixed(2)} V:${this.currentCandle.Volume.toFixed(6)}`);

    // Complete candle if we've reached the tick count
    if (this.tickCount >= this.ticksPerCandle) {
      console.log(`✅ [${this.id}] Candle complete! Total candles: ${this.data.length + 1}`);
      this.completeCandle();
    }

    // Update renderer with live price
    this.renderer.updateLivePrice(price, size);
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
    this.tickCount = 0;
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

      this.currentCandle = null;
      this.tickCount = 0;
    }
  }
}

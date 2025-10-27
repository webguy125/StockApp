/**
 * 100 Tick Timeframe
 * Creates a new candle after every 100 trades
 */

import { Timeframe1tick } from './1tick.js';

export class Timeframe100ticks extends Timeframe1tick {
  constructor() {
    super();
    this.id = '100ticks';
    this.name = '100 ticks';
    this.ticksPerCandle = 100;
  }
}

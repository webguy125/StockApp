/**
 * 1000 Tick Timeframe
 * Creates a new candle after every 1000 trades
 */

import { Timeframe1tick } from './1tick.js';

export class Timeframe1000ticks extends Timeframe1tick {
  constructor() {
    super();
    this.id = '1000ticks';
    this.name = '1000 ticks';
    this.ticksPerCandle = 1000;
  }
}

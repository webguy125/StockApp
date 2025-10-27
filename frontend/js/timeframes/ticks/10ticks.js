/**
 * 10 Tick Timeframe
 * Creates a new candle after every 10 trades
 */

import { Timeframe1tick } from './1tick.js';

export class Timeframe10ticks extends Timeframe1tick {
  constructor() {
    super();
    this.id = '10ticks';
    this.name = '10 ticks';
    this.ticksPerCandle = 10;
  }
}

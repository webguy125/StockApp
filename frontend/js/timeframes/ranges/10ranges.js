/**
 * 10 Range Timeframe
 * Creates a new candle when price moves $10 from opening price
 */

import { Timeframe1range } from './1range.js';

export class Timeframe10ranges extends Timeframe1range {
  constructor() {
    super();
    this.id = '10ranges';
    this.name = '10 ranges';
    this.rangeSize = 10;
  }
}

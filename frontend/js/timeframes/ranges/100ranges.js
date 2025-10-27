/**
 * 100 Range Timeframe
 * Creates a new candle when price moves $100 from opening price
 */

import { Timeframe1range } from './1range.js';

export class Timeframe100ranges extends Timeframe1range {
  constructor() {
    super();
    this.id = '100ranges';
    this.name = '100 ranges';
    this.rangeSize = 100;
  }
}

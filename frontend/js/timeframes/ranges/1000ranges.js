/**
 * 1000 Range Timeframe
 * Creates a new candle when price moves $1000 from opening price
 */

import { Timeframe1range } from './1range.js';

export class Timeframe1000ranges extends Timeframe1range {
  constructor() {
    super();
    this.id = '1000ranges';
    this.name = '1000 ranges';
    this.rangeSize = 1000;
  }
}

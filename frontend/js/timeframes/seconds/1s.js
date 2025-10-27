/**
 * 1 Second Timeframe
 * Displays 1-second candlestick chart with live updates
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe1s extends BaseTimeframe {
  constructor() {
    super({
      id: '1s',
      name: '1 second',
      interval: '1s',
      period: '1h',        // Show 1 hour of 1s candles
      category: 'seconds',
      isCustom: false
    });
  }
}

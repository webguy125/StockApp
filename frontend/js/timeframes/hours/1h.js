/**
 * 1 Hour Timeframe
 * Displays 1-hour candlestick chart with live updates
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe1h extends BaseTimeframe {
  constructor() {
    super({
      id: '1h',
      name: '1 hour',
      interval: '1h',
      period: '1mo',       // Show 1 month of 1h candles
      category: 'hours',
      isCustom: false
    });
  }
}

/**
 * 2 Minute Timeframe
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe2m extends BaseTimeframe {
  constructor() {
    super({
      id: '2m',
      name: '2 minutes',
      interval: '2m',
      period: '2d',
      category: 'minutes',
      isCustom: false
    });
  }
}

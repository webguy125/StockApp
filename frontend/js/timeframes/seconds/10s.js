/**
 * 10 Second Timeframe
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe10s extends BaseTimeframe {
  constructor() {
    super({
      id: '10s',
      name: '10 seconds',
      interval: '10s',
      period: '1h',
      category: 'seconds',
      isCustom: false
    });
  }
}

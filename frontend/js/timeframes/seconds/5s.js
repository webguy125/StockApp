/**
 * 5 Second Timeframe
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe5s extends BaseTimeframe {
  constructor() {
    super({
      id: '5s',
      name: '5 seconds',
      interval: '5s',
      period: '1h',
      category: 'seconds',
      isCustom: false
    });
  }
}

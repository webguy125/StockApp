/**
 * 15 Second Timeframe
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe15s extends BaseTimeframe {
  constructor() {
    super({
      id: '15s',
      name: '15 seconds',
      interval: '15s',
      period: '1h',
      category: 'seconds',
      isCustom: false
    });
  }
}

/**
 * 30 Second Timeframe
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe30s extends BaseTimeframe {
  constructor() {
    super({
      id: '30s',
      name: '30 seconds',
      interval: '30s',
      period: '2h',
      category: 'seconds',
      isCustom: false
    });
  }
}

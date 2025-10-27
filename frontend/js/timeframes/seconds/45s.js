/**
 * 45 Second Timeframe
 */

import { BaseTimeframe } from '../BaseTimeframe.js';

export class Timeframe45s extends BaseTimeframe {
  constructor() {
    super({
      id: '45s',
      name: '45 seconds',
      interval: '45s',
      period: '2h',
      category: 'seconds',
      isCustom: false
    });
  }
}

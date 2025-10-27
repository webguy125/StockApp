/** 45 Minute Timeframe */
import { BaseTimeframe } from '../BaseTimeframe.js';
export class Timeframe45m extends BaseTimeframe {
  constructor() {
    super({ id: '45m', name: '45 minutes', interval: '45m', period: '21d', category: 'minutes', isCustom: false });
  }
}

/** 3 Minute Timeframe */
import { BaseTimeframe } from '../BaseTimeframe.js';
export class Timeframe3m extends BaseTimeframe {
  constructor() {
    super({ id: '3m', name: '3 minutes', interval: '3m', period: '3d', category: 'minutes', isCustom: false });
  }
}

/** 10 Minute Timeframe */
import { BaseTimeframe } from '../BaseTimeframe.js';
export class Timeframe10m extends BaseTimeframe {
  constructor() {
    super({ id: '10m', name: '10 minutes', interval: '10m', period: '5d', category: 'minutes', isCustom: false });
  }
}

/** 3 Hour Timeframe */
import { BaseTimeframe } from '../BaseTimeframe.js';
export class Timeframe3h extends BaseTimeframe {
  constructor() {
    super({ id: '3h', name: '3 hours', interval: '3h', period: '3mo', category: 'hours', isCustom: false });
  }
}

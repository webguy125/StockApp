/** 2 Hour Timeframe */
import { BaseTimeframe } from '../BaseTimeframe.js';
export class Timeframe2h extends BaseTimeframe {
  constructor() {
    super({ id: '2h', name: '2 hours', interval: '2h', period: '2mo', category: 'hours', isCustom: false });
  }
}

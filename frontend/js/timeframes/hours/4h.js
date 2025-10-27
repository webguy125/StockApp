/** 4 Hour Timeframe */
import { BaseTimeframe } from '../BaseTimeframe.js';
export class Timeframe4h extends BaseTimeframe {
  constructor() {
    super({ id: '4h', name: '4 hours', interval: '4h', period: '6mo', category: 'hours', isCustom: false });
  }
}

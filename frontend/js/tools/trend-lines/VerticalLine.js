/**
 * Vertical Line Tool
 * Draw a vertical line at a specific time/date
 */

export class VerticalLine {
  constructor() {
    this.id = 'vertical-line';
    this.name = 'Vertical Line';
    this.category = 'trendLines';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Line properties
    this.lineColor = '#9c27b0';
    this.lineWidth = 2;
    this.lineStyle = 'solid';
    this.extend = 'both'; // 'up', 'down', 'both'
  }

  /**
   * Activate this tool
   */
  activate(canvas) {
    this.isActive = true;
    canvas.style.cursor = this.cursorStyle;
  }

  /**
   * Deactivate this tool
   */
  deactivate(canvas) {
    this.isActive = false;
    canvas.style.cursor = 'default';
  }

  /**
   * Handle click - place vertical line
   */
  onClick(event, chartState) {
    return {
      action: 'place-vertical-line',
      x: event.clientX,
      color: this.lineColor,
      width: this.lineWidth,
      style: this.lineStyle,
      extend: this.extend,
      id: crypto.randomUUID()
    };
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    return {
      action: 'preview-vertical-line',
      x: event.clientX,
      color: this.lineColor,
      width: this.lineWidth,
      style: this.lineStyle,
      extend: this.extend
    };
  }

  /**
   * Set line color
   */
  setColor(color) {
    this.lineColor = color;
  }

  /**
   * Set line width
   */
  setWidth(width) {
    this.lineWidth = Math.max(1, Math.min(10, width));
  }

  /**
   * Set line style
   */
  setStyle(style) {
    const validStyles = ['solid', 'dashed', 'dotted'];
    if (validStyles.includes(style)) {
      this.lineStyle = style;
    }
  }

  /**
   * Set extension direction
   */
  setExtend(direction) {
    const validDirections = ['up', 'down', 'both'];
    if (validDirections.includes(direction)) {
      this.extend = direction;
    }
  }
}

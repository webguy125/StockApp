/**
 * Horizontal Line Tool
 * Draw a horizontal line at a specific price level
 */

export class HorizontalLine {
  constructor() {
    this.id = 'horizontal-line';
    this.name = 'Horizontal Line';
    this.category = 'trendLines';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Line properties
    this.lineColor = '#ff9800';
    this.lineWidth = 2;
    this.lineStyle = 'solid';
    this.extend = 'both'; // 'left', 'right', 'both'
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
   * Handle click - place horizontal line
   */
  onClick(event, chartState) {
    return {
      action: 'place-horizontal-line',
      y: event.clientY,
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
      action: 'preview-horizontal-line',
      y: event.clientY,
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
    const validDirections = ['left', 'right', 'both'];
    if (validDirections.includes(direction)) {
      this.extend = direction;
    }
  }
}

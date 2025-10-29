/**
 * Extended Line Tool
 * Draw a line that extends infinitely in both directions
 */

export class ExtendedLine {
  constructor() {
    this.id = 'extended-line';
    this.name = 'Extended Line';
    this.category = 'trendLines';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#f44336';
    this.lineWidth = 2;
    this.lineStyle = 'dashed';
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
    this.isDrawing = false;
    this.startPoint = null;
  }

  /**
   * Handle mouse down - start or finish drawing
   */
  onMouseDown(event, chartState) {
    if (!this.isDrawing) {
      // First click - start the line
      this.isDrawing = true;
      this.startPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-extended-line',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - finish the line
      this.isDrawing = false;

      const line = {
        action: 'finish-extended-line',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        color: this.lineColor,
        width: this.lineWidth,
        style: this.lineStyle,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return line;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      return {
        action: 'preview-extended-line',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        color: this.lineColor,
        width: this.lineWidth,
        style: this.lineStyle
      };
    }
    return null;
  }

  /**
   * Handle escape key - cancel drawing
   */
  onKeyDown(event, chartState) {
    if (event.key === 'Escape' && this.isDrawing) {
      this.isDrawing = false;
      this.startPoint = null;
      return {
        action: 'cancel-drawing'
      };
    }
    return null;
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
}

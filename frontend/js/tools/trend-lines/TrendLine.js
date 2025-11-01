/**
 * Trend Line Tool
 * Draw a line between two points on the chart
 */

export class TrendLine {
  constructor() {
    this.id = 'trend-line';
    this.name = 'Trend Line';
    this.category = 'trendLines';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;
    this.endPoint = null;

    // Line properties
    this.lineColor = '#2196f3';
    this.lineWidth = 2;
    this.lineStyle = 'solid'; // 'solid', 'dashed', 'dotted'
  }

  /**
   * Activate this tool
   */
  activate(canvas) {
    this.isActive = true;
    this.isDrawing = false;  // Reset drawing state
    this.startPoint = null;  // Clear any previous points
    this.endPoint = null;
    canvas.style.cursor = this.cursorStyle;
    console.log('✅ Trend line tool activated');
  }

  /**
   * Deactivate this tool
   */
  deactivate(canvas) {
    console.log('🛑 Trend line tool deactivated');
    this.isActive = false;
    canvas.style.cursor = 'default';
    this.isDrawing = false;
    this.startPoint = null;
    this.endPoint = null;
  }

  /**
   * Handle mouse down - start drawing
   */
  onMouseDown(event, chartState) {
    if (!this.isActive) {
      console.warn('⚠️ Trend line tool not active!');
      return null;
    }

    // Use canvas-relative coordinates
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    console.log(`🖱️ Mouse down at (${x}, ${y}), isDrawing: ${this.isDrawing}`);

    if (!this.isDrawing) {
      // First click - start the line
      this.isDrawing = true;
      this.startPoint = { x, y };
      console.log('✏️ Started drawing trend line');
      return {
        action: 'start-trend-line',
        x,
        y
      };
    } else {
      // Second click - finish the line
      console.log('✅ Finishing trend line drawing');
      this.isDrawing = false;
      this.endPoint = { x, y };

      const line = {
        action: 'finish-trend-line',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: this.endPoint.x,
        endY: this.endPoint.y,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        style: this.lineStyle,
        id: crypto.randomUUID()
      };

      // Reset for next line
      this.startPoint = null;
      this.endPoint = null;

      return line;
    }
  }

  /**
   * Handle mouse move - show preview of line
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      // Use canvas-relative coordinates
      const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
      const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

      return {
        action: 'preview-trend-line',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
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
      this.endPoint = null;
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

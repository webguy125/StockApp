/**
 * Rectangle Tool
 * Draw rectangles on the chart
 */

export class Rectangle {
  constructor() {
    this.id = 'rectangle';
    this.name = 'Rectangle';
    this.category = 'shapes';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Shape properties
    this.lineColor = '#2196f3';
    this.lineWidth = 2;
    this.fillColor = 'rgba(33, 150, 243, 0.1)';
    this.filled = true;
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
    // Fix cursor offset - use canvas coordinates
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    if (!this.isDrawing) {
      // First click - start the rectangle
      this.isDrawing = true;
      this.startPoint = { x, y };
      return {
        action: 'start-rectangle',
        x,
        y
      };
    } else {
      // Second click - finish the rectangle
      this.isDrawing = false;

      const rectangle = {
        action: 'finish-rectangle',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        fillColor: this.fillColor,
        filled: this.filled,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return rectangle;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      // Fix cursor offset - use canvas coordinates
      const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
      const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

      return {
        action: 'preview-rectangle',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        fillColor: this.fillColor,
        filled: this.filled
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
   * Set fill color
   */
  setFillColor(color) {
    this.fillColor = color;
  }

  /**
   * Toggle filled
   */
  toggleFilled() {
    this.filled = !this.filled;
  }
}

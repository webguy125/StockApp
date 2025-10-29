/**
 * Ellipse Tool
 * Draw ellipses on the chart
 */

export class Ellipse {
  constructor() {
    this.id = 'ellipse';
    this.name = 'Ellipse';
    this.category = 'shapes';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Shape properties
    this.lineColor = '#ff9800';
    this.lineWidth = 2;
    this.fillColor = 'rgba(255, 152, 0, 0.1)';
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
    if (!this.isDrawing) {
      // First click - start the ellipse
      this.isDrawing = true;
      this.startPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-ellipse',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - finish the ellipse
      this.isDrawing = false;

      const ellipse = {
        action: 'finish-ellipse',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        fillColor: this.fillColor,
        filled: this.filled,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return ellipse;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      return {
        action: 'preview-ellipse',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
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

/**
 * Circle Tool
 * Draw circles on the chart
 */

export class Circle {
  constructor() {
    this.id = 'circle';
    this.name = 'Circle';
    this.category = 'shapes';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.centerPoint = null;

    // Shape properties
    this.lineColor = '#00c853';
    this.lineWidth = 2;
    this.fillColor = 'rgba(0, 200, 83, 0.1)';
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
    this.centerPoint = null;
  }

  /**
   * Handle mouse down - start or finish drawing
   */
  onMouseDown(event, chartState) {
    if (!this.isDrawing) {
      // First click - set center
      this.isDrawing = true;
      this.centerPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-circle',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - set radius and finish
      this.isDrawing = false;

      const dx = event.clientX - this.centerPoint.x;
      const dy = event.clientY - this.centerPoint.y;
      const radius = Math.sqrt(dx * dx + dy * dy);

      const circle = {
        action: 'finish-circle',
        centerX: this.centerPoint.x,
        centerY: this.centerPoint.y,
        radius: radius,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        fillColor: this.fillColor,
        filled: this.filled,
        id: crypto.randomUUID()
      };

      this.centerPoint = null;
      return circle;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.centerPoint) {
      const dx = event.clientX - this.centerPoint.x;
      const dy = event.clientY - this.centerPoint.y;
      const radius = Math.sqrt(dx * dx + dy * dy);

      return {
        action: 'preview-circle',
        centerX: this.centerPoint.x,
        centerY: this.centerPoint.y,
        radius: radius,
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
      this.centerPoint = null;
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

/**
 * Fibonacci Spiral Tool
 * Draw logarithmic spiral based on Fibonacci golden ratio
 */

export class FibonacciSpiral {
  constructor() {
    this.id = 'fibonacci-spiral';
    this.name = 'Fibonacci Spiral';
    this.category = 'fibonacci';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#e91e63';
    this.lineWidth = 2;
    this.showSquares = true; // Show Fibonacci squares
    this.clockwise = true;
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
      // First click - start the spiral
      this.isDrawing = true;
      this.startPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-fibonacci-spiral',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - finish the spiral
      this.isDrawing = false;

      const spiral = {
        action: 'finish-fibonacci-spiral',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showSquares: this.showSquares,
        clockwise: this.clockwise,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return spiral;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      return {
        action: 'preview-fibonacci-spiral',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showSquares: this.showSquares,
        clockwise: this.clockwise
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
   * Toggle squares display
   */
  toggleSquares() {
    this.showSquares = !this.showSquares;
  }

  /**
   * Toggle spiral direction
   */
  toggleDirection() {
    this.clockwise = !this.clockwise;
  }
}

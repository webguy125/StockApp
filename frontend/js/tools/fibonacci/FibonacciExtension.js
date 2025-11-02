/**
 * Fibonacci Extension Tool
 * Draw Fibonacci extension levels for target projections
 * Levels: 0%, 61.8%, 100%, 161.8%, 261.8%, 423.6%
 */

export class FibonacciExtension {
  constructor() {
    this.id = 'fibonacci-extension';
    this.name = 'Fibonacci Extension';
    this.category = 'fibonacci';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state (requires 3 points)
    this.drawingStep = 0; // 0: not started, 1: point 1, 2: point 2, 3: point 3
    this.point1 = null;
    this.point2 = null;

    // Line properties
    this.lineColor = '#9c27b0';
    this.lineWidth = 1;
    this.showLabels = true;

    // Extension levels
    this.levels = [0, 0.618, 1.0, 1.618, 2.618, 4.236];
    this.levelColors = {
      0: '#808080',
      0.618: '#2196f3',
      1.0: '#00c853',
      1.618: '#ff9800',
      2.618: '#f44336',
      4.236: '#9c27b0'
    };
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
    this.reset();
  }

  /**
   * Reset drawing state
   */
  reset() {
    this.drawingStep = 0;
    this.point1 = null;
    this.point2 = null;
  }

  /**
   * Handle mouse down - progress through 3-point drawing
   */
  onMouseDown(event, chartState) {
    // Use canvas-relative coordinates (provided by tool panel)
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    if (this.drawingStep === 0) {
      // First click - set point 1
      this.drawingStep = 1;
      this.point1 = { x, y };
      return {
        action: 'start-fibonacci-extension-p1',
        x,
        y
      };
    } else if (this.drawingStep === 1) {
      // Second click - set point 2
      this.drawingStep = 2;
      this.point2 = { x, y };
      return {
        action: 'start-fibonacci-extension-p2',
        x1: this.point1.x,
        y1: this.point1.y,
        x2: x,
        y2: y
      };
    } else if (this.drawingStep === 2) {
      // Third click - set point 3 and finish
      this.drawingStep = 0;

      const extension = {
        action: 'finish-fibonacci-extension',
        point1: this.point1,
        point2: this.point2,
        point3: { x, y },
        levels: this.levels,
        levelColors: this.levelColors,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        id: crypto.randomUUID()
      };

      this.reset();
      return extension;
    }
    return null;
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    // Use canvas-relative coordinates (provided by tool panel)
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    if (this.drawingStep === 1 && this.point1) {
      return {
        action: 'preview-fibonacci-extension-p1',
        x1: this.point1.x,
        y1: this.point1.y,
        x2: x,
        y2: y
      };
    } else if (this.drawingStep === 2 && this.point2) {
      return {
        action: 'preview-fibonacci-extension',
        point1: this.point1,
        point2: this.point2,
        point3: { x, y },
        levels: this.levels,
        levelColors: this.levelColors,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels
      };
    }
    return null;
  }

  /**
   * Handle escape key - cancel drawing
   */
  onKeyDown(event, chartState) {
    if (event.key === 'Escape' && this.drawingStep > 0) {
      this.reset();
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
   * Toggle labels
   */
  toggleLabels() {
    this.showLabels = !this.showLabels;
  }
}

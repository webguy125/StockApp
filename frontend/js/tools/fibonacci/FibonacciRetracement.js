/**
 * Fibonacci Retracement Tool
 * Draw Fibonacci retracement levels between two price points
 * Levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
 */

export class FibonacciRetracement {
  constructor() {
    this.id = 'fibonacci-retracement';
    this.name = 'Fibonacci Retracement';
    this.category = 'fibonacci';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#2196f3';
    this.lineWidth = 1;
    this.showLabels = true;
    this.showLevels = true;

    // Fibonacci levels (percentages)
    this.levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
    this.levelColors = {
      0: '#808080',
      0.236: '#00c853',
      0.382: '#2196f3',
      0.5: '#ff9800',
      0.618: '#f44336',
      0.786: '#9c27b0',
      1.0: '#808080'
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
    this.isDrawing = false;
    this.startPoint = null;
  }

  /**
   * Handle mouse down - start or finish drawing
   */
  onMouseDown(event, chartState) {
    // Use canvas-relative coordinates (provided by tool panel)
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    if (!this.isDrawing) {
      // First click - start the retracement
      this.isDrawing = true;
      this.startPoint = { x, y };
      return {
        action: 'start-fibonacci-retracement',
        x,
        y
      };
    } else {
      // Second click - finish the retracement
      this.isDrawing = false;

      const retracement = {
        action: 'finish-fibonacci-retracement',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        levels: this.levels,
        levelColors: this.levelColors,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        showLevels: this.showLevels,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return retracement;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      // Use canvas-relative coordinates (provided by tool panel)
      const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
      const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

      return {
        action: 'preview-fibonacci-retracement',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        levels: this.levels,
        levelColors: this.levelColors,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        showLevels: this.showLevels
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
   * Toggle labels
   */
  toggleLabels() {
    this.showLabels = !this.showLabels;
  }

  /**
   * Toggle level lines
   */
  toggleLevels() {
    this.showLevels = !this.showLevels;
  }
}

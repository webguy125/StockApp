/**
 * Fibonacci Arcs Tool
 * Draw curved arcs at Fibonacci levels
 * Levels: 38.2%, 50%, 61.8%
 */

export class FibonacciArcs {
  constructor() {
    this.id = 'fibonacci-arcs';
    this.name = 'Fibonacci Arcs';
    this.category = 'fibonacci';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#4caf50';
    this.lineWidth = 1;
    this.showLabels = true;

    // Arc levels
    this.levels = [0.382, 0.5, 0.618];
    this.levelColors = {
      0.382: '#2196f3',
      0.5: '#ff9800',
      0.618: '#f44336'
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
    if (!this.isDrawing) {
      // First click - start the arcs
      this.isDrawing = true;
      this.startPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-fibonacci-arcs',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - finish the arcs
      this.isDrawing = false;

      const arcs = {
        action: 'finish-fibonacci-arcs',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        levels: this.levels,
        levelColors: this.levelColors,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return arcs;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      return {
        action: 'preview-fibonacci-arcs',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
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
}

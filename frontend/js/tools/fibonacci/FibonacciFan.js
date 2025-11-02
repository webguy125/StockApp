/**
 * Fibonacci Fan Tool
 * Draw diagonal trend lines at Fibonacci angles
 * Levels: 38.2%, 50%, 61.8%
 */

export class FibonacciFan {
  constructor() {
    this.id = 'fibonacci-fan';
    this.name = 'Fibonacci Fan';
    this.category = 'fibonacci';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#00bcd4';
    this.lineWidth = 1;
    this.showLabels = true;

    // Fan levels
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
    // Use canvas-relative coordinates (provided by tool panel)
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    if (!this.isDrawing) {
      // First click - start the fan
      this.isDrawing = true;
      this.startPoint = { x, y };
      return {
        action: 'start-fibonacci-fan',
        x,
        y
      };
    } else {
      // Second click - finish the fan
      this.isDrawing = false;

      const fan = {
        action: 'finish-fibonacci-fan',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        levels: this.levels,
        levelColors: this.levelColors,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return fan;
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
        action: 'preview-fibonacci-fan',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
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

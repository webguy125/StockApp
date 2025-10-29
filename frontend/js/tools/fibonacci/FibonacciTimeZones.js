/**
 * Fibonacci Time Zones Tool
 * Draw vertical lines at Fibonacci time intervals
 * Sequence: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144
 */

export class FibonacciTimeZones {
  constructor() {
    this.id = 'fibonacci-time-zones';
    this.name = 'Fibonacci Time Zones';
    this.category = 'fibonacci';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Line properties
    this.lineColor = '#673ab7';
    this.lineWidth = 1;
    this.showLabels = true;

    // Fibonacci sequence for time zones
    this.sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
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
  }

  /**
   * Handle click - place time zones starting point
   */
  onClick(event, chartState) {
    return {
      action: 'place-fibonacci-time-zones',
      x: event.clientX,
      y: event.clientY,
      sequence: this.sequence,
      lineColor: this.lineColor,
      lineWidth: this.lineWidth,
      showLabels: this.showLabels,
      id: crypto.randomUUID()
    };
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    return {
      action: 'preview-fibonacci-time-zones',
      x: event.clientX,
      y: event.clientY,
      sequence: this.sequence,
      lineColor: this.lineColor,
      lineWidth: this.lineWidth,
      showLabels: this.showLabels
    };
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
   * Set maximum sequence length
   */
  setMaxSequence(count) {
    this.sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144].slice(0, count);
  }
}

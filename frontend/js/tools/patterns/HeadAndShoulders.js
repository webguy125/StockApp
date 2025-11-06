/**
 * Head and Shoulders Pattern Tool
 * Mark head and shoulders chart pattern (reversal pattern)
 * Requires 5 points: left shoulder, head, right shoulder, and neckline
 */

export class HeadAndShoulders {
  constructor() {
    this.id = 'head-and-shoulders';
    this.name = 'Head & Shoulders';
    this.category = 'patterns';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state (requires 5 points)
    this.drawingStep = 0;
    this.points = [];

    // Line properties
    this.lineColor = '#ff5722';
    this.lineWidth = 2;
    this.showLabels = true;
    this.fillOpacity = 0.1;
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
    this.points = [];
  }

  /**
   * Handle mouse down - collect 5 points
   */
  onMouseDown(event, chartState) {
    // Fix cursor offset - use canvas coordinates
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    this.points.push({ x, y });
    this.drawingStep++;

    if (this.drawingStep < 5) {
      return {
        action: `head-shoulders-point-${this.drawingStep}`,
        points: [...this.points]
      };
    } else {
      // Fifth point - complete the pattern
      const pattern = {
        action: 'finish-head-and-shoulders',
        points: [...this.points],
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        fillOpacity: this.fillOpacity,
        id: crypto.randomUUID()
      };

      this.reset();
      return pattern;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.drawingStep > 0) {
      // Fix cursor offset - use canvas coordinates
      const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
      const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

      return {
        action: 'preview-head-and-shoulders',
        points: [...this.points],
        currentX: x,
        currentY: y,
        step: this.drawingStep,
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

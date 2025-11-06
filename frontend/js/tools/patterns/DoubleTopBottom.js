/**
 * Double Top/Bottom Pattern Tool
 * Mark double top or double bottom patterns (reversal patterns)
 * Requires 3 points: peak 1, valley/peak (middle), peak 2
 */

export class DoubleTopBottom {
  constructor() {
    this.id = 'double-top-bottom';
    this.name = 'Double Top/Bottom';
    this.category = 'patterns';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state (requires 3 points)
    this.drawingStep = 0;
    this.points = [];

    // Line properties
    this.lineColor = '#f44336';
    this.lineWidth = 2;
    this.showLabels = true;
    this.patternType = 'top'; // 'top' or 'bottom'
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
   * Handle mouse down - collect 3 points
   */
  onMouseDown(event, chartState) {
    // Fix cursor offset - use canvas coordinates
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    this.points.push({ x, y });
    this.drawingStep++;

    if (this.drawingStep < 3) {
      return {
        action: `double-top-bottom-point-${this.drawingStep}`,
        points: [...this.points]
      };
    } else {
      // Third point - complete the pattern
      const pattern = {
        action: 'finish-double-top-bottom',
        points: [...this.points],
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        patternType: this.patternType,
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
        action: 'preview-double-top-bottom',
        points: [...this.points],
        currentX: x,
        currentY: y,
        step: this.drawingStep,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        patternType: this.patternType
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
   * Set pattern type
   */
  setPatternType(type) {
    const validTypes = ['top', 'bottom'];
    if (validTypes.includes(type)) {
      this.patternType = type;
    }
  }

  /**
   * Toggle labels
   */
  toggleLabels() {
    this.showLabels = !this.showLabels;
  }
}

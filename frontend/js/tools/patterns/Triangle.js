/**
 * Triangle Pattern Tool
 * Draw triangle patterns (ascending, descending, symmetric)
 * Requires 4 points to define two converging trend lines
 */

export class Triangle {
  constructor() {
    this.id = 'triangle';
    this.name = 'Triangle';
    this.category = 'patterns';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state (requires 4 points)
    this.drawingStep = 0;
    this.points = [];

    // Line properties
    this.lineColor = '#9c27b0';
    this.lineWidth = 2;
    this.showLabels = true;
    this.fillOpacity = 0.1;
    this.patternType = 'symmetric'; // 'ascending', 'descending', 'symmetric'
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
   * Handle mouse down - collect 4 points
   */
  onMouseDown(event, chartState) {
    // Fix cursor offset - use canvas coordinates
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    this.points.push({ x, y });
    this.drawingStep++;

    if (this.drawingStep < 4) {
      return {
        action: `triangle-point-${this.drawingStep}`,
        points: [...this.points]
      };
    } else {
      // Fourth point - complete the pattern
      const pattern = {
        action: 'finish-triangle',
        points: [...this.points],
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showLabels: this.showLabels,
        fillOpacity: this.fillOpacity,
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
        action: 'preview-triangle',
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
   * Set pattern type
   */
  setPatternType(type) {
    const validTypes = ['ascending', 'descending', 'symmetric'];
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

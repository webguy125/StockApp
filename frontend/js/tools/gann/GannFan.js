/**
 * Gann Fan Tool
 * Draw Gann angles from a pivot point
 * Standard angles: 1x8, 1x4, 1x3, 1x2, 1x1, 2x1, 3x1, 4x1, 8x1
 */

export class GannFan {
  constructor() {
    this.id = 'gann-fan';
    this.name = 'Gann Fan';
    this.category = 'gann';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#ff9800';
    this.lineWidth = 1;
    this.showLabels = true;

    // Gann angles (rise/run ratios)
    this.angles = {
      '1x8': 1/8,
      '1x4': 1/4,
      '1x3': 1/3,
      '1x2': 1/2,
      '1x1': 1,
      '2x1': 2,
      '3x1': 3,
      '4x1': 4,
      '8x1': 8
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
        action: 'start-gann-fan',
        x,
        y
      };
    } else {
      // Second click - finish the fan (determines direction/trend)
      this.isDrawing = false;

      const fan = {
        action: 'finish-gann-fan',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        angles: this.angles,
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
        action: 'preview-gann-fan',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: x,
        endY: y,
        angles: this.angles,
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

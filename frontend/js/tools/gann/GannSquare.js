/**
 * Gann Square Tool
 * Draw a Gann square with price and time divisions
 * Square of 9 or Square of 144
 */

export class GannSquare {
  constructor() {
    this.id = 'gann-square';
    this.name = 'Gann Square';
    this.category = 'gann';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#4caf50';
    this.lineWidth = 1;
    this.divisions = 9; // Square of 9 by default
    this.showLabels = true;
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
      // First click - start the square
      this.isDrawing = true;
      this.startPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-gann-square',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - finish the square
      this.isDrawing = false;

      const square = {
        action: 'finish-gann-square',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        divisions: this.divisions,
        showLabels: this.showLabels,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return square;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      return {
        action: 'preview-gann-square',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        divisions: this.divisions,
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
   * Set number of divisions
   */
  setDivisions(count) {
    this.divisions = Math.max(3, Math.min(144, count));
  }

  /**
   * Toggle labels
   */
  toggleLabels() {
    this.showLabels = !this.showLabels;
  }
}

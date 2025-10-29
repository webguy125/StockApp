/**
 * Gann Box Tool
 * Draw a Gann box with internal division lines
 * Box divided into quarters with diagonal lines
 */

export class GannBox {
  constructor() {
    this.id = 'gann-box';
    this.name = 'Gann Box';
    this.category = 'gann';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#00bcd4';
    this.lineWidth = 1;
    this.showDiagonals = true;
    this.showQuarters = true;
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
      // First click - start the box
      this.isDrawing = true;
      this.startPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-gann-box',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - finish the box
      this.isDrawing = false;

      const box = {
        action: 'finish-gann-box',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showDiagonals: this.showDiagonals,
        showQuarters: this.showQuarters,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return box;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      return {
        action: 'preview-gann-box',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        showDiagonals: this.showDiagonals,
        showQuarters: this.showQuarters
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
   * Toggle diagonals
   */
  toggleDiagonals() {
    this.showDiagonals = !this.showDiagonals;
  }

  /**
   * Toggle quarter divisions
   */
  toggleQuarters() {
    this.showQuarters = !this.showQuarters;
  }
}

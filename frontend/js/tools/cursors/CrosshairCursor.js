/**
 * Crosshair Cursor Tool
 * Precise price and time tracking with crosshair lines
 */

export class CrosshairCursor {
  constructor() {
    this.id = 'crosshair-cursor';
    this.name = 'Crosshair';
    this.category = 'cursors';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Crosshair state
    this.showHorizontalLine = true;
    this.showVerticalLine = true;
    this.showPriceLabel = true;
    this.showTimeLabel = true;
  }

  /**
   * Activate this cursor
   */
  activate(canvas) {
    this.isActive = true;
    canvas.style.cursor = this.cursorStyle;
    // console.log(`✅ Activated: ${this.name} (Crosshair Tracking Tool)`);
  }

  /**
   * Deactivate this cursor
   */
  deactivate(canvas) {
    this.isActive = false;
    canvas.style.cursor = 'default';
  }

  /**
   * Handle mouse move - draw crosshair
   */
  onMouseMove(event, chartState) {
    return {
      action: 'draw-crosshair',
      x: event.clientX,
      y: event.clientY,
      showHorizontal: this.showHorizontalLine,
      showVertical: this.showVerticalLine,
      showPriceLabel: this.showPriceLabel,
      showTimeLabel: this.showTimeLabel
    };
  }

  /**
   * Handle mouse down - lock crosshair position
   */
  onMouseDown(event, chartState) {
    return {
      action: 'lock-crosshair',
      x: event.clientX,
      y: event.clientY
    };
  }

  /**
   * Handle mouse up - unlock crosshair
   */
  onMouseUp(event, chartState) {
    return {
      action: 'unlock-crosshair'
    };
  }

  /**
   * Toggle horizontal line
   */
  toggleHorizontalLine() {
    this.showHorizontalLine = !this.showHorizontalLine;
  }

  /**
   * Toggle vertical line
   */
  toggleVerticalLine() {
    this.showVerticalLine = !this.showVerticalLine;
  }

  /**
   * Toggle price label
   */
  togglePriceLabel() {
    this.showPriceLabel = !this.showPriceLabel;
  }

  /**
   * Toggle time label
   */
  toggleTimeLabel() {
    this.showTimeLabel = !this.showTimeLabel;
  }
}

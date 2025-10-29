/**
 * Dot Cursor Tool
 * Mark specific points on the chart
 */

export class DotCursor {
  constructor() {
    this.id = 'dot-cursor';
    this.name = 'Dot';
    this.category = 'cursors';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Dot properties
    this.dotColor = '#00bfff';
    this.dotSize = 6;
  }

  /**
   * Activate this cursor
   */
  activate(canvas) {
    this.isActive = true;
    canvas.style.cursor = this.cursorStyle;
    // console.log(`✅ Activated: ${this.name} (Dot Placement Tool)`);
  }

  /**
   * Deactivate this cursor
   */
  deactivate(canvas) {
    this.isActive = false;
    canvas.style.cursor = 'default';
  }

  /**
   * Handle click - place a dot
   */
  onClick(event, chartState) {
    return {
      action: 'place-dot',
      x: event.clientX,
      y: event.clientY,
      color: this.dotColor,
      size: this.dotSize,
      id: crypto.randomUUID()
    };
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    return {
      action: 'preview-dot',
      x: event.clientX,
      y: event.clientY,
      color: this.dotColor,
      size: this.dotSize
    };
  }

  /**
   * Set dot color
   */
  setColor(color) {
    this.dotColor = color;
  }

  /**
   * Set dot size
   */
  setSize(size) {
    this.dotSize = Math.max(2, Math.min(20, size));
  }
}

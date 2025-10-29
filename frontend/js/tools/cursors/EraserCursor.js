/**
 * Eraser Cursor Tool
 * Delete drawings by clicking on them
 */

export class EraserCursor {
  constructor() {
    this.id = 'eraser-cursor';
    this.name = 'Eraser';
    this.category = 'cursors';
    this.cursorStyle = 'pointer';
    this.isActive = false;
    this.hoverRadius = 10; // Detection radius in pixels
  }

  /**
   * Activate this cursor
   */
  activate(canvas) {
    this.isActive = true;
    canvas.style.cursor = this.cursorStyle;
    // console.log(`✅ Activated: ${this.name}`);
  }

  /**
   * Deactivate this cursor
   */
  deactivate(canvas) {
    this.isActive = false;
    canvas.style.cursor = 'default';
  }

  /**
   * Handle mouse move - highlight drawing under cursor
   */
  onMouseMove(event, chartState) {
    return {
      action: 'highlight-drawing',
      x: event.clientX,
      y: event.clientY,
      radius: this.hoverRadius
    };
  }

  /**
   * Handle click - delete drawing under cursor
   */
  onClick(event, chartState) {
    return {
      action: 'delete-drawing',
      x: event.clientX,
      y: event.clientY,
      radius: this.hoverRadius
    };
  }

  /**
   * Set detection radius
   */
  setRadius(radius) {
    this.hoverRadius = Math.max(5, Math.min(50, radius));
  }
}

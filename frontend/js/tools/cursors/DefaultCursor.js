/**
 * Default Cursor Tool
 * Used for selection, panning, and general chart interaction
 */

export class DefaultCursor {
  constructor() {
    this.id = 'default-cursor';
    this.name = 'Default Cursor';
    this.category = 'cursors';
    this.cursorStyle = 'default';
    this.isActive = false;
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
   * Handle mouse down
   */
  onMouseDown(event, chartState) {
    // Default behavior: start panning
    return {
      action: 'pan-start',
      x: event.clientX,
      y: event.clientY
    };
  }

  /**
   * Handle mouse move
   */
  onMouseMove(event, chartState) {
    // Update cursor position for hover detection
    return {
      action: 'hover',
      x: event.clientX,
      y: event.clientY
    };
  }

  /**
   * Handle mouse up
   */
  onMouseUp(event, chartState) {
    return {
      action: 'pan-end'
    };
  }

  /**
   * Handle click (for selection)
   */
  onClick(event, chartState) {
    // Check if clicked on a drawing
    return {
      action: 'check-selection',
      x: event.clientX,
      y: event.clientY
    };
  }
}

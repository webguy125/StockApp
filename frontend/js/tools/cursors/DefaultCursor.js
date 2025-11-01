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
    // Return null to let canvas-renderer handle panning and selection
    // SelectionManager will handle clicks on drawings
    return null;
  }

  /**
   * Handle mouse move
   */
  onMouseMove(event, chartState) {
    // Return null to let canvas-renderer handle hover/crosshair
    return null;
  }

  /**
   * Handle mouse up
   */
  onMouseUp(event, chartState) {
    // Return null to let canvas-renderer handle pan end
    return null;
  }

  /**
   * Handle click (for selection)
   */
  onClick(event, chartState) {
    // Return null to let SelectionManager handle selection
    return null;
  }
}

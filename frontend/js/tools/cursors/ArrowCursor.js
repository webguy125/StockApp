/**
 * Arrow Cursor Tool
 * Place directional arrows on the chart
 */

export class ArrowCursor {
  constructor() {
    this.id = 'arrow-cursor';
    this.name = 'Arrow';
    this.category = 'cursors';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Arrow properties
    this.arrowColor = null; // Let canvas-renderer auto-color based on direction
    this.arrowSize = 20;
    this.direction = 'up'; // 'up', 'down', 'left', 'right'
  }

  /**
   * Activate this cursor
   */
  activate(canvas) {
    this.isActive = true;
    canvas.style.cursor = this.cursorStyle;
    // console.log(`✅ Activated: ${this.name} (Arrow Placement Tool)`);
  }

  /**
   * Deactivate this cursor
   */
  deactivate(canvas) {
    this.isActive = false;
    canvas.style.cursor = 'default';
  }

  /**
   * Handle click - place an arrow
   */
  onClick(event, chartState) {
    return {
      action: 'place-arrow',
      x: event.clientX,
      y: event.clientY,
      color: this.arrowColor,
      size: this.arrowSize,
      direction: this.direction,
      id: crypto.randomUUID()
    };
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    return {
      action: 'preview-arrow',
      x: event.clientX,
      y: event.clientY,
      color: this.arrowColor,
      size: this.arrowSize,
      direction: this.direction
    };
  }

  /**
   * Set arrow color
   */
  setColor(color) {
    this.arrowColor = color;
  }

  /**
   * Set arrow size
   */
  setSize(size) {
    this.arrowSize = Math.max(10, Math.min(50, size));
  }

  /**
   * Set arrow direction
   */
  setDirection(direction) {
    const validDirections = ['up', 'down', 'left', 'right'];
    if (validDirections.includes(direction)) {
      this.direction = direction;
    }
  }
}

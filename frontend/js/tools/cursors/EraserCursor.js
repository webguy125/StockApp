/**
 * Eraser Cursor Tool
 * Delete drawings by clicking on them
 */

export class EraserCursor {
  constructor() {
    this.id = 'eraser-cursor';
    this.name = 'Eraser';
    this.category = 'cursors';
    // Custom cursor: pointer with eraser icon
    // Using data URL for a simple eraser icon SVG
    this.cursorStyle = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'%3E%3Cpath d='M16.24 3.56L21.19 8.5C21.97 9.29 21.97 10.55 21.19 11.34L12 20.53C10.44 22.09 7.91 22.09 6.34 20.53L2.81 17C2.03 16.21 2.03 14.95 2.81 14.16L13.41 3.56C14.2 2.78 15.46 2.78 16.24 3.56M4.22 15.58L7.76 19.11C8.54 19.9 9.8 19.9 10.59 19.11L14.12 15.58L9.17 10.63L4.22 15.58Z' fill='%23ff6b6b'/%3E%3C/svg%3E"), pointer`;
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

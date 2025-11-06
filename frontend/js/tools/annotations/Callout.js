/**
 * Callout Tool
 * Add callout boxes with arrow pointers
 */

export class Callout {
  constructor() {
    this.id = 'callout';
    this.name = 'Callout';
    this.category = 'annotations';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state (requires 2 points: pointer and text box)
    this.isDrawing = false;
    this.pointerPoint = null;

    // Callout properties
    this.textColor = '#ffffff';
    this.backgroundColor = 'rgba(33, 150, 243, 0.9)';
    this.borderColor = '#2196f3';
    this.fontSize = 14;
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
    this.pointerPoint = null;
  }

  /**
   * Handle mouse down - set pointer or text box position
   */
  onMouseDown(event, chartState) {
    // Fix cursor offset - use canvas coordinates
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    if (!this.isDrawing) {
      // First click - set pointer position
      this.isDrawing = true;
      this.pointerPoint = { x, y };
      return {
        action: 'start-callout',
        x,
        y
      };
    } else {
      // Second click - set text box position and finish
      this.isDrawing = false;

      const callout = {
        action: 'finish-callout',
        pointerX: this.pointerPoint.x,
        pointerY: this.pointerPoint.y,
        textX: x,
        textY: y,
        text: 'Callout', // Default text, will be editable
        textColor: this.textColor,
        backgroundColor: this.backgroundColor,
        borderColor: this.borderColor,
        fontSize: this.fontSize,
        id: crypto.randomUUID()
      };

      this.pointerPoint = null;
      return callout;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.pointerPoint) {
      // Fix cursor offset - use canvas coordinates
      const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
      const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

      return {
        action: 'preview-callout',
        pointerX: this.pointerPoint.x,
        pointerY: this.pointerPoint.y,
        textX: x,
        textY: y,
        text: 'Callout',
        textColor: this.textColor,
        backgroundColor: this.backgroundColor,
        borderColor: this.borderColor,
        fontSize: this.fontSize
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
      this.pointerPoint = null;
      return {
        action: 'cancel-drawing'
      };
    }
    return null;
  }

  /**
   * Set text color
   */
  setTextColor(color) {
    this.textColor = color;
  }

  /**
   * Set background color
   */
  setBackgroundColor(color) {
    this.backgroundColor = color;
  }
}

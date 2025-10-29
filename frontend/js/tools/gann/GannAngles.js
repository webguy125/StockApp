/**
 * Gann Angles Tool
 * Draw specific Gann angle lines (1x1, 2x1, etc.)
 */

export class GannAngles {
  constructor() {
    this.id = 'gann-angles';
    this.name = 'Gann Angles';
    this.category = 'gann';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.startPoint = null;

    // Line properties
    this.lineColor = '#f44336';
    this.lineWidth = 2;
    this.selectedAngle = '1x1'; // Default to 1x1 angle (45 degrees)
    this.extendBoth = true; // Extend line in both directions
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
      // First click - start the angle line
      this.isDrawing = true;
      this.startPoint = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-gann-angles',
        x: event.clientX,
        y: event.clientY
      };
    } else {
      // Second click - finish the angle line (determines direction)
      this.isDrawing = false;

      const angle = {
        action: 'finish-gann-angles',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        angleType: this.selectedAngle,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        extendBoth: this.extendBoth,
        id: crypto.randomUUID()
      };

      this.startPoint = null;
      return angle;
    }
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.startPoint) {
      return {
        action: 'preview-gann-angles',
        startX: this.startPoint.x,
        startY: this.startPoint.y,
        endX: event.clientX,
        endY: event.clientY,
        angleType: this.selectedAngle,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        extendBoth: this.extendBoth
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
   * Set selected angle
   */
  setAngle(angleType) {
    const validAngles = ['1x8', '1x4', '1x3', '1x2', '1x1', '2x1', '3x1', '4x1', '8x1'];
    if (validAngles.includes(angleType)) {
      this.selectedAngle = angleType;
    }
  }

  /**
   * Toggle extend both directions
   */
  toggleExtend() {
    this.extendBoth = !this.extendBoth;
  }
}

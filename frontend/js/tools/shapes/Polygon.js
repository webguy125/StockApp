/**
 * Polygon Tool
 * Draw multi-point polygons on the chart
 * Double-click or press Enter to finish
 */

export class Polygon {
  constructor() {
    this.id = 'polygon';
    this.name = 'Polygon';
    this.category = 'shapes';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.isDrawing = false;
    this.points = [];

    // Shape properties
    this.lineColor = '#9c27b0';
    this.lineWidth = 2;
    this.fillColor = 'rgba(156, 39, 176, 0.1)';
    this.filled = true;
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
    this.reset();
  }

  /**
   * Reset drawing state
   */
  reset() {
    this.isDrawing = false;
    this.points = [];
  }

  /**
   * Handle mouse down - add point
   */
  onMouseDown(event, chartState) {
    this.isDrawing = true;
    this.points.push({ x: event.clientX, y: event.clientY });

    return {
      action: 'polygon-add-point',
      points: [...this.points]
    };
  }

  /**
   * Handle double click - finish polygon
   */
  onDoubleClick(event, chartState) {
    if (this.points.length >= 3) {
      const polygon = {
        action: 'finish-polygon',
        points: [...this.points],
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        fillColor: this.fillColor,
        filled: this.filled,
        id: crypto.randomUUID()
      };

      this.reset();
      return polygon;
    }
    return null;
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    if (this.isDrawing && this.points.length > 0) {
      return {
        action: 'preview-polygon',
        points: [...this.points],
        currentX: event.clientX,
        currentY: event.clientY,
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        fillColor: this.fillColor,
        filled: this.filled
      };
    }
    return null;
  }

  /**
   * Handle key press - Enter to finish, Escape to cancel
   */
  onKeyDown(event, chartState) {
    if (event.key === 'Enter' && this.points.length >= 3) {
      const polygon = {
        action: 'finish-polygon',
        points: [...this.points],
        lineColor: this.lineColor,
        lineWidth: this.lineWidth,
        fillColor: this.fillColor,
        filled: this.filled,
        id: crypto.randomUUID()
      };

      this.reset();
      return polygon;
    } else if (event.key === 'Escape' && this.isDrawing) {
      this.reset();
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
   * Set fill color
   */
  setFillColor(color) {
    this.fillColor = color;
  }

  /**
   * Toggle filled
   */
  toggleFilled() {
    this.filled = !this.filled;
  }
}

/**
 * Price Label Tool
 * Add price level labels to the chart
 */

export class PriceLabel {
  constructor() {
    this.id = 'price-label';
    this.name = 'Price Label';
    this.category = 'annotations';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Label properties
    this.labelColor = '#00c853';
    this.textColor = '#ffffff';
    this.fontSize = 12;
    this.showLine = true; // Show horizontal line at price level
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
  }

  /**
   * Handle click - place price label
   */
  onClick(event, chartState) {
    return {
      action: 'place-price-label',
      x: event.clientX,
      y: event.clientY,
      labelColor: this.labelColor,
      textColor: this.textColor,
      fontSize: this.fontSize,
      showLine: this.showLine,
      id: crypto.randomUUID()
    };
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    return {
      action: 'preview-price-label',
      y: event.clientY,
      labelColor: this.labelColor,
      textColor: this.textColor,
      fontSize: this.fontSize,
      showLine: this.showLine
    };
  }

  /**
   * Set label color
   */
  setLabelColor(color) {
    this.labelColor = color;
  }

  /**
   * Set text color
   */
  setTextColor(color) {
    this.textColor = color;
  }

  /**
   * Toggle horizontal line
   */
  toggleLine() {
    this.showLine = !this.showLine;
  }
}

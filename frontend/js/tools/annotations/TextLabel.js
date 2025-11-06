/**
 * Text Label Tool
 * Add text labels to the chart
 */

export class TextLabel {
  constructor() {
    this.id = 'text-label';
    this.name = 'Text Label';
    this.category = 'annotations';
    this.cursorStyle = 'text';
    this.isActive = false;

    // Text properties
    this.fontSize = 14;
    this.fontFamily = 'Arial';
    this.textColor = '#ffffff';
    this.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    this.showBackground = true;
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
   * Handle click - place text label
   */
  onClick(event, chartState) {
    // Fix cursor offset - use canvas coordinates
    const x = event.canvasX !== undefined ? event.canvasX : event.clientX;
    const y = event.canvasY !== undefined ? event.canvasY : event.clientY;

    return {
      action: 'place-text-label',
      x,
      y,
      text: 'Text', // Default text, will be editable
      fontSize: this.fontSize,
      fontFamily: this.fontFamily,
      textColor: this.textColor,
      backgroundColor: this.backgroundColor,
      showBackground: this.showBackground,
      id: crypto.randomUUID()
    };
  }

  /**
   * Set font size
   */
  setFontSize(size) {
    this.fontSize = Math.max(8, Math.min(72, size));
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

  /**
   * Toggle background
   */
  toggleBackground() {
    this.showBackground = !this.showBackground;
  }
}

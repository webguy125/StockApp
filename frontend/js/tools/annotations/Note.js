/**
 * Note Tool
 * Add sticky notes to the chart
 */

export class Note {
  constructor() {
    this.id = 'note';
    this.name = 'Note';
    this.category = 'annotations';
    this.cursorStyle = 'pointer';
    this.isActive = false;

    // Note properties
    this.noteColor = '#ffeb3b'; // Yellow sticky note
    this.textColor = '#000000';
    this.fontSize = 12;
    this.width = 150;
    this.height = 100;
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
   * Handle click - place note
   */
  onClick(event, chartState) {
    return {
      action: 'place-note',
      x: event.clientX,
      y: event.clientY,
      text: 'Note', // Default text, will be editable
      noteColor: this.noteColor,
      textColor: this.textColor,
      fontSize: this.fontSize,
      width: this.width,
      height: this.height,
      id: crypto.randomUUID()
    };
  }

  /**
   * Handle mouse move - show preview
   */
  onMouseMove(event, chartState) {
    return {
      action: 'preview-note',
      x: event.clientX,
      y: event.clientY,
      noteColor: this.noteColor,
      width: this.width,
      height: this.height
    };
  }

  /**
   * Set note color
   */
  setNoteColor(color) {
    this.noteColor = color;
  }

  /**
   * Set text color
   */
  setTextColor(color) {
    this.textColor = color;
  }

  /**
   * Set size
   */
  setSize(width, height) {
    this.width = Math.max(100, Math.min(300, width));
    this.height = Math.max(80, Math.min(200, height));
  }
}

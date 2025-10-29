/**
 * Parallel Channel Tool
 * Draw two parallel trend lines forming a channel
 */

export class ParallelChannel {
  constructor() {
    this.id = 'parallel-channel';
    this.name = 'Parallel Channel';
    this.category = 'trendLines';
    this.cursorStyle = 'crosshair';
    this.isActive = false;

    // Drawing state
    this.drawingStep = 0; // 0: not started, 1: first line started, 2: first line complete, 3: second line
    this.firstLineStart = null;
    this.firstLineEnd = null;

    // Line properties
    this.lineColor = '#00bcd4';
    this.lineWidth = 2;
    this.lineStyle = 'solid';
    this.fillOpacity = 0.1;
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
    this.drawingStep = 0;
    this.firstLineStart = null;
    this.firstLineEnd = null;
  }

  /**
   * Handle mouse down - progress through channel drawing steps
   */
  onMouseDown(event, chartState) {
    if (this.drawingStep === 0) {
      // First click - start first line
      this.drawingStep = 1;
      this.firstLineStart = { x: event.clientX, y: event.clientY };
      return {
        action: 'start-parallel-channel-line1',
        x: event.clientX,
        y: event.clientY
      };
    } else if (this.drawingStep === 1) {
      // Second click - finish first line
      this.drawingStep = 2;
      this.firstLineEnd = { x: event.clientX, y: event.clientY };
      return {
        action: 'finish-parallel-channel-line1',
        startX: this.firstLineStart.x,
        startY: this.firstLineStart.y,
        endX: this.firstLineEnd.x,
        endY: this.firstLineEnd.y
      };
    } else if (this.drawingStep === 2) {
      // Third click - place parallel line and finish channel
      this.drawingStep = 0;

      const channel = {
        action: 'finish-parallel-channel',
        line1Start: this.firstLineStart,
        line1End: this.firstLineEnd,
        parallelY: event.clientY,
        parallelX: event.clientX,
        color: this.lineColor,
        width: this.lineWidth,
        style: this.lineStyle,
        fillOpacity: this.fillOpacity,
        id: crypto.randomUUID()
      };

      this.reset();
      return channel;
    }
    return null;
  }

  /**
   * Handle mouse move - show preview at each step
   */
  onMouseMove(event, chartState) {
    if (this.drawingStep === 1 && this.firstLineStart) {
      // Preview first line
      return {
        action: 'preview-parallel-channel-line1',
        startX: this.firstLineStart.x,
        startY: this.firstLineStart.y,
        endX: event.clientX,
        endY: event.clientY,
        color: this.lineColor,
        width: this.lineWidth,
        style: this.lineStyle
      };
    } else if (this.drawingStep === 2 && this.firstLineEnd) {
      // Preview parallel line
      return {
        action: 'preview-parallel-channel',
        line1Start: this.firstLineStart,
        line1End: this.firstLineEnd,
        parallelY: event.clientY,
        parallelX: event.clientX,
        color: this.lineColor,
        width: this.lineWidth,
        style: this.lineStyle,
        fillOpacity: this.fillOpacity
      };
    }
    return null;
  }

  /**
   * Handle escape key - cancel drawing
   */
  onKeyDown(event, chartState) {
    if (event.key === 'Escape' && this.drawingStep > 0) {
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
   * Set line width
   */
  setWidth(width) {
    this.lineWidth = Math.max(1, Math.min(10, width));
  }

  /**
   * Set line style
   */
  setStyle(style) {
    const validStyles = ['solid', 'dashed', 'dotted'];
    if (validStyles.includes(style)) {
      this.lineStyle = style;
    }
  }

  /**
   * Set fill opacity
   */
  setFillOpacity(opacity) {
    this.fillOpacity = Math.max(0, Math.min(1, opacity));
  }
}

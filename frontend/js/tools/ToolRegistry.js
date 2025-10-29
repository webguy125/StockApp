/**
 * Drawing Tools Registry
 * Central registry for all TradingView-style drawing tools
 * Manages tool categories, instances, and state
 */

export class ToolRegistry {
  constructor() {
    this.categories = {
      cursors: [],
      trendLines: [],
      fibonacci: [],
      gann: [],
      patterns: [],
      forecasting: [],
      shapes: [],
      annotations: []
    };

    this.activeTool = null;
    this.drawings = []; // All active drawings on the chart
  }

  /**
   * Register a tool in a specific category
   */
  register(category, tool) {
    if (!this.categories[category]) {
      console.error(`❌ Unknown category: ${category}`);
      return false;
    }

    this.categories[category].push(tool);
    // console.log(`✅ Registered tool: ${tool.name} in ${category}`);
    return true;
  }

  /**
   * Get all tools in a category
   */
  getCategory(category) {
    return this.categories[category] || [];
  }

  /**
   * Get all available tools
   */
  getAllTools() {
    return this.categories;
  }

  /**
   * Set active drawing tool
   */
  setActiveTool(tool) {
    this.activeTool = tool;
    // console.log(`🖊️ Active tool: ${tool ? tool.name : 'none'}`);
  }

  /**
   * Get active tool
   */
  getActiveTool() {
    return this.activeTool;
  }

  /**
   * Add a drawing to the chart
   */
  addDrawing(drawing) {
    this.drawings.push(drawing);
    return drawing;
  }

  /**
   * Remove a drawing
   */
  removeDrawing(drawingId) {
    const index = this.drawings.findIndex(d => d.id === drawingId);
    if (index !== -1) {
      this.drawings.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Get all drawings
   */
  getDrawings() {
    return this.drawings;
  }

  /**
   * Clear all drawings
   */
  clearDrawings() {
    this.drawings = [];
  }
}

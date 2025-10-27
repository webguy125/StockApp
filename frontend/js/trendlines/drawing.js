/**
 * Trendline Drawing Module
 * Handles drawing mode, pan mode, and line deletion
 */

import { state } from '../core/state.js';
import { updateShapesColors } from './selection.js';

/**
 * Enable drawing mode for trendlines
 * NOTE: Disabled - using canvas renderer now, not Plotly
 */
export function enableDrawing(plotId = "tos-plot") {
  const plotDiv = document.getElementById(plotId);
  if (!plotDiv) return;
  state.isDrawingMode = true;
  // Plotly.relayout(plotId, { dragmode: 'drawline' });  // DISABLED - no longer using Plotly
  state.selectedLineId = null;
  // updateShapesColors(plotId);  // DISABLED - no longer using Plotly
}

/**
 * Enable pan mode (disable drawing)
 * NOTE: Disabled - using canvas renderer now, not Plotly
 */
export function enablePan(plotId = "tos-plot") {
  const plotDiv = document.getElementById(plotId);
  if (!plotDiv) return;
  state.isDrawingMode = false;
  // Plotly.relayout(plotId, { dragmode: 'pan' });  // DISABLED - no longer using Plotly
  state.selectedLineId = null;
  // updateShapesColors(plotId);  // DISABLED - no longer using Plotly
}

/**
 * Clear all trendlines for current symbol
 * NOTE: Disabled - using canvas renderer now, not Plotly
 */
export function clearAllLines(plotId = "tos-plot") {
  if (!state.currentSymbol) return;

  if (!confirm("Clear all trendlines for " + state.currentSymbol + "?")) return;

  fetch("/clear_lines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol: state.currentSymbol })
  })
  .then(() => {
    state.drawnLines = [];
    state.selectedLineId = null;
    state.suppressSave = true;
    // Plotly.relayout(plotId, { shapes: [], annotations: [] });  // DISABLED - no longer using Plotly
    state.suppressSave = false;
  })
  .catch(err => console.error("Error clearing lines:", err));
}

/**
 * Delete the currently selected line
 * NOTE: Disabled - using canvas renderer now, not Plotly
 */
export function deleteSelectedLine(plotId = "tos-plot") {
  if (!state.selectedLineId) return;

  const plotDiv = document.getElementById(plotId);
  if (!plotDiv || !plotDiv.layout) return;

  const shapes = (plotDiv.layout.shapes || []).filter(s => s._id !== state.selectedLineId);
  const annotations = (plotDiv.layout.annotations || []).filter(a => a._lineId !== state.selectedLineId);

  state.drawnLines = state.drawnLines.filter(l => l.id !== state.selectedLineId);

  fetch("/delete_line", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol: state.currentSymbol, id: state.selectedLineId })
  }).catch(err => console.error("Error deleting line:", err));

  state.selectedLineId = null;
  state.suppressSave = true;
  // Plotly.relayout(plotId, { shapes, annotations });  // DISABLED - no longer using Plotly
  state.suppressSave = false;
}

/**
 * Hide popup menu
 */
export function hidePopup() {
  const popup = document.getElementById("popupMenu");
  popup.style.display = "none";
}

// Make globally accessible for onclick handlers
window.enableDrawing = enableDrawing;
window.enablePan = enablePan;
window.clearAllLines = clearAllLines;
window.deleteSelectedLine = deleteSelectedLine;
window.hidePopup = hidePopup;

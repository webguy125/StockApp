"""
Helper script to complete the refactoring of StockApp into modular structure.
This script creates all remaining modular files for frontend and backend.
"""

import os

# Define all files to create
files = {
    # Frontend JS modules (remaining)
    'frontend/js/trendlines/annotations.js': '''// Trendline Annotations (Volume Labels)
import { getCurrentTheme, getChartData } from '../core/state.js';
import { formatVolume } from './geometry.js';
import { setSelectedLineId, updateShapesColors } from './selection.js';

export function createAnnotation(line, chartData) {
  const midX = new Date((new Date(line.x0).getTime() + new Date(line.x1).getTime()) / 2);
  const midY = (line.y0 + line.y1) / 2;
  const priceRange = Math.abs(line.y1 - line.y0);
  const baseOffset = Math.max(priceRange * 0.08, 5);

  let yPosition = midY + baseOffset;
  let verticalAlign = 'bottom';

  if (chartData && chartData.length > 0) {
    const nearbyCandlesAbove = chartData.filter(candle => {
      return candle.High > midY && candle.High < (midY + baseOffset * 2);
    });

    const nearbyCandlesBelow = chartData.filter(candle => {
      return candle.Low < midY && candle.Low > (midY - baseOffset * 2);
    });

    if (nearbyCandlesAbove.length > nearbyCandlesBelow.length) {
      yPosition = midY - baseOffset;
      verticalAlign = 'top';
    }
  }

  return {
    x: midX,
    y: yPosition,
    text: formatVolume(line.volume),
    showarrow: false,
    font: {
      color: getCurrentTheme() === 'light' ? "#2563eb" : "#60a5fa",
      size: 14,
      family: 'Arial, sans-serif',
      weight: 'bold'
    },
    xanchor: 'center',
    yanchor: verticalAlign,
    bgcolor: getCurrentTheme() === 'light' ? 'rgba(255, 255, 255, 0.95)' : 'rgba(30, 41, 59, 0.95)',
    borderpad: 6,
    bordercolor: getCurrentTheme() === 'light' ? '#2563eb' : '#60a5fa',
    borderwidth: 2,
    captureevents: true,
    clicktoshow: false
  };
}

export function setupAnnotationClicks() {
  setTimeout(() => {
    console.log('Setting up annotation clicks...');
    const plotDiv = document.getElementById("plot");

    const allTextElements = document.querySelectorAll('#plot svg text');
    console.log(`Found ${allTextElements.length} SVG text elements`);

    const layoutAnns = (plotDiv.layout && plotDiv.layout.annotations) || [];
    const volumeTexts = layoutAnns
      .filter(a => a._lineId)
      .map(a => a.text);

    console.log('Volume texts to look for:', volumeTexts);

    let foundCount = 0;
    allTextElements.forEach(textEl => {
      const text = textEl.textContent || textEl.innerText;

      if (volumeTexts.includes(text)) {
        foundCount++;
        textEl.style.cursor = 'pointer';
        textEl.style.pointerEvents = 'all';

        const newEl = textEl.cloneNode(true);
        textEl.parentNode.replaceChild(newEl, textEl);

        newEl.addEventListener('click', function(e) {
          e.stopPropagation();
          e.preventDefault();
          console.log('✓ Volume label clicked:', text);

          const layoutAnns = plotDiv.layout.annotations || [];
          for (let a of layoutAnns) {
            if (a._lineId && a.text === text) {
              setSelectedLineId(a._lineId);
              updateShapesColors();
              console.log('✓ Selected line via volume label:', a._lineId);
              return;
            }
          }
        });

        console.log(`✓ Added click handler to volume label: "${text}"`);
      }
    });

    console.log(`Attached ${foundCount} annotation click handlers`);
  }, 500);
}
''',

    'frontend/js/trendlines/drawing.js': '''// Trendline Drawing Mode Management
import { state, setSelectedLineId } from '../core/state.js';
import { updateShapesColors } from './selection.js';
import { getCurrentSymbol } from '../core/state.js';

export function enableDrawing() {
  state.isDrawingMode = true;
  Plotly.relayout("plot", { dragmode: 'drawline' });
  setSelectedLineId(null);
  updateShapesColors();
}

export function enablePan() {
  state.isDrawingMode = false;
  Plotly.relayout("plot", { dragmode: 'pan' });
  setSelectedLineId(null);
  updateShapesColors();
}

export function clearAllLines() {
  const symbol = getCurrentSymbol();
  if (!symbol) return;

  if (!confirm("Clear all trendlines for " + symbol + "?")) return;

  fetch("/clear_lines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol: symbol })
  })
  .then(() => {
    state.drawnLines = [];
    setSelectedLineId(null);
    state.suppressSave = true;
    Plotly.relayout("plot", { shapes: [], annotations: [] });
    state.suppressSave = false;
  })
  .catch(err => console.error("Error clearing lines:", err));
}

export function deleteSelectedLine() {
  const selectedLineId = state.selectedLineId;
  if (!selectedLineId) return;

  const plotDiv = document.getElementById("plot");
  const shapes = (plotDiv.layout.shapes || []).filter(s => s._id !== selectedLineId);
  const annotations = (plotDiv.layout.annotations || []).filter(a => a._lineId !== selectedLineId);

  state.drawnLines = state.drawnLines.filter(l => l.id !== selectedLineId);

  fetch("/delete_line", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol: getCurrentSymbol(), id: selectedLineId })
  }).catch(err => console.error("Error deleting line:", err));

  setSelectedLineId(null);
  state.suppressSave = true;
  Plotly.relayout("plot", { shapes, annotations });
  state.suppressSave = false;
}

export function hidePopup() {
  const popup = document.getElementById("popupMenu");
  if (popup) {
    popup.style.display = "none";
  }
}

// Make functions globally available for onclick handlers
window.enableDrawing = enableDrawing;
window.enablePan = enablePan;
window.clearAllLines = clearAllLines;
''',
}

def create_files():
    """Create all files defined in the files dictionary."""
    for filepath, content in files.items():
        full_path = os.path.join('C:', 'StockApp', filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {filepath}")

if __name__ == "__main__":
    create_files()
    print("\\nRefactoring helper completed!")

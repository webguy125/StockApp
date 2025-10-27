/**
 * Trendline Handlers Module
 * Handles plotly events and line saving/loading
 */

import { state } from '../core/state.js';
import { createAnnotation, setupAnnotationClicks } from './annotations.js';
import { updateShapesColors } from './selection.js';
import { getDataCoords, distanceToSegment } from './geometry.js';
import { hidePopup, deleteSelectedLine } from './drawing.js';

/**
 * Load saved trendlines for a symbol
 */
export async function loadSavedLines(symbol) {
  try {
    const response = await fetch(`/lines/${symbol}`);
    const savedLines = await response.json();

    if (savedLines.length > 0) {
      const shapes = savedLines.map(line => ({
        type: "line",
        x0: line.x0,
        x1: line.x1,
        y0: line.y0,
        y1: line.y1,
        line: { color: state.currentTheme === 'light' ? "black" : "#9ca3af", width: 2 },
        _id: line.id
      }));

      const annotations = savedLines.map(line => {
        const ann = createAnnotation(line, state.chartData);
        ann._lineId = line.id;
        return ann;
      });

      state.drawnLines = savedLines;
      state.suppressSave = true;
      await Plotly.relayout("plot", { shapes, annotations });
      state.suppressSave = false;
    }
  } catch (err) {
    console.error("Error loading saved lines:", err);
  }
}

/**
 * Setup plotly event handlers for trendlines
 */
export function setupPlotlyHandlers(plotId = "plot") {
  const plotDiv = document.getElementById(plotId);

  if (!plotDiv) {
    console.warn(`Plot element with id "${plotId}" not found`);
    return;
  }

  // Handle new/modified lines
  plotDiv.on('plotly_relayout', function(event) {
    if (state.relayoutGuard || state.suppressSave) return;
    state.relayoutGuard = true;
    setTimeout(() => { state.relayoutGuard = false; }, 100);

    const shapes = plotDiv.layout.shapes || [];
    const annotations = plotDiv.layout.annotations || [];

    shapes.forEach((shape) => {
      if (shape.type !== "line") return;
      if (!shape._id) shape._id = crypto.randomUUID();

      const { x0, x1, y0, y1 } = shape;
      const existing = state.drawnLines.find(l => l.id === shape._id);

      if (existing && existing.x0 === x0 && existing.x1 === x1 && existing.y0 === y0 && existing.y1 === y1) {
        return;
      }

      // Fetch volume for this line
      fetch("/volume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: state.currentSymbol,
          start_date: x0,
          end_date: x1,
          interval: state.currentInterval
        })
      })
      .then(res => res.json())
      .then(data => {
        const avgVolume = data.avg_volume || 0;
        const lineData = {
          id: shape._id,
          x0, x1, y0, y1,
          volume: avgVolume
        };

        if (existing) {
          Object.assign(existing, lineData);
        } else {
          state.drawnLines.push(lineData);
        }

        const annotation = createAnnotation(lineData, state.chartData);
        annotation._lineId = shape._id;

        const existingAnnIdx = annotations.findIndex(a => a._lineId === shape._id);
        if (existingAnnIdx >= 0) {
          annotations[existingAnnIdx] = annotation;
        } else {
          annotations.push(annotation);
        }

        state.suppressSave = true;
        Plotly.relayout(plotId, { annotations });
        state.suppressSave = false;

        // Reattach annotation click handlers
        setupAnnotationClicks(plotId);

        // Save to backend
        fetch("/save_line", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbol: state.currentSymbol,
            line: lineData
          })
        }).catch(err => console.error("Error saving line:", err));
      })
      .catch(err => console.error("Error fetching volume:", err));
    });
  });

  // Handle line selection via click
  plotDiv.on('plotly_click', function(event) {
    console.log('plotly_click event:', event);

    const coords = getDataCoords(event.event, plotId);
    if (!coords) {
      console.log('Could not get data coords from click');
      return;
    }

    console.log('Click coords:', coords);

    const clickX = new Date(coords.x).getTime();
    const clickY = coords.y;

    const xaxis = plotDiv._fullLayout.xaxis;
    const yaxis = plotDiv._fullLayout.yaxis;

    if (!xaxis || !yaxis) {
      console.log('No axis found');
      return;
    }

    const xRange = xaxis.range;
    const yRange = yaxis.range;

    console.log('X range:', xRange, 'Y range:', yRange);

    const xScale = 1 / ((new Date(xRange[1]).getTime() - new Date(xRange[0]).getTime()) / 100);
    const yScale = 1 / ((yRange[1] - yRange[0]) / 100);

    console.log('Scales:', xScale, yScale);

    let closestLine = null;
    let minDistance = Infinity;

    const shapes = plotDiv.layout.shapes || [];
    console.log(`Checking ${shapes.length} shapes for proximity`);

    shapes.forEach((shape, idx) => {
      if (shape.type !== "line") {
        console.log(`Shape ${idx} is not a line (${shape.type})`);
        return;
      }
      if (!shape._id) {
        console.log(`Shape ${idx} has no _id`);
        return;
      }

      const x0Time = new Date(shape.x0).getTime();
      const x1Time = new Date(shape.x1).getTime();
      const y0 = shape.y0;
      const y1 = shape.y1;

      console.log(`Line ${idx}: (${x0Time}, ${y0}) to (${x1Time}, ${y1})`);

      const dist = distanceToSegment(clickX, clickY, x0Time, y0, x1Time, y1, xScale, yScale);

      console.log(`Distance to line ${idx}:`, dist);

      if (dist < minDistance) {
        minDistance = dist;
        closestLine = shape;
      }
    });

    console.log('Closest distance found:', minDistance);

    // Check if closest line is within threshold (50 pixels)
    if (closestLine && minDistance < 50) {
      state.selectedLineId = closestLine._id;
      updateShapesColors(plotId);
      console.log('✓ Selected line via click, distance:', minDistance, 'id:', state.selectedLineId);
    } else {
      console.log('✗ No line close enough (min:', minDistance, 'threshold: 50)');
      state.selectedLineId = null;
      updateShapesColors(plotId);
    }

    hidePopup();
  });

  // Handle annotation clicks via Plotly event (backup method)
  plotDiv.on('plotly_clickannotation', function(event, data) {
    console.log('plotly_clickannotation event:', event, data);

    const annotationData = data || event;

    if (!annotationData || annotationData.index === undefined) {
      console.log('No annotation data available');
      return;
    }

    const annotation = plotDiv.layout.annotations[annotationData.index];
    if (annotation && annotation._lineId) {
      state.selectedLineId = annotation._lineId;
      updateShapesColors(plotId);
      console.log('Selected line via plotly_clickannotation:', state.selectedLineId);
    }
    hidePopup();
  });

  // Handle right-click context menu
  plotDiv.on('plotly_contextmenu', function(event) {
    event.event.preventDefault();

    const popup = document.getElementById("popupMenu");

    if (state.selectedLineId) {
      popup.innerHTML = '<div onclick="deleteSelectedLine(); hidePopup();">Delete Line</div>';
    } else {
      hidePopup();
      return false;
    }

    popup.style.left = event.event.pageX + 'px';
    popup.style.top = event.event.pageY + 'px';
    popup.style.display = 'block';

    return false;
  });

  // Handle Delete key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Delete' && state.selectedLineId) {
      deleteSelectedLine();
    }
  });

  // Setup annotation DOM click handlers
  setupAnnotationClicks(plotId);
}

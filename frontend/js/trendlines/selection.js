// Trendline Selection and Highlighting
import { getSelectedLineId, setSelectedLineId, getCurrentTheme, state } from '../core/state.js';
import { distanceToSegment, getDataCoords } from './geometry.js';

// Drag handles removed per user request - circles were not positioning correctly on trendlines

export function updateShapesColors(plotId = "tos-plot") {
  const plotDiv = document.getElementById(plotId);
  if (!plotDiv) {
    // console.log('⚠️ Plot element not found:', plotId);
    return;
  }
  if (!plotDiv.layout || !plotDiv.layout.shapes) {
    // console.log('⚠️ No layout or shapes found');
    return;
  }

  const selectedLineId = getSelectedLineId();
  // console.log('📊 Updating shapes colors. Selected ID:', selectedLineId);

  const shapes = plotDiv.layout.shapes.map((shape, idx) => {
    if (shape.type !== "line") return shape;

    const isSelected = shape._id === selectedLineId;
    const newColor = isSelected ? "#ef4444" : (getCurrentTheme() === 'light' ? "black" : "#9ca3af");
    const newWidth = isSelected ? 3 : 2;

    console.log(`Shape ${idx}: _id=${shape._id}, isSelected=${isSelected}, newColor=${newColor}, newWidth=${newWidth}`);

    return {
      ...shape,
      line: {
        ...shape.line,
        color: newColor,
        width: newWidth
      },
      editable: true  // Always editable
    };
  });

  // Also update annotation colors to indicate selection
  const annotations = (plotDiv.layout.annotations || []).map(ann => {
    if (!ann._lineId) return ann;
    const isSelected = ann._lineId === selectedLineId;
    return {
      ...ann,
      font: {
        ...ann.font,
        color: isSelected ? "#ef4444" : (getCurrentTheme() === 'light' ? "#2563eb" : "#60a5fa"),
        size: isSelected ? 16 : 14
      },
      bordercolor: isSelected ? "#ef4444" : (getCurrentTheme() === 'light' ? '#2563eb' : '#60a5fa'),
      borderwidth: isSelected ? 3 : 2,
      bgcolor: isSelected
        ? 'rgba(239, 68, 68, 0.1)'
        : (getCurrentTheme() === 'light' ? 'rgba(255, 255, 255, 0.95)' : 'rgba(30, 41, 59, 0.95)'),
      captureevents: true  // Make annotations clickable
    };
  });

  state.suppressSave = true;
  // console.log('🔄 Calling Plotly.relayout with', shapes.length, 'shapes and', annotations.length, 'annotations');

  Plotly.relayout(plotId, { shapes, annotations }).then(() => {
    // console.log('✅ Plotly.relayout completed successfully. Selected line:', selectedLineId);

    // Verify the update actually applied
    const plotDiv = document.getElementById(plotId);
    const currentShapes = plotDiv.layout.shapes || [];
    console.log('📋 Current shapes after relayout:', currentShapes.length);
    currentShapes.forEach((s, i) => {
      if (s.type === "line" && s._id) {
        console.log(`  Shape ${i}: _id=${s._id}, color=${s.line?.color}, width=${s.line?.width}`);
      }
    });

    // Drag handles removed - red highlighting and thicker line width provide visual feedback

    state.suppressSave = false;
  }).catch(err => {
    console.error('❌ Error updating shapes:', err);
    state.suppressSave = false;
  });
}

export function handleLineClick(event, plotId = "tos-plot") {
  console.log('plotly_click event:', event);

  const coords = getDataCoords(event.event, plotId);
  if (!coords) {
    console.log('Could not get data coords from click');
    return;
  }

  console.log('Click coords:', coords);

  const clickX = new Date(coords.x).getTime();
  const clickY = coords.y;

  const plotDiv = document.getElementById("plot");
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
      closestLine = shape;  // Always track the closest line
    }
  });

  console.log('Closest distance found:', minDistance);

  // Check if closest line is within threshold (50 pixels)
  if (closestLine && minDistance < 50) {
    setSelectedLineId(closestLine._id);
    updateShapesColors(plotId);
    console.log('✓ Selected line via click, distance:', minDistance, 'id:', closestLine._id);
  } else {
    console.log('✗ No line close enough (min:', minDistance, 'threshold: 50)');
    setSelectedLineId(null);
    updateShapesColors(plotId);
  }
}

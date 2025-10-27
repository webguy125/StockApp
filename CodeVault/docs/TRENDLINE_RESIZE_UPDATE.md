# 🔧 Adding Trendlines & Resize Controls

## Features to Add:

### 1. **Trendline Drawing with Volume Labels**
- Draw trendlines from point A to point B
- Automatic volume calculation for the span
- Smart label positioning to avoid candle overlap
- Persistent storage across sessions
- Right-click to delete selected line

### 2. **Manual Panel Resize**
- Drag to resize main chart vs indicator panels
- Adjustable RSI panel height
- Adjustable MACD panel height
- Save preferred sizes

---

## Quick Fix: Use Classic UI for Trendlines

**The easiest solution**: The trendline feature with volume labels is already fully working in the classic UI!

### Access the Classic UI with Trendlines:
**URL**: http://127.0.0.1:5000/classic

This version has:
- ✅ Full trendline drawing (just draw on the chart)
- ✅ Volume labels showing average volume from point A to B
- ✅ Smart positioning to avoid candles
- ✅ Click to select, right-click to delete
- ✅ Delete key support
- ✅ Persistent storage

### How to Use Trendlines in Classic UI:
1. Go to http://127.0.0.1:5000/classic
2. Enter a symbol and load the chart
3. **Draw a trendline** - Click and drag on the chart
4. **Volume label appears** - Shows average volume between the two points
5. **Click a line** - It turns red when selected
6. **Right-click selected line** - Choose "Delete Selected Line"
7. **Or press Delete key** - Removes selected line

---

## Alternative: Add Trendlines to Complete UI

If you want trendlines in the complete UI (http://127.0.0.1:5000/), you need to add the following code:

### Step 1: Add Drawing Mode Control in Sidebar

Add after the "Quick Actions" section in `index_complete.html`:

```html
<!-- Drawing Tools -->
<div class="section">
  <div class="section-title">Drawing Tools</div>
  <button class="btn-secondary" onclick="enableDrawing()">✏️ Draw Trendlines</button>
  <button class="btn-secondary" onclick="enablePan()">🖐️ Pan Mode</button>
  <div style="font-size: 11px; margin-top: 8px; opacity: 0.7;">
    <strong>Trendline Tips:</strong><br>
    • Click & drag to draw<br>
    • Click line to select<br>
    • Right-click to delete<br>
    • Delete key removes selected
  </div>
</div>
```

### Step 2: Add Helper Functions

Add these JavaScript functions before the `loadChart()` function:

```javascript
// Trendline state
let relayoutGuard = false;
let suppressSave = false;

// Drawing mode toggle
function enableDrawing() {
  const plotDiv = document.getElementById('plot');
  if (plotDiv.layout) {
    Plotly.relayout('plot', { dragmode: 'drawline' });
  }
}

function enablePan() {
  const plotDiv = document.getElementById('plot');
  if (plotDiv.layout) {
    Plotly.relayout('plot', { dragmode: 'pan' });
  }
}

// Volume formatting
function formatVolume(volume) {
  if (volume >= 1000000) {
    const millions = volume / 1000000;
    if (millions >= 100) return Math.round(millions) + 'M';
    if (millions >= 10) return (Math.round(millions * 10) / 10) + 'M';
    return (Math.round(millions * 100) / 100) + 'M';
  } else if (volume >= 1000) {
    const thousands = volume / 1000;
    if (thousands >= 100) return Math.round(thousands) + 'K';
    if (thousands >= 10) return (Math.round(thousands * 10) / 10) + 'K';
    return (Math.round(thousands * 100) / 100) + 'K';
  }
  return Math.round(volume).toString();
}

// Create volume annotation
function createAnnotation(line, chartData) {
  const midX = new Date((new Date(line.x0).getTime() + new Date(line.x1).getTime()) / 2);
  const midY = (line.y0 + line.y1) / 2;
  const priceRange = Math.abs(line.y1 - line.y0);
  const baseOffset = Math.max(priceRange * 0.08, 5);

  let yPosition = midY + baseOffset;
  let verticalAlign = 'bottom';

  if (chartData && chartData.length > 0) {
    const midTime = midX.getTime();
    const timeRange = new Date(line.x1).getTime() - new Date(line.x0).getTime();
    const searchRadius = timeRange * 0.1;

    const nearbyCandlesAbove = chartData.filter(candle => {
      const candleTime = new Date(candle.Date).getTime();
      const timeDiff = Math.abs(candleTime - midTime);
      if (timeDiff > searchRadius) return false;
      return candle.High > midY && candle.High < (midY + baseOffset * 2);
    });

    const nearbyCandlesBelow = chartData.filter(candle => {
      const candleTime = new Date(candle.Date).getTime();
      const timeDiff = Math.abs(candleTime - midTime);
      if (timeDiff > searchRadius) return false;
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
      color: currentTheme === 'light' ? "#2563eb" : "#60a5fa",
      size: 11,
      family: "Arial, sans-serif"
    },
    xanchor: 'center',
    yanchor: verticalAlign,
    bgcolor: currentTheme === 'light' ? 'rgba(255, 255, 255, 0.8)' : 'rgba(55, 65, 81, 0.8)',
    borderpad: 3
  };
}

// Line selection helpers
function getDataCoords(e) {
  const gd = document.getElementById('plot');
  const bb = gd.getBoundingClientRect();
  const px = e.clientX - bb.left;
  const py = e.clientY - bb.top;
  const layout = gd._fullLayout;
  if (!layout) return null;
  const xInPlot = px - layout.margin.l;
  const yInPlot = py - layout.margin.t;
  const xCoord = layout.xaxis.p2c(xInPlot);
  if (!xCoord) return null;
  const xData = new Date(xCoord).getTime();
  const yData = layout.yaxis.p2c(yInPlot);
  return { x: xData, y: yData };
}

function distanceToSegment(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) return Math.hypot(p.x - a.x, p.y - a.y);
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
  t = Math.max(0, Math.min(1, t));
  const projX = a.x + t * dx;
  const projY = a.y + t * dy;
  return Math.hypot(p.x - projX, p.y - projY);
}

function updateShapesColors() {
  const plotDiv = document.getElementById('plot');
  const layout = plotDiv.layout;
  if (!layout.shapes) return;
  const updatedShapes = layout.shapes.map(shape => ({
    ...shape,
    line: {
      ...shape.line,
      color: (shape._id === selectedLineId) ? 'red' : (currentTheme === 'light' ? 'black' : '#9ca3af')
    }
  }));
  Plotly.relayout('plot', { shapes: updatedShapes });
}

function deleteSelectedLine() {
  if (selectedLineId === null) return;
  drawnLines = drawnLines.filter(line => line.id !== selectedLineId);
  const updatedShapes = drawnLines.map(line => ({
    type: "line",
    x0: line.x0,
    x1: line.x1,
    y0: line.y0,
    y1: line.y1,
    line: { color: currentTheme === 'light' ? "black" : "#9ca3af" },
    _id: line.id
  }));
  const updatedAnnotations = drawnLines.map(line => createAnnotation(line, chartData));
  Plotly.relayout("plot", { shapes: updatedShapes, annotations: updatedAnnotations });

  // Delete from backend
  fetch("/delete_line", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol: currentSymbol, line_id: selectedLineId })
  });
  selectedLineId = null;
}

function hidePopup() {
  document.getElementById("popupMenu").style.display = "none";
}

// Keyboard shortcut
document.addEventListener("keydown", function(e) {
  if (e.key === "Delete" && selectedLineId !== null) {
    deleteSelectedLine();
  }
});

// Click outside to close menu
document.addEventListener('click', function(e) {
  const menu = document.getElementById('popupMenu');
  if (menu.style.display !== 'none' && !menu.contains(e.target)) {
    hidePopup();
  }
});
```

### Step 3: Add Trendline Event Handlers

Add these event handlers at the END of `loadChart()` function, after `Plotly.newPlot()`:

```javascript
// After Plotly.newPlot(), add:

const plotDiv = document.getElementById("plot");

// Load saved lines
fetch(`/lines/${symbol}`)
  .then(res => res.json())
  .then(savedLines => {
    if (savedLines.length > 0) {
      const shapes = savedLines.map(line => ({
        type: "line",
        x0: line.x0,
        x1: line.x1,
        y0: line.y0,
        y1: line.y1,
        line: { color: currentTheme === 'light' ? "black" : "#9ca3af" },
        _id: line.id
      }));
      const annotations = savedLines.map(line => createAnnotation(line, chartData));
      drawnLines = savedLines;
      suppressSave = true;
      Plotly.relayout("plot", { shapes, annotations });
      suppressSave = false;
    }
  });

// Handle line drawing
plotDiv.on('plotly_relayout', function(event) {
  if (relayoutGuard || suppressSave) return;
  relayoutGuard = true;
  setTimeout(() => { relayoutGuard = false; }, 100);

  const shapes = plotDiv.layout.shapes || [];
  shapes.forEach((shape) => {
    if (shape.type !== "line") return;
    if (!shape._id) shape._id = crypto.randomUUID();

    const { x0, x1, y0, y1 } = shape;
    const existing = drawnLines.find(l => l.id === shape._id);
    const coordsChanged = !existing || existing.x0 !== x0 || existing.x1 !== x1 || existing.y0 !== y0 || existing.y1 !== y1;

    if (!coordsChanged) return;

    // Fetch volume for this line
    fetch("/volume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol: currentSymbol,
        start_date: x0,
        end_date: x1,
        interval: currentInterval
      })
    })
    .then(res => res.json())
    .then(data => {
      const avgVolume = data.avg_volume;
      const lineData = { x0, x1, y0, y1, volume: avgVolume };
      const annotation = createAnnotation(lineData, chartData);
      const layout = plotDiv.layout;
      const annotations = layout.annotations || [];

      if (existing) {
        annotations[drawnLines.indexOf(existing)] = annotation;
        Object.assign(existing, { x0, x1, y0, y1, volume: avgVolume });
      } else {
        annotations.push(annotation);
        drawnLines.push({ id: shape._id, x0, x1, y0, y1, volume: avgVolume });
      }

      Plotly.relayout("plot", { annotations });

      // Save to backend
      fetch("/save_line", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: currentSymbol,
          line: { id: shape._id, x0, x1, y0, y1, volume: avgVolume }
        })
      });
    });
  });
});

// Handle line selection via click
plotDiv.addEventListener('click', function(e) {
  const pos = getDataCoords(e);
  if (!pos) return;

  const clicked_time = pos.x;
  const clickedY = pos.y;
  const layout = plotDiv._fullLayout;
  const x_min = new Date(layout.xaxis.range[0]).getTime();
  const x_max = new Date(layout.xaxis.range[1]).getTime();
  const y_min = layout.yaxis.range[0];
  const y_max = layout.yaxis.range[1];

  const clicked_xn = (clicked_time - x_min) / (x_max - x_min);
  const clicked_yn = (clickedY - y_min) / (y_max - y_min);

  let closestId = null;
  let minDistance = Infinity;
  const shapes = plotDiv.layout.shapes || [];

  shapes.forEach((shape) => {
    if (shape.type !== "line") return;
    const x0_time = new Date(shape.x0).getTime();
    const x1_time = new Date(shape.x1).getTime();
    const x0n = (x0_time - x_min) / (x_max - x_min);
    const x1n = (x1_time - x_min) / (x_max - x_min);
    const y0n = (shape.y0 - y_min) / (y_max - y_min);
    const y1n = (shape.y1 - y_min) / (y_max - y_min);

    const dist = distanceToSegment(
      { x: clicked_xn, y: clicked_yn },
      { x: x0n, y: y0n },
      { x: x1n, y: y1n }
    );

    if (dist < minDistance) {
      minDistance = dist;
      closestId = shape._id;
    }
  });

  const threshold = 0.05;
  if (minDistance < threshold) {
    if (closestId === selectedLineId) {
      selectedLineId = null;
    } else {
      selectedLineId = closestId;
    }
  } else {
    selectedLineId = null;
  }

  updateShapesColors();
});

// Handle right-click context menu
plotDiv.addEventListener("contextmenu", function(e) {
  e.preventDefault();
  const menu = document.getElementById("popupMenu");
  menu.innerHTML = '';

  if (selectedLineId !== null) {
    const deleteDiv = document.createElement('div');
    deleteDiv.textContent = '🗑️ Delete Selected Line';
    deleteDiv.onclick = function() {
      deleteSelectedLine();
      hidePopup();
    };
    menu.appendChild(deleteDiv);
    menu.style.left = e.pageX + "px";
    menu.style.top = e.pageY + "px";
    menu.style.display = "block";
  }
});
```

### Step 4: Change Default Drag Mode

In the layout configuration in `loadChart()`, change:

```javascript
dragmode: 'pan',  // Change this line
```

To:

```javascript
dragmode: 'drawline',  // Enable trendline drawing by default
```

---

## For Manual Resize of Panels

To add manual resize controls for chart/indicator panel heights:

### Add Resize Controls in Sidebar:

```html
<!-- Panel Heights -->
<div class="section">
  <div class="section-title">Panel Heights</div>
  <div class="input-group">
    <label>Main Chart: <span id="mainChartPercent">70</span>%</label>
    <input type="range" id="mainChartHeight" min="40" max="90" value="70"
           oninput="updatePanelHeights()" style="width: 100%;">
  </div>
  <div class="input-group">
    <label>RSI: <span id="rsiPercent">15</span>%</label>
    <input type="range" id="rsiHeight" min="10" max="30" value="15"
           oninput="updatePanelHeights()" style="width: 100%;">
  </div>
  <div class="input-group">
    <label>MACD: <span id="macdPercent">15</span>%</label>
    <input type="range" id="macdHeight" min="10" max="30" value="15"
           oninput="updatePanelHeights()" style="width: 100%;">
  </div>
</div>
```

### Add Resize Function:

```javascript
function updatePanelHeights() {
  const mainHeight = parseInt(document.getElementById('mainChartHeight').value) / 100;
  const rsiHeight = parseInt(document.getElementById('rsiHeight').value) / 100;
  const macdHeight = parseInt(document.getElementById('macdHeight').value) / 100;

  document.getElementById('mainChartPercent').textContent = Math.round(mainHeight * 100);
  document.getElementById('rsiPercent').textContent = Math.round(rsiHeight * 100);
  document.getElementById('macdPercent').textContent = Math.round(macdHeight * 100);

  const hasRSI = activeIndicators.RSI;
  const hasMACD = activeIndicators.MACD;

  if (!hasRSI && !hasMACD) return;

  let update = {};

  if (hasRSI && hasMACD) {
    const rsiBottom = macdHeight;
    const rsiTop = rsiBottom + rsiHeight;
    update['yaxis.domain'] = [rsiTop, 1];
    update['yaxis2.domain'] = [rsiBottom, rsiTop];
    update['yaxis3.domain'] = [0, rsiBottom];
  } else if (hasRSI) {
    update['yaxis.domain'] = [rsiHeight, 1];
    update['yaxis2.domain'] = [0, rsiHeight];
  } else if (hasMACD) {
    update['yaxis.domain'] = [macdHeight, 1];
    update['yaxis3.domain'] = [0, macdHeight];
  }

  Plotly.relayout('plot', update);
}
```

---

## Summary

**Easiest Solution**: Use the classic UI at http://127.0.0.1:5000/classic - it already has full trendline functionality!

**For Complete UI**: Add the code snippets above to `index_complete.html` to enable:
1. Trendline drawing with volume labels
2. Manual panel height adjustment
3. Line selection and deletion

All the backend APIs for trendlines are already working - you just need to add the frontend event handlers!

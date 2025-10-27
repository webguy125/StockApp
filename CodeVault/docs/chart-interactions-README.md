# Chart Interactions Module

This folder contains modular, isolated components for chart interactions. Each module is independent and can be updated without affecting other chart functionality.

## Modules

### 1. **crosshair.js** - Crosshair Management
Handles the vertical and horizontal crosshair lines that appear on hover.

**Usage:**
```javascript
import { Crosshair } from './chart-interactions/crosshair.js';

const crosshair = new Crosshair('plot-id');
const config = crosshair.getConfig(); // Get Plotly config for crosshair

// Control crosshair
crosshair.enable();
crosshair.disable();
crosshair.toggle();
crosshair.setColor('#ff0000');
```

**Methods:**
- `getConfig()` - Returns Plotly layout configuration for crosshair
- `enable()` - Enable crosshair
- `disable()` - Disable crosshair
- `toggle()` - Toggle crosshair on/off
- `setColor(color)` - Change crosshair color

---

### 2. **pan-zoom.js** - Pan and Zoom Management
Manages chart panning and zooming functionality.

**Usage:**
```javascript
import { PanZoom } from './chart-interactions/pan-zoom.js';

const panZoom = new PanZoom('plot-id');
const config = panZoom.getConfig(); // Get Plotly config
const layoutConfig = panZoom.getLayoutConfig(); // Get layout config

// Control pan/zoom
panZoom.setPan();
panZoom.setZoom();
panZoom.toggleMode();
panZoom.resetZoom();
panZoom.zoomToRange({ min: '2024-01-01', max: '2024-12-31' }, { min: 100, max: 200 });
```

**Methods:**
- `getConfig()` - Returns Plotly config for pan/zoom
- `getLayoutConfig()` - Returns Plotly layout config
- `setPan()` - Set drag mode to pan
- `setZoom()` - Set drag mode to zoom
- `toggleMode()` - Toggle between pan and zoom
- `enableScrollZoom()` / `disableScrollZoom()` - Control scroll zoom
- `resetZoom()` - Reset zoom to show all data
- `zoomToRange(xRange, yRange)` - Zoom to specific range

---

### 3. **keyboard-controls.js** - Keyboard Shortcuts
Manages keyboard shortcuts for chart interactions.

**Usage:**
```javascript
import { KeyboardControls } from './chart-interactions/keyboard-controls.js';

const keyboardControls = new KeyboardControls('plot-id', {
  panZoom: panZoomInstance  // Optional: pass PanZoom instance
});

// Control keyboard shortcuts
keyboardControls.enable();
keyboardControls.disable();
keyboardControls.toggle();
keyboardControls.destroy(); // Clean up event listeners
```

**Built-in Shortcuts:**
- `Ctrl` - Toggle between pan and zoom mode
- `R` - Reset zoom to show all data
- `+` / `=` - Zoom in
- `-` / `_` - Zoom out

**Methods:**
- `registerListener(name, handler)` - Add custom keyboard shortcut
- `unregisterListener(name)` - Remove keyboard shortcut
- `enable()` / `disable()` / `toggle()` - Control shortcuts
- `destroy()` - Clean up all event listeners

---

## Integration Example

```javascript
import { SimpleRenderer } from './chart-renderers/simple-renderer.js';

// Renderer automatically initializes all interaction modules
const renderer = new SimpleRenderer();

// Render chart
await renderer.render(data, 'BTC-USD');

// Access interaction modules if needed
const interactions = renderer.getInteractions();
interactions.crosshair.setColor('#00ff00');
interactions.panZoom.resetZoom();
interactions.keyboardControls.disable();

// Clean up when done
renderer.destroy();
```

## Benefits of Modular Design

1. **Isolation** - Each module is independent and self-contained
2. **Testability** - Each module can be tested separately
3. **Maintainability** - Changes to one module don't break others
4. **Reusability** - Modules can be used in different chart types
5. **Extensibility** - Easy to add new interaction modules

## File Structure

```
chart-interactions/
├── README.md              # This file
├── crosshair.js           # Crosshair management
├── pan-zoom.js            # Pan and zoom management
└── keyboard-controls.js   # Keyboard shortcuts
```

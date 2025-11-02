/**
 * SelectionManager - Handles selection, moving, and resizing of drawn objects
 * Works with all drawing tools (trend lines, shapes, annotations, etc.)
 * Keeps canvas-renderer.js modular by extracting selection logic
 */

export class SelectionManager {
  constructor(renderer) {
    this.renderer = renderer; // Reference to CanvasRenderer instance

    // Selection state
    this.selectedDrawing = null;
    this.isMovingDrawing = false;
    this.isDraggingHandle = false;
    this.draggedHandle = null; // 'start' or 'end'
    this.dragOffset = { x: 0, y: 0 };

    // UI settings
    this.selectionColor = '#ffeb3b'; // Yellow highlight
    this.handleRadius = 6;
    this.hitThreshold = 50; // Pixels for click detection (increased for testing)

    // Last mouse position (for delta calculations)
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    // Context menu state
    this.contextMenu = null;

    this.setupKeyboardEvents();
    this.setupContextMenu();
  }

  /**
   * Setup keyboard event handlers (Delete key)
   */
  setupKeyboardEvents() {
    window.addEventListener('keydown', (e) => this.onKeyDown(e));
  }

  /**
   * Setup context menu (right-click menu)
   */
  setupContextMenu() {
    // Create context menu element
    this.contextMenu = document.createElement('div');
    this.contextMenu.className = 'drawing-context-menu';
    this.contextMenu.style.display = 'none';
    this.contextMenu.innerHTML = `
      <div class="context-menu-item" data-action="copy">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <path fill="currentColor" d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"/>
        </svg>
        <span>Duplicate</span>
      </div>
      <div class="context-menu-item" data-action="delete">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/>
        </svg>
        <span>Delete</span>
      </div>
      <div class="context-menu-divider"></div>
      <div class="context-menu-item context-menu-toggle" data-action="toggle-persistent">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <path fill="currentColor" d="M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2M12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4M11,16.5L6.5,12L7.91,10.59L11,13.67L16.59,8.09L18,9.5L11,16.5Z"/>
        </svg>
        <span>Keep Tool Active</span>
        <div class="toggle-indicator"></div>
      </div>
    `;
    document.body.appendChild(this.contextMenu);

    // Click handlers for menu items
    this.contextMenu.addEventListener('click', (e) => {
      const item = e.target.closest('.context-menu-item');
      if (!item) return;

      const action = item.dataset.action;
      if (action === 'copy' && this.selectedDrawing) {
        this.copyDrawing(this.selectedDrawing);
        this.hideContextMenu();
      } else if (action === 'delete' && this.selectedDrawing) {
        this.deleteSelectedDrawing();
        this.hideContextMenu();
      } else if (action === 'toggle-persistent') {
        // Toggle persistent mode
        const toolPanel = window.toolRegistry?.toolPanel;
        if (toolPanel) {
          toolPanel.persistentMode = !toolPanel.persistentMode;
          // Update both context menu indicator AND header checkbox
          this.updatePersistentModeIndicator();
          toolPanel.updatePersistentModeUI(); // Sync header checkbox
          console.log(`✅ Keep Tool Active: ${toolPanel.persistentMode ? 'ON' : 'OFF'}`);
        }
        // Don't hide menu immediately for toggle - let user see the change
        setTimeout(() => this.hideContextMenu(), 150);
      }
    });

    // Hide context menu when clicking elsewhere
    document.addEventListener('click', () => {
      this.hideContextMenu();
    });

    // Prevent default context menu on canvas, show custom menu instead
    if (this.renderer.canvas) {
      this.renderer.canvas.addEventListener('contextmenu', (e) => {
        // Only show context menu if we have a selected drawing
        if (this.selectedDrawing) {
          e.preventDefault();
          this.showContextMenu(e.clientX, e.clientY);
        }
      });
    }
  }

  /**
   * Show context menu at position
   */
  showContextMenu(x, y) {
    if (!this.contextMenu) return;

    // Update persistent mode indicator before showing
    this.updatePersistentModeIndicator();

    this.contextMenu.style.left = x + 'px';
    this.contextMenu.style.top = y + 'px';
    this.contextMenu.style.display = 'block';
  }

  /**
   * Update the persistent mode toggle indicator in context menu
   */
  updatePersistentModeIndicator() {
    if (!this.contextMenu) return;

    const toolPanel = window.toolRegistry?.toolPanel;
    const toggleItem = this.contextMenu.querySelector('[data-action="toggle-persistent"]');
    if (toggleItem && toolPanel) {
      if (toolPanel.persistentMode) {
        toggleItem.classList.add('active');
      } else {
        toggleItem.classList.remove('active');
      }
    }
  }

  /**
   * Hide context menu
   */
  hideContextMenu() {
    if (this.contextMenu) {
      this.contextMenu.style.display = 'none';
    }
  }

  /**
   * Copy/duplicate a drawing
   */
  copyDrawing(drawing) {
    // Create a deep copy of the drawing
    const copy = JSON.parse(JSON.stringify(drawing));

    // Generate new ID
    copy.id = crypto.randomUUID();

    // Offset the copy slightly so it's visible
    const offset = 5; // Offset in data indices and price units
    if (copy.action === 'place-dot' || copy.action === 'place-arrow') {
      copy.chartIndex += offset;
      copy.chartPrice += offset * (this.renderer.maxPrice - this.renderer.minPrice) / 100; // 5% of price range
    } else if (copy.action === 'finish-trend-line' || copy.action.includes('ray-line') || copy.action.includes('extended-line')) {
      copy.startIndex += offset;
      copy.endIndex += offset;
      const priceOffset = offset * (this.renderer.maxPrice - this.renderer.minPrice) / 100;
      copy.startPrice += priceOffset;
      copy.endPrice += priceOffset;
    } else if (copy.action && copy.action.includes('horizontal-line')) {
      copy.price += offset * (this.renderer.maxPrice - this.renderer.minPrice) / 100;
    } else if (copy.action && copy.action.includes('vertical-line')) {
      copy.chartIndex += offset;
    } else if (copy.action && copy.action.includes('parallel-channel')) {
      copy.startIndex += offset;
      copy.endIndex += offset;
      const priceOffset = offset * (this.renderer.maxPrice - this.renderer.minPrice) / 100;
      copy.startPrice += priceOffset;
      copy.endPrice += priceOffset;
      copy.parallelPrice += priceOffset;
    } else if (copy.action && copy.action.includes('fibonacci-extension')) {
      // Fibonacci Extension has 3 points
      copy.point1Index += offset;
      copy.point2Index += offset;
      copy.point3Index += offset;
      const priceOffset = offset * (this.renderer.maxPrice - this.renderer.minPrice) / 100;
      copy.point1Price += priceOffset;
      copy.point2Price += priceOffset;
      copy.point3Price += priceOffset;
    } else if (copy.action && copy.action.includes('fibonacci-time-zones')) {
      // Fibonacci Time Zones only has chartIndex
      copy.chartIndex += offset;
    } else if (copy.action && copy.action.includes('fibonacci-')) {
      // Other Fibonacci tools (Retracement, Fan, Arcs, Spiral)
      copy.startIndex += offset;
      copy.endIndex += offset;
      const priceOffset = offset * (this.renderer.maxPrice - this.renderer.minPrice) / 100;
      copy.startPrice += priceOffset;
      copy.endPrice += priceOffset;
    } else if (copy.action && copy.action.includes('gann-')) {
      // Gann tools (Fan, Box, Square, Angles)
      copy.startIndex += offset;
      copy.endIndex += offset;
      const priceOffset = offset * (this.renderer.maxPrice - this.renderer.minPrice) / 100;
      copy.startPrice += priceOffset;
      copy.endPrice += priceOffset;
    }

    // Add to drawings array
    this.renderer.drawings.push(copy);

    // Save to backend
    this.renderer.saveDrawing(copy);

    // Select the new copy
    this.selectedDrawing = copy;

    // Redraw
    this.renderer.draw();

    console.log('✅ Drawing duplicated');
  }

  /**
   * Delete the selected drawing
   */
  deleteSelectedDrawing() {
    if (!this.selectedDrawing) return;

    const drawingId = this.selectedDrawing.id;
    const index = this.renderer.drawings.indexOf(this.selectedDrawing);
    if (index !== -1) {
      this.renderer.drawings.splice(index, 1);
      this.selectedDrawing = null;
      this.renderer.draw();

      // Delete from backend
      this.renderer.deleteDrawing(drawingId);
    }
  }

  /**
   * Handle mouse down - check for selection or handle dragging
   * Returns true if selection manager handled the event
   */
  onMouseDown(e, mouseX, mouseY) {
    // Store position for delta calculations
    this.lastMouseX = mouseX;
    this.lastMouseY = mouseY;

    // Get the active tool from ToolRegistry
    const activeTool = window.toolRegistry?.getActiveTool();

    // Check if clicking on any drawing
    const clickedDrawing = this.findDrawingAtPoint(mouseX, mouseY);

    // If no drawing clicked and we have a selection, deselect it
    if (!clickedDrawing && this.selectedDrawing) {
      this.selectedDrawing = null;
      this.renderer.draw();
      // Continue to let other tools handle the click
    }

    // Only handle selection/movement when default tool is active
    const isDefaultTool = !activeTool || activeTool.id === 'default' || activeTool.id === 'default-cursor';

    if (!isDefaultTool) {
      return false; // Let the active tool handle it
    }

    // Check if clicking on selected drawing's handle (only with default tool)
    if (this.selectedDrawing) {
      const handle = this.getHandleAtPoint(this.selectedDrawing, mouseX, mouseY);
      if (handle) {
        this.isDraggingHandle = true;
        this.draggedHandle = handle;
        return true; // Handled
      }
    }

    // Handle clicking on a drawing (only with default tool)
    if (clickedDrawing) {
      this.selectedDrawing = clickedDrawing;
      this.isMovingDrawing = true;

      // Calculate offset from drawing start point
      // Handle different coordinate structures
      let screenStart;
      if (clickedDrawing.startIndex !== undefined && clickedDrawing.startPrice !== undefined) {
        screenStart = this.drawingToScreen(clickedDrawing.startIndex, clickedDrawing.startPrice);
      } else if (clickedDrawing.chartIndex !== undefined && clickedDrawing.chartPrice !== undefined) {
        screenStart = this.drawingToScreen(clickedDrawing.chartIndex, clickedDrawing.chartPrice);
      } else if (clickedDrawing.point1Index !== undefined && clickedDrawing.point1Price !== undefined) {
        screenStart = this.drawingToScreen(clickedDrawing.point1Index, clickedDrawing.point1Price);
      } else {
        screenStart = { x: mouseX, y: mouseY }; // Fallback
      }

      this.dragOffset = {
        x: mouseX - screenStart.x,
        y: mouseY - screenStart.y
      };

      this.renderer.draw(); // Redraw with selection highlight
      return true; // Handled
    }

    return false; // Not handled
  }

  /**
   * Handle mouse move - move or resize selected drawing
   * Returns true if selection manager handled the event
   */
  onMouseMove(e, mouseX, mouseY) {
    // Update cursor style based on hover
    if (!this.isMovingDrawing && !this.isDraggingHandle) {
      this.updateCursor(mouseX, mouseY);
    }

    // Handle moving entire drawing
    if (this.isMovingDrawing && this.selectedDrawing) {
      const deltaX = mouseX - this.lastMouseX;
      const deltaY = mouseY - this.lastMouseY;

      // Convert pixel deltas to chart coordinate deltas
      const oldIndex = this.renderer.xToIndex(this.lastMouseX);
      const newIndex = this.renderer.xToIndex(mouseX);
      const deltaIndex = newIndex - oldIndex;

      const oldPrice = this.renderer.yToPrice(this.lastMouseY);
      const newPrice = this.renderer.yToPrice(mouseY);
      const deltaPrice = newPrice - oldPrice;

      // Update drawing coordinates based on type
      const drawing = this.selectedDrawing;
      if (drawing.action === 'place-dot' || drawing.action === 'place-arrow') {
        // Dots and arrows use single chartIndex/chartPrice
        drawing.chartIndex += deltaIndex;
        drawing.chartPrice += deltaPrice;

        // For arrows, update direction based on new position
        if (drawing.action === 'place-arrow' && this.renderer.data.length > 0) {
          const nearestIndex = Math.round(drawing.chartIndex);
          if (nearestIndex >= 0 && nearestIndex < this.renderer.data.length) {
            const candle = this.renderer.data[nearestIndex];
            const candleHigh = candle.High;
            const candleLow = candle.Low;

            // Update arrow direction based on position relative to candle
            if (drawing.chartPrice < candleLow) {
              drawing.direction = 'up';
              drawing.color = null; // Clear color to allow auto-coloring
            } else if (drawing.chartPrice > candleHigh) {
              drawing.direction = 'down';
              drawing.color = null; // Clear color to allow auto-coloring
            } else {
              // If inside the candle body, use proximity to decide
              const candleMid = (candleHigh + candleLow) / 2;
              drawing.direction = drawing.chartPrice < candleMid ? 'up' : 'down';
              drawing.color = null; // Clear color to allow auto-coloring
            }
          }
        }
      } else if (drawing.action && drawing.action.includes('horizontal-line')) {
        // Horizontal lines use price only
        drawing.price += deltaPrice;

        // Re-detect support/resistance at new position
        const srType = this.renderer.detectSupportResistance(drawing.price);
        if (srType === 'support') {
          drawing.lineColor = '#00c851'; // Green for support
        } else if (srType === 'resistance') {
          drawing.lineColor = '#ff4444'; // Red for resistance
        } else {
          drawing.lineColor = '#ff9800'; // Orange for neutral
        }
      } else if (drawing.action && drawing.action.includes('vertical-line')) {
        // Vertical lines use chartIndex only
        drawing.chartIndex += deltaIndex;
      } else if (drawing.action && drawing.action.includes('parallel-channel')) {
        // Parallel channel - move both lines together
        drawing.startIndex += deltaIndex;
        drawing.endIndex += deltaIndex;
        drawing.startPrice += deltaPrice;
        drawing.endPrice += deltaPrice;
        drawing.parallelPrice += deltaPrice;
      } else if (drawing.action && drawing.action.includes('fibonacci-extension')) {
        // Fibonacci Extension - move all 3 points
        drawing.point1Index += deltaIndex;
        drawing.point1Price += deltaPrice;
        drawing.point2Index += deltaIndex;
        drawing.point2Price += deltaPrice;
        drawing.point3Index += deltaIndex;
        drawing.point3Price += deltaPrice;
      } else if (drawing.action && drawing.action.includes('fibonacci-time-zones')) {
        // Fibonacci Time Zones - move start index only
        drawing.chartIndex += deltaIndex;
      } else if (drawing.action && drawing.action.includes('fibonacci-')) {
        // Other Fibonacci tools (Retracement, Fan, Arcs, Spiral) use startIndex/endIndex
        drawing.startIndex += deltaIndex;
        drawing.endIndex += deltaIndex;
        drawing.startPrice += deltaPrice;
        drawing.endPrice += deltaPrice;
      } else if (drawing.action && drawing.action.includes('gann-')) {
        // Gann tools (Fan, Box, Square, Angles) use startIndex/endIndex
        drawing.startIndex += deltaIndex;
        drawing.endIndex += deltaIndex;
        drawing.startPrice += deltaPrice;
        drawing.endPrice += deltaPrice;
      } else {
        // Trend lines, ray, extended line use startIndex/endIndex
        drawing.startIndex += deltaIndex;
        drawing.endIndex += deltaIndex;
        drawing.startPrice += deltaPrice;
        drawing.endPrice += deltaPrice;
      }

      this.renderer.draw(); // Redraw with updated position

      this.lastMouseX = mouseX;
      this.lastMouseY = mouseY;
      return true; // Handled
    }

    // Handle dragging handle to resize
    if (this.isDraggingHandle && this.selectedDrawing) {
      const newIndex = this.renderer.xToIndex(mouseX);
      const newPrice = this.renderer.yToPrice(mouseY);

      if (this.draggedHandle === 'start') {
        this.selectedDrawing.startIndex = newIndex;
        this.selectedDrawing.startPrice = newPrice;
      } else if (this.draggedHandle === 'end') {
        this.selectedDrawing.endIndex = newIndex;
        this.selectedDrawing.endPrice = newPrice;
      }

      this.renderer.draw(); // Redraw with updated size

      this.lastMouseX = mouseX;
      this.lastMouseY = mouseY;
      return true; // Handled
    }

    return false; // Not handled
  }

  /**
   * Handle mouse up - finish moving/resizing
   * Returns true if selection manager handled the event
   */
  onMouseUp(e, mouseX, mouseY) {
    if (this.isMovingDrawing || this.isDraggingHandle) {
      this.isMovingDrawing = false;
      this.isDraggingHandle = false;
      this.draggedHandle = null;

      // Save updated drawing to backend
      if (this.selectedDrawing) {
        this.renderer.saveDrawing(this.selectedDrawing);
      }

      return true; // Handled
    }

    return false; // Not handled
  }

  /**
   * Handle keyboard events (Delete key)
   */
  onKeyDown(e) {
    if (e.key === 'Delete' && this.selectedDrawing) {
      const drawingId = this.selectedDrawing.id;
      const index = this.renderer.drawings.indexOf(this.selectedDrawing);
      if (index !== -1) {
        this.renderer.drawings.splice(index, 1);
        this.selectedDrawing = null;
        this.renderer.draw();

        // Delete from backend
        this.renderer.deleteDrawing(drawingId);
      }

      e.preventDefault();
      e.stopPropagation();
    }
  }

  /**
   * Update cursor based on hover position
   */
  updateCursor(mouseX, mouseY) {
    const canvas = this.renderer.canvas;
    if (!canvas) return;

    // Check if hovering over selected drawing's handles
    if (this.selectedDrawing) {
      const handle = this.getHandleAtPoint(this.selectedDrawing, mouseX, mouseY);
      if (handle) {
        canvas.style.cursor = 'pointer';
        return;
      }
    }

    // Check if hovering over any drawing
    const drawing = this.findDrawingAtPoint(mouseX, mouseY);
    if (drawing) {
      canvas.style.cursor = 'move';
    } else {
      canvas.style.cursor = 'default';
    }
  }

  /**
   * Find drawing at given point (hit detection)
   */
  findDrawingAtPoint(x, y) {
    // Search in reverse order (most recent drawings first)
    for (let i = this.renderer.drawings.length - 1; i >= 0; i--) {
      const drawing = this.renderer.drawings[i];

      if (drawing.action === 'finish-trend-line') {
        const hit = this.isTrendLineHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action === 'place-dot') {
        const hit = this.isDotHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action === 'place-arrow') {
        const hit = this.isArrowHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('horizontal-line')) {
        const hit = this.isHorizontalLineHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('vertical-line')) {
        const hit = this.isVerticalLineHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('ray-line') || drawing.action.includes('extended-line')) {
        const hit = this.isTrendLineHit(drawing, x, y); // Ray and extended use same hit detection as trend line
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('parallel-channel')) {
        const hit = this.isParallelChannelHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('fibonacci-retracement')) {
        const hit = this.isFibonacciRetracementHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('fibonacci-extension')) {
        const hit = this.isFibonacciExtensionHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('fibonacci-fan')) {
        const hit = this.isFibonacciFanHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('fibonacci-arcs')) {
        const hit = this.isFibonacciArcsHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('fibonacci-time-zones')) {
        const hit = this.isFibonacciTimeZonesHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('fibonacci-spiral')) {
        const hit = this.isFibonacciSpiralHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('gann-fan')) {
        const hit = this.isGannFanHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('gann-box')) {
        const hit = this.isGannBoxHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('gann-square')) {
        const hit = this.isGannSquareHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      } else if (drawing.action.includes('gann-angles')) {
        const hit = this.isGannAnglesHit(drawing, x, y);
        if (hit) {
          return drawing;
        }
      }
      // TODO: Add hit detection for other drawing types (Patterns, Shapes, etc.)
    }

    return null;
  }

  /**
   * Check if trend line is hit by point
   */
  isTrendLineHit(drawing, x, y) {
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    const distance = this.pointToLineDistance(x, y, start.x, start.y, end.x, end.y);
    return distance < this.hitThreshold;
  }

  /**
   * Check if dot is hit by point
   */
  isDotHit(drawing, x, y) {
    const pos = this.drawingToScreen(drawing.chartIndex, drawing.chartPrice);
    const size = drawing.size || 6;
    const distance = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
    return distance < size + this.hitThreshold;
  }

  /**
   * Check if arrow is hit by point
   */
  isArrowHit(drawing, x, y) {
    const pos = this.drawingToScreen(drawing.chartIndex, drawing.chartPrice);
    const size = drawing.size || 20;
    // Use a rectangular hit box around the arrow
    const halfSize = size / 2 + this.hitThreshold;
    return Math.abs(x - pos.x) < halfSize && Math.abs(y - pos.y) < halfSize;
  }

  /**
   * Check if horizontal line is hit by point
   */
  isHorizontalLineHit(drawing, x, y) {
    const lineY = this.renderer.priceToY(drawing.price);
    // Check if y is within hit threshold of the line
    return Math.abs(y - lineY) < this.hitThreshold;
  }

  /**
   * Check if vertical line is hit by point
   */
  isVerticalLineHit(drawing, x, y) {
    const lineX = this.renderer.indexToX(drawing.chartIndex);
    // Check if x is within hit threshold of the line
    return Math.abs(x - lineX) < this.hitThreshold;
  }

  /**
   * Check if parallel channel is hit by point
   */
  isParallelChannelHit(drawing, x, y) {
    // Check if hit on either of the two parallel lines
    const line1Start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const line1End = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    const distance1 = this.pointToLineDistance(x, y, line1Start.x, line1Start.y, line1End.x, line1End.y);
    if (distance1 < this.hitThreshold) return true;

    // Check second parallel line
    const parallelY = this.renderer.priceToY(drawing.parallelPrice);
    const offset = parallelY - line1Start.y;
    const distance2 = this.pointToLineDistance(x, y, line1Start.x, line1Start.y + offset, line1End.x, line1End.y + offset);
    return distance2 < this.hitThreshold;
  }

  /**
   * Check if Fibonacci Retracement is hit
   */
  isFibonacciRetracementHit(drawing, x, y) {
    // Check main diagonal line
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);
    const distance = this.pointToLineDistance(x, y, start.x, start.y, end.x, end.y);
    if (distance < this.hitThreshold) return true;

    // Check any horizontal level lines
    const priceRange = drawing.endPrice - drawing.startPrice;
    for (const level of drawing.levels) {
      const levelPrice = drawing.startPrice + priceRange * level;
      const levelY = this.renderer.priceToY(levelPrice);
      if (Math.abs(y - levelY) < this.hitThreshold) return true;
    }
    return false;
  }

  /**
   * Check if Fibonacci Extension is hit
   */
  isFibonacciExtensionHit(drawing, x, y) {
    // Check connecting lines P1→P2→P3
    const p1 = this.drawingToScreen(drawing.point1Index, drawing.point1Price);
    const p2 = this.drawingToScreen(drawing.point2Index, drawing.point2Price);
    const p3 = this.drawingToScreen(drawing.point3Index, drawing.point3Price);

    const distance1 = this.pointToLineDistance(x, y, p1.x, p1.y, p2.x, p2.y);
    if (distance1 < this.hitThreshold) return true;

    const distance2 = this.pointToLineDistance(x, y, p2.x, p2.y, p3.x, p3.y);
    if (distance2 < this.hitThreshold) return true;

    // Check extension level lines
    const swingPrice = drawing.point2Price - drawing.point1Price;
    for (const level of drawing.levels) {
      const levelPrice = drawing.point3Price + swingPrice * level;
      const levelY = this.renderer.priceToY(levelPrice);
      if (Math.abs(y - levelY) < this.hitThreshold && x >= p3.x) return true;
    }
    return false;
  }

  /**
   * Check if Fibonacci Fan is hit
   */
  isFibonacciFanHit(drawing, x, y) {
    // Check main line
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);
    const distance = this.pointToLineDistance(x, y, start.x, start.y, end.x, end.y);
    if (distance < this.hitThreshold) return true;

    // Check fan lines
    const priceRange = drawing.endPrice - drawing.startPrice;
    const chartRight = this.renderer.width - this.renderer.margin.right;
    for (const level of drawing.levels) {
      const fanPrice = drawing.startPrice + priceRange * level;
      const fanY = this.renderer.priceToY(fanPrice);
      const slope = (fanY - start.y) / (end.x - start.x);
      const extendedY = start.y + slope * (chartRight - start.x);
      const fanDistance = this.pointToLineDistance(x, y, start.x, start.y, chartRight, extendedY);
      if (fanDistance < this.hitThreshold) return true;
    }
    return false;
  }

  /**
   * Check if Fibonacci Arcs is hit
   */
  isFibonacciArcsHit(drawing, x, y) {
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const radius = Math.sqrt(dx * dx + dy * dy);

    // Check if point is near any arc
    const pointDistance = Math.sqrt((x - start.x) ** 2 + (y - start.y) ** 2);
    for (const level of drawing.levels) {
      const arcRadius = radius * level;
      if (Math.abs(pointDistance - arcRadius) < this.hitThreshold) return true;
    }
    return false;
  }

  /**
   * Check if Fibonacci Time Zones is hit
   */
  isFibonacciTimeZonesHit(drawing, x, y) {
    // Check if near any vertical time zone line
    for (const fib of drawing.sequence) {
      const fibIndex = drawing.chartIndex + fib;
      const lineX = this.renderer.indexToX(fibIndex);
      if (Math.abs(x - lineX) < this.hitThreshold) return true;
    }
    return false;
  }

  /**
   * Check if Fibonacci Spiral is hit
   */
  isFibonacciSpiralHit(drawing, x, y) {
    // Check main rectangle
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    const left = Math.min(start.x, end.x);
    const right = Math.max(start.x, end.x);
    const top = Math.min(start.y, end.y);
    const bottom = Math.max(start.y, end.y);

    // Check if near rectangle edges
    if (x >= left && x <= right && (Math.abs(y - top) < this.hitThreshold || Math.abs(y - bottom) < this.hitThreshold)) return true;
    if (y >= top && y <= bottom && (Math.abs(x - left) < this.hitThreshold || Math.abs(x - right) < this.hitThreshold)) return true;

    return false;
  }

  /**
   * Check if Gann Fan is hit
   */
  isGannFanHit(drawing, x, y) {
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);
    const trendDirection = end.y < start.y ? -1 : 1;
    const chartRight = this.renderer.width - this.renderer.margin.right;

    // Check each Gann angle line
    for (const [name, ratio] of Object.entries(drawing.angles)) {
      const dx = chartRight - start.x;
      const dy = dx * ratio * trendDirection;
      const distance = this.pointToLineDistance(x, y, start.x, start.y, start.x + dx, start.y + dy);
      if (distance < this.hitThreshold) return true;
    }
    return false;
  }

  /**
   * Check if Gann Box is hit
   */
  isGannBoxHit(drawing, x, y) {
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    const left = Math.min(start.x, end.x);
    const right = Math.max(start.x, end.x);
    const top = Math.min(start.y, end.y);
    const bottom = Math.max(start.y, end.y);

    // Check box outline
    if (x >= left && x <= right && (Math.abs(y - top) < this.hitThreshold || Math.abs(y - bottom) < this.hitThreshold)) return true;
    if (y >= top && y <= bottom && (Math.abs(x - left) < this.hitThreshold || Math.abs(x - right) < this.hitThreshold)) return true;

    // Check quarter divisions if enabled
    if (drawing.showQuarters) {
      const midX = (left + right) / 2;
      const midY = (top + bottom) / 2;
      if (x >= left && x <= right && Math.abs(y - midY) < this.hitThreshold) return true;
      if (y >= top && y <= bottom && Math.abs(x - midX) < this.hitThreshold) return true;
    }

    // Check diagonals if enabled
    if (drawing.showDiagonals) {
      const dist1 = this.pointToLineDistance(x, y, left, top, right, bottom);
      const dist2 = this.pointToLineDistance(x, y, right, top, left, bottom);
      if (dist1 < this.hitThreshold || dist2 < this.hitThreshold) return true;
    }

    return false;
  }

  /**
   * Check if Gann Square is hit
   */
  isGannSquareHit(drawing, x, y) {
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    const width = end.x - start.x;
    const height = end.y - start.y;
    const size = Math.min(Math.abs(width), Math.abs(height));

    const left = start.x;
    const right = start.x + size;
    const top = start.y;
    const bottom = start.y + size;

    // Check square outline
    if (x >= left && x <= right && (Math.abs(y - top) < this.hitThreshold || Math.abs(y - bottom) < this.hitThreshold)) return true;
    if (y >= top && y <= bottom && (Math.abs(x - left) < this.hitThreshold || Math.abs(x - right) < this.hitThreshold)) return true;

    // Check grid divisions
    const cellSize = size / drawing.divisions;
    for (let i = 1; i < drawing.divisions; i++) {
      // Check vertical lines
      const lineX = start.x + i * cellSize;
      if (y >= top && y <= bottom && Math.abs(x - lineX) < this.hitThreshold) return true;

      // Check horizontal lines
      const lineY = start.y + i * cellSize;
      if (x >= left && x <= right && Math.abs(y - lineY) < this.hitThreshold) return true;
    }

    return false;
  }

  /**
   * Check if Gann Angles is hit
   */
  isGannAnglesHit(drawing, x, y) {
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    // Calculate angle based on type
    const angleRatios = {
      '1x8': 1/8, '1x4': 1/4, '1x3': 1/3, '1x2': 1/2,
      '1x1': 1, '2x1': 2, '3x1': 3, '4x1': 4, '8x1': 8
    };

    const ratio = angleRatios[drawing.angleType] || 1;
    const direction = end.y < start.y ? -1 : 1;
    const chartRight = this.renderer.width - this.renderer.margin.right;
    const dx = chartRight - start.x;
    const dy = dx * ratio * direction;

    const distance = this.pointToLineDistance(x, y, start.x, start.y, start.x + dx, start.y + dy);
    return distance < this.hitThreshold;
  }

  /**
   * Get handle at point (for resizing)
   */
  getHandleAtPoint(drawing, x, y) {
    if (!drawing) return null;

    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    const distStart = Math.sqrt((x - start.x) ** 2 + (y - start.y) ** 2);
    const distEnd = Math.sqrt((x - end.x) ** 2 + (y - end.y) ** 2);

    if (distStart < this.handleRadius * 2) return 'start';
    if (distEnd < this.handleRadius * 2) return 'end';

    return null;
  }

  /**
   * Convert drawing coordinates to screen coordinates
   */
  drawingToScreen(dataIndex, price) {
    const x = this.renderer.indexToX(dataIndex);
    const y = this.renderer.priceToY(price);
    return { x, y };
  }

  /**
   * Calculate distance from point to line segment
   */
  pointToLineDistance(px, py, x1, y1, x2, y2) {
    const A = px - x1;
    const B = py - y1;
    const C = x2 - x1;
    const D = y2 - y1;

    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    let param = -1;

    if (lenSq !== 0) param = dot / lenSq;

    let xx, yy;
    if (param < 0) {
      xx = x1;
      yy = y1;
    } else if (param > 1) {
      xx = x2;
      yy = y2;
    } else {
      xx = x1 + param * C;
      yy = y1 + param * D;
    }

    const dx = px - xx;
    const dy = py - yy;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Draw selection highlight and handles
   * Called by canvas-renderer during draw cycle
   */
  drawSelection() {
    if (!this.selectedDrawing) return;

    const ctx = this.renderer.ctx;

    const drawing = this.selectedDrawing;

    if (drawing.action === 'finish-trend-line') {
      this.drawTrendLineSelection(ctx, drawing);
    } else if (drawing.action === 'place-dot') {
      this.drawDotSelection(ctx, drawing);
    } else if (drawing.action === 'place-arrow') {
      this.drawArrowSelection(ctx, drawing);
    } else if (drawing.action.includes('horizontal-line')) {
      this.drawHorizontalLineSelection(ctx, drawing);
    } else if (drawing.action.includes('vertical-line')) {
      this.drawVerticalLineSelection(ctx, drawing);
    } else if (drawing.action.includes('ray-line') || drawing.action.includes('extended-line')) {
      this.drawTrendLineSelection(ctx, drawing); // Ray and extended use same selection as trend line
    } else if (drawing.action.includes('parallel-channel')) {
      this.drawParallelChannelSelection(ctx, drawing);
    }
    // TODO: Add selection drawing for other types
  }

  /**
   * Draw selection highlight for trend line
   */
  drawTrendLineSelection(ctx, drawing) {
    const start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const end = this.drawingToScreen(drawing.endIndex, drawing.endPrice);

    ctx.save();

    // Draw highlight outline (thicker, yellow)
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = (drawing.lineWidth || 2) + 4;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();

    // Draw drag handles at endpoints with arrows
    ctx.fillStyle = this.selectionColor;
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = 2;

    // Calculate line angle for arrow direction
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    const arrowSize = 12;

    // Start handle with arrow pointing along the line
    this.drawArrowHandle(ctx, start.x, start.y, angle, arrowSize);

    // End handle with arrow pointing along the line
    this.drawArrowHandle(ctx, end.x, end.y, angle, arrowSize);

    ctx.restore();
  }

  /**
   * Draw an arrow-shaped handle at a point
   */
  drawArrowHandle(ctx, x, y, angle, size) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    // Draw circle at the center first (as base)
    ctx.fillStyle = this.selectionColor;
    ctx.beginPath();
    ctx.arc(0, 0, this.handleRadius, 0, Math.PI * 2);
    ctx.fill();

    // Draw arrow triangle on top, pointing right (direction of line)
    ctx.fillStyle = '#000000'; // Black arrow for contrast
    ctx.beginPath();
    ctx.moveTo(size, 0); // Arrow tip
    ctx.lineTo(-size / 3, -size / 2); // Top wing
    ctx.lineTo(-size / 3, size / 2); // Bottom wing
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  /**
   * Draw selection highlight for dot
   */
  drawDotSelection(ctx, drawing) {
    const pos = this.drawingToScreen(drawing.chartIndex, drawing.chartPrice);
    const size = drawing.size || 6;

    ctx.save();

    // Draw yellow outline around dot
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, size + 5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Draw selection highlight for arrow
   */
  drawArrowSelection(ctx, drawing) {
    const pos = this.drawingToScreen(drawing.chartIndex, drawing.chartPrice);
    const size = drawing.size || 20;

    ctx.save();

    // Draw yellow outline box around arrow
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, size * 0.7, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Draw selection highlight for horizontal line
   */
  drawHorizontalLineSelection(ctx, drawing) {
    const y = this.renderer.priceToY(drawing.price);
    const chartLeft = this.renderer.margin.left;
    const chartRight = this.renderer.width - this.renderer.margin.right;

    ctx.save();

    // Draw yellow highlight outline (thicker)
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = (drawing.lineWidth || 2) + 4;
    ctx.beginPath();
    ctx.moveTo(chartLeft, y);
    ctx.lineTo(chartRight, y);
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Draw selection highlight for vertical line
   */
  drawVerticalLineSelection(ctx, drawing) {
    const x = this.renderer.indexToX(drawing.chartIndex);
    const chartTop = this.renderer.margin.top;
    const chartBottom = this.renderer.height - this.renderer.margin.bottom;

    ctx.save();

    // Draw yellow highlight outline (thicker)
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = (drawing.lineWidth || 2) + 4;
    ctx.beginPath();
    ctx.moveTo(x, chartTop);
    ctx.lineTo(x, chartBottom);
    ctx.stroke();

    ctx.restore();

    // Draw tooltip with date and stock info
    this.drawVerticalLineTooltip(ctx, drawing, x, chartTop);
  }

  /**
   * Draw tooltip for vertical line showing date and OHLCV data
   */
  drawVerticalLineTooltip(ctx, drawing, x, chartTop) {
    const nearestIndex = Math.round(drawing.chartIndex);

    // Check if we have valid data at this index
    if (nearestIndex < 0 || nearestIndex >= this.renderer.data.length) {
      return; // Out of range
    }

    const candle = this.renderer.data[nearestIndex];
    if (!candle) return;

    // Format the tooltip text
    const date = candle.Date || 'N/A';
    const open = candle.Open?.toFixed(2) || 'N/A';
    const high = candle.High?.toFixed(2) || 'N/A';
    const low = candle.Low?.toFixed(2) || 'N/A';
    const close = candle.Close?.toFixed(2) || 'N/A';
    const volume = candle.Volume ? this.formatVolume(candle.Volume) : 'N/A';

    const lines = [
      `Date: ${date}`,
      `Open: $${open}`,
      `High: $${high}`,
      `Low: $${low}`,
      `Close: $${close}`,
      `Volume: ${volume}`
    ];

    // Measure text to determine tooltip size
    ctx.font = '12px monospace';
    const lineHeight = 16;
    const padding = 8;
    const maxWidth = Math.max(...lines.map(line => ctx.measureText(line).width));
    const tooltipWidth = maxWidth + padding * 2;
    const tooltipHeight = lines.length * lineHeight + padding * 2;

    // Position tooltip to the right of the line
    const offset = 15; // Pixels to the right of the line
    let tooltipX = x + offset;
    let tooltipY = chartTop + 10;

    // Keep tooltip within canvas bounds
    const canvasWidth = this.renderer.width;
    const chartRight = this.renderer.width - this.renderer.margin.right;

    // If tooltip would go off right edge, position it to the left instead
    if (tooltipX + tooltipWidth > chartRight - 10) {
      tooltipX = x - tooltipWidth - offset;
    }

    // Ensure it doesn't go off left edge either
    if (tooltipX < this.renderer.margin.left + 10) {
      tooltipX = this.renderer.margin.left + 10;
    }

    ctx.save();

    // Draw tooltip background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = 2;
    this.roundRect(ctx, tooltipX, tooltipY, tooltipWidth, tooltipHeight, 6);
    ctx.fill();
    ctx.stroke();

    // Draw tooltip text
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    lines.forEach((line, i) => {
      ctx.fillText(line, tooltipX + padding, tooltipY + padding + i * lineHeight);
    });

    ctx.restore();
  }

  /**
   * Format volume for display (e.g., 1.2M, 3.4K)
   */
  formatVolume(volume) {
    if (volume >= 1000000000) {
      return (volume / 1000000000).toFixed(1) + 'B';
    } else if (volume >= 1000000) {
      return (volume / 1000000).toFixed(1) + 'M';
    } else if (volume >= 1000) {
      return (volume / 1000).toFixed(1) + 'K';
    } else {
      return volume.toString();
    }
  }

  /**
   * Draw a rounded rectangle
   */
  roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  /**
   * Draw selection highlight for parallel channel
   */
  drawParallelChannelSelection(ctx, drawing) {
    const line1Start = this.drawingToScreen(drawing.startIndex, drawing.startPrice);
    const line1End = this.drawingToScreen(drawing.endIndex, drawing.endPrice);
    const parallelY = this.renderer.priceToY(drawing.parallelPrice);
    const offset = parallelY - line1Start.y;

    ctx.save();

    // Draw yellow highlight on both lines (thicker)
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = (drawing.lineWidth || 2) + 4;

    // Line 1
    ctx.beginPath();
    ctx.moveTo(line1Start.x, line1Start.y);
    ctx.lineTo(line1End.x, line1End.y);
    ctx.stroke();

    // Line 2 (parallel)
    ctx.beginPath();
    ctx.moveTo(line1Start.x, line1Start.y + offset);
    ctx.lineTo(line1End.x, line1End.y + offset);
    ctx.stroke();

    // Draw drag handles at endpoints
    ctx.fillStyle = this.selectionColor;
    ctx.strokeStyle = this.selectionColor;
    ctx.lineWidth = 2;

    // Calculate line angle for arrow direction
    const angle = Math.atan2(line1End.y - line1Start.y, line1End.x - line1Start.x);
    const arrowSize = 12;

    // Handles on line 1
    this.drawArrowHandle(ctx, line1Start.x, line1Start.y, angle, arrowSize);
    this.drawArrowHandle(ctx, line1End.x, line1End.y, angle, arrowSize);

    ctx.restore();
  }

  /**
   * Clear selection
   */
  clearSelection() {
    if (this.selectedDrawing) {
      this.selectedDrawing = null;
      this.renderer.draw();
    }
  }

  /**
   * Get selected drawing
   */
  getSelectedDrawing() {
    return this.selectedDrawing;
  }
}

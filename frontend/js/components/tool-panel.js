/**
 * Drawing Tools Panel Component
 * Manages the collapsible tool panel with cursor and drawing tools
 */

import { ToolRegistry } from '../tools/ToolRegistry.js';
import {
  DefaultCursor,
  CrosshairCursor,
  EraserCursor,
  DotCursor,
  ArrowCursor
} from '../tools/cursors/index.js';
import {
  TrendLine,
  HorizontalLine,
  VerticalLine,
  RayLine,
  ExtendedLine,
  ParallelChannel
} from '../tools/trend-lines/index.js';
import {
  FibonacciRetracement,
  FibonacciExtension,
  FibonacciFan,
  FibonacciArcs,
  FibonacciTimeZones,
  FibonacciSpiral
} from '../tools/fibonacci/index.js';
import {
  GannFan,
  GannBox,
  GannSquare,
  GannAngles
} from '../tools/gann/index.js';
import {
  HeadAndShoulders,
  Triangle,
  Wedge,
  DoubleTopBottom
} from '../tools/patterns/index.js';
import {
  Rectangle,
  Circle,
  Ellipse,
  Polygon
} from '../tools/shapes/index.js';
import {
  TextLabel,
  Callout,
  Note,
  PriceLabel
} from '../tools/annotations/index.js';

export class ToolPanel {
  constructor(containerElement) {
    this.container = containerElement;
    this.canvas = null; // Will be set when canvas is available
    this.registry = new ToolRegistry();

    // Expose registry globally so SelectionManager can access it
    window.toolRegistry = this.registry;
    // Also expose this panel instance for persistent mode access
    this.registry.toolPanel = this;

    this.isCollapsed = false;
    this.panelElement = null;
    this.currentTool = null;
    this.persistentMode = false; // Persistent tool selection mode

    this.initializeTools();
    this.initializeUI();
    this.loadState();
    this.findCanvas(); // Try to find canvas in container
  }

  /**
   * Find or wait for canvas element in the container
   */
  findCanvas() {
    if (!this.container) {
      console.warn('⚠️ Tool panel: No container element');
      return;
    }

    // Try to find canvas directly
    this.canvas = this.container.querySelector('canvas');

    if (this.canvas) {
      this.attachCanvasListeners();
      this.setActiveTool('default'); // Activate default cursor now that canvas exists
      this.updateActiveToolItem('default');
      // console.log('✅ Tool panel found canvas element');
    } else {
      // console.log('⏳ Tool panel waiting for canvas to be created...');
      // Wait for canvas to be created (use MutationObserver)
      const observer = new MutationObserver(() => {
        this.canvas = this.container.querySelector('canvas');
        if (this.canvas) {
          this.attachCanvasListeners();
          this.setActiveTool('default'); // Activate default cursor now that canvas exists
          this.updateActiveToolItem('default');
          console.log('✅ Tool panel connected to canvas');
          observer.disconnect();
        }
      });

      observer.observe(this.container, {
        childList: true,
        subtree: true
      });
    }
  }

  /**
   * Initialize all drawing tools
   */
  initializeTools() {
    // Create cursor tools
    this.tools = {
      // Cursors
      default: new DefaultCursor(),
      crosshair: new CrosshairCursor(),
      eraser: new EraserCursor(),
      dot: new DotCursor(),
      arrow: new ArrowCursor(),

      // Trend Lines
      'trend-line': new TrendLine(),
      'horizontal-line': new HorizontalLine(),
      'vertical-line': new VerticalLine(),
      'ray-line': new RayLine(),
      'extended-line': new ExtendedLine(),
      'parallel-channel': new ParallelChannel(),

      // Fibonacci Tools
      'fibonacci-retracement': new FibonacciRetracement(),
      'fibonacci-extension': new FibonacciExtension(),
      'fibonacci-fan': new FibonacciFan(),
      'fibonacci-arcs': new FibonacciArcs(),
      'fibonacci-time-zones': new FibonacciTimeZones(),
      'fibonacci-spiral': new FibonacciSpiral(),

      // Gann Tools
      'gann-fan': new GannFan(),
      'gann-box': new GannBox(),
      'gann-square': new GannSquare(),
      'gann-angles': new GannAngles(),

      // Pattern Tools
      'head-and-shoulders': new HeadAndShoulders(),
      'triangle': new Triangle(),
      'wedge': new Wedge(),
      'double-top-bottom': new DoubleTopBottom(),

      // Geometric Shapes
      'rectangle': new Rectangle(),
      'circle': new Circle(),
      'ellipse': new Ellipse(),
      'polygon': new Polygon(),

      // Annotations
      'text-label': new TextLabel(),
      'callout': new Callout(),
      'note': new Note(),
      'price-label': new PriceLabel()
    };

    // Register all tools
    Object.values(this.tools).forEach(tool => {
      this.registry.register(tool.category, tool);
    });

    // Don't activate default cursor yet - wait for canvas to be available
  }

  /**
   * Create the UI elements for the tool panel
   */
  initializeUI() {
    // Create main panel container
    this.panelElement = document.createElement('div');
    this.panelElement.className = 'tos-tools-panel';
    this.panelElement.innerHTML = `
      <div class="tos-tools-header">
        <div class="tos-tools-title">TOOLS</div>
        <label class="tos-persistent-toggle" title="Keep tool active after use (ESC to deactivate)">
          <input type="checkbox" id="persistent-mode-toggle">
          <span class="toggle-label">Keep Active</span>
        </label>
      </div>

      <div class="tos-tools-list">
        <!-- Cursors Category -->
        <button class="tos-tool-category-btn" data-category="cursors" data-tooltip="Cursors">
          <svg viewBox="0 0 24 24">
            <path d="M10.07,14.27C10.57,14.03 11.16,14.25 11.4,14.75L13.7,19.74L15.5,18.89L13.19,13.91C12.95,13.41 13.17,12.81 13.67,12.58L13.95,12.5L16.25,12.05L8,5.12V15.9L9.82,14.43L10.07,14.27M13.64,21.97C13.14,22.21 12.54,22 12.31,21.5L10.13,16.76L7.62,18.78C7.45,18.92 7.24,19 7,19A1,1 0 0,1 6,18V3A1,1 0 0,1 7,2C7.24,2 7.47,2.09 7.64,2.23L7.65,2.22L19.14,11.86C19.57,12.22 19.62,12.85 19.27,13.27C19.12,13.45 18.91,13.57 18.7,13.61L15.54,14.23L17.74,18.96C18,19.46 17.76,20.05 17.26,20.28L13.64,21.97Z"/>
          </svg>
        </button>

        <!-- Trend Lines Category -->
        <button class="tos-tool-category-btn" data-category="trendLines" data-tooltip="Trend Lines">
          <svg viewBox="0 0 24 24">
            <path d="M3.5,18.5L9.5,12.5L13.5,16.5L22,6.92L20.59,5.5L13.5,13.5L9.5,9.5L2,17L3.5,18.5Z"/>
          </svg>
        </button>

        <!-- Fibonacci Category -->
        <button class="tos-tool-category-btn" data-category="fibonacci" data-tooltip="Fibonacci">
          <svg viewBox="0 0 24 24">
            <path d="M2,2V6H4V4H6V2H2M7,2V4H9V6H11V8H13V6H15V4H17V2H7M18,2V6H20V4H22V2H18M2,7V11H4V9H6V7H2M18,7V9H20V11H22V7H18M2,12V16H4V14H6V12H2M7,12V14H9V16H11V18H13V16H15V14H17V12H7M18,12V14H20V16H22V12H18M2,17V21H4V19H6V17H2M18,17V19H20V21H22V17H18Z"/>
          </svg>
        </button>

        <!-- Gann Category -->
        <button class="tos-tool-category-btn" data-category="gann" data-tooltip="Gann">
          <svg viewBox="0 0 24 24">
            <path d="M2,2V22H22V2H2M20,20H4V4H20V20M6,6V18H18V6H6M8,8H16V16H8V8Z"/>
          </svg>
        </button>

        <!-- Patterns Category -->
        <button class="tos-tool-category-btn" data-category="patterns" data-tooltip="Patterns">
          <svg viewBox="0 0 24 24">
            <path d="M12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,10A2,2 0 0,0 10,12A2,2 0 0,0 12,14A2,2 0 0,0 14,12A2,2 0 0,0 12,10Z"/>
          </svg>
        </button>

        <!-- Shapes Category -->
        <button class="tos-tool-category-btn" data-category="shapes" data-tooltip="Shapes">
          <svg viewBox="0 0 24 24">
            <path d="M19,19H5V5H19M19,3H5A2,2 0 0,0 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V5A2,2 0 0,0 19,3Z"/>
          </svg>
        </button>

        <!-- Annotations Category -->
        <button class="tos-tool-category-btn" data-category="annotations" data-tooltip="Annotations">
          <svg viewBox="0 0 24 24">
            <path d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M6,9H18V11H6M14,14H6V12H14M18,8H6V6H18"/>
          </svg>
        </button>
      </div>

      <!-- Collapse/Expand Button -->
      <button class="tos-tools-collapse-btn" title="Collapse/Expand Tools Panel">
        <svg viewBox="0 0 24 24">
          <path d="M15.41,16.58L10.83,12L15.41,7.41L14,6L8,12L14,18L15.41,16.58Z"/>
        </svg>
      </button>
    `;

    // Insert panel after the left panel (watchlist/news)
    const leftPanel = document.querySelector('.tos-left-panel');
    if (leftPanel && leftPanel.parentElement) {
      leftPanel.parentElement.insertBefore(this.panelElement, leftPanel.nextSibling);
    }

    // Create dropdown panels for each category
    this.createDropdowns();

    // Attach event listeners
    this.attachEventListeners();
  }

  /**
   * Create dropdown panels for each tool category
   */
  createDropdowns() {
    const categories = {
      cursors: {
        title: 'Cursors',
        tools: [
          { id: 'default', name: 'Default', icon: 'M10.07,14.27C10.57,14.03 11.16,14.25 11.4,14.75L13.7,19.74L15.5,18.89L13.19,13.91C12.95,13.41 13.17,12.81 13.67,12.58L13.95,12.5L16.25,12.05L8,5.12V15.9L9.82,14.43L10.07,14.27M13.64,21.97C13.14,22.21 12.54,22 12.31,21.5L10.13,16.76L7.62,18.78C7.45,18.92 7.24,19 7,19A1,1 0 0,1 6,18V3A1,1 0 0,1 7,2C7.24,2 7.47,2.09 7.64,2.23L7.65,2.22L19.14,11.86C19.57,12.22 19.62,12.85 19.27,13.27C19.12,13.45 18.91,13.57 18.7,13.61L15.54,14.23L17.74,18.96C18,19.46 17.76,20.05 17.26,20.28L13.64,21.97Z' },
          { id: 'crosshair', name: 'Crosshair', icon: 'M3,11H7A5,5 0 0,1 12,6V2H13V6A5,5 0 0,1 18,11H22V12H18A5,5 0 0,1 13,17V21H12V17A5,5 0 0,1 7,12H3V11M12,8A4,4 0 0,0 8,12H11V13H8A4,4 0 0,0 12,17V14H13V17A4,4 0 0,0 17,13H14V12H17A4,4 0 0,0 13,8V11H12V8Z' },
          { id: 'eraser', name: 'Eraser', icon: 'M16.24,3.56L21.19,8.5C21.97,9.29 21.97,10.55 21.19,11.34L12,20.53C10.44,22.09 7.91,22.09 6.34,20.53L2.81,17C2.03,16.21 2.03,14.95 2.81,14.16L13.41,3.56C14.2,2.78 15.46,2.78 16.24,3.56M4.22,15.58L7.76,19.11C8.54,19.9 9.8,19.9 10.59,19.11L14.12,15.58L9.17,10.63L4.22,15.58Z' },
          { id: 'dot', name: 'Dot', icon: 'M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2Z' },
          { id: 'arrow', name: 'Arrow', icon: 'M12,2L7,12H10V22H14V12H17L12,2Z' }
        ]
      },
      trendLines: {
        title: 'Trend Lines',
        tools: [
          { id: 'trend-line', name: 'Trend Line', icon: 'M3.5,18.5L9.5,12.5L13.5,16.5L22,6.92L20.59,5.5L13.5,13.5L9.5,9.5L2,17L3.5,18.5Z' },
          { id: 'horizontal-line', name: 'Horizontal Line', icon: 'M2,12H22V14H2V12Z' },
          { id: 'vertical-line', name: 'Vertical Line', icon: 'M12,2H14V22H12V2Z' },
          { id: 'ray-line', name: 'Ray', icon: 'M2,12L12,2V8H22V16H12V22L2,12Z' },
          { id: 'extended-line', name: 'Extended Line', icon: 'M2,12L7,7V10H17V7L22,12L17,17V14H7V17L2,12Z' },
          { id: 'parallel-channel', name: 'Parallel Channel', icon: 'M3,4L9,4L22,18L16,18L3,4M3,8L9,14L16,14L22,20L16,20L3,8Z' }
        ]
      },
      fibonacci: {
        title: 'Fibonacci',
        tools: [
          { id: 'fibonacci-retracement', name: 'Fib Retracement', icon: 'M3,21V19H7V17H3V15H9V13H3V11H9V9H3V7H9V5H3V3H11V5H13V3H15V7H13V9H15V13H13V15H15V17H13V19H15V21H13V19H11V21H9V17H11V15H9V13H11V11H9V9H11V7H9V5H11V7H13V11H11V13H13V15H11V17H13V21H11M15,9V7H17V9H15M15,15V13H17V15H15M17,11V9H19V11H17M17,17V15H19V17H17M19,13V11H21V13H19M19,19V17H21V19H19Z' },
          { id: 'fibonacci-extension', name: 'Fib Extension', icon: 'M2,2V4H7V8H2V10H9V14H2V16H9V20H2V22H11V16H13V22H15V20H13V16H15V10H13V8H15V4H13V2H11V8H9V2H7V4H9V8H7V14H9V16H7V20H9V22H7V16H2M11,10V14H15V16H17V14H15V10H17V8H15V10H11M17,16V20H22V16H20V18H19V16H17M19,2V8H22V6H20V4H22V2H19Z' },
          { id: 'fibonacci-fan', name: 'Fib Fan', icon: 'M2,2V22L22,2H2M4,4H8.58L4,8.58V4M4,11.41L13.41,2H16.58L4,14.58V11.41M4,17.41L19.41,2H22V4.58L4,22V17.41Z' },
          { id: 'fibonacci-arcs', name: 'Fib Arcs', icon: 'M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22A10,10 0 0,1 2,12A10,10 0 0,1 12,2M12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4M12,6A6,6 0 0,1 18,12A6,6 0 0,1 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6M12,8A4,4 0 0,0 8,12A4,4 0 0,0 12,16A4,4 0 0,0 16,12A4,4 0 0,0 12,8Z' },
          { id: 'fibonacci-time-zones', name: 'Fib Time Zones', icon: 'M12,20A8,8 0 0,0 20,12A8,8 0 0,0 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20M12,2A10,10 0 0,1 22,12A10,10 0 0,1 12,22C6.47,22 2,17.5 2,12A10,10 0 0,1 12,2M12.5,7V12.25L17,14.92L16.25,16.15L11,13V7H12.5Z' },
          { id: 'fibonacci-spiral', name: 'Fib Spiral', icon: 'M6.5,2C5.67,2 5,2.67 5,3.5C5,4.33 5.67,5 6.5,5C7.33,5 8,4.33 8,3.5C8,2.67 7.33,2 6.5,2M9,5V7H11V13H9V15H15V13H13V9H15V11H17V7H15V9H13V7H15V5H9M18,9V11H20V13H18V15H20V17H18V19H20V21H22V17H20V15H22V11H20V9H18Z' }
        ]
      },
      gann: {
        title: 'Gann',
        tools: [
          { id: 'gann-fan', name: 'Gann Fan', icon: 'M2,2V22L22,2H2M4,4H10.17L4,10.17V4M4,13.66L15.66,2H18.83L4,16.83V13.66M4,20.34L22,2.34V5.51L4,23.51V20.34Z' },
          { id: 'gann-box', name: 'Gann Box', icon: 'M4,4H20V20H4V4M6,6V18H18V6H6M8,8H16V16H8V8Z' },
          { id: 'gann-square', name: 'Gann Square', icon: 'M2,2V22H22V2H2M4,4H20V20H4V4M6,6V18H18V6H6M8,8H16V16H8V8M10,10V14H14V10H10M11,11H13V13H11V11Z' },
          { id: 'gann-angles', name: 'Gann Angles', icon: 'M2,2L22,22M2,22L22,2M12,2V22M2,12H22' }
        ]
      },
      patterns: {
        title: 'Patterns',
        tools: [
          { id: 'head-and-shoulders', name: 'Head & Shoulders', icon: 'M12,2L8,6L4,6L2,10L2,14L4,14L4,10L8,10L12,6L16,10L20,10L20,14L22,14L22,10L20,6L16,6L12,2M4,16V22H8V16H4M10,16V22H14V16H10M16,16V22H20V16H16Z' },
          { id: 'triangle', name: 'Triangle', icon: 'M1,21H23L12,2M12,6L19.53,19H4.47' },
          { id: 'wedge', name: 'Wedge', icon: 'M2,21L12,3L22,21H2M12,9L6,19H18L12,9Z' },
          { id: 'double-top-bottom', name: 'Double Top/Bottom', icon: 'M4,4L8,10L12,4L16,10L20,4M4,14L8,20L12,14L16,20L20,14' }
        ]
      },
      shapes: {
        title: 'Shapes',
        tools: [
          { id: 'rectangle', name: 'Rectangle', icon: 'M4,6V18H20V6M18,16H6V8H18V16Z' },
          { id: 'circle', name: 'Circle', icon: 'M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20Z' },
          { id: 'ellipse', name: 'Ellipse', icon: 'M12,4A8,3 0 0,0 4,7A8,3 0 0,0 12,10A8,3 0 0,0 20,7A8,3 0 0,0 12,4M4,9A8,3 0 0,0 12,12A8,3 0 0,0 20,9V11A8,3 0 0,1 12,14A8,3 0 0,1 4,11V9M4,13A8,3 0 0,0 12,16A8,3 0 0,0 20,13V15A8,3 0 0,1 12,18A8,3 0 0,1 4,15V13Z' },
          { id: 'polygon', name: 'Polygon', icon: 'M2,2V8H4.28L5.57,16H4V22H10V20.06L15,20.05V22H21V16H19.17L20,9H22V3H16V6.53L14.8,8H9.59L8,5.82V2M4,4H6V6H4M18,5H20V7H18M6,18H8V20H6M17,18H19V20H17Z' }
        ]
      },
      annotations: {
        title: 'Annotations',
        tools: [
          { id: 'text-label', name: 'Text Label', icon: 'M18.5,4L19.66,8.35L18.7,8.61C18.25,7.74 17.79,6.87 17.26,6.43C16.73,6 16.11,6 15.5,6H13V16.5C13,17 13,17.5 13.33,17.75C13.67,18 14.33,18 15,18V19H9V18C9.67,18 10.33,18 10.67,17.75C11,17.5 11,17 11,16.5V6H8.5C7.89,6 7.27,6 6.74,6.43C6.21,6.87 5.75,7.74 5.3,8.61L4.34,8.35L5.5,4H18.5Z' },
          { id: 'callout', name: 'Callout', icon: 'M4,2H20A2,2 0 0,1 22,4V16A2,2 0 0,1 20,18H16L12,22L8,18H4A2,2 0 0,1 2,16V4A2,2 0 0,1 4,2M6,5V7H18V5H6M6,9V11H18V9H6M6,13V15H15V13H6Z' },
          { id: 'note', name: 'Note', icon: 'M3,3V21H21V3H3M18,18H6V17H18V18M18,16H6V15H18V16M18,12H6V6H18V12Z' },
          { id: 'price-label', name: 'Price Label', icon: 'M2,12H4V17H20V12H22V17A2,2 0 0,1 20,19H4A2,2 0 0,1 2,17V12M12,2L6.5,7.5L7.91,8.91L11,5.83V15H13V5.83L16.09,8.91L17.5,7.5L12,2Z' }
        ]
      }
    };

    this.dropdowns = {};

    Object.entries(categories).forEach(([categoryKey, categoryData]) => {
      const dropdown = document.createElement('div');
      dropdown.className = 'tos-tool-dropdown';
      dropdown.dataset.category = categoryKey;

      let toolsHTML = '';
      if (categoryData.tools.length > 0) {
        toolsHTML = categoryData.tools.map(tool => `
          <div class="tos-tool-item" data-tool="${tool.id}">
            <div class="tos-tool-item-icon">
              <svg viewBox="0 0 24 24">
                <path d="${tool.icon}"/>
              </svg>
            </div>
            <div class="tos-tool-item-label">${tool.name}</div>
          </div>
        `).join('');
      } else {
        toolsHTML = '<div style="padding: 12px; text-align: center; color: var(--tos-text-muted); font-size: 11px;">Coming Soon</div>';
      }

      dropdown.innerHTML = `
        <div class="tos-tool-dropdown-header">${categoryData.title}</div>
        ${toolsHTML}
      `;

      this.panelElement.appendChild(dropdown);
      this.dropdowns[categoryKey] = dropdown;
    });
  }

  /**
   * Attach event listeners to UI elements
   */
  attachEventListeners() {
    // Category button clicks
    const categoryButtons = this.panelElement.querySelectorAll('.tos-tool-category-btn');
    categoryButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const category = btn.dataset.category;
        this.toggleDropdown(category, btn);
      });
    });

    // Tool item clicks in dropdowns
    Object.values(this.dropdowns).forEach(dropdown => {
      const toolItems = dropdown.querySelectorAll('.tos-tool-item');
      toolItems.forEach(item => {
        item.addEventListener('click', () => {
          const toolName = item.dataset.tool;
          this.setActiveTool(toolName);
          this.updateActiveToolItem(toolName);
          this.hideAllDropdowns();
        });
      });
    });

    // Collapse/expand button
    const collapseBtn = this.panelElement.querySelector('.tos-tools-collapse-btn');
    collapseBtn.addEventListener('click', () => {
      this.toggleCollapse();
    });

    // Persistent mode toggle in header
    const persistentToggle = this.panelElement.querySelector('#persistent-mode-toggle');
    if (persistentToggle) {
      persistentToggle.addEventListener('change', (e) => {
        this.persistentMode = e.target.checked;
        this.updatePersistentModeUI();
        console.log(`✅ Keep Tool Active: ${this.persistentMode ? 'ON' : 'OFF'}`);
      });
    }

    // Click outside to close dropdowns
    document.addEventListener('click', (e) => {
      if (!this.panelElement.contains(e.target)) {
        this.hideAllDropdowns();
      }
    });

    // ESC key to deactivate persistent mode and return to default cursor
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.persistentMode) {
        this.persistentMode = false;
        this.updatePersistentModeUI();
        this.setActiveTool('default');
        this.updateActiveToolItem('default');
        console.log('✅ Persistent mode disabled (ESC key)');
      }
    });

    // Canvas event delegation
    this.attachCanvasListeners();
  }

  /**
   * Toggle dropdown visibility
   */
  toggleDropdown(category, button) {
    const dropdown = this.dropdowns[category];
    const isVisible = dropdown.classList.contains('visible');

    // Hide all dropdowns first
    this.hideAllDropdowns();

    // Show this dropdown if it wasn't visible
    if (!isVisible) {
      dropdown.classList.add('visible');

      // Position dropdown next to the button
      const buttonRect = button.getBoundingClientRect();
      const panelRect = this.panelElement.getBoundingClientRect();
      dropdown.style.top = `${buttonRect.top - panelRect.top}px`;
    }
  }

  /**
   * Hide all dropdowns
   */
  hideAllDropdowns() {
    Object.values(this.dropdowns).forEach(dropdown => {
      dropdown.classList.remove('visible');
    });
  }

  /**
   * Attach event listeners to the canvas for tool interactions
   */
  attachCanvasListeners() {
    if (!this.canvas) return;

    this.canvas.addEventListener('click', (e) => {
      const activeTool = this.registry.getActiveTool();
      if (activeTool && activeTool.onClick) {
        const result = activeTool.onClick(e, {});
        this.handleToolAction(result);
      }
    });

    this.canvas.addEventListener('mousemove', (e) => {
      const activeTool = this.registry.getActiveTool();
      if (activeTool && activeTool.onMouseMove) {
        // Add canvas-relative coordinates to event
        const rect = this.canvas.getBoundingClientRect();
        e.canvasX = e.clientX - rect.left;
        e.canvasY = e.clientY - rect.top;

        const result = activeTool.onMouseMove(e, {});
        this.handleToolAction(result);
      }
    });

    this.canvas.addEventListener('mousedown', (e) => {
      const activeTool = this.registry.getActiveTool();
      if (activeTool && activeTool.onMouseDown) {
        // Add canvas-relative coordinates to event
        const rect = this.canvas.getBoundingClientRect();
        e.canvasX = e.clientX - rect.left;
        e.canvasY = e.clientY - rect.top;

        const result = activeTool.onMouseDown(e, {});
        this.handleToolAction(result);
      }
    });

    this.canvas.addEventListener('mouseup', (e) => {
      const activeTool = this.registry.getActiveTool();
      if (activeTool && activeTool.onMouseUp) {
        // Add canvas-relative coordinates to event
        const rect = this.canvas.getBoundingClientRect();
        e.canvasX = e.clientX - rect.left;
        e.canvasY = e.clientY - rect.top;

        const result = activeTool.onMouseUp(e, {});
        this.handleToolAction(result);
      }
    });

    // Note: Right-click context menu handling is done in SelectionManager
    // We removed the auto-toggle on right-click to avoid confusion
  }

  /**
   * Handle tool action results
   */
  handleToolAction(result) {
    if (!result) return;

    // Emit custom event for the chart renderer to handle
    const event = new CustomEvent('tool-action', {
      detail: result
    });
    this.canvas.dispatchEvent(event);

    // Check if this is a completed drawing action
    const completedActions = [
      'finish-trend-line', 'place-horizontal-line', 'place-vertical-line',
      'finish-ray-line', 'finish-extended-line', 'finish-parallel-channel',
      'finish-fibonacci-retracement', 'finish-fibonacci-extension',
      'finish-fibonacci-fan', 'finish-fibonacci-arcs',
      'finish-fibonacci-time-zones', 'finish-fibonacci-spiral',
      'finish-gann-fan', 'finish-gann-box', 'finish-gann-square', 'finish-gann-angles',
      'finish-head-and-shoulders', 'finish-triangle', 'finish-wedge', 'finish-double-top-bottom',
      'finish-rectangle', 'finish-circle', 'finish-ellipse', 'finish-polygon',
      'place-text-label', 'finish-callout', 'place-note', 'place-price-label'
    ];

    if (completedActions.includes(result.action)) {
      // Auto-switch back to default tool (pan mode) ONLY if persistent mode is OFF
      if (!this.persistentMode) {
        console.log('✅ Drawing completed, switching to default tool');
        this.setActiveTool('default');
        this.updateActiveToolItem('default');
      } else {
        console.log('✅ Drawing completed, staying in current tool (persistent mode ON)');
      }
    }

    // console.log('🔧 Tool action:', result.action, result);
  }

  /**
   * Set the active tool
   */
  setActiveTool(toolName) {
    // Check if canvas is available
    if (!this.canvas) {
      console.warn('⚠️ Cannot activate tool: canvas not available yet');
      return;
    }

    console.log(`🔄 Switching to tool: ${toolName} (current: ${this.currentTool?.id || 'none'})`);

    // Deactivate current tool
    if (this.currentTool) {
      console.log(`⏹️ Deactivating current tool: ${this.currentTool.name}`);
      this.currentTool.deactivate(this.canvas);
    }

    // Activate new tool
    const newTool = this.tools[toolName];
    if (newTool) {
      console.log(`▶️ Activating new tool: ${newTool.name}`);
      this.registry.setActiveTool(newTool);
      newTool.activate(this.canvas);
      this.currentTool = newTool;
      this.saveState();
    } else {
      console.error(`❌ Tool not found: ${toolName}`);
    }
  }

  /**
   * Update the active tool item visual state
   */
  updateActiveToolItem(toolName) {
    // Update all tool items across all dropdowns
    Object.values(this.dropdowns).forEach(dropdown => {
      const items = dropdown.querySelectorAll('.tos-tool-item');
      items.forEach(item => {
        if (item.dataset.tool === toolName) {
          item.classList.add('active');
        } else {
          item.classList.remove('active');
        }
      });
    });
  }

  /**
   * Update persistent mode UI (called when mode changes)
   * Syncs both header checkbox and context menu indicator
   */
  updatePersistentModeUI() {
    // Update header checkbox
    const persistentToggle = this.panelElement?.querySelector('#persistent-mode-toggle');
    if (persistentToggle) {
      persistentToggle.checked = this.persistentMode;
    }

    // Update context menu indicator (via SelectionManager)
    // The SelectionManager will pick up the change when the menu is next shown
  }

  /**
   * Toggle panel collapse state
   */
  toggleCollapse() {
    this.isCollapsed = !this.isCollapsed;

    if (this.isCollapsed) {
      this.panelElement.classList.add('collapsed');
    } else {
      this.panelElement.classList.remove('collapsed');
    }

    this.saveState();
  }

  /**
   * Save panel state to localStorage
   */
  saveState() {
    localStorage.setItem('toolPanelCollapsed', this.isCollapsed.toString());
    if (this.currentTool) {
      localStorage.setItem('toolPanelActiveTool', this.currentTool.id);
    }
  }

  /**
   * Load panel state from localStorage
   */
  loadState() {
    const collapsed = localStorage.getItem('toolPanelCollapsed');
    if (collapsed === 'true') {
      this.isCollapsed = true;
      this.panelElement.classList.add('collapsed');
    }

    const activeTool = localStorage.getItem('toolPanelActiveTool');
    if (activeTool) {
      const toolEntry = Object.entries(this.tools).find(([_, tool]) => tool.id === activeTool);
      if (toolEntry) {
        this.setActiveTool(toolEntry[0]);
        this.updateActiveToolItem(toolEntry[0]);
      }
    }
  }

  /**
   * Get the tool registry
   */
  getRegistry() {
    return this.registry;
  }

  /**
   * Get all placed drawings
   */
  getDrawings() {
    return this.registry.getDrawings();
  }

  /**
   * Add a drawing to the registry
   */
  addDrawing(drawing) {
    return this.registry.addDrawing(drawing);
  }

  /**
   * Remove a drawing from the registry
   */
  removeDrawing(drawingId) {
    return this.registry.removeDrawing(drawingId);
  }

  /**
   * Clear all drawings
   */
  clearDrawings() {
    this.registry.clearDrawings();
  }
}

/**
 * Initialize the tool panel
 */
export function initializeToolPanel(containerElement) {
  // console.log('🔧 Initializing drawing tools panel...');
  const containerDiv = containerElement || document.getElementById('tos-plot');
  const toolPanel = new ToolPanel(containerDiv);
  // console.log('✅ Drawing tools panel initialized');
  return toolPanel;
}

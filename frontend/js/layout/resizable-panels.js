/**
 * Resizable Panels Module
 * Handles drag-to-resize functionality for panel layout
 */

export class ResizablePanel {
  constructor(element, options = {}) {
    this.element = element;
    this.minWidth = options.minWidth || 200;
    this.maxWidth = options.maxWidth || 600;
    this.minHeight = options.minHeight || 150;
    this.maxHeight = options.maxHeight || 600;
    this.direction = options.direction || 'vertical'; // 'vertical' or 'horizontal'
    this.onResize = options.onResize || null;

    this.isResizing = false;
    this.startX = 0;
    this.startY = 0;
    this.startWidth = 0;
    this.startHeight = 0;

    this.initResizer();
  }

  initResizer() {
    const resizer = document.createElement('div');

    if (this.direction === 'vertical') {
      resizer.className = 'tos-resizer tos-resizer-vertical';
    } else {
      resizer.className = 'tos-resizer tos-resizer-horizontal';
    }

    this.resizer = resizer;
    this.element.appendChild(resizer);

    // Add event listeners
    resizer.addEventListener('mousedown', this.startResize.bind(this));
    resizer.addEventListener('dblclick', this.resetSize.bind(this));
  }

  startResize(e) {
    e.preventDefault();
    this.isResizing = true;

    this.startX = e.clientX;
    this.startY = e.clientY;
    this.startWidth = this.element.offsetWidth;
    this.startHeight = this.element.offsetHeight;

    this.resizer.classList.add('resizing');
    document.body.style.cursor = this.direction === 'vertical' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';

    // Bind events to document for smooth dragging
    this.boundResize = this.resize.bind(this);
    this.boundStopResize = this.stopResize.bind(this);

    document.addEventListener('mousemove', this.boundResize);
    document.addEventListener('mouseup', this.boundStopResize);
  }

  resize(e) {
    if (!this.isResizing) return;

    e.preventDefault();

    if (this.direction === 'vertical') {
      const deltaX = e.clientX - this.startX;
      let newWidth = this.startWidth;

      // Determine if we're resizing from left or right edge
      const isRightEdge = this.resizer.classList.contains('tos-resizer-vertical') &&
                          this.element.classList.contains('tos-left-panel');

      if (isRightEdge) {
        newWidth = this.startWidth + deltaX;
      } else {
        newWidth = this.startWidth - deltaX;
      }

      // Clamp to min/max
      newWidth = Math.max(this.minWidth, Math.min(this.maxWidth, newWidth));

      this.element.style.width = newWidth + 'px';
    } else {
      const deltaY = e.clientY - this.startY;
      let newHeight = this.startHeight + deltaY;

      // Clamp to min/max
      newHeight = Math.max(this.minHeight, Math.min(this.maxHeight, newHeight));

      this.element.style.height = newHeight + 'px';
    }

    // Callback for custom handling
    if (this.onResize) {
      this.onResize();
    }
  }

  stopResize(e) {
    if (!this.isResizing) return;

    this.isResizing = false;
    this.resizer.classList.remove('resizing');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';

    document.removeEventListener('mousemove', this.boundResize);
    document.removeEventListener('mouseup', this.boundStopResize);

    // Save panel size to localStorage
    this.saveSize();
  }

  resetSize() {
    // Reset to default size based on element type
    if (this.element.classList.contains('tos-left-panel')) {
      this.element.style.width = '300px';
    } else if (this.element.classList.contains('tos-right-panel')) {
      this.element.style.width = '350px';
    } else if (this.element.classList.contains('tos-watchlist')) {
      this.element.style.flex = '1';
    } else if (this.element.classList.contains('tos-news-feed')) {
      this.element.style.minHeight = '150px';
    }

    this.saveSize();

    if (this.onResize) {
      this.onResize();
    }
  }

  saveSize() {
    const panelId = this.element.id || this.element.className;
    if (!panelId) return;

    const sizeData = {
      width: this.element.style.width,
      height: this.element.style.height,
      flex: this.element.style.flex
    };

    localStorage.setItem(`tos-panel-${panelId}`, JSON.stringify(sizeData));
  }

  loadSize() {
    const panelId = this.element.id || this.element.className;
    if (!panelId) return;

    const savedSize = localStorage.getItem(`tos-panel-${panelId}`);
    if (!savedSize) return;

    try {
      const sizeData = JSON.parse(savedSize);
      if (sizeData.width) this.element.style.width = sizeData.width;
      if (sizeData.height) this.element.style.height = sizeData.height;
      if (sizeData.flex) this.element.style.flex = sizeData.flex;
    } catch (error) {
      console.error('Error loading saved panel size:', error);
    }
  }

  destroy() {
    if (this.resizer && this.resizer.parentNode) {
      this.resizer.parentNode.removeChild(this.resizer);
    }

    document.removeEventListener('mousemove', this.boundResize);
    document.removeEventListener('mouseup', this.boundStopResize);
  }
}

/**
 * Initialize resizable panels for the TOS layout
 */
export function initializeResizablePanels() {
  const leftPanel = document.querySelector('.tos-left-panel');
  const rightPanel = document.querySelector('.tos-right-panel');
  const watchlist = document.querySelector('.tos-watchlist');
  const newsFeed = document.querySelector('.tos-news-feed');

  const panels = [];

  if (leftPanel) {
    const leftResizer = new ResizablePanel(leftPanel, {
      direction: 'vertical',
      minWidth: 200,
      maxWidth: 500
    });
    leftResizer.loadSize();
    panels.push(leftResizer);
  }

  if (rightPanel) {
    const rightResizer = new ResizablePanel(rightPanel, {
      direction: 'vertical',
      minWidth: 250,
      maxWidth: 600
    });
    rightResizer.loadSize();
    panels.push(rightResizer);
  }

  if (watchlist && newsFeed) {
    const watchlistResizer = new ResizablePanel(watchlist, {
      direction: 'horizontal',
      minHeight: 150,
      maxHeight: 600
    });
    watchlistResizer.loadSize();
    panels.push(watchlistResizer);
  }

  return panels;
}

/**
 * Toggle panel visibility
 */
export function togglePanel(panelClass) {
  const panel = document.querySelector(`.${panelClass}`);
  if (!panel) return;

  panel.classList.toggle('collapsed');

  // Save state to localStorage
  const isCollapsed = panel.classList.contains('collapsed');
  localStorage.setItem(`tos-panel-${panelClass}-collapsed`, isCollapsed);
}

/**
 * Load panel collapse states from localStorage
 */
export function loadPanelStates() {
  const panels = [
    { selector: '.tos-left-panel', key: 'tos-left-panel' },
    { selector: '.tos-right-panel', key: 'tos-right-panel' }
  ];

  panels.forEach(({ selector, key }) => {
    const panel = document.querySelector(selector);
    if (!panel) return;

    const isCollapsed = localStorage.getItem(`tos-panel-${key}-collapsed`) === 'true';
    if (isCollapsed) {
      panel.classList.add('collapsed');
    }
  });
}

/**
 * ORD Volume UI Controller
 * Completely segregated UI logic for ORD Volume feature
 * Handles mode switching, user interactions, and modal management
 *
 * NO SHARED CODE - Standalone implementation
 */

import { ORDVolumeAnalysis } from './ORDVolumeAnalysis.js';
import { ORDVolumeRenderer } from './ORDVolumeRenderer.js';

export class ORDVolumeController {
  constructor() {
    this.mode = 'auto'; // 'draw' or 'auto'
    this.lineCount = 100; // Default for auto mode (was 3)
    this.sensitivity = 'high'; // 'normal' or 'high' (lookback 2 or 1)
    this.isActive = false;
    this.isDrawing = false;
    this.currentLines = [];
    this.analysisResult = null;
    this.renderer = null;
    this.candles = [];
    this.symbol = '';

    // Modal elements
    this.modal = null;
    this.drawModeBtn = null;
    this.autoModeBtn = null;
    this.lineCountInput = null;
    this.analyzeBtn = null;
    this.cancelBtn = null;
    this.drawCanvas = null;
    this.tempLines = []; // Temporary lines during drawing

    this._initModal();
  }

  /**
   * Initialize the ORD Volume modal UI
   * @private
   */
  _initModal() {
    // Create modal HTML
    const modalHTML = `
      <div id="ord-volume-modal" class="ord-volume-modal" style="display: none;">
        <div class="ord-volume-modal-content">
          <div class="ord-volume-modal-header">
            <h3>ORD Volume Analysis</h3>
            <span class="ord-volume-close">&times;</span>
          </div>
          <div class="ord-volume-modal-body">
            <!-- Mode Selection -->
            <div class="ord-volume-mode-selection">
              <label>Mode:</label>
              <div class="ord-volume-mode-buttons">
                <button id="ord-draw-mode-btn" class="ord-mode-btn active">Draw</button>
                <button id="ord-auto-mode-btn" class="ord-mode-btn">Auto</button>
              </div>
            </div>

            <!-- Line Count (visible in Auto mode) -->
            <div id="ord-line-count-group" class="ord-volume-setting" style="display: none;">
              <label for="ord-line-count">Number of Trendlines (3-100):</label>
              <input type="number" id="ord-line-count" min="3" max="100" value="100" />
            </div>

            <!-- Properties Section -->
            <div class="ord-volume-properties">
              <h4 style="margin: 15px 0 10px 0; font-size: 14px; color: #888;">Analysis Options</h4>

              <!-- Sensitivity Toggle -->
              <div class="ord-volume-checkbox-group">
                <label>
                  <input type="checkbox" id="ord-high-sensitivity" checked />
                  <span>High Sensitivity (More Swing Points)</span>
                </label>
              </div>

              <h4 style="margin: 15px 0 10px 0; font-size: 14px; color: #888;">Display Options</h4>
              <div class="ord-volume-checkbox-group">
                <label>
                  <input type="checkbox" id="ord-show-signals" checked />
                  <span>Show Trade Signals</span>
                </label>
              </div>
            </div>

            <!-- Instructions -->
            <div id="ord-draw-instructions" class="ord-volume-instructions">
              <p>Click on the chart to draw trendlines. Minimum 3 lines required (Initial, Correction, Retest).</p>
              <p>Press ESC to cancel current line. Click "Analyze" when done.</p>
            </div>

            <div id="ord-auto-instructions" class="ord-volume-instructions" style="display: none;">
              <p>Automatically detect trendlines based on price structure and analyze volume.</p>
            </div>

            <!-- Status -->
            <div id="ord-status" class="ord-volume-status"></div>
          </div>
          <div class="ord-volume-modal-footer">
            <button id="ord-clear-btn" class="ord-btn-danger" style="margin-right: auto;">Clear ORD Volume</button>
            <button id="ord-analyze-btn" class="ord-btn-primary">Analyze</button>
            <button id="ord-cancel-btn" class="ord-btn-secondary">Cancel</button>
          </div>
        </div>
      </div>
    `;

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHTML);

    // Get references
    this.modal = document.getElementById('ord-volume-modal');
    this.drawModeBtn = document.getElementById('ord-draw-mode-btn');
    this.autoModeBtn = document.getElementById('ord-auto-mode-btn');
    this.lineCountInput = document.getElementById('ord-line-count');
    this.analyzeBtn = document.getElementById('ord-analyze-btn');
    this.cancelBtn = document.getElementById('ord-cancel-btn');
    this.clearBtn = document.getElementById('ord-clear-btn');

    // Add modal styles
    this._addModalStyles();

    // Bind events
    this._bindEvents();
  }

  /**
   * Add CSS styles for the modal
   * @private
   */
  _addModalStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .ord-volume-modal {
        display: none;
        position: fixed;
        z-index: 10000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
      }

      .ord-volume-modal-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: var(--tos-bg-primary, #1e1e1e);
        border: 1px solid var(--tos-border-color, #333);
        border-radius: 8px;
        width: 500px;
        max-width: 90%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
      }

      .ord-volume-modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        border-bottom: 1px solid var(--tos-border-color, #333);
      }

      .ord-volume-modal-header h3 {
        margin: 0;
        color: var(--tos-text-primary, #ffffff);
        font-size: 18px;
        font-weight: 600;
      }

      .ord-volume-close {
        font-size: 28px;
        font-weight: bold;
        color: var(--tos-text-secondary, #999);
        cursor: pointer;
        line-height: 1;
      }

      .ord-volume-close:hover {
        color: var(--tos-text-primary, #ffffff);
      }

      .ord-volume-modal-body {
        padding: 20px;
      }

      .ord-volume-mode-selection {
        margin-bottom: 20px;
      }

      .ord-volume-mode-selection label {
        display: block;
        margin-bottom: 10px;
        color: var(--tos-text-primary, #ffffff);
        font-weight: 600;
      }

      .ord-volume-mode-buttons {
        display: flex;
        gap: 10px;
      }

      .ord-mode-btn {
        flex: 1;
        padding: 10px;
        border: 1px solid var(--tos-border-color, #333);
        background: var(--tos-bg-secondary, #2a2a2a);
        color: var(--tos-text-primary, #ffffff);
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
      }

      .ord-mode-btn:hover {
        background: var(--tos-bg-hover, #333);
      }

      .ord-mode-btn.active {
        background: #2196f3;
        border-color: #2196f3;
        color: #ffffff;
      }

      .ord-volume-setting {
        margin-bottom: 20px;
      }

      .ord-volume-setting label {
        display: block;
        margin-bottom: 8px;
        color: var(--tos-text-primary, #ffffff);
        font-size: 14px;
      }

      .ord-volume-setting input {
        width: 100%;
        padding: 8px;
        border: 1px solid var(--tos-border-color, #333);
        background: var(--tos-bg-secondary, #2a2a2a);
        color: var(--tos-text-primary, #ffffff);
        border-radius: 4px;
        font-size: 14px;
      }

      .ord-volume-instructions {
        margin-bottom: 20px;
        padding: 15px;
        background: rgba(33, 150, 243, 0.1);
        border-left: 3px solid #2196f3;
        border-radius: 4px;
      }

      .ord-volume-instructions p {
        margin: 8px 0;
        color: var(--tos-text-secondary, #ccc);
        font-size: 13px;
        line-height: 1.5;
      }

      .ord-volume-status {
        min-height: 20px;
        margin-bottom: 10px;
        color: var(--tos-text-secondary, #ccc);
        font-size: 13px;
      }

      .ord-volume-checkbox-group {
        margin-bottom: 10px;
      }

      .ord-volume-checkbox-group label {
        display: flex;
        align-items: center;
        cursor: pointer;
        color: var(--tos-text-primary, #ffffff);
        font-size: 13px;
      }

      .ord-volume-checkbox-group input[type="checkbox"] {
        margin-right: 8px;
        cursor: pointer;
      }

      .ord-volume-modal-footer {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        padding: 20px;
        border-top: 1px solid var(--tos-border-color, #333);
      }

      .ord-btn-primary, .ord-btn-secondary {
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }

      .ord-btn-primary {
        background: #2196f3;
        color: #ffffff;
      }

      .ord-btn-primary:hover {
        background: #1976d2;
      }

      .ord-btn-secondary {
        background: var(--tos-bg-secondary, #2a2a2a);
        color: var(--tos-text-primary, #ffffff);
        border: 1px solid var(--tos-border-color, #333);
      }

      .ord-btn-secondary:hover {
        background: var(--tos-bg-hover, #333);
      }

      .ord-btn-danger {
        background: #dc3545;
        color: #ffffff;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }

      .ord-btn-danger:hover {
        background: #c82333;
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Bind event listeners
   * @private
   */
  _bindEvents() {
    // Close button
    const closeBtn = this.modal.querySelector('.ord-volume-close');
    closeBtn.addEventListener('click', () => this.closeModal());

    // Click outside to close
    this.modal.addEventListener('click', (e) => {
      if (e.target === this.modal) {
        this.closeModal();
      }
    });

    // Mode buttons
    this.drawModeBtn.addEventListener('click', () => this.setMode('draw'));
    this.autoModeBtn.addEventListener('click', () => this.setMode('auto'));

    // Analyze button
    this.analyzeBtn.addEventListener('click', () => this.analyze());

    // Cancel button
    this.cancelBtn.addEventListener('click', () => this.closeModal());

    // Clear button
    this.clearBtn.addEventListener('click', () => this.clearORDVolume());

    // Line count input
    this.lineCountInput.addEventListener('change', (e) => {
      let value = parseInt(e.target.value);
      if (value < 3) value = 3;
      if (value > 100) value = 100;
      this.lineCount = value;
      e.target.value = value;
    });

    // Sensitivity checkbox
    const sensitivityCheckbox = document.getElementById('ord-high-sensitivity');
    if (sensitivityCheckbox) {
      sensitivityCheckbox.addEventListener('change', (e) => {
        this.sensitivity = e.target.checked ? 'high' : 'normal';
        console.log(`[ORD Controller] Sensitivity: ${this.sensitivity} (lookback ${this.sensitivity === 'high' ? 1 : 2})`);

        // Save preference
        localStorage.setItem('ordVolumeSensitivity', this.sensitivity);
      });

      // Load saved preference
      const savedSensitivity = localStorage.getItem('ordVolumeSensitivity');
      if (savedSensitivity) {
        this.sensitivity = savedSensitivity;
        sensitivityCheckbox.checked = (savedSensitivity === 'high');
      }
    }

    // Trade signals checkbox
    const showSignalsCheckbox = document.getElementById('ord-show-signals');
    if (showSignalsCheckbox) {
      showSignalsCheckbox.addEventListener('change', (e) => {
        if (window.ordVolumeBridge) {
          window.ordVolumeBridge.showTradeSignals = e.target.checked;
          console.log(`[ORD Controller] Trade signals ${e.target.checked ? 'enabled' : 'disabled'}`);

          // Save preference
          localStorage.setItem('ordVolumeShowSignals', e.target.checked);

          // Redraw chart
          if (window.tosApp && window.tosApp.activeChartType) {
            if (window.tosApp.activeChartType === 'timeframe') {
              const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
              if (currentTimeframe && currentTimeframe.renderer && currentTimeframe.renderer.draw) {
                currentTimeframe.renderer.draw();
              }
            } else if (window.tosApp.activeChartType === 'tick') {
              const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
              if (currentTickChart && currentTickChart.renderer && currentTickChart.renderer.draw) {
                currentTickChart.renderer.draw();
              }
            }
          }
        }
      });

      // Load saved preference
      const savedPreference = localStorage.getItem('ordVolumeShowSignals');
      if (savedPreference !== null) {
        const showSignals = savedPreference === 'true';
        showSignalsCheckbox.checked = showSignals;
        if (window.ordVolumeBridge) {
          window.ordVolumeBridge.showTradeSignals = showSignals;
        }
      }
    }
  }

  /**
   * Open the ORD Volume modal
   * @param {Array} candles - OHLCV candle data
   * @param {String} symbol - Current symbol
   * @param {Object} renderer - ORD Volume renderer instance
   */
  open(candles, symbol, renderer) {
    this.candles = candles;
    this.symbol = symbol;
    this.renderer = renderer;
    this.currentLines = [];
    this.tempLines = [];
    this.isActive = true;

    // Show modal
    this.modal.style.display = 'block';

    // Set initial mode display
    this.setMode(this.mode);

    // Update status
    this._updateStatus('Ready');
  }

  /**
   * Close the modal
   */
  closeModal() {
    this.modal.style.display = 'none';
    this.isActive = false;
    this.isDrawing = false;
    this.currentLines = [];
    this.tempLines = [];

    // Clear drawing mode if active
    if (this.renderer) {
      this.renderer.clearDrawingMode();
    }
  }

  /**
   * Set analysis mode
   * @param {String} mode - 'draw' or 'auto'
   */
  setMode(mode) {
    this.mode = mode;

    // Update button states
    if (mode === 'draw') {
      this.drawModeBtn.classList.add('active');
      this.autoModeBtn.classList.remove('active');
      document.getElementById('ord-line-count-group').style.display = 'none';
      document.getElementById('ord-draw-instructions').style.display = 'block';
      document.getElementById('ord-auto-instructions').style.display = 'none';

      // In draw mode, close the modal immediately and start drawing
      console.log('[ORD Volume] Entering draw mode - closing modal');
      this.isActive = false; // Allow reopening modal while in draw mode
      this.modal.style.display = 'none';

      // Enable drawing mode on renderer
      if (this.renderer) {
        this.renderer.enableDrawMode();

        // Set callback to run analysis after each line is drawn
        this.renderer.onLineDrawn = (count) => {
          console.log(`[ORD Volume] ${count} lines drawn, running live analysis...`);
          // Run analysis automatically after each line (if at least 3 lines)
          if (count >= 3) {
            this._runDrawModeAnalysis();
          }
        };

        console.log('[ORD Volume] Draw mode enabled - draw lines on chart (analysis runs automatically)');
      }

      // DON'T show instructions panel - just let user draw
      // this._showDrawingInstructions(); // REMOVED

    } else {
      this.autoModeBtn.classList.add('active');
      this.drawModeBtn.classList.remove('active');
      document.getElementById('ord-line-count-group').style.display = 'block';
      document.getElementById('ord-auto-instructions').style.display = 'block';
      document.getElementById('ord-draw-instructions').style.display = 'none';

      // In auto mode, show the analyze button
      this.analyzeBtn.style.display = 'block';

      // Disable drawing mode on renderer
      if (this.renderer) {
        this.renderer.clearDrawingMode();
        console.log('[ORD Volume] Auto mode enabled');
      }
    }
  }

  /**
   * Perform analysis
   */
  analyze() {
    try {
      // Validate candles
      if (!this.candles || this.candles.length === 0) {
        this._updateStatus('Error: No candle data available', 'error');
        return;
      }

      console.log(`[ORD Volume] Analyzing ${this.candles.length} candles`);

      // Check minimum data requirements for ORD Volume
      const MIN_BARS_REQUIRED = 100;  // Absolute minimum for basic analysis
      const RECOMMENDED_BARS = 500;   // Recommended for accurate ORD Volume signals
      const MAX_BARS_ALLOWED = 4500;  // Maximum before performance degrades (allows Daily ~4000 bars)

      if (this.candles.length < MIN_BARS_REQUIRED) {
        this._updateStatus(`Error: Insufficient data. Need at least ${MIN_BARS_REQUIRED} bars, have ${this.candles.length}. Try a longer timeframe (Daily recommended).`, 'error');
        alert(`ORD Volume Analysis Error\n\nInsufficient data for analysis.\n\nRequired: ${MIN_BARS_REQUIRED}+ bars\nAvailable: ${this.candles.length} bars\n\nSuggestion: Use Daily or Weekly timeframe for best results.\nORD Volume methodology requires significant historical data.`);
        return;
      }

      // IMPORTANT: Block Auto mode only if too much data, Draw mode is OK
      if (this.mode === 'auto' && this.candles.length > MAX_BARS_ALLOWED) {
        this._updateStatus(`Error: Auto mode not available - too much data (${this.candles.length} bars)`, 'error');
        alert(`ORD Volume Auto Mode Not Available\n\nDataset too large for Auto mode.\n\nMaximum: ${MAX_BARS_ALLOWED} bars\nYou have: ${this.candles.length} bars\n\n✅ SOLUTION: Use Draw mode instead!\nDraw mode works on any timeframe (manually draw 3+ trendlines).\n\n❌ Auto mode requires swing point detection which freezes on large datasets.\n\nClick "Draw" to switch modes.`);
        return;
      }

      // Draw mode can handle larger datasets (manual drawing, no swing detection)
      if (this.mode === 'draw' && this.candles.length > MAX_BARS_ALLOWED) {
        console.warn(`[ORD Volume] Draw mode with ${this.candles.length} bars (exceeds auto mode limit, but manual drawing is OK)`);
      }

      if (this.candles.length < RECOMMENDED_BARS) {
        console.warn(`[ORD Volume] WARNING: Only ${this.candles.length} bars available. ${RECOMMENDED_BARS}+ recommended for accurate signals.`);
        this._updateStatus(`Warning: Limited data (${this.candles.length} bars). 500+ bars recommended.`, 'warning');
      }

      // Create analyzer
      const analyzer = new ORDVolumeAnalysis(this.candles);

      // Set sensitivity (lookback period for swing detection)
      analyzer.setSensitivity(this.sensitivity);

      if (this.mode === 'draw') {
        // Get lines from renderer (user drawn)
        const drawnLines = this.renderer.getDrawnLines();

        if (drawnLines.length < 3) {
          this._updateStatus('Error: Draw at least 3 trendlines', 'error');
          return;
        }

        this._updateStatus('Analyzing drawn trendlines...');
        this.analysisResult = analyzer.analyzeDrawMode(drawnLines);
      } else {
        // Auto mode
        this._updateStatus(`Detecting ${this.lineCount} trendlines (${this.sensitivity} sensitivity)...`);
        this.analysisResult = analyzer.analyzeAutoMode(this.lineCount);
      }

      // Store result in bridge for persistent rendering
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.setChartRenderer(this.renderer.chartState);
        window.ordVolumeBridge.setAnalysis(this.analysisResult, this.candles);
        console.log('[ORD Volume] Analysis stored in bridge');
      }

      // Trigger chart redraw to show overlays
      if (this.renderer.chartState && this.renderer.chartState.draw) {
        this.renderer.chartState.draw();
      }

      // Save to backend
      this._saveAnalysis();

      // Update status
      this._updateStatus(`Analysis complete! Strength: ${this.analysisResult.strength}`, 'success');

      // Close modal after short delay
      setTimeout(() => this.closeModal(), 1500);

    } catch (error) {
      console.error('ORD Volume Analysis Error:', error);
      this._updateStatus(`Error: ${error.message}`, 'error');
    }
  }

  /**
   * Update status message
   * @private
   * @param {String} message
   * @param {String} type - 'info', 'success', 'error'
   */
  _updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('ord-status');
    statusEl.textContent = message;

    // Color coding
    if (type === 'success') {
      statusEl.style.color = '#00c853';
    } else if (type === 'error') {
      statusEl.style.color = '#ff1744';
    } else {
      statusEl.style.color = 'var(--tos-text-secondary, #ccc)';
    }
  }

  /**
   * Save analysis to backend
   * @private
   */
  async _saveAnalysis() {
    if (!this.analysisResult) return;

    try {
      const response = await fetch('http://127.0.0.1:5000/ord-volume/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symbol: this.symbol,
          analysis: this.analysisResult
        })
      });

      if (!response.ok) {
        console.error('Failed to save ORD Volume analysis');
      }
    } catch (error) {
      console.error('Error saving ORD Volume analysis:', error);
    }
  }

  /**
   * Load saved analysis from backend
   * @param {String} symbol
   */
  async loadAnalysis(symbol) {
    try {
      const response = await fetch(`http://127.0.0.1:5000/ord-volume/load/${symbol}`);

      if (!response.ok) {
        return null;
      }

      const data = await response.json();
      return data.analysis;
    } catch (error) {
      console.error('Error loading ORD Volume analysis:', error);
      return null;
    }
  }

  /**
   * Show on-screen drawing instructions
   * @private
   */
  _showDrawingInstructions() {
    // Create floating instruction panel
    let panel = document.getElementById('ord-draw-panel');

    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'ord-draw-panel';
      panel.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 20px;
        border-radius: 8px;
        border: 2px solid #2196f3;
        z-index: 10000;
        font-family: Arial, sans-serif;
        min-width: 300px;
      `;
      panel.innerHTML = `
        <h3 style="margin: 0 0 15px 0; color: #2196f3;">ORD Volume - Draw Mode</h3>
        <p style="margin: 5px 0;">📍 Click and drag to draw trendlines</p>
        <p style="margin: 5px 0;">✏️ Draw at least 3 lines (Initial, Correction, Retest)</p>
        <p style="margin: 5px 0;">⌨️ Press ESC to cancel current line</p>
        <p style="margin: 5px 0; padding-top: 10px; border-top: 1px solid #444;">
          <strong>Lines drawn: <span id="ord-line-counter">0</span>/3</strong>
        </p>
        <button id="ord-finish-drawing" style="
          width: 100%;
          margin-top: 15px;
          padding: 10px;
          background: #2196f3;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          font-weight: bold;
        ">Finish & Analyze</button>
      `;
      document.body.appendChild(panel);

      // Bind finish button
      document.getElementById('ord-finish-drawing').addEventListener('click', () => {
        this._finishDrawing();
      });
    }

    panel.style.display = 'block';
    this._updateLineCounter();
  }

  /**
   * Hide drawing instructions panel
   * @private
   */
  _hideDrawingInstructions() {
    const panel = document.getElementById('ord-draw-panel');
    if (panel) {
      panel.style.display = 'none';
    }
  }

  /**
   * Update line counter in drawing panel
   * @private
   */
  _updateLineCounter() {
    const counter = document.getElementById('ord-line-counter');
    if (counter && this.renderer) {
      const lineCount = this.renderer.getDrawnLines().length;
      counter.textContent = lineCount;

      // Update finish button state
      const finishBtn = document.getElementById('ord-finish-drawing');
      if (finishBtn) {
        if (lineCount >= 3) {
          finishBtn.disabled = false;
          finishBtn.style.opacity = '1';
        } else {
          finishBtn.disabled = true;
          finishBtn.style.opacity = '0.5';
        }
      }
    }
  }

  /**
   * Run analysis in draw mode (called automatically after each line)
   * @private
   */
  _runDrawModeAnalysis() {
    try {
      // Get lines from renderer (user drawn)
      const drawnLines = this.renderer.getDrawnLines();

      if (drawnLines.length < 3) {
        return; // Need at least 3 lines
      }

      console.log(`[ORD Volume] Running analysis on ${drawnLines.length} drawn lines...`);

      // Create analyzer
      const analyzer = new ORDVolumeAnalysis(this.candles);
      analyzer.setSensitivity(this.sensitivity);

      // Analyze drawn lines
      this.analysisResult = analyzer.analyzeDrawMode(drawnLines);

      // Store result in bridge for persistent rendering
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.setChartRenderer(this.renderer.chartState);
        window.ordVolumeBridge.setAnalysis(this.analysisResult, this.candles);
        console.log('[ORD Volume] Draw mode analysis complete, stored in bridge');
      }

      // Trigger chart redraw to show overlays
      if (this.renderer.chartState && this.renderer.chartState.draw) {
        this.renderer.chartState.draw();
      }

    } catch (error) {
      console.error('[ORD Volume] Error during draw mode analysis:', error);
    }
  }

  /**
   * Finish drawing and perform analysis (DEPRECATED - now runs automatically)
   * @private
   */
  _finishDrawing() {
    console.log('[ORD Volume] Finishing drawing mode');

    // Hide drawing panel
    this._hideDrawingInstructions();

    // Perform analysis
    this.analyze();
  }

  /**
   * Clear ORD Volume analysis and lines from current chart
   */
  clearORDVolume() {
    console.log('[ORD Volume] Clearing ORD Volume for current chart');

    // Clear from bridge (removes from current chart storage)
    if (window.ordVolumeBridge) {
      window.ordVolumeBridge.clearAnalysis();
    }

    // Clear renderer state
    if (this.renderer) {
      this.renderer.clearDrawingMode();
    }

    // Clear local state
    this.analysisResult = null;
    this.currentLines = [];
    this.tempLines = [];

    // Trigger chart redraw to remove overlays
    if (this.renderer && this.renderer.chartState && this.renderer.chartState.draw) {
      this.renderer.chartState.draw();
    }

    // Update status
    this._updateStatus('ORD Volume cleared', 'success');

    // Close modal after short delay
    setTimeout(() => this.closeModal(), 1000);

    console.log('[ORD Volume] Cleared successfully');
  }
}

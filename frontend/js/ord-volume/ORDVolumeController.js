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
    this.lineCount = 3; // Default for auto mode
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
              <label for="ord-line-count">Number of Trendlines (3-7):</label>
              <input type="number" id="ord-line-count" min="3" max="7" value="3" />
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

    // Line count input
    this.lineCountInput.addEventListener('change', (e) => {
      let value = parseInt(e.target.value);
      if (value < 3) value = 3;
      if (value > 7) value = 7;
      this.lineCount = value;
      e.target.value = value;
    });
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
    } else {
      this.autoModeBtn.classList.add('active');
      this.drawModeBtn.classList.remove('active');
      document.getElementById('ord-line-count-group').style.display = 'block';
      document.getElementById('ord-auto-instructions').style.display = 'block';
      document.getElementById('ord-draw-instructions').style.display = 'none';
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

      // Create analyzer
      const analyzer = new ORDVolumeAnalysis(this.candles);

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
        this._updateStatus(`Detecting ${this.lineCount} trendlines...`);
        this.analysisResult = analyzer.analyzeAutoMode(this.lineCount);
      }

      // Store result in bridge for persistent rendering
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.setChartRenderer(this.renderer.chartState);
        window.ordVolumeBridge.setAnalysis(this.analysisResult);
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
}

/**
 * Indicator Settings Modal
 * Dynamic settings UI generator for all indicators
 * Auto-generates forms based on indicator settings schema
 */

import { indicatorRegistry } from './IndicatorRegistry.js';

export class IndicatorSettingsModal {
  constructor() {
    this.modal = null;
    this.currentIndicator = null;
    this.formElements = new Map(); // field name -> input element

    // Drag state
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.modalStartX = 0;
    this.modalStartY = 0;

    this._initModal();
    this._attachEventListeners();
  }

  /**
   * Initialize the modal HTML
   * @private
   */
  _initModal() {
    const modalHTML = `
      <div id="indicator-settings-modal" class="indicator-modal" style="display: none;">
        <div class="indicator-modal-overlay"></div>
        <div class="indicator-modal-content">
          <div class="indicator-modal-header">
            <h2>Indicators</h2>
            <button class="indicator-modal-close">&times;</button>
          </div>

          <div class="indicator-modal-body">
            <!-- Indicator List -->
            <div class="indicator-list-panel">
              <h3>Available Indicators</h3>
              <div id="indicator-list" class="indicator-list">
                <!-- Populated dynamically -->
              </div>

              <div class="indicator-actions">
                <button id="indicator-export-all-btn" class="indicator-btn indicator-btn-secondary">
                  Export Settings
                </button>
                <button id="indicator-import-all-btn" class="indicator-btn indicator-btn-secondary">
                  Import Settings
                </button>
                <button id="indicator-reset-all-btn" class="indicator-btn indicator-btn-danger">
                  Reset All
                </button>
              </div>
            </div>

            <!-- Settings Panel -->
            <div class="indicator-settings-panel">
              <div id="indicator-settings-content">
                <p class="indicator-empty-state">Select an indicator to configure</p>
              </div>
            </div>
          </div>

          <div class="indicator-modal-footer">
            <button id="indicator-apply-btn" class="indicator-btn indicator-btn-primary">
              Apply
            </button>
            <button id="indicator-cancel-btn" class="indicator-btn indicator-btn-secondary">
              Cancel
            </button>
          </div>
        </div>
      </div>
    `;

    // Inject modal into DOM
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    this.modal = document.getElementById('indicator-settings-modal');

    // Inject styles
    this._injectStyles();
  }

  /**
   * Inject CSS styles for the modal
   * @private
   */
  _injectStyles() {
    const styles = `
      <style>
        .indicator-modal {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: 10000;
        }

        .indicator-modal-overlay {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: rgba(0, 0, 0, 0.7);
        }

        .indicator-modal-content {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: #1e1e1e;
          border-radius: 8px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
          width: 900px;
          height: 600px;
          min-width: 400px;
          min-height: 300px;
          display: flex;
          flex-direction: column;
          color: #e0e0e0;
          resize: both;
          overflow: hidden;
        }

        .indicator-modal-header {
          padding: 20px 24px;
          border-bottom: 1px solid #333;
          display: flex;
          justify-content: space-between;
          align-items: center;
          cursor: move;
          user-select: none;
        }

        .indicator-modal-header h2 {
          margin: 0;
          font-size: 24px;
          color: #4ECDC4;
        }

        .indicator-modal-close {
          background: none;
          border: none;
          color: #999;
          font-size: 32px;
          cursor: pointer;
          padding: 0;
          line-height: 1;
          transition: color 0.2s;
        }

        .indicator-modal-close:hover {
          color: #fff;
        }

        .indicator-modal-body {
          flex: 1;
          overflow: hidden;
          display: flex;
          gap: 20px;
          padding: 20px 24px;
        }

        .indicator-list-panel {
          width: 250px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .indicator-list-panel h3 {
          margin: 0 0 8px 0;
          font-size: 14px;
          color: #999;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .indicator-list {
          flex: 1;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .indicator-list-item {
          padding: 12px;
          background: #2a2a2a;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s;
          border: 2px solid transparent;
        }

        .indicator-list-item:hover {
          background: #333;
        }

        .indicator-list-item.active {
          background: #2a4a4a;
          border-color: #4ECDC4;
        }

        .indicator-list-item.enabled {
          border-left: 4px solid #4ECDC4;
        }

        .indicator-list-item-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 4px;
        }

        .indicator-list-item-name {
          font-weight: 600;
          font-size: 14px;
        }

        .indicator-list-item-toggle {
          width: 40px;
          height: 20px;
          background: #555;
          border-radius: 10px;
          position: relative;
          transition: background 0.2s;
          cursor: pointer;
        }

        .indicator-list-item-toggle.enabled {
          background: #4ECDC4;
        }

        .indicator-list-item-toggle::after {
          content: '';
          position: absolute;
          top: 2px;
          left: 2px;
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          transition: transform 0.2s;
        }

        .indicator-list-item-toggle.enabled::after {
          transform: translateX(20px);
        }

        .indicator-list-item-desc {
          font-size: 11px;
          color: #999;
          line-height: 1.3;
        }

        .indicator-list-item-tags {
          display: flex;
          gap: 4px;
          flex-wrap: wrap;
          margin-top: 6px;
        }

        .indicator-tag {
          font-size: 10px;
          padding: 2px 6px;
          background: #333;
          border-radius: 3px;
          color: #4ECDC4;
        }

        .indicator-settings-panel {
          flex: 1;
          overflow-y: auto;
          background: #2a2a2a;
          border-radius: 6px;
          padding: 20px;
        }

        .indicator-empty-state {
          text-align: center;
          color: #666;
          padding: 40px 20px;
        }

        .indicator-settings-form {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .indicator-setting-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .indicator-setting-label {
          font-size: 12px;
          font-weight: 600;
          color: #ccc;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .indicator-setting-help {
          font-size: 11px;
          color: #666;
          font-weight: normal;
        }

        .indicator-setting-input {
          padding: 8px 12px;
          background: #1e1e1e;
          border: 1px solid #444;
          border-radius: 4px;
          color: #e0e0e0;
          font-size: 14px;
        }

        .indicator-setting-input:focus {
          outline: none;
          border-color: #4ECDC4;
        }

        .indicator-setting-input[type="color"] {
          height: 40px;
          cursor: pointer;
        }

        .indicator-setting-input[type="checkbox"] {
          width: 18px;
          height: 18px;
          cursor: pointer;
        }

        .indicator-setting-value-display {
          font-size: 12px;
          color: #4ECDC4;
          font-family: monospace;
        }

        .indicator-actions {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .indicator-modal-footer {
          padding: 16px 24px;
          border-top: 1px solid #333;
          display: flex;
          justify-content: flex-end;
          gap: 12px;
        }

        .indicator-btn {
          padding: 10px 20px;
          border: none;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .indicator-btn-primary {
          background: #4ECDC4;
          color: #1e1e1e;
        }

        .indicator-btn-primary:hover {
          background: #45b8ad;
        }

        .indicator-btn-secondary {
          background: #555;
          color: #e0e0e0;
        }

        .indicator-btn-secondary:hover {
          background: #666;
        }

        .indicator-btn-danger {
          background: #ff6b6b;
          color: white;
        }

        .indicator-btn-danger:hover {
          background: #ff5252;
        }

        .indicator-settings-header {
          margin-bottom: 20px;
        }

        .indicator-settings-title {
          font-size: 20px;
          color: #4ECDC4;
          margin: 0 0 8px 0;
        }

        .indicator-settings-subtitle {
          font-size: 13px;
          color: #999;
          margin: 0;
        }

        /* Scrollbar styling */
        .indicator-list::-webkit-scrollbar,
        .indicator-settings-panel::-webkit-scrollbar {
          width: 8px;
        }

        .indicator-list::-webkit-scrollbar-track,
        .indicator-settings-panel::-webkit-scrollbar-track {
          background: #1e1e1e;
        }

        .indicator-list::-webkit-scrollbar-thumb,
        .indicator-settings-panel::-webkit-scrollbar-thumb {
          background: #555;
          border-radius: 4px;
        }

        .indicator-list::-webkit-scrollbar-thumb:hover,
        .indicator-settings-panel::-webkit-scrollbar-thumb:hover {
          background: #666;
        }
      </style>
    `;

    document.head.insertAdjacentHTML('beforeend', styles);
  }

  /**
   * Attach event listeners
   * @private
   */
  _attachEventListeners() {
    // Close button
    this.modal.querySelector('.indicator-modal-close').addEventListener('click', () => {
      this.close();
    });

    // Overlay click
    this.modal.querySelector('.indicator-modal-overlay').addEventListener('click', () => {
      this.close();
    });

    // Cancel button
    document.getElementById('indicator-cancel-btn').addEventListener('click', () => {
      this.close();
    });

    // Apply button
    document.getElementById('indicator-apply-btn').addEventListener('click', () => {
      this._applySettings();
      this.close();
    });

    // Export/Import/Reset buttons
    document.getElementById('indicator-export-all-btn').addEventListener('click', () => {
      this._exportSettings();
    });

    document.getElementById('indicator-import-all-btn').addEventListener('click', () => {
      this._importSettings();
    });

    document.getElementById('indicator-reset-all-btn').addEventListener('click', () => {
      if (confirm('Reset all indicators to default settings?')) {
        indicatorRegistry.resetAll();
        this._renderIndicatorList();
        this._renderSettings(this.currentIndicator);
      }
    });

    // Drag functionality
    const header = this.modal.querySelector('.indicator-modal-header');
    const modalContent = this.modal.querySelector('.indicator-modal-content');

    header.addEventListener('mousedown', (e) => {
      // Don't drag if clicking close button
      if (e.target.classList.contains('indicator-modal-close')) return;

      this.isDragging = true;
      this.dragStartX = e.clientX;
      this.dragStartY = e.clientY;

      // Get current position
      const rect = modalContent.getBoundingClientRect();
      this.modalStartX = rect.left;
      this.modalStartY = rect.top;

      // Remove transform centering when starting drag
      modalContent.style.transform = 'none';
      modalContent.style.left = `${this.modalStartX}px`;
      modalContent.style.top = `${this.modalStartY}px`;
    });

    document.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;

      const deltaX = e.clientX - this.dragStartX;
      const deltaY = e.clientY - this.dragStartY;

      modalContent.style.left = `${this.modalStartX + deltaX}px`;
      modalContent.style.top = `${this.modalStartY + deltaY}px`;
    });

    document.addEventListener('mouseup', () => {
      if (this.isDragging) {
        this.isDragging = false;
      }
    });
  }

  /**
   * Open the modal
   */
  open() {
    this._renderIndicatorList();
    this.modal.style.display = 'block';
    document.body.style.overflow = 'hidden';

    // Reset position to center when opening
    const modalContent = this.modal.querySelector('.indicator-modal-content');
    modalContent.style.transform = 'translate(-50%, -50%)';
    modalContent.style.left = '50%';
    modalContent.style.top = '50%';
  }

  /**
   * Close the modal
   */
  close() {
    this.modal.style.display = 'none';
    document.body.style.overflow = '';
    this.currentIndicator = null;
  }

  /**
   * Render the indicator list
   * @private
   */
  _renderIndicatorList() {
    const listEl = document.getElementById('indicator-list');
    const indicators = indicatorRegistry.getAll();

    console.log(`📊 [Modal] Rendering indicator list. Found ${indicators.length} indicators:`, indicators.map(i => i.name));

    listEl.innerHTML = indicators.map(ind => `
      <div class="indicator-list-item ${ind.enabled ? 'enabled' : ''}" data-indicator="${ind.name}">
        <div class="indicator-list-item-header">
          <span class="indicator-list-item-name">${ind.name}</span>
          <div class="indicator-list-item-toggle ${ind.enabled ? 'enabled' : ''}" data-toggle="${ind.name}"></div>
        </div>
        <p class="indicator-list-item-desc">${ind.description}</p>
        <div class="indicator-list-item-tags">
          ${ind.tags.map(tag => `<span class="indicator-tag">${tag}</span>`).join('')}
        </div>
      </div>
    `).join('');

    // Attach click handlers
    listEl.querySelectorAll('.indicator-list-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if (!e.target.classList.contains('indicator-list-item-toggle')) {
          const indName = item.dataset.indicator;
          this._selectIndicator(indName);
        }
      });
    });

    // Attach toggle handlers
    listEl.querySelectorAll('.indicator-list-item-toggle').forEach(toggle => {
      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const indName = toggle.dataset.toggle;
        indicatorRegistry.toggle(indName);
        this._renderIndicatorList();
      });
    });
  }

  /**
   * Select an indicator for editing
   * @param {string} name - Indicator name
   * @private
   */
  _selectIndicator(name) {
    this.currentIndicator = name;

    // Update UI
    document.querySelectorAll('.indicator-list-item').forEach(item => {
      item.classList.remove('active');
    });
    document.querySelector(`[data-indicator="${name}"]`)?.classList.add('active');

    // Render settings
    this._renderSettings(name);
  }

  /**
   * Render settings form for an indicator
   * @param {string} name - Indicator name
   * @private
   */
  _renderSettings(name) {
    const contentEl = document.getElementById('indicator-settings-content');

    if (!name) {
      contentEl.innerHTML = '<p class="indicator-empty-state">Select an indicator to configure</p>';
      return;
    }

    const indicator = indicatorRegistry.get(name);
    if (!indicator) return;

    const schema = indicator.getSettingsSchema();
    const currentSettings = indicator.getSettings();

    let formHTML = `
      <div class="indicator-settings-header">
        <h3 class="indicator-settings-title">${indicator.name} Settings</h3>
        <p class="indicator-settings-subtitle">${indicator.description}</p>
      </div>
      <div class="indicator-settings-form">
    `;

    this.formElements.clear();

    Object.entries(schema).forEach(([fieldName, fieldSchema]) => {
      const value = currentSettings[fieldName];
      let inputHTML = '';

      switch (fieldSchema.type) {
        case 'number':
          inputHTML = `
            <input type="number"
              id="ind-${fieldName}"
              class="indicator-setting-input"
              min="${fieldSchema.min}"
              max="${fieldSchema.max}"
              step="${fieldSchema.step}"
              value="${value}">
            <span class="indicator-setting-value-display">${value}</span>
          `;
          break;

        case 'color':
          inputHTML = `
            <input type="color"
              id="ind-${fieldName}"
              class="indicator-setting-input"
              value="${value}">
          `;
          break;

        case 'boolean':
          inputHTML = `
            <input type="checkbox"
              id="ind-${fieldName}"
              class="indicator-setting-input"
              ${value ? 'checked' : ''}>
          `;
          break;

        default:
          inputHTML = `
            <input type="text"
              id="ind-${fieldName}"
              class="indicator-setting-input"
              value="${value}">
          `;
      }

      formHTML += `
        <div class="indicator-setting-group">
          <label class="indicator-setting-label" for="ind-${fieldName}">
            ${fieldSchema.label}
            ${fieldSchema.description ? `<span class="indicator-setting-help">?</span>` : ''}
          </label>
          ${inputHTML}
          ${fieldSchema.description ? `<small style="color: #666; font-size: 11px;">${fieldSchema.description}</small>` : ''}
        </div>
      `;
    });

    formHTML += '</div>';

    contentEl.innerHTML = formHTML;

    // Store form element references
    Object.keys(schema).forEach(fieldName => {
      this.formElements.set(fieldName, document.getElementById(`ind-${fieldName}`));
    });

    // Add live value updates for number inputs
    this.formElements.forEach((el, fieldName) => {
      if (el.type === 'number') {
        el.addEventListener('input', (e) => {
          const valueDisplay = el.nextElementSibling;
          if (valueDisplay && valueDisplay.classList.contains('indicator-setting-value-display')) {
            valueDisplay.textContent = e.target.value;
          }
        });
      }
      // Add instant apply for checkboxes (especially for ML toggle)
      else if (el.type === 'checkbox') {
        // Capture the indicator name in closure to avoid this.currentIndicator becoming null
        const indicatorName = this.currentIndicator;
        el.addEventListener('change', (e) => {
          // Apply settings directly using the captured indicator name
          const indicator = indicatorRegistry.get(indicatorName);
          if (!indicator) return;

          // Read current checkbox value and apply immediately
          const newSettings = {};
          this.formElements.forEach((formEl, formFieldName) => {
            if (formEl.type === 'checkbox') {
              newSettings[formFieldName] = formEl.checked;
            } else if (formEl.type === 'number') {
              newSettings[formFieldName] = parseFloat(formEl.value);
            } else {
              newSettings[formFieldName] = formEl.value;
            }
          });

          indicatorRegistry.updateSettings(indicatorName, newSettings);
        });
      }
    });
  }

  /**
   * Apply current settings
   * @private
   */
  _applySettings() {
    if (!this.currentIndicator) return;

    const indicator = indicatorRegistry.get(this.currentIndicator);
    if (!indicator) return;

    const newSettings = {};

    this.formElements.forEach((el, fieldName) => {
      if (el.type === 'checkbox') {
        newSettings[fieldName] = el.checked;
      } else if (el.type === 'number') {
        newSettings[fieldName] = parseFloat(el.value);
      } else {
        newSettings[fieldName] = el.value;
      }
    });

    indicatorRegistry.updateSettings(this.currentIndicator, newSettings);
  }

  /**
   * Export all settings to JSON file
   * @private
   */
  _exportSettings() {
    const settings = indicatorRegistry.exportAllSettings();
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `indicator-settings-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Import settings from JSON file
   * @private
   */
  _importSettings() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const settings = JSON.parse(event.target.result);
            indicatorRegistry.importAllSettings(settings);
            this._renderIndicatorList();
            this._renderSettings(this.currentIndicator);
          } catch (error) {
            alert('Error importing settings: ' + error.message);
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  }
}

// Create singleton instance
export const indicatorSettingsModal = new IndicatorSettingsModal();

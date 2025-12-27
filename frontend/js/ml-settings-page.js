/**
 * ML Settings Page
 * Handles model configuration creation, activation, and management
 */

// Load models on page load
document.addEventListener('DOMContentLoaded', () => {
    loadModels();

    // Handle form submission
    document.getElementById('create-model-form').addEventListener('submit', (e) => {
        e.preventDefault();
        createModel();
    });
});

/**
 * Load all saved models
 */
async function loadModels() {
    try {
        const response = await fetch('/ml-configs');
        const data = await response.json();

        if (data.status === 'success') {
            renderModels(data.configurations);
        } else {
            console.error('Failed to load models:', data.message);
            showEmptyState();
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showEmptyState();
    }
}

/**
 * Render models list
 */
function renderModels(models) {
    const container = document.getElementById('models-list');
    const emptyState = document.getElementById('empty-models');

    if (!models || models.length === 0) {
        showEmptyState();
        return;
    }

    emptyState.style.display = 'none';
    container.innerHTML = '';

    models.forEach(model => {
        const card = createModelCard(model);
        container.appendChild(card);
    });
}

/**
 * Create a model card element
 */
function createModelCard(model) {
    const card = document.createElement('div');
    card.className = 'model-card' + (model.active ? ' active' : '');

    // Get human-readable labels
    const analysisLabel = getAnalysisTypeLabel(model.analysis_type);
    const winCriteriaLabel = getWinCriteriaLabel(model.win_criteria);

    card.innerHTML = `
        <div class="model-card-header">
            <div class="model-name">${model.name}</div>
            ${model.active ? '<div class="model-badge">ACTIVE</div>' : ''}
        </div>

        <div class="model-stats">
            <div class="model-stat">
                <div class="model-stat-value">${model.trades_count || 0}</div>
                <div class="model-stat-label">Trades</div>
            </div>
            <div class="model-stat">
                <div class="model-stat-value">${(model.win_rate || 0).toFixed(1)}%</div>
                <div class="model-stat-label">Win Rate</div>
            </div>
            <div class="model-stat">
                <div class="model-stat-value">${model.win_count || 0}/${model.loss_count || 0}</div>
                <div class="model-stat-label">W/L</div>
            </div>
        </div>

        <div class="model-config-summary">
            <div><strong>Analysis:</strong> ${analysisLabel}</div>
            <div><strong>Win Criteria:</strong> ${winCriteriaLabel}</div>
            <div><strong>Hold Period:</strong> ${model.hold_period_days || 14} days</div>
            <div><strong>Target:</strong> +${model.win_threshold_pct || 10}% / ${model.loss_threshold_pct || -5}%</div>
        </div>

        <div class="model-actions">
            ${!model.active ? `<button class="tos-btn-primary" onclick="activateModel('${model.name}')">✓ Activate</button>` : '<button class="tos-btn-secondary" disabled>Active Model</button>'}
            <button class="tos-btn-secondary" onclick="viewModelDetails('${model.name}')">📊 Details</button>
            ${!model.active ? `<button class="tos-btn-secondary" onclick="deleteModel('${model.name}')">🗑️ Delete</button>` : ''}
        </div>
    `;

    return card;
}

/**
 * Create new model
 */
async function createModel() {
    const name = document.getElementById('model-name').value.trim();
    const analysisType = document.getElementById('analysis-type').value;
    const winCriteria = document.getElementById('win-criteria').value;
    const holdPeriod = parseInt(document.getElementById('hold-period').value);
    const winThreshold = parseFloat(document.getElementById('win-threshold').value);
    const lossThreshold = parseFloat(document.getElementById('loss-threshold').value);

    // Get philosophy checkboxes
    const philosophy = [];
    const checkboxes = document.querySelectorAll('#create-model-form input[type="checkbox"]:checked');
    checkboxes.forEach(cb => philosophy.push(cb.value));

    // Validation
    if (!name) {
        alert('Please enter a model name');
        return;
    }

    if (!analysisType) {
        alert('Please select an analysis type');
        return;
    }

    if (!winCriteria) {
        alert('Please select win criteria');
        return;
    }

    // Create model
    try {
        const response = await fetch('/ml-config-create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                analysis_type: analysisType,
                philosophy,
                win_criteria: winCriteria,
                hold_period_days: holdPeriod,
                win_threshold_pct: winThreshold,
                loss_threshold_pct: lossThreshold
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert(`✅ Model "${name}" created successfully!\n\nThe model will start training with the next automation cycle.`);

            // Reset form
            document.getElementById('create-model-form').reset();

            // Reload models
            loadModels();
        } else {
            alert(`❌ Failed to create model:\n\n${data.message}`);
        }
    } catch (error) {
        console.error('Error creating model:', error);
        alert(`❌ Error creating model:\n\n${error.message}`);
    }
}

/**
 * Activate a model
 */
async function activateModel(name) {
    if (!confirm(`Activate model "${name}"?\n\nThis will become the active model for all future scans and learning cycles.`)) {
        return;
    }

    try {
        const response = await fetch('/ml-config-activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert(`✅ Model "${name}" is now active!\n\nNext automation cycle will use this model's configuration.`);
            loadModels();
        } else {
            alert(`❌ Failed to activate model:\n\n${data.message}`);
        }
    } catch (error) {
        console.error('Error activating model:', error);
        alert(`❌ Error activating model:\n\n${error.message}`);
    }
}

/**
 * Delete a model
 */
async function deleteModel(name) {
    if (!confirm(`Delete model "${name}"?\n\nThis will permanently delete the model configuration and all its data.\n\nThis action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/ml-config-delete/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.status === 'success') {
            alert(`✅ Model "${name}" deleted successfully.`);
            loadModels();
        } else {
            alert(`❌ Failed to delete model:\n\n${data.message}`);
        }
    } catch (error) {
        console.error('Error deleting model:', error);
        alert(`❌ Error deleting model:\n\n${error.message}`);
    }
}

/**
 * View model details
 */
async function viewModelDetails(name) {
    try {
        const response = await fetch(`/ml-config-performance/${encodeURIComponent(name)}`);
        const data = await response.json();

        if (data.status === 'success') {
            const model = data.configuration;

            const details = `
Model: ${model.name}
Analysis Type: ${getAnalysisTypeLabel(model.analysis_type)}
Win Criteria: ${getWinCriteriaLabel(model.win_criteria)}
Philosophy: ${(model.philosophy || []).join(', ') || 'None'}

Performance:
- Total Trades: ${model.trades_count || 0}
- Wins: ${model.win_count || 0}
- Losses: ${model.loss_count || 0}
- Win Rate: ${(model.win_rate || 0).toFixed(1)}%
- Avg Win: ${(model.avg_win_pct || 0).toFixed(1)}%
- Avg Loss: ${(model.avg_loss_pct || 0).toFixed(1)}%
- Profit Factor: ${(model.profit_factor || 0).toFixed(2)}
- Total P/L: $${(model.total_profit_loss || 0).toFixed(2)}

Settings:
- Hold Period: ${model.hold_period_days || 14} days
- Win Target: +${model.win_threshold_pct || 10}%
- Stop Loss: ${model.loss_threshold_pct || -5}%

Created: ${new Date(model.created).toLocaleString()}
Last Modified: ${new Date(model.modified).toLocaleString()}
Status: ${model.active ? 'ACTIVE' : 'Inactive'}
            `;

            alert(details);
        } else {
            alert(`❌ Failed to load model details:\n\n${data.message}`);
        }
    } catch (error) {
        console.error('Error loading model details:', error);
        alert(`❌ Error loading model details:\n\n${error.message}`);
    }
}

/**
 * Show empty state
 */
function showEmptyState() {
    document.getElementById('models-list').innerHTML = '';
    document.getElementById('empty-models').style.display = 'block';
}

/**
 * Get human-readable analysis type label
 */
function getAnalysisTypeLabel(type) {
    const labels = {
        'price_action': 'Pure Price Action',
        'volume_profile': 'Volume Profile',
        'raw_ohlcv': 'Raw OHLCV Data',
        'market_structure': 'Market Structure'
    };
    return labels[type] || type;
}

/**
 * Get human-readable win criteria label
 */
function getWinCriteriaLabel(criteria) {
    const labels = {
        'price_movement': 'Price Movement Only',
        'quality_of_move': 'Quality of Move',
        'speed_of_move': 'Speed of Move',
        'risk_adjusted': 'Risk-Adjusted'
    };
    return labels[criteria] || criteria;
}

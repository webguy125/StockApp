/**
 * Heat Map Page JavaScript
 * Handles dynamic loading and rendering of agent signal heat maps
 */

// State
let currentTimeframe = 'all';
let heatmapData = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadHeatmapData();
});

/**
 * Load heat map data from backend
 */
async function loadHeatmapData(timeframe = 'all') {
    try {
        showLoading();

        const url = `/heatmap-data?timeframe=${timeframe}`;
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (data.status === 'no_data') {
            showEmptyState(data.message);
            return;
        }

        if (data.status === 'success') {
            heatmapData = data.heatmaps;
            renderHeatmaps(data.heatmaps, data.summary);
            updateCounts(data.summary || data.heatmaps);
            updateMetadata(data.heatmaps.metadata);
            hideLoading();
        }

    } catch (error) {
        console.error('Error loading heat map data:', error);
        showEmptyState(`Error loading heat map data: ${error.message}`);
    }
}

/**
 * Render heat maps for all timeframes
 */
function renderHeatmaps(heatmaps, summary) {
    const timeframes = ['intraday', 'daily', 'monthly'];

    timeframes.forEach(timeframe => {
        const grid = document.getElementById(`grid-${timeframe}`);
        const section = document.getElementById(`section-${timeframe}`);

        if (!grid) return;

        // Clear existing content
        grid.innerHTML = '';

        const signals = heatmaps[timeframe] || [];

        if (signals.length === 0) {
            grid.innerHTML = '<div class="empty-grid-message">No signals available for this timeframe</div>';
            return;
        }

        // Render each signal
        signals.forEach(signal => {
            const cell = createSignalCell(signal);
            grid.appendChild(cell);
        });

        // Show section
        section.style.display = 'block';
    });
}

/**
 * Create a signal cell element
 */
function createSignalCell(signal) {
    const cell = document.createElement('div');
    cell.className = `signal-cell signal-${signal.recommendation.replace('_', '-')}`;
    cell.onclick = () => loadSymbolInChart(signal.symbol);

    // Symbol
    const symbolDiv = document.createElement('div');
    symbolDiv.className = 'signal-symbol';
    symbolDiv.textContent = signal.symbol;
    cell.appendChild(symbolDiv);

    // Emoji
    if (signal.emoji) {
        const emojiDiv = document.createElement('div');
        emojiDiv.className = 'signal-emoji';
        emojiDiv.textContent = signal.emoji;
        cell.appendChild(emojiDiv);
    }

    // Score
    const scoreDiv = document.createElement('div');
    scoreDiv.className = 'signal-score';
    scoreDiv.textContent = `Score: ${signal.score.toFixed(1)}`;
    cell.appendChild(scoreDiv);

    // Recommendation
    const recDiv = document.createElement('div');
    recDiv.className = 'signal-recommendation';
    recDiv.textContent = signal.recommendation.replace('_', ' ');
    cell.appendChild(recDiv);

    // Confidence bar
    const confDiv = document.createElement('div');
    confDiv.className = 'signal-confidence';
    const confBar = document.createElement('div');
    confBar.className = 'confidence-bar';
    confBar.style.width = `${signal.confidence * 100}%`;
    confDiv.appendChild(confBar);
    cell.appendChild(confDiv);

    // Tooltip
    const tooltip = createTooltip(signal);
    cell.appendChild(tooltip);

    return cell;
}

/**
 * Create tooltip for signal cell
 */
function createTooltip(signal) {
    const tooltip = document.createElement('div');
    tooltip.className = 'signal-tooltip';

    const rows = [
        { label: 'Symbol:', value: signal.symbol },
        { label: 'Score:', value: signal.score.toFixed(1) },
        { label: 'Confidence:', value: `${(signal.confidence * 100).toFixed(0)}%` },
        { label: 'Signal:', value: signal.recommendation.replace('_', ' ').toUpperCase() },
        { label: 'Timestamp:', value: formatTimestamp(signal.timestamp) }
    ];

    rows.forEach(row => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'tooltip-row';

        const label = document.createElement('span');
        label.className = 'tooltip-label';
        label.textContent = row.label;

        const value = document.createElement('span');
        value.className = 'tooltip-value';
        value.textContent = row.value;

        rowDiv.appendChild(label);
        rowDiv.appendChild(value);
        tooltip.appendChild(rowDiv);
    });

    return tooltip;
}

/**
 * Switch between timeframes
 */
function switchTimeframe(timeframe) {
    currentTimeframe = timeframe;

    // Update active tab
    document.querySelectorAll('.timeframe-tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.dataset.timeframe === timeframe) {
            tab.classList.add('active');
        }
    });

    // Show/hide sections based on selected timeframe
    const sections = ['intraday', 'daily', 'monthly'];

    if (timeframe === 'all') {
        // Show all sections
        sections.forEach(tf => {
            const section = document.getElementById(`section-${tf}`);
            if (section) section.classList.remove('hidden');
        });
    } else {
        // Show only selected section
        sections.forEach(tf => {
            const section = document.getElementById(`section-${tf}`);
            if (section) {
                if (tf === timeframe) {
                    section.classList.remove('hidden');
                } else {
                    section.classList.add('hidden');
                }
            }
        });
    }
}

/**
 * Update signal counts in tabs
 */
function updateCounts(summary) {
    if (summary.intraday_count !== undefined) {
        document.getElementById('count-intraday').textContent = summary.intraday_count;
        document.getElementById('count-daily').textContent = summary.daily_count;
        document.getElementById('count-monthly').textContent = summary.monthly_count;
        document.getElementById('meta-total').textContent = summary.total_unique_symbols || 0;
    } else {
        // Fallback: count from heatmaps object
        document.getElementById('count-intraday').textContent = (summary.intraday || []).length;
        document.getElementById('count-daily').textContent = (summary.daily || []).length;
        document.getElementById('count-monthly').textContent = (summary.monthly || []).length;
    }
}

/**
 * Update metadata footer
 */
function updateMetadata(metadata) {
    if (!metadata) return;

    document.getElementById('meta-timestamp').textContent = formatTimestamp(metadata.generated_at);
    document.getElementById('meta-source').textContent = metadata.source || 'Agent Learning Loop';
    document.getElementById('heatmap-footer').style.display = 'flex';
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '-';

    try {
        const date = new Date(timestamp);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (e) {
        return timestamp;
    }
}

/**
 * Load symbol in main chart
 */
function loadSymbolInChart(symbol) {
    // Store symbol in localStorage for chart to pick up
    localStorage.setItem('selectedSymbol', symbol);

    // Redirect to main chart page
    window.location.href = '/';
}

/**
 * Refresh heat maps
 */
async function refreshHeatmaps() {
    const refreshIcon = document.getElementById('refresh-icon');
    refreshIcon.classList.add('refreshing');

    try {
        // Optionally trigger agent refresh first
        // await fetch('/agent-signals/refresh', { method: 'POST' });

        // Reload data
        await loadHeatmapData(currentTimeframe);
    } catch (error) {
        console.error('Error refreshing:', error);
    } finally {
        refreshIcon.classList.remove('refreshing');
    }
}

/**
 * Run agents to generate new signals
 */
async function runAgents() {
    const emptyState = document.getElementById('empty-state');
    emptyState.innerHTML = '<div class="loading-spinner"></div><p>Running agents...</p>';

    try {
        const response = await fetch('/agent-signals/refresh', { method: 'POST' });
        const data = await response.json();

        if (data.status === 'success') {
            // Wait a moment for processing
            await new Promise(resolve => setTimeout(resolve, 2000));
            // Reload data
            await loadHeatmapData();
        } else {
            showEmptyState(`Error: ${data.message}`);
        }
    } catch (error) {
        showEmptyState(`Error running agents: ${error.message}`);
    }
}

/**
 * Show loading state
 */
function showLoading() {
    document.getElementById('loading-state').style.display = 'block';
    document.getElementById('empty-state').style.display = 'none';

    const sections = ['intraday', 'daily', 'monthly'];
    sections.forEach(tf => {
        const section = document.getElementById(`section-${tf}`);
        if (section) section.style.display = 'none';
    });
}

/**
 * Hide loading state
 */
function hideLoading() {
    document.getElementById('loading-state').style.display = 'none';
}

/**
 * Show empty state
 */
function showEmptyState(message) {
    hideLoading();
    const emptyState = document.getElementById('empty-state');
    emptyState.style.display = 'block';

    const sections = ['intraday', 'daily', 'monthly'];
    sections.forEach(tf => {
        const section = document.getElementById(`section-${tf}`);
        if (section) section.style.display = 'none';
    });

    // Update message if provided
    if (message) {
        const p = emptyState.querySelector('p');
        if (p) p.textContent = message;
    }
}

// Signal Heatmap JavaScript Module
class SignalHeatmap {
    constructor() {
        this.signals = [];
        this.isExpanded = true;
        this.container = document.getElementById('signal-heatmap');
        this.grid = document.getElementById('heatmap-grid');
        this.refreshInterval = null;

        // Initialize
        this.init();
    }

    init() {
        // Load signals on startup
        this.loadSignals();

        // Set up auto-refresh every 5 minutes
        this.refreshInterval = setInterval(() => {
            this.loadSignals();
        }, 5 * 60 * 1000);
    }

    async loadSignals() {
        try {
            this.showLoading();

            const response = await fetch('/agent-signals');
            const data = await response.json();

            if (data.status === 'success') {
                this.signals = data.signals;
                this.renderHeatmap();
            } else if (data.status === 'no_data') {
                this.showEmpty(data.message);
            } else {
                this.showError('Failed to load signals');
            }
        } catch (error) {
            console.error('Error loading agent signals:', error);
            this.showError('Error loading signals');
        }
    }

    renderHeatmap() {
        if (!this.signals || this.signals.length === 0) {
            this.showEmpty('No signals available');
            return;
        }

        // Clear existing content
        this.grid.innerHTML = '';

        // Separate stocks and crypto
        const cryptoSignals = this.signals.filter(s => s.symbol.endsWith('-USD'));
        const stockSignals = this.signals.filter(s => !s.symbol.endsWith('-USD'));

        // Create crypto section if there are crypto signals
        if (cryptoSignals.length > 0) {
            const cryptoHeader = document.createElement('div');
            cryptoHeader.className = 'heatmap-section-header';
            cryptoHeader.innerHTML = `<span>Crypto (${cryptoSignals.length})</span>`;
            this.grid.appendChild(cryptoHeader);

            const cryptoGrid = document.createElement('div');
            cryptoGrid.className = 'heatmap-section-grid';

            // Take top 10 crypto signals
            cryptoSignals.slice(0, 10).forEach(signal => {
                const cell = this.createSignalCell(signal);
                cryptoGrid.appendChild(cell);
            });

            this.grid.appendChild(cryptoGrid);
        }

        // Create stock section if there are stock signals
        if (stockSignals.length > 0) {
            const stockHeader = document.createElement('div');
            stockHeader.className = 'heatmap-section-header';
            stockHeader.innerHTML = `<span>Stocks (${stockSignals.length})</span>`;
            this.grid.appendChild(stockHeader);

            const stockGrid = document.createElement('div');
            stockGrid.className = 'heatmap-section-grid';

            // Take top 10 stock signals
            stockSignals.slice(0, 10).forEach(signal => {
                const cell = this.createSignalCell(signal);
                stockGrid.appendChild(cell);
            });

            this.grid.appendChild(stockGrid);
        }
    }

    createSignalCell(signal) {
        const cell = document.createElement('div');
        cell.className = `signal-cell signal-${signal.recommendation.replace('_', '-')}`;

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
        const tooltip = this.createTooltip(signal);
        cell.appendChild(tooltip);

        // Click handler - load symbol in chart or navigate to heat map page
        cell.addEventListener('click', (e) => {
            // If shift key is held, navigate to heat map page
            if (e.shiftKey) {
                window.location.href = '/heatmap';
            } else {
                this.loadSymbolInChart(signal.symbol);
            }
        });

        // Double-click handler - always navigate to heat map page
        cell.addEventListener('dblclick', () => {
            window.location.href = '/heatmap';
        });

        return cell;
    }

    createTooltip(signal) {
        const tooltip = document.createElement('div');
        tooltip.className = 'signal-tooltip';

        const rows = [
            { label: 'Symbol:', value: signal.symbol },
            { label: 'Score:', value: signal.score.toFixed(1) },
            { label: 'Confidence:', value: `${(signal.confidence * 100).toFixed(0)}%` },
            { label: 'Signal:', value: signal.recommendation.replace('_', ' ').toUpperCase() }
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

    loadSymbolInChart(symbol) {
        // Check if tosApp is available
        if (window.tosApp && window.tosApp.chart) {
            // Update symbol input
            const symbolInput = document.getElementById('symbolInput');
            if (symbolInput) {
                symbolInput.value = symbol;
            }

            // Trigger chart update
            window.tosApp.chart.updateSymbol(symbol);
        } else {
            // Fallback - just update the input
            const symbolInput = document.getElementById('symbolInput');
            if (symbolInput) {
                symbolInput.value = symbol;
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                symbolInput.dispatchEvent(event);
            }
        }
    }

    showLoading() {
        this.grid.innerHTML = '<div class="heatmap-loading"></div>';
    }

    showEmpty(message) {
        this.grid.innerHTML = `<div class="heatmap-empty">${message || 'No signals available'}</div>`;
    }

    showError(message) {
        this.grid.innerHTML = `<div class="heatmap-empty" style="color: var(--error-color);">${message}</div>`;
    }

    async refreshSignals() {
        const refreshBtn = document.querySelector('.tos-signal-heatmap .tos-panel-icon-btn[title="Refresh Signals"]');
        if (refreshBtn) {
            refreshBtn.classList.add('refreshing');
        }

        try {
            // First trigger agent system refresh
            const refreshResponse = await fetch('/agent-signals/refresh', { method: 'POST' });
            const refreshData = await refreshResponse.json();

            if (refreshData.status === 'success') {
                // Wait a bit for agents to complete
                await new Promise(resolve => setTimeout(resolve, 2000));
                // Then load the new signals
                await this.loadSignals();
            } else {
                this.showError('Failed to refresh signals');
            }
        } catch (error) {
            console.error('Error refreshing signals:', error);
            this.showError('Error refreshing signals');
        } finally {
            if (refreshBtn) {
                refreshBtn.classList.remove('refreshing');
            }
        }
    }

    toggle() {
        this.isExpanded = !this.isExpanded;

        if (this.isExpanded) {
            this.container.classList.remove('collapsed');
            this.container.classList.add('expanded');
        } else {
            this.container.classList.add('collapsed');
            this.container.classList.remove('expanded');
        }

        // Update toggle button
        const toggleBtn = document.querySelector('.tos-signal-heatmap .tos-panel-icon-btn[title="Expand/Collapse"]');
        if (toggleBtn) {
            toggleBtn.textContent = this.isExpanded ? '▼' : '▶';
        }
    }

    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }
}

// Global functions for HTML onclick handlers
function toggleHeatmap() {
    if (window.signalHeatmap) {
        window.signalHeatmap.toggle();
    }
}

function refreshAgentSignals() {
    if (window.signalHeatmap) {
        window.signalHeatmap.refreshSignals();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize signal heatmap
    window.signalHeatmap = new SignalHeatmap();
});
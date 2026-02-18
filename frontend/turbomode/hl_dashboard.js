// HL Dashboard JavaScript - READ-ONLY, ADVISORY-ONLY Analytics Display

let currentSymbol = 'AAPL';

// Load HL data for a symbol
async function loadHLData() {
    const symbolInput = document.getElementById('symbolInput');
    const symbol = symbolInput.value.trim().toUpperCase();

    if (!symbol) {
        showError('Please enter a valid symbol');
        return;
    }

    currentSymbol = symbol;

    // Show loading state
    document.getElementById('content').innerHTML = '<div class="loading">Loading HL analytics for ' + symbol + '...</div>';

    try {
        const response = await fetch(`/api/hl/${symbol}`);

        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }

        const hlData = await response.json();

        if (hlData.error) {
            throw new Error(hlData.error);
        }

        renderHLDashboard(hlData);

    } catch (error) {
        console.error('[HL DASHBOARD] Error loading data:', error);
        showError(`Failed to load HL analytics: ${error.message}`);
    }
}

// Render the complete HL dashboard
function renderHLDashboard(hlData) {
    const content = document.getElementById('content');

    const html = `
        ${renderHLSummary(hlData)}
        ${renderHLBlocks(hlData)}
    `;

    content.innerHTML = html;
}

// Render HL summary section
function renderHLSummary(hlData) {
    const bias = hlData.hl_bias || 'neutral';
    const confidence = hlData.hl_confidence || 0.5;
    const summary = hlData.hl_summary || 'No summary available';

    const biasClass = bias.toLowerCase();
    const confidencePct = (confidence * 100).toFixed(0);

    return `
        <div class="hl-summary">
            <h2>${hlData.symbol} - HL Analytics</h2>

            <div class="hl-bias-container">
                <div class="hl-bias-box">
                    <div class="hl-bias-label">HL Bias</div>
                    <div class="hl-bias-value ${biasClass}">${bias.toUpperCase()}</div>
                </div>
                <div class="hl-bias-box">
                    <div class="hl-bias-label">HL Confidence</div>
                    <div class="hl-bias-value">${confidencePct}%</div>
                    <div class="hl-confidence-bar">
                        <div class="hl-confidence-fill" style="width: ${confidencePct}%"></div>
                    </div>
                </div>
            </div>

            <div class="hl-summary-text">
                ${summary}
            </div>
        </div>
    `;
}

// Render all HL blocks
function renderHLBlocks(hlData) {
    return `
        <div class="blocks-grid">
            ${renderTrendBlock(hlData.trend)}
            ${renderVolumeBlock(hlData.volume)}
            ${renderVolatilityBlock(hlData.volatility)}
            ${renderProbabilityBlock(hlData.probability_drift)}
            ${renderStructureBlock(hlData.structure)}
        </div>
    `;
}

// Render Trend block
function renderTrendBlock(trend) {
    if (!trend) {
        return '<div class="hl-block"><h3>📈 Trend Analytics</h3><p>No trend data available</p></div>';
    }

    const direction = trend.trend_direction || 'neutral';
    const strength = trend.trend_strength || 0.5;
    const strengthPct = (strength * 100).toFixed(0);
    const alignment = trend.multi_tf_alignment ? 'Yes' : 'No';
    const emaAlignment = trend.ema_alignment || 'mixed';
    const swingStructure = trend.swing_structure || 'choppy';
    const commentary = trend.commentary || 'No commentary available';

    const directionClass = direction === 'uptrend' ? 'strong' : direction === 'downtrend' ? 'weak' : 'neutral';

    return `
        <div class="hl-block">
            <h3><span class="block-icon">📈</span> Trend Analytics</h3>

            <div class="metric-row">
                <span class="metric-label">Direction</span>
                <span class="metric-value ${directionClass}">${direction.toUpperCase()}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Strength</span>
                <span class="metric-value">${strengthPct}%</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Multi-TF Alignment</span>
                <span class="metric-value">${alignment}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">EMA Alignment</span>
                <span class="metric-value">${emaAlignment}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Swing Structure</span>
                <span class="metric-value">${swingStructure}</span>
            </div>

            <div class="commentary-box">
                <div class="commentary-text">${commentary}</div>
            </div>
        </div>
    `;
}

// Render Volume block
function renderVolumeBlock(volume) {
    if (!volume) {
        return '<div class="hl-block"><h3>📊 Volume Analytics</h3><p>No volume data available</p></div>';
    }

    const initialVol = volume.initial_wave_volume || 0;
    const correctionVol = volume.correction_wave_volume || 0;
    const retestVol = volume.retest_wave_volume || 0;
    const retestStrength = volume.retest_strength_pct || 0;
    const classification = volume.classification || 'unavailable';
    const commentary = volume.commentary || 'No commentary available';

    const classificationClass = classification === 'strong' ? 'strong' : classification === 'weak' ? 'weak' : 'neutral';

    return `
        <div class="hl-block">
            <h3><span class="block-icon">📊</span> Volume Analytics (ORD)</h3>

            <div class="metric-row">
                <span class="metric-label">Initial Wave Volume</span>
                <span class="metric-value">${formatNumber(initialVol)}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Correction Wave Volume</span>
                <span class="metric-value">${formatNumber(correctionVol)}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Retest Wave Volume</span>
                <span class="metric-value">${formatNumber(retestVol)}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Retest Strength</span>
                <span class="metric-value">${retestStrength.toFixed(1)}%</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Classification</span>
                <span class="metric-value ${classificationClass}">${classification.toUpperCase()}</span>
            </div>

            <div class="commentary-box">
                <div class="commentary-text">${commentary}</div>
            </div>
        </div>
    `;
}

// Render Volatility block
function renderVolatilityBlock(volatility) {
    if (!volatility) {
        return '<div class="hl-block"><h3>⚡ Volatility Analytics</h3><p>No volatility data available</p></div>';
    }

    const atr = volatility.atr_current || 0;
    const percentile = volatility.atr_percentile || 0.5;
    const percentilePct = (percentile * 100).toFixed(0);
    const state = volatility.volatility_state || 'stable';
    const regime = volatility.regime || 'normal';
    const commentary = volatility.commentary || 'No commentary available';

    const regimeClass = regime === 'extreme' || regime === 'elevated' ? 'weak' : regime === 'low' ? 'neutral' : 'strong';

    return `
        <div class="hl-block">
            <h3><span class="block-icon">⚡</span> Volatility Analytics</h3>

            <div class="metric-row">
                <span class="metric-label">ATR (Current)</span>
                <span class="metric-value">$${atr.toFixed(2)}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">ATR Percentile</span>
                <span class="metric-value">${percentilePct}%</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Volatility State</span>
                <span class="metric-value">${state}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Regime</span>
                <span class="metric-value ${regimeClass}">${regime.toUpperCase()}</span>
            </div>

            <div class="commentary-box">
                <div class="commentary-text">${commentary}</div>
            </div>
        </div>
    `;
}

// Render Probability Drift block
function renderProbabilityBlock(probability) {
    if (!probability) {
        return '<div class="hl-block"><h3>🎯 Probability Drift</h3><p>No probability data available</p></div>';
    }

    const initialProb = probability.initial_probability || 0.5;
    const liveProb = probability.live_probability || 0.5;
    const delta = probability.delta || 0;
    const direction = probability.drift_direction || 'stable';
    const commentary = probability.commentary || 'No commentary available';

    const initialPct = (initialProb * 100).toFixed(1);
    const livePct = (liveProb * 100).toFixed(1);
    const deltaPct = (delta * 100).toFixed(1);

    const directionClass = direction === 'strengthening' ? 'strong' : direction === 'weakening' ? 'weak' : 'neutral';
    const deltaSign = delta >= 0 ? '+' : '';

    return `
        <div class="hl-block">
            <h3><span class="block-icon">🎯</span> Probability Drift</h3>

            <div class="metric-row">
                <span class="metric-label">Initial Probability</span>
                <span class="metric-value">${initialPct}%</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Live Probability</span>
                <span class="metric-value">${livePct}%</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Delta</span>
                <span class="metric-value ${directionClass}">${deltaSign}${deltaPct}%</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Drift Direction</span>
                <span class="metric-value ${directionClass}">${direction.toUpperCase()}</span>
            </div>

            <div class="commentary-box">
                <div class="commentary-text">${commentary}</div>
            </div>
        </div>
    `;
}

// Render Structure block
function renderStructureBlock(structure) {
    if (!structure) {
        return '<div class="hl-block"><h3>🏗️ Market Structure</h3><p>No structure data available</p></div>';
    }

    const currentPrice = structure.current_price || 0;
    const support = structure.key_support || 0;
    const resistance = structure.key_resistance || 0;
    const distSupport = structure.distance_to_support_pct || 0;
    const distResistance = structure.distance_to_resistance_pct || 0;
    const state = structure.structure_state || 'holding';
    const commentary = structure.commentary || 'No commentary available';

    const stateClass = state === 'holding' ? 'strong' : state === 'breaking' ? 'weak' : 'neutral';

    return `
        <div class="hl-block">
            <h3><span class="block-icon">🏗️</span> Market Structure</h3>

            <div class="metric-row">
                <span class="metric-label">Current Price</span>
                <span class="metric-value">$${currentPrice.toFixed(2)}</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Key Support</span>
                <span class="metric-value">$${support.toFixed(2)} (${distSupport.toFixed(1)}% below)</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Key Resistance</span>
                <span class="metric-value">$${resistance.toFixed(2)} (${distResistance.toFixed(1)}% above)</span>
            </div>

            <div class="metric-row">
                <span class="metric-label">Structure State</span>
                <span class="metric-value ${stateClass}">${state.toUpperCase()}</span>
            </div>

            <div class="commentary-box">
                <div class="commentary-text">${commentary}</div>
            </div>
        </div>
    `;
}

// Helper: Format large numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    } else {
        return num.toFixed(0);
    }
}

// Show error message
function showError(message) {
    const content = document.getElementById('content');
    content.innerHTML = `
        <div class="error">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

// Allow Enter key to trigger load
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('symbolInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            loadHLData();
        }
    });
});

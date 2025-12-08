/**
 * ORD Volume Integration Script
 * Completely segregated integration for ORD Volume feature
 * Wires up button to ORD Volume system
 *
 * NO SHARED CODE - Standalone initialization
 */

import { ORDVolumeController } from './ORDVolumeController.js';
import { ORDVolumeRenderer } from './ORDVolumeRenderer.js';
import { ORDVolumeBridge } from './ord-volume-bridge.js';
import { volumeAccumulator } from '../services/VolumeAccumulator.js';

/**
 * Initialize ORD Volume feature
 * Call this function when the app loads
 */
export function initializeORDVolume() {
  // Debug logging disabled for performance
  // console.log('[ORD Volume] Initializing segregated ORD Volume system...');

  // Create controller and renderer instances
  const ordVolumeController = new ORDVolumeController();
  let ordVolumeRenderer = null;

  // Wire up the ORD Volume button
  const ordVolumeBtn = document.getElementById('btn-ord-volume');

  if (!ordVolumeBtn) {
    console.error('[ORD Volume] Button not found: btn-ord-volume');
    return;
  }

  ordVolumeBtn.addEventListener('click', () => {
    // Debug logging disabled for performance
    // console.log('[ORD Volume] Button clicked');

    // Note: 1m and 5m timeframes are allowed now (for Draw mode only)
    // Auto mode will be blocked in the controller if too much data

    // CRITICAL: Check data length BEFORE extraction to prevent freeze
    const MAX_CANDLES = 4500;
    let dataLength = 0;
    let chartType = 'unknown';

    try {
      if (window.tosApp?.activeChartType === 'timeframe') {
        chartType = 'timeframe';
        const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
        dataLength = currentTimeframe?.data?.length || 0;
        // Debug logging disabled for performance
        // console.log(`[ORD Volume] Pre-check: Timeframe chart has ${dataLength} candles`);
      } else if (window.tosApp?.activeChartType === 'tick') {
        // BLOCK tick charts completely - incompatible with ORD Volume methodology
        console.error('[ORD Volume] BLOCKED: Tick charts are not compatible with ORD Volume analysis');
        alert(`ORD Volume Not Available on Tick Charts\n\n❌ Tick charts are incompatible with ORD Volume methodology.\n\nWhy?\n• ORD Volume analyzes completed price swings over time\n• Tick charts update continuously with every trade\n• Analysis would become invalid with each new tick\n• Lines would need to be redrawn constantly\n\n✅ SOLUTION: Switch to a timeframe chart\n• Daily (1d) - RECOMMENDED for ORD Volume\n• Weekly (1wk) - Also excellent\n• Hourly (1h) - Works but needs more data\n• Avoid: 1-minute, 5-minute (too much data)`);
        return;
      } else {
        console.warn(`[ORD Volume] Unknown chart type: ${window.tosApp?.activeChartType}`);
      }

      // CRITICAL: If we can't determine data length, BLOCK to be safe
      if (dataLength === 0) {
        console.error('[ORD Volume] WARNING: Could not determine data length - blocking for safety');
        alert(`ORD Volume Error - Cannot Verify Data Size\n\nUnable to verify dataset size for safety.\n\nThis is a safety measure to prevent browser freezes.\n\nTry:\n1. Reload the chart\n2. Use a Daily or Weekly timeframe\n3. Check browser console for errors`);
        return;
      }

      if (dataLength > MAX_CANDLES) {
        alert(`ORD Volume Error - Dataset Too Large\n\n${chartType} chart has ${dataLength.toLocaleString()} candles\nMaximum allowed: 4,500 candles\n\nThis timeframe has too many bars and will freeze the browser.\n\n✅ Use Daily (1d) or Weekly (1wk) timeframe instead.\n❌ Avoid: Tick charts, 1-minute, 5-minute timeframes with long history.`);
        console.error(`[ORD Volume] BLOCKED at pre-check: ${dataLength} candles exceeds maximum of ${MAX_CANDLES}`);
        return; // Abort before any data extraction
      }

      // Debug logging disabled for performance
      // console.log(`[ORD Volume] ✓ Pre-check passed: ${dataLength} candles (max ${MAX_CANDLES})`);
    } catch (e) {
      console.error('[ORD Volume] Error during pre-check:', e);
      // CRITICAL: If pre-check fails, BLOCK to be safe
      alert(`ORD Volume Error - Safety Check Failed\n\nCannot verify data size due to error.\n\nError: ${e.message}\n\nBlocking for safety to prevent browser freeze.`);
      return;
    }

    // Get current chart data (need to extract from existing chart)
    const candles = extractCandleData();
    const symbol = extractCurrentSymbol();
    const canvas = extractChartCanvas();
    const chartState = extractChartState();

    // Check if extraction was aborted due to too much data
    if (!candles || candles.length === 0) {
      // Check if this was a "too much data" case vs "no data" case
      const currentTimeframe = window.tosApp?.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
      const currentTickChart = window.tosApp?.tickChartRegistry?.get(window.tosApp.currentTickChartId);
      const dataLength = currentTimeframe?.data?.length || currentTickChart?.data?.length || 0;

      if (dataLength > 4500) {
        alert(`ORD Volume Error - Dataset Too Large\n\nTimeframe has ${dataLength.toLocaleString()} candles\nMaximum allowed: 4,500 candles\n\nThis timeframe has too many bars and will freeze the browser.\n\n✅ Use Daily (1d) or Weekly (1wk) timeframe instead.\n❌ Avoid: 15-minute, 5-minute, 1-minute timeframes.`);
        console.error(`[ORD Volume] BLOCKED: ${dataLength} candles exceeds maximum of 4500`);
      } else {
        alert('Please load chart data before using ORD Volume');
      }
      return;
    }

    // Debug logging disabled for performance
    // console.log(`[ORD Volume] ✓ Candle count OK: ${candles.length} candles (max 4500)`);


    // ALWAYS recreate renderer to ensure we have the latest canvas/chartState
    if (canvas) {
      ordVolumeRenderer = new ORDVolumeRenderer(canvas, chartState);

      // Register renderer with bridge for draw mode
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.setORDVolumeRenderer(ordVolumeRenderer);
        // Debug logging disabled for performance
        // console.log('[ORD Volume] Renderer created and registered with bridge');
      }
    } else {
      alert('Chart canvas not found');
      return;
    }

    // Open ORD Volume modal
    ordVolumeController.open(candles, symbol, ordVolumeRenderer);
  });

  // Auto-load saved ORD Volume data when chart symbol changes
  async function autoLoadORDVolume() {
    const symbol = extractCurrentSymbol();
    if (!symbol) return;

    // Debug logging disabled for performance
    // console.log(`[ORD Volume] Auto-loading for symbol: ${symbol}`);
    const analysis = await ordVolumeController.loadAnalysis(symbol);

    if (analysis && window.ordVolumeBridge) {
      // Debug logging disabled for performance
      // console.log('[ORD Volume] Loaded saved analysis, displaying on chart');
      // Note: candles not available during auto-load, signals won't render until next analysis run
      const timeframeId = window.tosApp?.currentTimeframeId || window.tosApp?.currentTickChartId || null;
      window.ordVolumeBridge.setAnalysis(analysis, null, symbol, timeframeId);

      // Trigger chart redraw
      const canvas = extractChartCanvas();
      if (canvas) {
        requestAnimationFrame(() => {
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
        });
      }
    }
  }

  // Expose auto-load function globally for chart integrations
  window.autoLoadORDVolume = autoLoadORDVolume;

  // ===== CHART SWITCH DETECTION =====
  // Monitor timeframe/symbol changes and auto-load/clear ORD Volume analysis
  let lastSymbol = null;
  let lastTimeframeId = null;

  function handleChartSwitch() {
    if (!window.tosApp) return;

    const currentSymbol = window.tosApp.currentSymbol;
    const currentTimeframeId = window.tosApp.currentTimeframeId || window.tosApp.currentTickChartId;

    // Check if symbol changed (different asset = clear all analyses)
    if (lastSymbol && currentSymbol !== lastSymbol) {
      console.log(`[ORD Volume] 🔄 Symbol changed: ${lastSymbol} → ${currentSymbol}. Clearing all analyses.`);
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.clearAllAnalyses();
      }
      lastSymbol = currentSymbol;
      lastTimeframeId = currentTimeframeId;
      return; // Don't try to load analysis for new symbol yet
    }

    // Check if timeframe changed (same symbol, different timeframe)
    if (lastTimeframeId && currentTimeframeId !== lastTimeframeId) {
      console.log(`[ORD Volume] 🔄 Timeframe changed: ${lastTimeframeId} → ${currentTimeframeId}`);

      // Clear current display (but keep storage)
      if (window.ordVolumeBridge) {
        console.log(`[ORD Volume] 🧹 Clearing current display...`);
        window.ordVolumeBridge.clearCurrentChartDisplay();

        // Try to load analysis for new timeframe
        console.log(`[ORD Volume] 📂 Checking for saved analysis...`);
        checkAndLoadAnalysis(currentSymbol, currentTimeframeId);
      }
    }

    // Update tracking
    lastSymbol = currentSymbol;
    lastTimeframeId = currentTimeframeId;
  }

  /**
   * Check if saved analysis exists for current chart and load it
   * Shows re-analyze popup if analysis is stale (>3 candles old)
   */
  function checkAndLoadAnalysis(symbol, timeframeId) {
    if (!window.ordVolumeBridge) {
      console.log(`[ORD Volume] ❌ Bridge not available`);
      return;
    }

    console.log(`[ORD Volume] 🔍 Getting metadata for current chart key...`);
    const metadata = window.ordVolumeBridge.getAnalysisWithMetadata();

    if (!metadata) {
      // No saved analysis for this chart - clean state
      console.log(`[ORD Volume] ℹ️ No saved analysis for ${symbol} @ ${timeframeId}`);
      return;
    }

    console.log(`[ORD Volume] ✅ Found saved analysis:`, {
      symbol: metadata.symbol,
      timeframe: metadata.timeframeId,
      candleCount: metadata.candleCount,
      timestamp: new Date(metadata.timestamp).toLocaleString()
    });

    // Get current candle count
    const currentCandles = extractCandleData();
    const currentCandleCount = currentCandles ? currentCandles.length : 0;

    console.log(`[ORD Volume] 📊 Current candles: ${currentCandleCount}, Saved candles: ${metadata.candleCount}`);

    // Check if stale
    const isStale = window.ordVolumeBridge.isAnalysisStale(metadata.candleCount, currentCandleCount);

    if (isStale) {
      const candleDiff = Math.abs(currentCandleCount - metadata.candleCount);
      console.log(`[ORD Volume] ⚠️ Analysis is STALE: ${candleDiff} candle difference`);

      // Show re-analyze popup
      showReanalyzePopup(symbol, timeframeId, candleDiff, metadata);
    } else {
      // Fresh analysis - load it
      console.log(`[ORD Volume] ✅ Loading fresh analysis for ${symbol} @ ${timeframeId}`);

      // CRITICAL: Call getAnalysis() to restore the analysis AND set isActive = true
      const loadedAnalysis = window.ordVolumeBridge.getAnalysis();

      // FAILSAFE: Manually ensure isActive is true (in case getAnalysis didn't set it)
      if (loadedAnalysis && window.ordVolumeBridge) {
        window.ordVolumeBridge.isActive = true;
        console.log(`[ORD Volume] 🎨 Analysis loaded and FORCED isActive=true`);
      }

      console.log(`[ORD Volume] 🎨 Final state: isActive=${window.ordVolumeBridge.isActive}, hasAnalysis=${!!loadedAnalysis}`);

      // Trigger redraw to show the analysis
      requestAnimationFrame(() => {
        if (window.tosApp.activeChartType === 'timeframe') {
          const currentTimeframe = window.tosApp.timeframeRegistry?.get(timeframeId);
          if (currentTimeframe?.renderer?.draw) {
            console.log(`[ORD Volume] 🖼️ Calling renderer.draw() for timeframe ${timeframeId}`);
            currentTimeframe.renderer.draw();
          } else {
            console.log(`[ORD Volume] ❌ Renderer not found for timeframe ${timeframeId}`);
          }
        }
      });
    }
  }

  /**
   * Show popup asking user if they want to re-analyze stale data
   */
  function showReanalyzePopup(symbol, timeframeId, candleDiff, metadata) {
    const message = `ORD Volume Analysis is ${candleDiff} candle${candleDiff > 1 ? 's' : ''} old.\n\n` +
                    `Do you want to re-analyze ${symbol} on ${timeframeId}?\n\n` +
                    `• Re-analyze Now - Run fresh analysis with latest data\n` +
                    `• Keep Old - Display existing analysis (may be outdated)\n` +
                    `• Cancel - Show clean chart`;

    // Use confirm dialog (simple approach)
    // TODO: Could create custom modal for better UX
    const userChoice = confirm(message + "\n\n[OK = Re-analyze, Cancel = Keep Old]");

    if (userChoice) {
      // User chose to re-analyze
      console.log(`[ORD Volume] User chose to RE-ANALYZE`);

      // Simulate ORD Volume button click to open modal
      const ordBtn = document.getElementById('btn-ord-volume');
      if (ordBtn) {
        ordBtn.click();
      }
    } else {
      // User chose to keep old analysis
      console.log(`[ORD Volume] User chose to KEEP OLD analysis`);
      const loadedAnalysis = window.ordVolumeBridge.getAnalysis(); // Load old analysis

      // FAILSAFE: Ensure isActive is true
      if (loadedAnalysis && window.ordVolumeBridge) {
        window.ordVolumeBridge.isActive = true;
        console.log(`[ORD Volume] 🎨 Old analysis loaded and FORCED isActive=true`);
      }

      // Trigger redraw
      requestAnimationFrame(() => {
        if (window.tosApp.activeChartType === 'timeframe') {
          const currentTimeframe = window.tosApp.timeframeRegistry?.get(timeframeId);
          if (currentTimeframe?.renderer?.draw) {
            currentTimeframe.renderer.draw();
          }
        }
      });
    }
  }

  // ===== AUTO-UPDATE ORD VOLUME ON NEW CANDLES =====
  // Register new candle callbacks for all timeframes to auto-reanalyze
  const intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '1d'];

  intervals.forEach(interval => {
    volumeAccumulator.registerNewCandleCallback(interval, (candleInterval) => {
      console.log(`🔔 [ORD Volume] New candle callback triggered for ${candleInterval}`);

      // Only auto-update if this interval matches the current timeframe
      const currentTimeframeId = window.tosApp?.currentTimeframeId;
      console.log(`   Current timeframe: ${currentTimeframeId}, Candle interval: ${candleInterval}`);

      if (currentTimeframeId !== candleInterval) {
        console.log(`   ⏭️ Skipping - not the active timeframe`);
        return; // Not the active timeframe, skip
      }

      // Check if ORD Volume is active for this timeframe
      const chartKey = `timeframe:${candleInterval}`;
      const hasAnalysis = window.ordVolumeBridge?.analysisStore?.has(chartKey);
      console.log(`   Chart key: ${chartKey}, Has analysis: ${hasAnalysis}`);

      if (!hasAnalysis) {
        console.log(`   ⏭️ Skipping - no ORD Volume analysis for this timeframe`);
        return; // No ORD Volume analysis active for this timeframe, skip
      }

      console.log(`📊 [ORD Volume Auto-Update] New ${candleInterval} candle detected - waiting 2s for chart to update...`);

      // Wait 2 seconds for the chart to add the new candle to its data array
      // The new candle callback fires immediately when the first trade arrives,
      // but the chart needs time to create the new candle object
      setTimeout(() => {
        console.log(`📊 [ORD Volume Auto-Update] Re-analyzing now...`);

        // Automatically re-run ORD Volume analysis
        const candles = extractCandleData();
        const symbol = extractCurrentSymbol();
        const canvas = extractChartCanvas();
        const chartState = extractChartState();

        if (!candles || candles.length === 0) {
          console.warn('[ORD Volume Auto-Update] No candles available, skipping');
          return;
        }

        // Create renderer if needed
        let renderer = null;
        if (canvas) {
          renderer = new ORDVolumeRenderer(canvas, chartState);
          if (window.ordVolumeBridge) {
            window.ordVolumeBridge.setORDVolumeRenderer(renderer);
          }
        }

        // CRITICAL: Update controller's candles with fresh data
        // The analyze() method uses this.candles, not parameters
        ordVolumeController.candles = candles;
        ordVolumeController.symbol = symbol;
        ordVolumeController.renderer = renderer;
        console.log(`📊 [ORD Volume Auto-Update] Updated controller with ${candles.length} fresh candles`);

        // Run analysis silently (without opening modal)
        ordVolumeController.analyze();

        console.log(`✅ [ORD Volume Auto-Update] Analysis complete for ${symbol} @ ${candleInterval}`);

        // CRITICAL: Force bridge to reload fresh analysis from storage
        // The bridge caches analysis in memory, so we must clear it and reload
        if (window.ordVolumeBridge) {
          window.ordVolumeBridge.currentAnalysis = null; // Clear stale cache
          window.ordVolumeBridge.getAnalysis(); // Reload fresh analysis from storage
          console.log(`🔄 [ORD Volume Auto-Update] Forced reload of fresh analysis from storage`);
        }

        // CRITICAL: Trigger chart redraw to show the updated analysis
        requestAnimationFrame(() => {
          const currentTimeframe = window.tosApp.timeframeRegistry?.get(candleInterval);
          if (currentTimeframe?.renderer?.draw) {
            currentTimeframe.renderer.draw();
            console.log(`🎨 [ORD Volume Auto-Update] Chart redrawn with new analysis`);
          }
        });
      }, 2000); // 2 second delay
    });
  });

  console.log('[ORD Volume] Auto-update registered for all timeframes');

  // ===== INTRA-CANDLE VOLUME UPDATES =====
  // Update ORD Volume analysis every 15 seconds if volume has changed
  // This makes the percentage labels update in real-time as trades come in
  let lastVolumeUpdate = {};  // Track last volume for each interval

  setInterval(() => {
    if (!window.tosApp?.activeChartType || window.tosApp.activeChartType !== 'timeframe') {
      return;
    }

    const currentTimeframeId = window.tosApp?.currentTimeframeId;
    if (!currentTimeframeId) return;

    // Check if ORD Volume is active for this timeframe
    const chartKey = `timeframe:${currentTimeframeId}`;
    const hasAnalysis = window.ordVolumeBridge?.analysisStore?.has(chartKey);

    if (!hasAnalysis) return;

    // Get current volume for this interval
    const currentVolume = volumeAccumulator.getVolume(currentTimeframeId);
    const lastVolume = lastVolumeUpdate[currentTimeframeId] || 0;

    // Calculate percentage change (dynamic threshold works for any symbol)
    const volumeChange = Math.abs(currentVolume - lastVolume);
    const percentageChange = lastVolume > 0 ? (volumeChange / lastVolume) * 100 : 100;

    // Update if volume changed by at least 3% (works for BTC, stocks, anything)
    if (percentageChange >= 3 || (currentVolume > 0 && lastVolume === 0)) {
      console.log(`📊 [ORD Volume Intra-Update] Volume changed ${percentageChange.toFixed(1)}% on ${currentTimeframeId} - updating analysis...`);

      // Extract fresh data
      const candles = extractCandleData();
      const symbol = extractCurrentSymbol();
      const canvas = extractChartCanvas();
      const chartState = extractChartState();

      if (!candles || candles.length === 0) return;

      // Create renderer
      let renderer = null;
      if (canvas) {
        renderer = new ORDVolumeRenderer(canvas, chartState);
        if (window.ordVolumeBridge) {
          window.ordVolumeBridge.setORDVolumeRenderer(renderer);
        }
      }

      // Update controller with fresh data
      ordVolumeController.candles = candles;
      ordVolumeController.symbol = symbol;
      ordVolumeController.renderer = renderer;

      // Run analysis
      ordVolumeController.analyze();

      // Force bridge reload
      if (window.ordVolumeBridge) {
        window.ordVolumeBridge.currentAnalysis = null;
        window.ordVolumeBridge.getAnalysis();
      }

      // Trigger redraw
      requestAnimationFrame(() => {
        const currentTimeframe = window.tosApp.timeframeRegistry?.get(currentTimeframeId);
        if (currentTimeframe?.renderer?.draw) {
          currentTimeframe.renderer.draw();
        }
      });

      // Update last volume
      lastVolumeUpdate[currentTimeframeId] = currentVolume;
    }
  }, 15000); // Check every 15 seconds

  console.log('[ORD Volume] Intra-candle updates enabled (15s interval)');

  // Start monitoring chart switches (poll every 500ms)
  setInterval(handleChartSwitch, 500);

  // Initial check
  if (window.tosApp) {
    lastSymbol = window.tosApp.currentSymbol;
    lastTimeframeId = window.tosApp.currentTimeframeId || window.tosApp.currentTickChartId;
  }

  // Debug logging disabled for performance
  // console.log('[ORD Volume] Initialization complete');
}

/**
 * Extract candle data from existing chart
 * This function needs to be customized based on how the main app stores candle data
 * @returns {Array} Array of OHLCV objects
 */
function extractCandleData() {
  const MAX_CANDLES = 4500; // MUST match the limit in ord-volume-integration.js

  // Method 1: Get data from active timeframe
  if (window.tosApp && window.tosApp.activeChartType === 'timeframe') {
    const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
    if (currentTimeframe && currentTimeframe.data && currentTimeframe.data.length > 0) {
      const dataLength = currentTimeframe.data.length;
      // Debug logging disabled for performance
      // console.log(`[ORD Volume] Found ${dataLength} candles from timeframe ${window.tosApp.currentTimeframeId}`);

      // CHECK BEFORE CONVERSION to prevent freeze during data processing
      if (dataLength > MAX_CANDLES) {
        console.error(`[ORD Volume] Dataset too large: ${dataLength} candles exceeds maximum ${MAX_CANDLES}. Aborting extraction.`);
        return []; // Return empty to trigger the alert in the caller
      }

      return convertToOHLCV(currentTimeframe.data);
    }
  }

  // Method 2: Get data from active tick chart
  if (window.tosApp && window.tosApp.activeChartType === 'tick') {
    const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
    if (currentTickChart && currentTickChart.data && currentTickChart.data.length > 0) {
      const dataLength = currentTickChart.data.length;
      // Debug logging disabled for performance
      // console.log(`[ORD Volume] Found ${dataLength} candles from tick chart ${window.tosApp.currentTickChartId}`);

      // CHECK BEFORE CONVERSION to prevent freeze during data processing
      if (dataLength > MAX_CANDLES) {
        console.error(`[ORD Volume] Dataset too large: ${dataLength} candles exceeds maximum ${MAX_CANDLES}. Aborting extraction.`);
        return []; // Return empty to trigger the alert in the caller
      }

      return convertToOHLCV(currentTickChart.data);
    }
  }

  // Method 3: Check if there's a global chartData variable
  if (window.chartData) {
    return convertToOHLCV(window.chartData);
  }

  // Method 4: Try to extract from Plotly chart
  try {
    const chartDiv = document.getElementById('chart');
    if (chartDiv && chartDiv.data && chartDiv.data[0]) {
      const trace = chartDiv.data[0];
      if (trace.open && trace.high && trace.low && trace.close) {
        const candles = [];
        for (let i = 0; i < trace.open.length; i++) {
          candles.push({
            open: trace.open[i],
            high: trace.high[i],
            low: trace.low[i],
            close: trace.close[i],
            volume: trace.volume ? trace.volume[i] : 0
          });
        }
        return candles;
      }
    }
  } catch (e) {
    console.error('[ORD Volume] Error extracting candle data:', e);
  }

  console.error('[ORD Volume] Could not find candle data');
  return [];
}

/**
 * Convert chart data to OHLCV format
 * @param {Object} chartData - Raw chart data
 * @returns {Array} OHLCV array
 */
function convertToOHLCV(chartData) {
  if (!chartData) return [];

  // If already in correct format (array of objects with OHLCV)
  if (Array.isArray(chartData) && chartData.length > 0) {
    const first = chartData[0];

    // Check if it has OHLCV properties
    if (first.open !== undefined && first.high !== undefined &&
        first.low !== undefined && first.close !== undefined) {
      // Already in correct format, just ensure volume exists
      return chartData.map(candle => ({
        open: parseFloat(candle.open) || candle.Open || 0,
        high: parseFloat(candle.high) || candle.High || 0,
        low: parseFloat(candle.low) || candle.Low || 0,
        close: parseFloat(candle.close) || candle.Close || 0,
        volume: parseFloat(candle.volume) || candle.Volume || 0
      }));
    }

    // Check for capitalized keys (Date, Open, High, Low, Close, Volume)
    if (first.Open !== undefined && first.High !== undefined) {
      return chartData.map(candle => ({
        open: parseFloat(candle.Open) || 0,
        high: parseFloat(candle.High) || 0,
        low: parseFloat(candle.Low) || 0,
        close: parseFloat(candle.Close) || 0,
        volume: parseFloat(candle.Volume) || 0
      }));
    }
  }

  // Handle object with arrays
  if (chartData.open && Array.isArray(chartData.open)) {
    const candles = [];
    for (let i = 0; i < chartData.open.length; i++) {
      candles.push({
        open: parseFloat(chartData.open[i]) || 0,
        high: parseFloat(chartData.high[i]) || 0,
        low: parseFloat(chartData.low[i]) || 0,
        close: parseFloat(chartData.close[i]) || 0,
        volume: parseFloat(chartData.volume ? chartData.volume[i] : 0) || 0
      });
    }
    return candles;
  }

  return [];
}

/**
 * Extract current symbol from the chart
 * @returns {String} Current symbol
 */
function extractCurrentSymbol() {
  // Method 1: Check input field
  const symbolInput = document.getElementById('tos-symbol-input');
  if (symbolInput && symbolInput.value) {
    return symbolInput.value;
  }

  // Method 2: Check tosApp
  if (window.tosApp && window.tosApp.currentSymbol) {
    return window.tosApp.currentSymbol;
  }

  // Method 3: Default
  return 'BTC-USD';
}

/**
 * Extract chart canvas element
 * @returns {HTMLCanvasElement} Canvas element
 */
function extractChartCanvas() {
  // Method 1: Get from active timeframe renderer
  if (window.tosApp && window.tosApp.activeChartType === 'timeframe') {
    const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
    if (currentTimeframe && currentTimeframe.renderer && currentTimeframe.renderer.canvas) {
      // Debug logging disabled for performance
      // console.log('[ORD Volume] Found canvas from timeframe renderer');
      return currentTimeframe.renderer.canvas;
    }
  }

  // Method 2: Get from active tick chart renderer
  if (window.tosApp && window.tosApp.activeChartType === 'tick') {
    const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
    if (currentTickChart && currentTickChart.renderer && currentTickChart.renderer.canvas) {
      // Debug logging disabled for performance
      // console.log('[ORD Volume] Found canvas from tick chart renderer');
      return currentTickChart.renderer.canvas;
    }
  }

  // Method 3: Direct canvas lookup
  const canvas = document.getElementById('chartCanvas');
  if (canvas) return canvas;

  // Method 4: Query selector
  const canvasQuery = document.querySelector('canvas');
  if (canvasQuery) return canvasQuery;

  console.error('[ORD Volume] Could not find canvas element');
  return null;
}

/**
 * Extract chart state (for coordinate conversion)
 * @returns {Object} Chart state object with conversion functions
 */
function extractChartState() {
  // Method 1: Get from active timeframe renderer
  if (window.tosApp && window.tosApp.activeChartType === 'timeframe') {
    const currentTimeframe = window.tosApp.timeframeRegistry?.get(window.tosApp.currentTimeframeId);
    if (currentTimeframe && currentTimeframe.renderer) {
      const renderer = currentTimeframe.renderer;
      // Debug logging disabled for performance
      // console.log('[ORD Volume] Found chart state from timeframe renderer');
      return {
        xToIndex: renderer.xToIndex ? renderer.xToIndex.bind(renderer) : null,
        yToPrice: renderer.yToPrice ? renderer.yToPrice.bind(renderer) : null,
        indexToX: renderer.indexToX ? renderer.indexToX.bind(renderer) : null,
        priceToY: renderer.priceToY ? renderer.priceToY.bind(renderer) : null
      };
    }
  }

  // Method 2: Get from active tick chart renderer
  if (window.tosApp && window.tosApp.activeChartType === 'tick') {
    const currentTickChart = window.tosApp.tickChartRegistry?.get(window.tosApp.currentTickChartId);
    if (currentTickChart && currentTickChart.renderer) {
      const renderer = currentTickChart.renderer;
      // Debug logging disabled for performance
      // console.log('[ORD Volume] Found chart state from tick chart renderer');
      return {
        xToIndex: renderer.xToIndex ? renderer.xToIndex.bind(renderer) : null,
        yToPrice: renderer.yToPrice ? renderer.yToPrice.bind(renderer) : null,
        indexToX: renderer.indexToX ? renderer.indexToX.bind(renderer) : null,
        priceToY: renderer.priceToY ? renderer.priceToY.bind(renderer) : null
      };
    }
  }

  console.warn('[ORD Volume] Could not find chart state, using fallback');
  // Return empty state (renderer will use fallback)
  return {};
}

/**
 * Load saved ORD Volume analysis when chart loads
 * @param {String} symbol - Symbol to load analysis for
 */
export async function loadSavedORDVolume(symbol) {
  // This function can be called when chart loads to restore saved ORD Volume overlays
  // Debug logging disabled for performance
  // console.log(`[ORD Volume] Loading saved analysis for ${symbol}...`);

  // Implementation would fetch from backend and render
  // For now, this is a placeholder for future enhancement
}

// Auto-initialize when script loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeORDVolume);
} else {
  // DOM already loaded
  initializeORDVolume();
}

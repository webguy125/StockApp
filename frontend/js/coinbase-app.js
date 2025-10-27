/**
 * Coinbase Chart Application
 * Clean implementation using Coinbase REST + WebSocket
 */

class CoinbaseChart {
  constructor() {
    this.symbol = 'BTC-USD';
    this.historicalData = [];
    this.liveCandle = null;
    this.socket = null;
    this.lastPrice = null;
  }

  async initialize() {
    console.log('🚀 Starting Coinbase Chart App');

    // Load historical data
    await this.loadHistoricalData();

    // Setup WebSocket for live updates
    this.setupWebSocket();
  }

  async loadHistoricalData() {
    console.log(`📊 Loading historical data for ${this.symbol}`);

    try {
      const response = await fetch(`/data/coinbase/${this.symbol}`);
      const data = await response.json();

      if (!data || data.length === 0) {
        console.error('❌ No historical data received');
        return;
      }

      this.historicalData = data;
      console.log(`✅ Loaded ${data.length} historical candles`);
      console.log(`📅 Date range: ${data[0].Date} to ${data[data.length - 1].Date}`);

      // Check if we already have today's candle in the data
      const lastCandle = data[data.length - 1];
      const today = new Date().toISOString().split('T')[0];

      console.log(`📌 Last historical candle: ${lastCandle.Date}`);
      console.log(`📌 Today's date (JS): ${today}`);
      console.log(`📌 Dates match? ${lastCandle.Date === today}`);

      // Only create a live candle if we don't have data for today yet
      if (lastCandle.Date !== today) {
        this.liveCandle = {
          Date: today,
          Open: lastCandle.Close,  // Today opens at yesterday's close
          High: lastCandle.Close,
          Low: lastCandle.Close,
          Close: lastCandle.Close,
          Volume: 0
        };
        console.log(`🔴 Created NEW live candle for ${today} starting at $${this.liveCandle.Open.toFixed(2)}`);
        console.log(`🔴 Live candle OHLC: O=${this.liveCandle.Open.toFixed(2)} H=${this.liveCandle.High.toFixed(2)} L=${this.liveCandle.Low.toFixed(2)} C=${this.liveCandle.Close.toFixed(2)}`);
      } else {
        // Today's candle already exists in historical data, use it as live candle
        this.liveCandle = {...lastCandle};
        console.log(`🟢 Using existing candle for ${today} as live candle`);
        console.log(`🟢 Live candle OHLC: O=${this.liveCandle.Open.toFixed(2)} H=${this.liveCandle.High.toFixed(2)} L=${this.liveCandle.Low.toFixed(2)} C=${this.liveCandle.Close.toFixed(2)}`);
      }

      // Render initial chart
      this.renderChart();

    } catch (error) {
      console.error('❌ Error loading data:', error);
    }
  }

  setupWebSocket() {
    console.log('🔌 Connecting to WebSocket...');

    // Connect to Socket.IO
    this.socket = io();

    this.socket.on('connect', () => {
      console.log('✅ WebSocket connected');

      // Subscribe to ticker updates for our symbol
      this.socket.emit('subscribe', {
        type: 'ticker',
        symbol: this.symbol
      });
    });

    this.socket.on('ticker_update', (data) => {
      console.log(`📡 TICKER UPDATE RECEIVED: ${data.symbol} @ $${data.price}`);

      // Only process updates for our symbol
      if (data.symbol === this.symbol || data.symbol.includes(this.symbol.split('-')[0])) {
        console.log(`✅ Processing ticker update for ${this.symbol}`);
        this.updateLiveCandle(data.price);
      } else {
        console.log(`⏭️ Skipping ticker for ${data.symbol} (watching ${this.symbol})`);
      }
    });

    this.socket.on('disconnect', () => {
      console.log('❌ WebSocket disconnected');
    });
  }

  updateLiveCandle(price) {
    if (!this.liveCandle) return;

    // Update live candle OHLC
    this.liveCandle.Close = price;
    this.liveCandle.High = Math.max(this.liveCandle.High, price);
    this.liveCandle.Low = Math.min(this.liveCandle.Low, price);

    console.log(`📈 Live update: $${price.toFixed(2)} (H:${this.liveCandle.High.toFixed(2)} L:${this.liveCandle.Low.toFixed(2)})`);

    // Update chart
    this.updateChartLiveCandle();
  }

  renderChart() {
    // Check if live candle date already exists in historical data
    const lastHistoricalDate = this.historicalData[this.historicalData.length - 1]?.Date;
    const liveCandleDate = this.liveCandle?.Date;

    // Only add live candle if it's for a new date
    let allData;
    if (liveCandleDate && liveCandleDate !== lastHistoricalDate) {
      allData = [...this.historicalData, this.liveCandle];
      console.log(`Chart: Added live candle for ${liveCandleDate}`);
    } else if (liveCandleDate === lastHistoricalDate) {
      // Replace last historical candle with live candle (it has updated values)
      allData = [...this.historicalData.slice(0, -1), this.liveCandle];
      console.log(`Chart: Replaced historical candle with live candle for ${liveCandleDate}`);
    } else {
      allData = this.historicalData;
    }

    // Extract arrays for Plotly
    const dates = allData.map(d => d.Date);
    const opens = allData.map(d => d.Open);
    const highs = allData.map(d => d.High);
    const lows = allData.map(d => d.Low);
    const closes = allData.map(d => d.Close);
    const volumes = allData.map(d => d.Volume || 0);

    // Debug: Show last 5 dates with full details
    console.log('Last 5 candles being rendered:');
    allData.slice(-5).forEach((d, i) => {
      console.log(`  [${allData.length - 5 + i}] ${d.Date}: O=${d.Open.toFixed(2)} H=${d.High.toFixed(2)} L=${d.Low.toFixed(2)} C=${d.Close.toFixed(2)}`);
    });

    // Check for duplicates
    const dateSet = new Set(dates);
    if (dateSet.size !== dates.length) {
      console.error('⚠️ DUPLICATE DATES DETECTED IN CHART DATA!');
      const counts = {};
      dates.forEach(d => counts[d] = (counts[d] || 0) + 1);
      Object.entries(counts).forEach(([date, count]) => {
        if (count > 1) console.error(`  ${date} appears ${count} times`);
      });
    } else {
      console.log(`✓ All ${dates.length} dates are unique`);
    }

    const newCandles = allData.length - this.historicalData.length;
    if (newCandles > 0) {
      console.log(`📊 Rendering ${allData.length} candles (${this.historicalData.length} historical + ${newCandles} live)`);
    } else {
      console.log(`📊 Rendering ${allData.length} candles (all historical, live updates to last candle)`);
    }

    // Create candlestick trace
    const candleTrace = {
      x: dates,
      open: opens,
      high: highs,
      low: lows,
      close: closes,
      type: 'candlestick',
      name: this.symbol,
      increasing: { line: { color: '#00c851' } },
      decreasing: { line: { color: '#ff4444' } }
    };

    // Create volume trace
    const volumeTrace = {
      x: dates,
      y: volumes,
      type: 'bar',
      name: 'Volume',
      yaxis: 'y2',
      marker: { color: 'rgba(0, 188, 212, 0.5)' }
    };

    // Layout
    const layout = {
      title: `${this.symbol} - Daily Chart (Live)`,
      xaxis: {
        type: 'date',
        gridcolor: '#404040',
        color: '#a0a0a0'
      },
      yaxis: {
        title: 'Price',
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: [0.3, 1]
      },
      yaxis2: {
        title: 'Volume',
        gridcolor: '#404040',
        color: '#a0a0a0',
        side: 'right',
        domain: [0, 0.25]
      },
      plot_bgcolor: '#1a1a1a',
      paper_bgcolor: '#1a1a1a',
      font: { color: '#e0e0e0' },
      showlegend: false
    };

    // Render with Plotly
    Plotly.newPlot('chart-container', [candleTrace, volumeTrace], layout, {
      responsive: true,
      displayModeBar: true,
      scrollZoom: true
    });

    console.log('✅ Chart rendered');
  }

  updateChartLiveCandle() {
    const plotDiv = document.getElementById('chart-container');
    if (!plotDiv || !plotDiv.data) return;

    // Find candlestick trace
    const candleTrace = plotDiv.data[0];
    if (!candleTrace) return;

    // Update the last candle (live candle)
    const lastIndex = candleTrace.close.length - 1;

    candleTrace.close[lastIndex] = this.liveCandle.Close;
    candleTrace.high[lastIndex] = this.liveCandle.High;
    candleTrace.low[lastIndex] = this.liveCandle.Low;

    // Update chart
    Plotly.restyle('chart-container', {
      close: [candleTrace.close],
      high: [candleTrace.high],
      low: [candleTrace.low]
    }, [0]);
  }

  changeSymbol(newSymbol) {
    console.log(`🔄 Changing symbol to ${newSymbol}`);
    this.symbol = newSymbol;
    this.historicalData = [];
    this.liveCandle = null;

    // Reload everything
    this.loadHistoricalData();
  }
}

// Start the app
const app = new CoinbaseChart();
window.coinbaseApp = app;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => app.initialize());
} else {
  app.initialize();
}
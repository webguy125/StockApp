/**
 * Chart Indicators Module
 * Handles technical indicator calculations and trace building
 */

import { state } from '../core/state.js';
import { convertToCST } from './loader.js';

/**
 * Build Plotly traces for all active indicators
 */
export async function buildIndicatorTraces(symbol, data) {
  const indicatorConfigs = Object.keys(state.activeIndicators).map(type => ({
    type: type,
    params: state.activeIndicators[type].period ? { period: parseInt(state.activeIndicators[type].period) } : {}
  }));

  const indicatorResponse = await fetch("/indicators", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol: symbol,
      period: state.currentPeriod,
      interval: state.currentInterval,
      indicators: indicatorConfigs
    })
  });

  const indicatorData = await indicatorResponse.json();
  const traces = [];

  // SMA
  if (indicatorData.SMA_20) {
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.SMA_20,
      type: 'scatter',
      mode: 'lines',
      name: 'SMA(20)',
      line: { color: '#f59e0b', width: 2 }
    });
  }

  // EMA
  if (indicatorData.EMA_20) {
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.EMA_20,
      type: 'scatter',
      mode: 'lines',
      name: 'EMA(20)',
      line: { color: '#10b981', width: 2 }
    });
  }

  // VWAP
  if (indicatorData.VWAP) {
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.VWAP,
      type: 'scatter',
      mode: 'lines',
      name: 'VWAP',
      line: { color: '#8b5cf6', width: 2 }
    });
  }

  // Bollinger Bands
  if (indicatorData.BB_upper) {
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.BB_upper,
      type: 'scatter',
      mode: 'lines',
      name: 'BB Upper',
      line: { color: '#6b7280', width: 1, dash: 'dot' }
    });
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.BB_middle,
      type: 'scatter',
      mode: 'lines',
      name: 'BB Middle',
      line: { color: '#6b7280', width: 1 }
    });
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.BB_lower,
      type: 'scatter',
      mode: 'lines',
      name: 'BB Lower',
      line: { color: '#6b7280', width: 1, dash: 'dot' }
    });
  }

  // RSI
  if (indicatorData.RSI) {
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.RSI,
      type: 'scatter',
      mode: 'lines',
      name: 'RSI(14)',
      line: { color: '#ef4444', width: 2 },
      xaxis: 'x',
      yaxis: 'y2'
    });
  }

  // MACD
  if (indicatorData.MACD) {
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.MACD,
      type: 'scatter',
      mode: 'lines',
      name: 'MACD',
      line: { color: '#3b82f6', width: 2 },
      xaxis: 'x',
      yaxis: 'y3'
    });
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.MACD_signal,
      type: 'scatter',
      mode: 'lines',
      name: 'Signal',
      line: { color: '#f59e0b', width: 2 },
      xaxis: 'x',
      yaxis: 'y3'
    });
    traces.push({
      x: data.map(row => convertToCST(row.Date)),
      y: indicatorData.MACD_histogram,
      type: 'bar',
      name: 'Histogram',
      marker: { color: '#10b981' },
      xaxis: 'x',
      yaxis: 'y3'
    });
  }

  return traces;
}

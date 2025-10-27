// Global State Management
export const state = {
  currentSymbol: null,
  currentPeriod: null,
  currentInterval: '1d',
  drawnLines: [],
  selectedLineId: null,
  chartData: [],
  activeIndicators: {},
  currentTheme: localStorage.getItem('chartTheme') || 'dark', // Load from localStorage, default to dark
  authToken: null,
  currentUser: null,
  suppressSave: false,
  relayoutGuard: false,
  isDrawingMode: false
};

// State getters
export function getState() {
  return state;
}

export function setState(updates) {
  Object.assign(state, updates);
}

// Specific state accessors
export function getCurrentSymbol() {
  return state.currentSymbol;
}

export function setCurrentSymbol(symbol) {
  state.currentSymbol = symbol;
}

export function getSelectedLineId() {
  return state.selectedLineId;
}

export function setSelectedLineId(id) {
  state.selectedLineId = id;
}

export function getCurrentTheme() {
  return state.currentTheme;
}

export function setCurrentTheme(theme) {
  state.currentTheme = theme;
}

export function getChartData() {
  return state.chartData;
}

export function setChartData(data) {
  state.chartData = data;
}

export function getDrawnLines() {
  return state.drawnLines;
}

export function setDrawnLines(lines) {
  state.drawnLines = lines;
}

export function addDrawnLine(line) {
  state.drawnLines.push(line);
}

export function removeDrawnLine(id) {
  state.drawnLines = state.drawnLines.filter(l => l.id !== id);
}

export function getActiveIndicators() {
  return state.activeIndicators;
}

export function setActiveIndicator(indicator, config) {
  state.activeIndicators[indicator] = config;
}

export function removeActiveIndicator(indicator) {
  delete state.activeIndicators[indicator];
}

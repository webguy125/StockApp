// Theme Management
import { state } from './state.js';

export function initializeThemeToggle() {
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }
}

export function toggleTheme() {
  const newTheme = state.currentTheme === 'light' ? 'dark' : 'light';
  state.currentTheme = newTheme;
  applyTheme(newTheme);
}

export function applyTheme(theme) {
  document.body.className = theme + '-theme';
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.textContent = theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode';
  }

  // Reload chart if one is loaded to apply theme
  const currentSymbol = document.getElementById('symbolInput')?.value;
  if (currentSymbol) {
    // Import dynamically to avoid circular dependency
    import('../chart/loader.js').then(module => {
      module.loadChart();
    });
  }
}

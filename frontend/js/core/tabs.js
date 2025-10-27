// Tab Management

export function initializeTabSwitching(onTabChange) {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
      const tabName = this.getAttribute('data-tab');
      if (!tabName) return;

      switchTab(tabName);

      // Call the callback if provided
      if (onTabChange) {
        onTabChange(tabName);
      }
    });
  });
}

export function switchTab(tabName) {
  // Deactivate all tabs and content
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  // Activate selected tab
  const tabElement = document.querySelector(`[data-tab="${tabName}"]`);
  if (tabElement) {
    tabElement.classList.add('active');
  }

  const contentElement = document.getElementById(tabName + '-tab');
  if (contentElement) {
    contentElement.classList.add('active');
  }
}

export function showAnalysisTab() {
  switchTab('analysis');
}

export function showPluginsTab() {
  switchTab('plugins');
}

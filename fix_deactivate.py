#!/usr/bin/env python3
"""
Fix all timeframe files to properly destroy renderer when deactivating
"""
import os
import re

# Files to fix
files_to_fix = [
    r'frontend\js\timeframes\minutes\5m.js',
    r'frontend\js\timeframes\minutes\15m.js',
    r'frontend\js\timeframes\minutes\1h.js',
    r'frontend\js\timeframes\days\1d.js',
    r'frontend\js\timeframes\days\1w.js',
    r'frontend\js\timeframes\days\1mo.js',
    r'frontend\js\timeframes\days\3mo.js',
]

# Pattern to search for
old_pattern = r'(\s+this\.isActive = false;)\n\n(\s+// Unsubscribe from WebSocket)'

# Replacement
new_text = r'\1\n\n    // Destroy the renderer to remove the canvas from DOM\n    if (this.renderer) {\n      this.renderer.destroy();\n    }\n\n\2'

for file_path in files_to_fix:
    full_path = os.path.join(r'C:\StockApp', file_path)
    if not os.path.exists(full_path):
        print(f'[ERROR] File not found: {full_path}')
        continue

    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already fixed
    if 'this.renderer.destroy()' in content and 'deactivate()' in content:
        print(f'[OK] Already fixed: {file_path}')
        continue

    # Apply fix
    new_content = re.sub(old_pattern, new_text, content)

    if new_content == content:
        print(f'[SKIP] No changes: {file_path}')
    else:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'[FIXED] Fixed: {file_path}')

print('\n[DONE] All files processed!')

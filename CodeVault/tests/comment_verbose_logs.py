#!/usr/bin/env python3
"""
Comment out verbose console.log statements in JavaScript files
"""

import re
import os

# Patterns to comment out - keep ERROR and WARN messages only
# Comment out all INFO-level console.log that contain emojis or verbose status messages
VERBOSE_PATTERNS = [
    # Emoji-based messages (all of them)
    r"console\.log\([^)]*[💾📊📡✅💰🚀🔄🆕📅🔺🔻🕯️⏰❌🖱️]",

    # Specific verbose message patterns
    r"console\.log\(['\"`]Ticker update:",
    r"console\.log\(['\"`]Chart not ready",
    r"console\.log\(['\"`]Time check:",
    r"console\.log\(['\"`]WebSocket connected['\"`]\)",
    r"console\.log\(['\"`]Server response:",
    r"console\.log\(['\"`]Subscription:",
    r"console\.log\(['\"`]WebSocket real-time mode",
    r"console\.log\(['\"`]LOADING NEW SYMBOL",
    r"console\.log\(['\"`]FINISHED LOADING",
    r"console\.log\(['\"`]Loading saved indicators",
    r"console\.log\(['\"`]After loadSavedIndicators",
    r"console\.log\(['\"`]Calling reloadChart",
    r"console\.log\(['\"`]>>> restoreSavedIndicators",
    r"console\.log\(['\"`]>>> No indicators to restore",
    r"console\.log\(['\"`]>>> Restoring",
    r"console\.log\(['\"`]>>> Indicators:",
    r"console\.log\(['\"`]>>> Cleared",
    r"console\.log\(['\"`]>>> Re-adding",
    r"console\.log\(['\"`]>>> Successfully added",
    r"console\.log\(['\"`]>>> Finished restoring",
    r"console\.log\(['\"`]>>> Final activeIndicators",
    r"console\.log\(['\"`]Initializing",
    r"console\.log\(['\"`]Starting ThinkorSwim",
    r"console\.log\(['\"`]Platform ready",
    r"console\.log\(['\"`]ThinkorSwim-Style Platform initialized",
    r"console\.log\(['\"`]Hover hide CSS",
    r"console\.log\(['\"`]Data bar added",
    r"console\.log\(['\"`]TOS-style chart scaling",
    r"console\.log\(['\"`]Chart type changed",
    r"console\.log\(['\"`]Clearing all indicators",
    r"console\.log\(['\"`]Chart updated successfully",
    r"console\.log\(['\"`]Stopped all streams",
    r"console\.log\(['\"`]Socket state:",
    r"console\.log\(['\"`]Timeframe changed",
    r"console\.log\(['\"`]Indicators to restore",
    r"console\.log\(['\"`]Loaded .* saved indicators",
    r"console\.log\(['\"`]Saved .* indicators",
    r"console\.log\(['\"`]Updating existing chart",
    r"console\.log\(['\"`]Creating new chart",
    r"console\.log\(['\"`]Received data for",
    r"console\.log\(['\"`]Full response data",
    r"console\.log\(['\"`]Tracking indicator",
    r"console\.log\(['\"`]Active indicators count",
    r"console\.log\(['\"`]Indicator .* will use yaxis",
    r"console\.log\(['\"`]Current plot has",
    r"console\.log\(['\"`]Creating.*trace",
    r"console\.log\(['\"`]Data length:",
    r"console\.log\(['\"`]First .* data points",
    r"console\.log\(['\"`]Assigned to subplot",
    r"console\.log\(['\"`]Trace created",
    r"console\.log\(['\"`]Adding trace",
    r"console\.log\(['\"`]Created .* traces",
    r"console\.log\(['\"`]Indicator .* added with",
    r"console\.log\(\[",  # Arrays like [CANDLE UPDATE]
]

def comment_out_verbose_logs(filepath):
    """Comment out verbose console.log statements in a file"""
    print(f"Processing {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    lines = content.split('\n')
    modified_lines = []
    changes = 0

    for line in lines:
        # Check if this line should be commented out
        should_comment = False
        for pattern in VERBOSE_PATTERNS:
            if re.search(pattern, line):
                should_comment = True
                break

        if should_comment and not line.strip().startswith('//'):
            # Comment out the line
            # Preserve indentation
            indent = len(line) - len(line.lstrip())
            modified_line = ' ' * indent + '// ' + line.lstrip()
            modified_lines.append(modified_line)
            changes += 1
        else:
            modified_lines.append(line)

    if changes > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines))
        print(f"  OK Commented out {changes} verbose console.log statements")
    else:
        print(f"  - No changes needed")

    return changes

# Files to process
files_to_process = [
    r'C:\StockApp\frontend\js\tos-app.js',
    r'C:\StockApp\frontend\js\components\watchlist.js',
    r'C:\StockApp\frontend\js\trendlines\selection.js',
]

total_changes = 0
for filepath in files_to_process:
    if os.path.exists(filepath):
        changes = comment_out_verbose_logs(filepath)
        total_changes += changes
    else:
        print(f"File not found: {filepath}")

print(f"\nDONE: Commented out {total_changes} verbose console.log statements")

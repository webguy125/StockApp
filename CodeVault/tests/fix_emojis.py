import re

# Read file
with open('backend/api_server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all emojis
emoji_map = {
    '📤': '>>UPLOAD',
    '📥': '>>DOWNLOAD',
    '🚀': '>>ROCKET',
    '📊': '>>DATA',
    '❌': '>>ERROR',
    '✅': '>>SUCCESS',
    '🎯': '>>TARGET',
    '💾': '>>CACHE',
    '🚨': '>>ALERT',
    '🎉': '>>CELEBRATION',
    '💡': '>>IDEA',
    '👀': '>>WATCH',
    '🗑️': '>>DELETE',
    '🗑': '>>DELETE',
    '🔃': '>>REFRESH',
    '🔮': '>>PREDICT',
    '🔖': '>>BOOKMARK'
}

# Replace all emojis
for emoji, replacement in emoji_map.items():
    content = content.replace(emoji, replacement)

# Write back
with open('backend/api_server.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all emoji characters!")

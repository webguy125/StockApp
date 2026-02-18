import os
from datetime import datetime

# Log file path
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
LOG_FILE = os.path.join(LOG_DIR, 'scanner.log')


def log(message):
    """
    Log a message to both console and log file with timestamp.

    Args:
        message: Message string to log
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f'[{timestamp}] {message}'

    # Print to console
    print(log_line)

    # Write to log file
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
    except Exception as e:
        print(f'[LOG ERROR] Failed to write to log file: {e}')

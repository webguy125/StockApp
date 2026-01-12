"""
Scheduler Logging System
Provides file-based logging with rotation, per-task log files, and archiving

Features:
- Per-task log files (task_1_ingestion.log, task_2_training.log, etc.)
- Automatic rotation at 10MB per file
- Keeps last 5 rotated files per task
- Archives old logs (handled by Task 6)
- Both file and console output

Author: TurboMode System
Date: 2026-01-06
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import Optional

# Log directory
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Format for log messages
LOG_FORMAT = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class SchedulerLogger:
    """
    Manages logging for scheduler tasks

    Features:
    - Per-task log files
    - Rotation at 10MB
    - Console + file output
    - Archive support
    """

    def __init__(self):
        self.loggers = {}
        self.handlers = {}

    def get_task_logger(self, task_id: int, task_name: str) -> logging.Logger:
        """
        Get or create a logger for a specific task

        Args:
            task_id: Task ID (1-6)
            task_name: Task name (for filename)

        Returns:
            Logger instance
        """
        logger_name = f'task_{task_id}'

        if logger_name in self.loggers:
            return self.loggers[logger_name]

        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger

        # Create file handler with rotation
        log_filename = self._sanitize_filename(f'task_{task_id}_{task_name}.log')
        log_path = os.path.join(LOG_DIR, log_filename)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,  # Keep last 5 rotated files
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Store logger and handlers
        self.loggers[logger_name] = logger
        self.handlers[logger_name] = {
            'file': file_handler,
            'console': console_handler
        }

        return logger

    def get_scheduler_logger(self) -> logging.Logger:
        """
        Get the main scheduler logger

        Returns:
            Logger instance
        """
        logger_name = 'scheduler'

        if logger_name in self.loggers:
            return self.loggers[logger_name]

        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Create file handler
        log_path = os.path.join(LOG_DIR, 'scheduler.log')

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10,  # Keep more for scheduler logs
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Store
        self.loggers[logger_name] = logger
        self.handlers[logger_name] = {
            'file': file_handler,
            'console': console_handler
        }

        return logger

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for filesystem

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Replace spaces and special chars
        return filename.lower().replace(' ', '_').replace('-', '_')

    def close_all(self):
        """Close all log handlers"""
        for logger_name, handlers in self.handlers.items():
            for handler in handlers.values():
                handler.close()

    def get_log_files(self) -> list:
        """
        Get list of all log files

        Returns:
            List of tuples (filename, size_bytes, modified_time)
        """
        log_files = []

        if not os.path.exists(LOG_DIR):
            return log_files

        for filename in os.listdir(LOG_DIR):
            if filename.endswith('.log'):
                filepath = os.path.join(LOG_DIR, filename)
                stat = os.stat(filepath)

                log_files.append({
                    'filename': filename,
                    'path': filepath,
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        return log_files

    def get_old_logs(self, days: int = 30) -> list:
        """
        Get log files older than specified days

        Args:
            days: Age threshold in days

        Returns:
            List of old log file paths
        """
        from datetime import timedelta

        threshold = datetime.now() - timedelta(days=days)
        old_logs = []

        for log_info in self.get_log_files():
            modified_time = datetime.fromisoformat(log_info['modified'])
            if modified_time < threshold:
                old_logs.append(log_info['path'])

        return old_logs


# Global instance
_logger_manager = None


def get_logger_manager() -> SchedulerLogger:
    """Get global SchedulerLogger instance"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = SchedulerLogger()
    return _logger_manager


if __name__ == '__main__':
    # Test logging system
    print("=" * 80)
    print("SCHEDULER LOGGING SYSTEM - TEST")
    print("=" * 80)

    # Get logger manager
    manager = get_logger_manager()

    # Test task logger
    logger = manager.get_task_logger(1, 'Master Market Data Ingestion')
    logger.info("Test message for Task 1")
    logger.warning("Test warning")
    logger.error("Test error")

    # Test scheduler logger
    sched_logger = manager.get_scheduler_logger()
    sched_logger.info("Scheduler test message")

    # List log files
    print("\nLog files created:")
    for log_info in manager.get_log_files():
        print(f"  {log_info['filename']}: {log_info['size_mb']} MB")

    print("\n[OK] Logging system test complete!")

"""
Task Timeout Enforcement
Provides timeout decorator for scheduler tasks
Uses threading.Timer for cross-platform timeout support

Author: TurboMode System
Date: 2026-01-06
"""

import threading
import functools
from typing import Callable, Any, Dict


class TaskTimeoutError(Exception):
    """Exception raised when a task exceeds its timeout"""
    pass


def task_timeout(timeout_minutes: int):
    """
    Decorator to enforce timeout on task functions

    Args:
        timeout_minutes: Maximum execution time in minutes

    Raises:
        TaskTimeoutError: If task exceeds timeout

    Example:
        @task_timeout(30)
        def my_task():
            # Task logic here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = [None]
            exception = [None]
            timeout_seconds = timeout_minutes * 60

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                # Timeout exceeded
                error_msg = f"Task exceeded timeout of {timeout_minutes} minutes"
                raise TaskTimeoutError(error_msg)

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


def with_config_timeout(task_config: Dict[str, Any]):
    """
    Decorator that uses timeout from task config

    Args:
        task_config: Task configuration dict with 'timeout_minutes'

    Example:
        task_config = get_task_config(1)

        @with_config_timeout(task_config)
        def my_task():
            # Task logic here
            pass
    """
    timeout_minutes = task_config.get('timeout_minutes', 60)  # Default 60 min
    return task_timeout(timeout_minutes)


if __name__ == '__main__':
    # Test timeout enforcement
    import time

    print("=" * 80)
    print("TASK TIMEOUT - TEST")
    print("=" * 80)

    # Test 1: Task completes within timeout
    @task_timeout(1)  # 1 minute timeout
    def fast_task():
        print("[TEST 1] Running fast task (2 seconds)...")
        time.sleep(2)
        return {"status": "success"}

    try:
        result = fast_task()
        print(f"[OK] Fast task completed: {result}")
    except TaskTimeoutError as e:
        print(f"[ERROR] Fast task timed out: {e}")

    # Test 2: Task exceeds timeout
    @task_timeout(0.05)  # 3 second timeout (0.05 minutes)
    def slow_task():
        print("[TEST 2] Running slow task (10 seconds)...")
        time.sleep(10)
        return {"status": "success"}

    try:
        result = slow_task()
        print(f"[ERROR] Slow task should have timed out!")
    except TaskTimeoutError as e:
        print(f"[OK] Slow task correctly timed out: {e}")

    # Test 3: Task with exception
    @task_timeout(1)
    def error_task():
        print("[TEST 3] Running error task...")
        raise ValueError("Test error")

    try:
        result = error_task()
        print(f"[ERROR] Error task should have raised exception!")
    except ValueError as e:
        print(f"[OK] Error task correctly raised exception: {e}")
    except TaskTimeoutError:
        print(f"[ERROR] Error task timed out instead of raising exception!")

    print("\n[OK] Timeout enforcement test complete!")

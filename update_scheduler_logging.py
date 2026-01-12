"""
Script to update unified_scheduler.py with task-specific logging
"""

import re

# Read the file
with open('C:\\StockApp\\backend\\unified_scheduler.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the 6 task functions and their task IDs
tasks = [
    ('run_orchestrator', 2, 'TurboMode Training Orchestrator'),
    ('run_overnight_scanner', 3, 'Overnight Scanner'),
    ('run_backtest_generator', 4, 'Backtest Data Generator'),
    ('run_drift_monitor', 5, 'Drift Monitoring System'),
    ('run_weekly_maintenance', 6, 'Weekly Maintenance')
]

# For each task (excluding run_ingestion which is already done)
for func_name, task_id, task_name in tasks:
    # Find the function definition
    func_pattern = rf'(def {func_name}\(\).*?:\n.*?""".*?""")\n(.*?task_id = {task_id}\n.*?task_config = get_task_config\(task_id\))\n\n(.*?logger\.info\("=" \* 80\))'

    match = re.search(func_pattern, content, re.DOTALL)

    if match:
        # Replace the logger initialization part
        before = match.group(1) + '\n' + match.group(2)
        old_logger_part = match.group(3)

        # Create new logger initialization
        new_logger_part = f'''    # Get task-specific logger
    task_logger = logger_manager.get_task_logger(task_id, task_config['name'])

    task_logger.info("=" * 80)'''

        # Replace in content
        old_section = before + '\n\n' + old_logger_part
        new_section = before + '\n\n' + new_logger_part
        content = content.replace(old_section, new_section, 1)

        print(f"Updated {func_name} logger initialization")

        # Now replace all logger. references within this function with task_logger.
        # Find function start and next function start
        func_start = content.find(f'def {func_name}()')

        # Find the next function definition or end of file
        next_func_pattern = r'\n\n# ====.*?\n# TASK \d+:'
        next_func_match = re.search(next_func_pattern, content[func_start + len(func_name) + 10:])

        if next_func_match:
            func_end = func_start + len(func_name) + 10 + next_func_match.start()
        else:
            # Find the next function or scheduler management section
            scheduler_mgmt = content.find('\n# ============================================================================\n# SCHEDULER MANAGEMENT')
            if scheduler_mgmt > func_start:
                func_end = scheduler_mgmt
            else:
                func_end = len(content)

        # Extract function content
        func_content = content[func_start:func_end]

        # Replace logger. with task_logger. (but not logger_manager.)
        updated_func = func_content.replace('logger.info(', 'task_logger.info(')
        updated_func = updated_func.replace('logger.error(', 'task_logger.error(')
        updated_func = updated_func.replace('logger.warning(', 'task_logger.warning(')

        # Replace in main content
        content = content[:func_start] + updated_func + content[func_end:]

        print(f"Updated all logger calls in {func_name}")

# Write back
with open('C:\\StockApp\\backend\\unified_scheduler.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n[OK] All task functions updated with task-specific logging!")

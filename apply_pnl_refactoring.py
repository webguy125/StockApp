"""
Apply P/L calculation refactoring to source files.

This script modifies signal_closer.py and position_manager.py to use
the canonical calculate_pnl_pct() function from pnl_utils.py.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def refactor_signal_closer():
    """Refactor signal_closer.py to use calculate_pnl_pct()"""

    file_path = project_root / 'backend/turbomode/core_engine/signal_closer.py'

    with open(file_path, 'r') as f:
        content = f.read()

    # Step 1: Add import
    if 'from backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct' not in content:
        # Find the imports section
        import_line = 'from backend.turbomode.core_engine.training_symbols import get_symbol_metadata'
        if import_line in content:
            content = content.replace(
                import_line,
                import_line + '\nfrom backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct'
            )
            print("[OK] Added import to signal_closer.py")
        else:
            print("[ERROR] Could not find import section in signal_closer.py")
            return False
    else:
        print("[INFO] Import already exists in signal_closer.py")

    # Step 2: Replace P/L calculation
    old_pnl_calc = '        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0'
    new_pnl_calc = '        # Use canonical P/L calculation (handles BUY/SELL/HOLD correctly)\n        pnl_pct = calculate_pnl_pct(entry_price, current_price, signal_type)'

    if old_pnl_calc in content:
        content = content.replace(old_pnl_calc, new_pnl_calc)
        print("[OK] Replaced P/L calculation in signal_closer.py")
    else:
        print("[INFO] P/L calculation already refactored in signal_closer.py")

    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

    return True


def refactor_position_manager():
    """Refactor position_manager.py to use calculate_pnl_pct()"""

    file_path = project_root / 'backend/turbomode/core_engine/position_manager.py'

    with open(file_path, 'r') as f:
        content = f.read()

    # Step 1: Add import (find appropriate location)
    if 'from backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct' not in content:
        # Find last import line
        lines = content.split('\n')
        import_index = -1
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_index = i

        if import_index >= 0:
            lines.insert(import_index + 1, 'from backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct')
            content = '\n'.join(lines)
            print("[OK] Added import to position_manager.py")
        else:
            print("[ERROR] Could not find import section in position_manager.py")
            return False
    else:
        print("[INFO] Import already exists in position_manager.py")

    # Step 2: Replace P/L calculations
    # Long position
    old_long_calc = '                pnl_pct = (exit_price / entry_price - 1) * 100'
    new_long_calc = '                pnl_pct = calculate_pnl_pct(entry_price, exit_price, \"BUY\")  # Long position'

    if old_long_calc in content:
        content = content.replace(old_long_calc, new_long_calc)
        print("[OK] Replaced long P/L calculation in position_manager.py")

    # Short position
    old_short_calc = '                pnl_pct = (1 - exit_price / entry_price) * 100'
    new_short_calc = '                pnl_pct = calculate_pnl_pct(entry_price, exit_price, \"SELL\")  # Short position'

    if old_short_calc in content:
        content = content.replace(old_short_calc, new_short_calc)
        print("[OK] Replaced short P/L calculation in position_manager.py")

    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

    return True


if __name__ == '__main__':
    print("=" * 80)
    print("APPLYING P/L REFACTORING TO SOURCE FILES")
    print("=" * 80)
    print()

    print("Refactoring signal_closer.py...")
    success1 = refactor_signal_closer()
    print()

    print("Refactoring position_manager.py...")
    success2 = refactor_position_manager()
    print()

    if success1 and success2:
        print("=" * 80)
        print("REFACTORING COMPLETE")
        print("=" * 80)
        print()
        print("All P/L calculations now use the canonical calculate_pnl_pct() function.")
        print("Future SELL positions will calculate correctly.")
    else:
        print("=" * 80)
        print("REFACTORING FAILED")
        print("=" * 80)
        sys.exit(1)

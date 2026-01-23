"""
Add sys.path fix to all Python files in core_engine directory
"""
import os
from pathlib import Path

CORE_ENGINE_DIR = Path(r"C:\StockApp\backend\turbomode\core_engine")
SYS_PATH_BLOCK = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
"""

def file_has_sys_path_fix(filepath):
    """Check if file already has the sys.path fix"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return 'project_root = Path(__file__).resolve().parents[3]' in content

def add_sys_path_fix(filepath):
    """Add sys.path fix to the top of a Python file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find where to insert (after docstring and existing imports)
    insert_pos = 0
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track docstrings
        if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
            docstring_char = stripped[:3]
            if stripped.count(docstring_char) == 2 and len(stripped) > 6:
                # Single-line docstring
                insert_pos = i + 1
                continue
            else:
                in_docstring = True
                continue

        if in_docstring and docstring_char in stripped:
            in_docstring = False
            insert_pos = i + 1
            continue

        # Skip comments and blank lines at top
        if stripped.startswith('#') or not stripped:
            insert_pos = i + 1
            continue

        # Found first non-comment, non-docstring line
        break

    # Insert sys.path block
    lines.insert(insert_pos, '\n' + SYS_PATH_BLOCK + '\n')

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return True

def main():
    py_files = list(CORE_ENGINE_DIR.glob('*.py'))

    print(f"Found {len(py_files)} Python files in core_engine")
    print()

    updated = []
    skipped = []

    for py_file in sorted(py_files):
        if file_has_sys_path_fix(py_file):
            skipped.append(py_file.name)
            print(f"[SKIP] {py_file.name} - already has sys.path fix")
        else:
            add_sys_path_fix(py_file)
            updated.append(py_file.name)
            print(f"[ADD]  {py_file.name} - added sys.path fix")

    print()
    print("=" * 60)
    print(f"Updated: {len(updated)} files")
    print(f"Skipped: {len(skipped)} files")
    print("=" * 60)

if __name__ == '__main__':
    main()

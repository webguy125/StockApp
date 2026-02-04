import os
import re
import sys

print('=' * 80)
print('STOCKAPP PATH AND MODULE VALIDATION AUDIT')
print('=' * 80)
print()

# Step 1: Collect all Python files
python_files = []
scan_dirs = [
    'C:/StockApp/backend',
    'C:/StockApp/master_market_data'
]

for base_dir in scan_dirs:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

print(f'Step 1: Found {len(python_files)} Python files')
print()

# Step 2: Extract imports and references
imports = []
import_pattern = re.compile(r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))', re.MULTILINE)
path_join_pattern = re.compile(r'os\.path\.join\([^)]+\)|[\'"]((?:backend|master_market_data|scheduler)/[^\'"]+)[\'"]')

for py_file in python_files:
    try:
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract imports
        for match in import_pattern.finditer(content):
            from_module = match.group(1)
            import_modules = match.group(2)

            if from_module:
                imports.append({
                    'file': py_file,
                    'type': 'from_import',
                    'module': from_module,
                    'line': content[:match.start()].count('\n') + 1
                })
            elif import_modules:
                for mod in import_modules.split(','):
                    imports.append({
                        'file': py_file,
                        'type': 'import',
                        'module': mod.strip(),
                        'line': content[:match.start()].count('\n') + 1
                    })
    except Exception as e:
        print(f'[WARN] Failed to parse {py_file}: {e}')

print(f'Step 2: Extracted {len(imports)} import statements')
print()

# Step 3: Validate imports
print('Step 3: Validating imports and paths...')
print()

missing_modules = []
project_root = 'C:/StockApp'

for imp in imports:
    module = imp['module']

    # Skip standard library and third-party modules
    if not module.startswith(('backend', 'master_market_data', 'scheduler')):
        continue

    # Convert module path to file path
    module_parts = module.split('.')
    possible_paths = [
        os.path.join(project_root, *module_parts) + '.py',
        os.path.join(project_root, *module_parts, '__init__.py')
    ]

    exists = any(os.path.exists(p) for p in possible_paths)

    if not exists:
        missing_modules.append({
            'module': module,
            'referenced_in': imp['file'],
            'line': imp['line'],
            'expected_paths': possible_paths
        })

print(f'Found {len(missing_modules)} missing module references')
print()

# Check for shadow directories
shadow_dirs = []
backend_path = os.path.join(project_root, 'backend')
if os.path.exists(os.path.join(backend_path, 'backend')):
    shadow_dirs.append('backend/backend (duplicate nesting detected)')

# Step 4: Generate mismatch report
print('=' * 80)
print('MISMATCH REPORT')
print('=' * 80)
print()

if missing_modules:
    print(f'MISSING MODULES ({len(missing_modules)}):')
    print('-' * 80)
    for miss in missing_modules[:20]:  # Limit to first 20
        module = miss['module']
        ref_file = miss['referenced_in']
        line = miss['line']
        expected = miss['expected_paths'][0]
        print(f'  Module: {module}')
        print(f'  Referenced in: {ref_file}:{line}')
        print(f'  Expected: {expected}')
        print()
    if len(missing_modules) > 20:
        print(f'  ... and {len(missing_modules) - 20} more')
        print()
else:
    print('[OK] No missing module imports detected')
    print()

if shadow_dirs:
    print(f'SHADOW DIRECTORIES ({len(shadow_dirs)}):')
    print('-' * 80)
    for shadow in shadow_dirs:
        print(f'  {shadow}')
    print()
else:
    print('[OK] No shadow directories detected')
    print()

# Step 5: Summary
print('=' * 80)
print('SUMMARY')
print('=' * 80)
print()
print(f'Total Python files scanned: {len(python_files)}')
print(f'Total import statements: {len(imports)}')
print(f'Missing module imports: {len(missing_modules)}')
print(f'Shadow directories: {len(shadow_dirs)}')
print()

if missing_modules or shadow_dirs:
    print('STATUS: ISSUES DETECTED')
    print()
    print('Recommended actions:')
    if missing_modules:
        print('  1. Review missing module imports')
        print('  2. Check for renamed/moved files')
        print('  3. Update import statements to match actual file locations')
    if shadow_dirs:
        print('  4. Remove duplicate nested directories')
else:
    print('STATUS: NO ISSUES DETECTED')
    print('All imports resolve correctly.')

print()
print('=' * 80)
print('AUDIT COMPLETE')
print('=' * 80)

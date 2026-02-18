"""
READ-ONLY Schema Subset Verification
Verifies that turbomode.db schema is a complete subset of database_schema.py
Uses strict text-based parsing (no inference or heuristics)
"""

import sqlite3
import re
from pathlib import Path

# Paths
DB_PATH = Path(r"C:\StockApp\backend\data\turbomode.db")
SCHEMA_PATH = Path(r"C:\StockApp\backend\turbomode\database_schema.py")

def extract_canonical_schema(schema_text):
    """
    Extract table schemas from database_schema.py using strict text-based parsing.
    Returns: dict[table_name] = list of (column_name, column_type) tuples
    """
    canonical = {}

    # Find all CREATE TABLE blocks
    # Pattern: CREATE TABLE IF NOT EXISTS table_name (
    pattern = r'CREATE TABLE IF NOT EXISTS (\w+)\s*\('
    matches = list(re.finditer(pattern, schema_text, re.IGNORECASE))

    for i, match in enumerate(matches):
        table_name = match.group(1)
        start_pos = match.end()

        # Find the end of this CREATE TABLE block (closing parenthesis before next CREATE or end of string)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(schema_text)

        # Extract the column definition block
        block = schema_text[start_pos:end_pos]

        # Find the closing parenthesis for this CREATE TABLE
        paren_count = 1
        actual_end = 0
        for j, char in enumerate(block):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    actual_end = j
                    break

        if actual_end > 0:
            block = block[:actual_end]

        # Parse columns from this block
        columns = []
        for line in block.split('\n'):
            line = line.strip()

            # Skip empty lines, comments, and constraint lines
            if not line:
                continue
            if line.startswith('--'):
                continue
            if line.startswith('/*'):
                continue
            if line.upper().startswith('UNIQUE('):
                continue
            if line.upper().startswith('PRIMARY KEY'):
                continue
            if line.upper().startswith('FOREIGN KEY'):
                continue

            # Remove trailing comma
            if line.endswith(','):
                line = line[:-1].strip()

            # Column definition pattern: "column_name TYPE [constraints]"
            # Extract first two tokens
            tokens = line.split()
            if len(tokens) >= 2:
                column_name = tokens[0]
                column_type = tokens[1].upper()

                # Skip if not a valid column (constraints, etc.)
                if column_name.upper() in ['UNIQUE', 'PRIMARY', 'FOREIGN', 'CHECK', 'DEFAULT', 'NOT', 'REFERENCES']:
                    continue

                # Valid SQLite types
                valid_types = ['INTEGER', 'REAL', 'TEXT', 'BLOB', 'NUMERIC']
                if any(t in column_type for t in valid_types):
                    columns.append((column_name, column_type))

        canonical[table_name] = columns

    return canonical

def get_database_schema(db_path):
    """
    Get actual schema from turbomode.db using PRAGMA table_info.
    Returns: dict[table_name] = list of (column_name, column_type) tuples
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get all user tables (exclude sqlite internal tables)
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)

    tables = [row[0] for row in cursor.fetchall()]

    db_schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = []
        for row in cursor.fetchall():
            col_name = row[1]
            col_type = row[2].upper()
            columns.append((col_name, col_type))
        db_schema[table] = columns

    conn.close()
    return db_schema

def verify_subset():
    """
    Verify database schema is a complete subset of canonical schema.
    """
    print("=" * 80)
    print("READ-ONLY SCHEMA SUBSET VERIFICATION")
    print("=" * 80)
    print()

    # Load canonical schema
    print("[1/5] Loading canonical schema from database_schema.py...")
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema_text = f.read()

    canonical = extract_canonical_schema(schema_text)
    print(f"      Found {len(canonical)} tables in canonical schema")
    for table, cols in canonical.items():
        print(f"        - {table}: {len(cols)} columns")
    print()

    # Load database schema
    print("[2/5] Loading actual schema from turbomode.db...")
    db_schema = get_database_schema(DB_PATH)
    print(f"      Found {len(db_schema)} tables in database")
    for table, cols in db_schema.items():
        print(f"        - {table}: {len(cols)} columns")
    print()

    # Check for missing tables
    print("[3/5] Checking for tables in database but not in canonical schema...")
    missing_tables = []
    for table in db_schema.keys():
        if table not in canonical:
            missing_tables.append(table)

    if missing_tables:
        print(f"      [WARN] Found {len(missing_tables)} table(s) in DB not in canonical schema:")
        for table in missing_tables:
            print(f"        - {table}")
    else:
        print("      [OK] All database tables exist in canonical schema")
    print()

    # Check for missing columns
    print("[4/5] Checking for columns in database but not in canonical schema...")
    missing_columns = {}

    for table, db_cols in db_schema.items():
        if table not in canonical:
            # Already reported as missing table
            continue

        canonical_cols = canonical[table]
        canonical_col_names = set([col[0] for col in canonical_cols])

        table_missing = []
        for col_name, col_type in db_cols:
            if col_name not in canonical_col_names:
                table_missing.append((col_name, col_type))

        if table_missing:
            missing_columns[table] = table_missing

    if missing_columns:
        print(f"      [FAIL] Found missing columns in {len(missing_columns)} table(s):")
        for table, cols in missing_columns.items():
            print(f"        Table: {table}")
            for col_name, col_type in cols:
                print(f"          - {col_name} {col_type}")
    else:
        print("      [OK] All database columns exist in canonical schema")
    print()

    # Final verdict
    print("[5/5] Final Verdict:")
    print("=" * 80)

    if missing_tables or missing_columns:
        print("[FAIL] Database schema is NOT a complete subset of canonical schema")
        print()
        print("Summary:")
        print(f"  - Missing tables: {len(missing_tables)}")
        print(f"  - Tables with missing columns: {len(missing_columns)}")

        if missing_tables:
            print()
            print("Missing Tables:")
            for table in missing_tables:
                print(f"  - {table}")

        if missing_columns:
            print()
            print("Missing Columns:")
            for table, cols in missing_columns.items():
                print(f"  Table: {table}")
                for col_name, col_type in cols:
                    print(f"    - {col_name} {col_type}")
    else:
        print("[PASS] Database schema is a complete subset of canonical schema")
        print()
        print("All tables and columns in turbomode.db exist in database_schema.py")

    print("=" * 80)

    return len(missing_tables) == 0 and len(missing_columns) == 0

if __name__ == '__main__':
    success = verify_subset()
    exit(0 if success else 1)

"""
Schema Manager for TurboMode

One script that:
0) Syncs canonical database_schema.py into the rebuild directory
1) Extracts schema from the REAL turbomode.db
2) Extracts schema from Python (database_schema.TurboModeDB)
3) Compares them and prints a diff
4) Archives old ingestion schema into timestamped folder
5) Copies newly verified schema into ingestion directory
6) Renames turbomode.db → turbomode_OLD.db
7) Ingestion rebuilds new DB

This is designed to be deterministic and contamination-proof.
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple


# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

# Real production DB
REAL_DB_PATH = r"C:\StockApp\backend\data\turbomode.db"

# Temp DB for Python schema
TEMP_PY_SCHEMA_DB_PATH = r"C:\StockApp\backend\data\turbomode_schema_temp.db"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DB_SCHEMA_SQL_PATH = SCRIPT_DIR / "extracted_schema_db.sql"
PY_SCHEMA_SQL_PATH = SCRIPT_DIR / "extracted_schema_python.sql"

# Canonical ingestion schema file
CANONICAL_SCHEMA_PATH = Path(r"C:\StockApp\backend\turbomode\database_schema.py")

# Backup folder (Option 2)
SCHEMA_BACKUP_DIR = SCRIPT_DIR / "schema_backups"

# Add turbomode dir for import
turbomode_dir = SCRIPT_DIR.parent
sys.path.insert(0, str(turbomode_dir))

try:
    from database_schema import TurboModeDB
except ImportError as e:
    print("[ERROR] Could not import TurboModeDB from database_schema.py")
    print("  Expected path:", os.path.join(str(turbomode_dir), "database_schema.py"))
    print("  Import error:", e)
    raise SystemExit(1)


# --------------------------------------------------------------------------------------
# STEP 0: Sync canonical schema into rebuild directory
# --------------------------------------------------------------------------------------

def sync_canonical_schema() -> None:
    print("=" * 80)
    print("STEP 0: SYNC CANONICAL PYTHON SCHEMA")
    print("=" * 80)

    target = SCRIPT_DIR / "database_schema.py"

    if not CANONICAL_SCHEMA_PATH.exists():
        print("[ERROR] Canonical schema file not found:")
        print("        ", CANONICAL_SCHEMA_PATH)
        raise SystemExit(1)

    target.write_text(CANONICAL_SCHEMA_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    print("[OK] Synced canonical schema into rebuild directory:")
    print("     ", target)
    print()


# --------------------------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    if sql is None:
        return ""
    s = sql.strip().rstrip(";")
    parts = s.split()
    return " ".join(parts).lower()


def fetch_schema_from_sqlite(db_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        raise SystemExit(1)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, sql FROM sqlite_master
        WHERE type='table' AND sql IS NOT NULL
        ORDER BY name
    """)
    tables = {name: sql for (name, sql) in cursor.fetchall()}

    cursor.execute("""
        SELECT name, sql FROM sqlite_master
        WHERE type='index' AND sql IS NOT NULL
        ORDER BY name
    """)
    indices = {name: sql for (name, sql) in cursor.fetchall()}

    conn.close()
    return tables, indices


def write_schema_sql_file(path: Path, tables: Dict[str, str], indices: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("-- Extracted Schema\n\n")
        for name, sql in tables.items():
            f.write(sql.strip().rstrip(";") + ";\n\n")
        for name, sql in indices.items():
            f.write(sql.strip().rstrip(";") + ";\n\n")


# --------------------------------------------------------------------------------------
# STEP 1: Extract schema from REAL DB
# --------------------------------------------------------------------------------------

def extract_schema_from_real_db():
    print("=" * 80)
    print("STEP 1: EXTRACT SCHEMA FROM REAL DB")
    print("=" * 80)
    print("DB path:", REAL_DB_PATH)
    print()

    tables, indices = fetch_schema_from_sqlite(REAL_DB_PATH)

    print(f"[OK] Found {len(tables)} tables in REAL DB")
    print(f"[OK] Found {len(indices)} indices in REAL DB")
    print()

    write_schema_sql_file(DB_SCHEMA_SQL_PATH, tables, indices)
    print(f"[OK] Saved REAL DB schema to: {DB_SCHEMA_SQL_PATH}")
    print()

    return tables, indices


# --------------------------------------------------------------------------------------
# STEP 2: Extract schema from Python (TurboModeDB)
# --------------------------------------------------------------------------------------

def build_temp_db_from_python_schema():
    if os.path.exists(TEMP_PY_SCHEMA_DB_PATH):
        os.remove(TEMP_PY_SCHEMA_DB_PATH)

    print("=" * 80)
    print("STEP 2: BUILD TEMP DB FROM PYTHON SCHEMA (TurboModeDB)")
    print("=" * 80)
    print("Temp DB path:", TEMP_PY_SCHEMA_DB_PATH)
    print()

    TurboModeDB(db_path=TEMP_PY_SCHEMA_DB_PATH)


def extract_schema_from_python():
    build_temp_db_from_python_schema()

    tables, indices = fetch_schema_from_sqlite(TEMP_PY_SCHEMA_DB_PATH)

    print(f"[OK] Found {len(tables)} tables in PYTHON schema")
    print(f"[OK] Found {len(indices)} indices in PYTHON schema")
    print()

    write_schema_sql_file(PY_SCHEMA_SQL_PATH, tables, indices)
    print(f"[OK] Saved PYTHON schema snapshot to: {PY_SCHEMA_SQL_PATH}")
    print()

    return tables, indices


# --------------------------------------------------------------------------------------
# STEP 3: Compare schemas
# --------------------------------------------------------------------------------------

def compare_schemas(db_tables, db_indices, py_tables, py_indices) -> bool:
    print("=" * 80)
    print("STEP 3: COMPARE SCHEMAS (REAL DB vs PYTHON)")
    print("=" * 80)
    print()

    ok = True

    db_table_names = set(db_tables.keys())
    py_table_names = set(py_tables.keys())

    only_in_db = sorted(db_table_names - py_table_names)
    only_in_py = sorted(py_table_names - db_table_names)

    if only_in_db:
        ok = False
        print("[DIFF] Tables in REAL DB but missing in PYTHON schema:")
        for name in only_in_db:
            print("  -", name)
        print()

    if only_in_py:
        ok = False
        print("[DIFF] Tables in PYTHON schema but missing in REAL DB:")
        for name in only_in_py:
            print("  -", name)
        print()

    for name in sorted(db_table_names & py_table_names):
        if normalize_sql(db_tables[name]) != normalize_sql(py_tables[name]):
            ok = False
            print(f"[DIFF] Table mismatch: {name}")
            print("  REAL DB:", db_tables[name].strip().replace("\n", " "))
            print("  PYTHON :", py_tables[name].strip().replace("\n", " "))
            print()

    db_index_names = set(db_indices.keys())
    py_index_names = set(py_indices.keys())

    only_idx_in_db = sorted(db_index_names - py_index_names)
    only_idx_in_py = sorted(py_index_names - db_index_names)

    if only_idx_in_db:
        ok = False
        print("[DIFF] Indices in REAL DB but missing in PYTHON schema:")
        for name in only_idx_in_db:
            print("  -", name)
        print()

    if only_idx_in_py:
        ok = False
        print("[DIFF] Indices in PYTHON schema but missing in REAL DB:")
        for name in only_idx_in_py:
            print("  -", name)
        print()

    for name in sorted(db_index_names & py_index_names):
        if normalize_sql(db_indices[name]) != normalize_sql(py_indices[name]):
            ok = False
            print(f"[DIFF] Index mismatch: {name}")
            print("  REAL DB:", db_indices[name].strip().replace("\n", " "))
            print("  PYTHON :", py_indices[name].strip().replace("\n", " "))
            print()

    if ok:
        print("[OK] Schemas MATCH")
    else:
        print("[WARNING] Schemas DO NOT MATCH")
    print()

    return ok


# --------------------------------------------------------------------------------------
# STEP 4: Archive old ingestion schema (Option B4)
# --------------------------------------------------------------------------------------

def archive_old_ingestion_schema():
    print("=" * 80)
    print("STEP 4: ARCHIVE OLD INGESTION SCHEMA")
    print("=" * 80)

    if not CANONICAL_SCHEMA_PATH.exists():
        print("[INFO] No ingestion schema found to archive.")
        print()
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = SCHEMA_BACKUP_DIR / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_path = archive_dir / "database_schema.py"

    CANONICAL_SCHEMA_PATH.replace(archived_path)

    print("[OK] Archived old ingestion schema to:")
    print("     ", archived_path)
    print()


# --------------------------------------------------------------------------------------
# STEP 5: Copy new verified schema into ingestion directory
# --------------------------------------------------------------------------------------

def install_new_ingestion_schema():
    print("=" * 80)
    print("STEP 5: INSTALL NEW VERIFIED SCHEMA FOR INGESTION")
    print("=" * 80)

    source = SCRIPT_DIR / "database_schema.py"

    if not source.exists():
        print("[ERROR] Verified schema missing in rebuild directory.")
        raise SystemExit(1)

    source.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    source.replace(CANONICAL_SCHEMA_PATH)

    print("[OK] Installed new ingestion schema:")
    print("     ", CANONICAL_SCHEMA_PATH)
    print()


# --------------------------------------------------------------------------------------
# STEP 6: Rename DB for ingestion rebuild
# --------------------------------------------------------------------------------------

def rename_real_db():
    print("=" * 80)
    print("STEP 6: RENAME REAL DB FOR INGESTION REBUILD")
    print("=" * 80)

    if not os.path.exists(REAL_DB_PATH):
        print("[INFO] REAL DB does not exist. Nothing to rename.")
        print()
        return

    base_old_path = REAL_DB_PATH.replace("turbomode.db", "turbomode_OLD.db")
    final_old_path = base_old_path

    counter = 1
    while os.path.exists(final_old_path):
        final_old_path = base_old_path.replace(".db", f"_{counter}.db")
        counter += 1

    os.rename(REAL_DB_PATH, final_old_path)

    print("[OK] Renamed REAL DB to:")
    print("     ", final_old_path)
    print()
    print("Ingestion will now rebuild turbomode.db using the new schema.")
    print()


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("TURBOMODE SCHEMA MANAGER")
    print("=" * 80)
    print()

    sync_canonical_schema()

    db_tables, db_indices = extract_schema_from_real_db()
    py_tables, py_indices = extract_schema_from_python()

    schemas_match = compare_schemas(db_tables, db_indices, py_tables, py_indices)

    if not schemas_match:
        print("Schemas DO NOT match. Aborting rebuild.")
        print("Fix database_schema.py to match REAL DB, then rerun.")
        print()
        return

    archive_old_ingestion_schema()
    install_new_ingestion_schema()
    rename_real_db()


if __name__ == "__main__":
    main()
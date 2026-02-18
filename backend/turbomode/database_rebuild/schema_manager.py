import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import os
import sqlite3
import shutil
from datetime import datetime

# Paths
DB_PATH = "C:/StockApp/backend/data/turbomode.db"
SCHEMA_FILE_PATH = "C:/StockApp/backend/turbomode/database_schema.py"
BACKUP_DIR = "C:/StockApp/backend/turbomode/database_rebuild/schema_backups"
EXTRACTED_SQL_PATH = "C:/StockApp/backend/turbomode/database_rebuild/extracted_schema_db.sql"

def archive_old_schema_file():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = os.path.join(BACKUP_DIR, f"database_schema_{timestamp}.py")
    if os.path.exists(SCHEMA_FILE_PATH):
        shutil.copy2(SCHEMA_FILE_PATH, backup_path)
        print(f"[STEP 0A] Archived schema to {backup_path}")
    else:
        print(f"[STEP 0A] No existing schema file to archive")

def rename_db_in_place():
    if not os.path.exists(DB_PATH):
        print(f"[STEP 0B] Database does not exist: {DB_PATH}")
        return None

    # Check database size - must be at least 1GB to be valid
    db_size_bytes = os.path.getsize(DB_PATH)
    db_size_gb = db_size_bytes / (1024**3)

    if db_size_gb < 1.0:
        print(f"[STEP 0B] Database too small ({db_size_gb:.2f} GB) - skipping rename")
        print(f"[STEP 0B] Minimum size: 1 GB")
        print(f"[STEP 0B] This prevents renaming empty/partial databases")
        return None

    base_name = "C:/StockApp/backend/data/turbomode_OLD"
    counter = 0
    new_path = f"{base_name}.db"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_name}_{counter}.db"

    os.rename(DB_PATH, new_path)
    print(f"[STEP 0B] Renamed DB to {new_path} (size: {db_size_gb:.2f} GB)")
    return new_path

def extract_schema_from_renamed_db(renamed_db_path):
    conn = sqlite3.connect(renamed_db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'index') AND sql IS NOT NULL AND name != 'sqlite_sequence' ORDER BY type DESC, name")
    rows = cursor.fetchall()
    conn.close()

    sql_statements = [row[0] for row in rows]

    os.makedirs(os.path.dirname(EXTRACTED_SQL_PATH), exist_ok=True)
    with open(EXTRACTED_SQL_PATH, 'w') as f:
        for sql in sql_statements:
            f.write(sql + ";\n\n")

    print(f"[STEP 1] Extracted {len(sql_statements)} SQL statements to {EXTRACTED_SQL_PATH}")
    return sql_statements

def generate_database_schema_py(sql_statements):
    schema_code = '''import sqlite3
import os

class TurboModeDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self._init_schema()

    def _init_schema(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

'''

    for sql in sql_statements:
        escaped_sql = sql.replace("'", "\\'").replace('"', '\\"')
        schema_code += f'        cursor.execute("""{sql}""")\n\n'

    schema_code += '''        self.conn.commit()
        cursor.close()
'''

    print(f"[STEP 2] Generated database_schema.py with {len(sql_statements)} statements")
    return schema_code

def install_new_schema_file(schema_code):
    with open(SCHEMA_FILE_PATH, 'w') as f:
        f.write(schema_code)
    print(f"[STEP 3] Installed new schema to {SCHEMA_FILE_PATH}")

def main():
    print("=" * 80)
    print("SCHEMA REBUILD MANAGER")
    print("=" * 80)

    archive_old_schema_file()
    renamed_db_path = rename_db_in_place()

    # If rename was skipped (database too small), use existing OLD database
    if renamed_db_path is None:
        old_db_path = "C:/StockApp/backend/data/turbomode_OLD.db"
        if os.path.exists(old_db_path):
            print(f"[INFO] Using existing OLD database: {old_db_path}")
            renamed_db_path = old_db_path
        else:
            print("[ERROR] No database to extract schema from")
            return

    sql_statements = extract_schema_from_renamed_db(renamed_db_path)
    schema_code = generate_database_schema_py(sql_statements)
    install_new_schema_file(schema_code)

    print("\n" + "=" * 80)
    print("[STEP 4] REBUILD COMPLETE")
    print("=" * 80)
    print("Next steps:")
    print("1. Run ingestion to rebuild a fresh turbomode.db")
    print("2. The old database is preserved at:", renamed_db_path)
    print("=" * 80)

if __name__ == '__main__':
    main()

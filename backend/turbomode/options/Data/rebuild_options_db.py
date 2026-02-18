import os
import sqlite3

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'options_intel.db')
SCHEMA_PATH = os.path.join(BASE_DIR, 'schema_options_intel.sql')


def rebuild_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        print('[OPTIONS_DB] Rebuild complete.')
    finally:
        conn.close()


if __name__ == '__main__':
    rebuild_database()

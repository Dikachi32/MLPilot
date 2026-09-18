"""
One-off schema migration for MLPilot.

Adds the columns that models/experiment.py gained but that an existing
SQLite file does not have (db.create_all() creates missing TABLES, never
missing COLUMNS).

Run from the project root, with the venv active:

    python migrate_db.py

Safe to run more than once — it inspects the table first and only adds
what is actually missing. It never drops or rewrites existing data.
"""
import sys
from sqlalchemy import inspect, text

from app import create_app
from extensions import db

# column name -> SQLite type for ALTER TABLE
REQUIRED_COLUMNS = {
    "experiments": {
        "dataset_shape": "VARCHAR(50)",
        "run_mode": "VARCHAR(20)",
        "result_json": "TEXT",
    },
}


def main():
    app = create_app()
    with app.app_context():
        db_path = db.engine.url.database
        print(f"Database in use: {db_path}\n")

        # Create any table that does not exist at all.
        db.create_all()

        inspector = inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        total_added = 0

        for table, columns in REQUIRED_COLUMNS.items():
            if table not in existing_tables:
                print(f"  {table}: table did not exist — created fresh by create_all()")
                continue

            existing_columns = {c["name"] for c in inspector.get_columns(table)}

            for column, coltype in columns.items():
                if column in existing_columns:
                    print(f"  {table}.{column:<16} already present — skipped")
                    continue
                try:
                    db.session.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                    )
                    db.session.commit()
                    print(f"  {table}.{column:<16} ADDED ({coltype})")
                    total_added += 1
                except Exception as e:
                    db.session.rollback()
                    print(f"  {table}.{column:<16} FAILED: {e}")
                    return 1

        # Verify the model and the table now agree.
        inspector = inspect(db.engine)
        final_columns = {c["name"] for c in inspector.get_columns("experiments")}
        model_columns = {c.name for c in db.metadata.tables["experiments"].columns}
        missing = model_columns - final_columns

        print()
        if missing:
            print(f"STILL MISSING: {sorted(missing)}")
            return 1

        print(f"Migration complete — {total_added} column(s) added.")
        print("experiments table now matches models/experiment.py.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
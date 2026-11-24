"""Clear long-term user memory before recording a demo, keeping a timestamped backup.

Run with the backend stopped:  python scripts/reset_demo_memory.py
Session checkpoints are left alone: a new login starts a new session anyway.
"""
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402


def main() -> None:
    db = Path(get_settings().memory_db_path)
    if not db.exists():
        print(f"No memory database at {db}; nothing to reset.")
        return
    backup = db.with_name(f"{db.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(db, backup)
    with sqlite3.connect(db) as con:
        counts = con.execute("SELECT username, COUNT(*) FROM user_memory GROUP BY username").fetchall()
        con.execute("DELETE FROM user_memory")
    print(f"Backed up to {backup}")
    print("Cleared long-term memory rows:", dict(counts) or "none")


if __name__ == "__main__":
    main()

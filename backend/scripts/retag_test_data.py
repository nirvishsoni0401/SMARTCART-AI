import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent.parent / "smartcart.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("Before retagging events:", c.execute("SELECT origin, COUNT(*) FROM interaction_events GROUP BY origin").fetchall())
print("Before retagging logs:", c.execute("SELECT origin, COUNT(*) FROM recommendation_logs GROUP BY origin").fetchall())

c.execute("UPDATE interaction_events SET origin = 'test' WHERE origin = 'live'")
c.execute("UPDATE recommendation_logs SET origin = 'test' WHERE origin = 'live'")
conn.commit()

print("After retagging events:", c.execute("SELECT origin, COUNT(*) FROM interaction_events GROUP BY origin").fetchall())
print("After retagging logs:", c.execute("SELECT origin, COUNT(*) FROM recommendation_logs GROUP BY origin").fetchall())

conn.close()

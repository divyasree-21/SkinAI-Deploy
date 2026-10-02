
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# Store the database in the project folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DB_PATH = DATA_DIR / "skinai_history.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS screening_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL
            )
        """)



def save_screening_result(
    predicted_class,
    confidence,
    probabilities
):
    created_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO screening_history (
                created_at,
                predicted_class,
                confidence,
                probabilities
            )
            VALUES (?, ?, ?, ?)
        """, (
            created_at,
            str(predicted_class),
            float(confidence),
            json.dumps(probabilities),
        ))

    return created_at

def get_recent_history(limit=20):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, created_at, predicted_class,
                   confidence, probabilities
            FROM screening_history
            ORDER BY id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()

    records = []

    for row in rows:
        records.append({
            "ID": row["id"],
            "Date and time": row["created_at"],
            "Predicted class": row["predicted_class"],
            "Confidence (%)": round(row["confidence"], 2),
            "Probabilities": json.loads(row["probabilities"]),
        })

    return records


def delete_all_history():
    with get_connection() as connection:
        connection.execute("DELETE FROM screening_history")


initialize_database()

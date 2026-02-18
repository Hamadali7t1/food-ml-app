import hashlib
import os
import sqlite3
from typing import Optional, Tuple

import pandas as pd

DB_PATH = "data/nutrition_app.db"


def _get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs("data", exist_ok=True)
    _migrate_users_table_if_needed()
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS intake_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                food_label TEXT NOT NULL,
                weight_grams REAL NOT NULL,
                confidence REAL NOT NULL,
                calories REAL NOT NULL,
                protein REAL NOT NULL,
                carbs REAL NOT NULL,
                fats REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )


def _migrate_users_table_if_needed():
    with _get_connection() as conn:
        cols = conn.execute("PRAGMA table_info(users)").fetchall()
        if not cols:
            return

        col_names = {row[1] for row in cols}
        required = {"id", "name", "email", "password_hash", "salt", "created_at"}
        if required.issubset(col_names):
            return

        conn.execute("DROP TABLE IF EXISTS intake_entries")
        conn.execute("DROP TABLE IF EXISTS users")


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return digest.hex()


def create_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    if not name or not email or not password:
        return False, "Name, email, and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    clean_name = name.strip()
    clean_email = email.strip().lower()
    if "@" not in clean_email or "." not in clean_email:
        return False, "Please enter a valid email address."

    salt = os.urandom(16)
    pwd_hash = _hash_password(password, salt)
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                (clean_name, clean_email, pwd_hash, salt.hex()),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "Email is already registered."


def verify_user(email: str, password: str) -> Tuple[bool, Optional[int], Optional[str], str]:
    clean_email = email.strip().lower()
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, password_hash, salt FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()

    if row is None:
        return False, None, None, "Invalid email or password."

    user_id, name, stored_hash, salt_hex = row
    candidate_hash = _hash_password(password, bytes.fromhex(salt_hex))
    if candidate_hash != stored_hash:
        return False, None, None, "Invalid email or password."
    return True, int(user_id), str(name), "Login successful."


def add_intake_entry(
    user_id: int,
    food_label: str,
    weight_grams: float,
    confidence: float,
    nutrients: dict,
):
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO intake_entries (
                user_id, food_label, weight_grams, confidence, calories, protein, carbs, fats
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                food_label,
                float(weight_grams),
                float(confidence),
                float(nutrients["Calories (kcal)"]),
                float(nutrients["Protein (g)"]),
                float(nutrients["Carbs (g)"]),
                float(nutrients["Fats (g)"]),
            ),
        )


def get_daily_totals(user_id: int) -> dict:
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(calories), 0),
                COALESCE(SUM(protein), 0),
                COALESCE(SUM(carbs), 0),
                COALESCE(SUM(fats), 0)
            FROM intake_entries
            WHERE user_id = ? AND date(created_at) = date('now', 'localtime')
            """,
            (user_id,),
        ).fetchone()

    return {
        "Calories": float(row[0]),
        "Protein": float(row[1]),
        "Carbs": float(row[2]),
        "Fats": float(row[3]),
    }


def get_hourly_totals(user_id: int) -> pd.DataFrame:
    query = """
        SELECT
            strftime('%H:00', created_at) AS hour,
            SUM(calories) AS calories,
            SUM(protein) AS protein,
            SUM(carbs) AS carbs,
            SUM(fats) AS fats
        FROM intake_entries
        WHERE user_id = ? AND date(created_at) = date('now', 'localtime')
        GROUP BY strftime('%H', created_at)
        ORDER BY hour
    """
    with _get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(user_id,))
    return df


def get_recent_entries(user_id: int, limit: int = 25) -> pd.DataFrame:
    query = """
        SELECT
            created_at,
            food_label,
            weight_grams,
            confidence,
            calories,
            protein,
            carbs,
            fats
        FROM intake_entries
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """
    with _get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(user_id, limit))
    return df


def clear_today_entries(user_id: int):
    with _get_connection() as conn:
        conn.execute(
            """
            DELETE FROM intake_entries
            WHERE user_id = ? AND date(created_at) = date('now', 'localtime')
            """,
            (user_id,),
        )


def truncate_database():
    with _get_connection() as conn:
        conn.execute("DELETE FROM intake_entries")
        conn.execute("DELETE FROM users")

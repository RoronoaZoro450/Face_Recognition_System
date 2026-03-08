import sqlite3
import csv
from datetime import datetime

class DatabaseManager:

    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.cursor = self.connection.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_records (
            sr INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL,
            name TEXT NOT NULL,
            similarity REAL,
            timestamp DATETIME DEFAULT (datetime('now','localtime'))
        )
        """)

        self.connection.commit()


    def insert_record(self, id, name, similarity):

        insert_record = """
        INSERT INTO login_records (id, name, similarity)
        VALUES (?, ?, ?)
        """

        self.cursor.execute(insert_record, (id, name, similarity))
        self.connection.commit()


    def export_csv(self, file_path):

        if not file_path.endswith(".csv"):
            file_path += ".csv"

        self.cursor.execute("SELECT sr, id, name, similarity, timestamp FROM login_records")
        rows = self.cursor.fetchall()

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Sr", "ID", "Name", "Similarity", "Timestamp"])
            writer.writerows(rows)

    def delete_record(self):
        delete_query = "DELETE FROM login_records"
        self.cursor.execute(delete_query)
        self.connection.commit()

    def close(self):
        self.connection.close()
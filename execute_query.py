import sqlite3
import pandas as pd

def execute_sql_query(db_path: str, sql_query: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql_query, conn)
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    finally:
        conn.close()
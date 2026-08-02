import os
import sqlite3
import pandas as pd

INPUT_FOLDER = "input"
DB_NAME = "data.db"

conn = sqlite3.connect(DB_NAME)

for file_name in os.listdir(INPUT_FOLDER):
    if file_name.endswith(".csv"):
        csv_path = os.path.join(INPUT_FOLDER, file_name)
        for encoding in ["utf-8", "cp1252", "latin1"]:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Read {file_name} using {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Failed to read {file_name}")
            continue
        table_name = os.path.splitext(file_name)[0]
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Loaded {file_name} into {table_name}")

conn.close()
import sqlite3
import pandas as pd

def get_table_info_markdown(db_path: str, table_names: list[str], top_k: int = 5) -> str:
    conn = sqlite3.connect(db_path)
    markdown_output = []
    for table_name in table_names:
        markdown_output.append(f"# Table: {table_name}\n")
        schema_df = pd.read_sql_query(f"PRAGMA table_info('{table_name}')", conn)
        markdown_output.append("## Schema\n")
        schema_md = (schema_df[["name", "type"]]
                     .rename(columns={"name": "Column Name", "type": "Data Type"})
                     .to_markdown(index=False))
        markdown_output.append(schema_md)
        markdown_output.append("\n")
        sample_df = pd.read_sql_query(f"SELECT * FROM '{table_name}' LIMIT {top_k}", conn)
        markdown_output.append(f"## Sample Rows (Top {top_k})\n")
        if len(sample_df) > 0:
            markdown_output.append(sample_df.to_markdown(index=False))
        else:
            markdown_output.append("No data found.")
        markdown_output.append("\n---\n")
    conn.close()
    return "\n".join(markdown_output)
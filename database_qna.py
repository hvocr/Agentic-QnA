import re
import sqlite3
import pandas as pd
from groq import Groq
import streamlit as st
from get_schema import get_table_info_markdown
from execute_query import execute_sql_query

class DatabaseQnA:
    def __init__(self, db_path: str = "data.db", table_names: list = None):
        self.db_path = db_path
        # If no table names provided, try to get them from the DB
        if table_names is None:
            self.table_names = self._get_table_names()
        else:
            self.table_names = table_names
        self.schema_context = get_table_info_markdown(self.db_path, self.table_names)

    def _get_table_names(self):
        """Fetch table names from the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def _clean_sql(self, sql: str) -> str:
        """Remove markdown code fences."""
        sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```", "", sql)
        return sql.strip()

    def _validate_sql(self, sql: str):
        """Ensure SQL is a SELECT query only."""
        sql_lower = sql.lower().strip()
        forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "replace"]
        for kw in forbidden:
            if re.search(rf"\b{kw}\b", sql_lower):
                raise Exception(f"Unsafe SQL detected: {kw}")
        if not sql_lower.startswith("select"):
            raise Exception("Only SELECT queries are allowed.")

    def _generate_sql(self, question: str, groq_api_key: str) -> str:
        """Generate SQL from question using Groq."""
        client = Groq(api_key=groq_api_key)
        prompt = f"""
You are an expert SQLite query generator.

Database Schema:
{self.schema_context}

**Strict Instructions**:
1. Generate ONLY a SQLite SELECT query.
2. Use LIKE for text matching with LOWER for case-insensitivity.
3. Never generate UPDATE, DELETE, INSERT, DROP, ALTER.
4. Return only the SQL statement, no explanation, no markdown.

Question: {question}
SQL:
"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        sql = response.choices[0].message.content
        sql = self._clean_sql(sql)
        self._validate_sql(sql)
        return sql

    def _generate_response(self, question: str, sql: str, result: list, groq_api_key: str) -> str:
        """Generate natural language answer from query result using Groq."""
        client = Groq(api_key=groq_api_key)
        prompt = f"""
You are a hospital assistant.

User Question: {question}

Generated SQL: {sql}

Query Result (JSON): {result}

Instructions:
1. Answer the user naturally using only the provided data.
2. If no records are found, clearly say so.
3. Do not mention SQL or JSON.
4. Keep response concise.
"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content

    def answer(self, question: str, groq_api_key: str) -> dict:
        """
        Answer a question using the database.
        Returns a dict with 'answer', 'source', 'chunks' (empty), and 'confidence'.
        """
        try:
            # 1. Generate SQL
            sql = self._generate_sql(question, groq_api_key)
            # 2. Execute
            results = execute_sql_query(self.db_path, sql)
            # 3. Generate natural response
            answer = self._generate_response(question, sql, results, groq_api_key)
            return {
                "answer": answer,
                "source": "Database",
                "chunks": [],  # No chunks for DB, but we could include the SQL if needed
                "confidence": 0.9
            }
        except Exception as e:
            return {
                "answer": f"Database error: {str(e)}",
                "source": "Database",
                "chunks": [],
                "confidence": 0.0
            }
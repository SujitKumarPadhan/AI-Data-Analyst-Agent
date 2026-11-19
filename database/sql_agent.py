
import sqlite3

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


# 1. SQL response structure
class SQLQuery(BaseModel):

    query: str = Field(
        description="A valid SQLite SELECT query"
    )


# 2. Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

structured_llm = llm.with_structured_output(
    SQLQuery,
    method="json_mode"
)


# 3. Execute SQL safely on uploaded dataset
def execute_sql(query, db_path):

    query = query.strip().rstrip(";").strip()

    if not query.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed!")

    if ";" in query:
        raise ValueError("Multiple SQL statements are not allowed!")

    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True
    )

    try:

        connection.execute("PRAGMA query_only = ON")

        cursor = connection.cursor()
        cursor.execute(query)

        columns = [
            description[0]
            for description in cursor.description
        ]

        results = cursor.fetchall()

        return columns, results

    finally:
        connection.close()


# 4. Dynamic SQL Agent
def run_sql_agent(user_question, db_path, columns):

    column_details = ", ".join(columns)

    response = structured_llm.invoke(
        f"""
        You are an expert SQLite Data Analyst.

        User Question:
        {user_question}

        Database Table:
        dataset

        Available Columns:
        {column_details}

        Instructions:
        1. Generate one valid SQLite SELECT query.
        2. Use only the available columns.
        3. Use table name dataset.
        4. Never generate INSERT, UPDATE, DELETE or DROP.
        5. Return only a valid SELECT query.

        Return JSON field "query".
        """
    )

    sql_query = response.query

    print("\nGenerated SQL:")
    print(sql_query)

    return execute_sql(
        sql_query,
        db_path
    )


if __name__ == "__main__":

    print("Dynamic SQL Agent is ready!")

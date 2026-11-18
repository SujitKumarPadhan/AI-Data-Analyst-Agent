
import os
import sqlite3

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()


# 1. Define SQL response structure
class SQLQuery(BaseModel):
    query: str = Field(
        description="A valid SQLite SELECT query"
    )


# 2. Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

structured_llm = llm.with_structured_output(
    SQLQuery,
    method="json_mode"
)


# 3. Execute SQL query
def execute_sql(query):

    query = query.strip()

    # Remove trailing semicolon
    query = query.rstrip(";").strip()

    # Allow only SELECT queries
    if not query.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed!")

    # Reject multiple SQL statements
    if ";" in query:
        raise ValueError("Multiple SQL statements are not allowed!")

    connection = sqlite3.connect(
        "file:data/sales.db?mode=ro",
        uri=True
    )

    try:
        connection.execute("PRAGMA query_only = ON")

        cursor = connection.cursor()
        cursor.execute(query)

        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()

        return columns, results

    finally:
        connection.close()


# 4. Display results in table format
def display_results(columns, results):

    if not results:
        print("\nNo records found.")
        return

    formatted_rows = [
        [str(value) for value in row]
        for row in results
    ]

    column_widths = []

    for index, column in enumerate(columns):
        max_width = max(
            len(column),
            max(len(row[index]) for row in formatted_rows)
        )
        column_widths.append(max_width)

    separator = "+".join(
        "-" * (width + 2)
        for width in column_widths
    )

    print("\n+" + separator + "+")

    header = " | ".join(
        column.ljust(column_widths[index])
        for index, column in enumerate(columns)
    )

    print("| " + header + " |")
    print("+" + separator + "+")

    for row in formatted_rows:

        row_data = " | ".join(
            value.ljust(column_widths[index])
            for index, value in enumerate(row)
        )

        print("| " + row_data + " |")

    print("+" + separator + "+")
    print(f"Total Records: {len(results)}")



# 5. Reusable SQL Agent Function

def run_sql_agent(user_question):

    # Generate SQL using Gemini
    response = structured_llm.invoke(
        f"""
        Generate one SQLite SELECT query.

        User Question: {user_question}

        Table: sales

        Columns:
        Order_ID, Date, Product, Category,
        Region, Quantity, Price, Profit

        Revenue = Quantity * Price

        Rules:
        - Generate only SELECT queries.
        - Never modify database records.
        - Use correct column names.
        - Return the response in JSON format.
        - The JSON must contain one field named "query".
        - The "query" field must contain the SQL SELECT query.
                
        """
    )

    sql_query = response.query

    print("\nGenerated SQL:")
    print(sql_query)

    # Execute SQL
    columns, results = execute_sql(sql_query)

    # Return results
    return columns, results


# 6. Interactive SQL Agent

if __name__ == "__main__":

    print("\nAI SQL Agent is Ready!")
    print("Ask your business questions.")
    print("Type 'exit' to stop.\n")

    while True:

        user_question = input(
            "Ask your business question (or type 'exit'): "
        )

        if user_question.lower().strip() == "exit":
            print("\nSQL Agent Stopped!")
            break

        try:

            columns, results = run_sql_agent(user_question)

            print("\nSQL Query Results:")
            display_results(columns, results)

        except Exception as e:

            print("\nError:", str(e))

        print("\n" + "-" * 50)

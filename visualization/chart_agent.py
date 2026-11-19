
import os
import sqlite3
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq


class ChartRequest(BaseModel):

    sql_query: str = Field(
        description="A valid SQLite SELECT query"
    )

    chart_type: Literal["bar", "line", "pie"]

    x_column: str
    y_column: str
    chart_title: str


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

chart_llm = llm.with_structured_output(
    ChartRequest,
    method="json_mode"
)


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


def generate_chart(
    chart_request,
    columns,
    results,
    chart_path
):

    if not results:
        raise ValueError("No data available for visualization.")

    if chart_request.x_column not in columns:
        raise ValueError("X column not found in SQL results!")

    if chart_request.y_column not in columns:
        raise ValueError("Y column not found in SQL results!")

    x_index = columns.index(chart_request.x_column)
    y_index = columns.index(chart_request.y_column)

    x_values = [row[x_index] for row in results]
    y_values = [float(row[y_index]) for row in results]

    plt.figure(figsize=(10, 6))

    if chart_request.chart_type == "bar":

        plt.bar(x_values, y_values)
        plt.xlabel(chart_request.x_column)
        plt.ylabel(chart_request.y_column)

    elif chart_request.chart_type == "line":

        plt.plot(x_values, y_values, marker="o")
        plt.xlabel(chart_request.x_column)
        plt.ylabel(chart_request.y_column)

    elif chart_request.chart_type == "pie":

        plt.pie(
            y_values,
            labels=x_values,
            autopct="%1.1f%%"
        )

    plt.title(chart_request.chart_title)

    if chart_request.chart_type != "pie":
        plt.xticks(rotation=30)

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(chart_path),
        exist_ok=True
    )

    plt.savefig(chart_path, dpi=150)

    plt.close()


def run_visualization_agent(
    user_question,
    db_path,
    columns
):

    column_details = ", ".join(columns)

    response = chart_llm.invoke(
        f"""
        You are a Data Visualization Agent.

        User Question:
        {user_question}

        Database Table:
        dataset

        Available Columns:
        {column_details}

        Instructions:

        1. Generate a valid SQLite SELECT query.
        2. Use only available columns.
        3. Use table name dataset.
        4. Choose bar, line, or pie chart.
        5. x_column must exist in query results.
        6. y_column must be numeric.
        7. Never generate modification queries.
        8. Return JSON with:
           sql_query, chart_type, x_column,
           y_column, chart_title.
        """
    )

    columns_result, results = execute_sql(
        response.sql_query,
        db_path
    )

    # Create a separate chart folder for each session
    session_folder = os.path.dirname(
        os.path.abspath(db_path)
    )

    chart_path = os.path.join(
        session_folder,
        "charts",
        "business_chart.png"
    )

    generate_chart(
        response,
        columns_result,
        results,
        chart_path
    )

    return (
        f"Chart generated successfully. "
        f"Chart Type: {response.chart_type}. "
        f"Chart Path: {chart_path}"
    )

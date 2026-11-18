
import os
import sqlite3

import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from langchain_groq import ChatGroq

load_dotenv()


# 1. Define chart response structure
class ChartRequest(BaseModel):

    sql_query: str = Field(
        description="A valid SQLite SELECT query"
    )

    chart_type: Literal["bar", "line", "pie"]

    x_column: str
    y_column: str
    chart_title: str


# 2. Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

chart_llm = llm.with_structured_output(
    ChartRequest,
    method="json_mode"
)


# 3. Execute SQL query safely
def execute_sql(query):

    query = query.strip().rstrip(";").strip()

    if not query.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed!")

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

        columns = [
            description[0]
            for description in cursor.description
        ]

        results = cursor.fetchall()

        return columns, results

    finally:
        connection.close()


# 4. Generate chart
def generate_chart(chart_request, columns, results):

    if not results:
        print("\nNo data available for visualization.")
        return

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

        plt.plot(
            x_values,
            y_values,
            marker="o"
        )

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

    os.makedirs("outputs/charts", exist_ok=True)

    chart_path = "outputs/charts/business_chart.png"

    plt.savefig(chart_path, dpi=150)

    print("\nChart saved successfully:", chart_path)

    plt.show()

    plt.close()


# 5. Reusable Visualization Agent Function

def run_visualization_agent(user_question):

    # Generate SQL and chart details
    response = chart_llm.invoke(
        f"""
        You are a Business Data Visualization Agent.

        User Question: {user_question}

        Database Table: sales

        Available Columns:
        Order_ID, Date, Product, Category,
        Region, Quantity, Price, Profit

        Revenue = Quantity * Price

        Instructions:

        1. Generate a valid SQLite SELECT query.
        2. Select only the columns needed for the chart.
        3. Choose the most suitable chart type:
           - bar: Compare categories or products.
           - line: Show monthly or time-based trends.
           - pie: Show percentage contribution.
        4. x_column must match a SQL result column.
        5. y_column must match a numeric SQL result column.
        6. Provide a meaningful chart title.
        7. Never generate INSERT, UPDATE, DELETE or DROP queries.

        Return the response in JSON format.
        The JSON must contain these fields:
        sql_query, chart_type, x_column, y_column, chart_title.
        """
    )

    print("\nGenerated SQL:")
    print(response.sql_query)

    print("\nSelected Chart Type:")
    print(response.chart_type)

    # Fetch database results
    columns, results = execute_sql(
        response.sql_query
    )

    print("\nSQL Query Results:")

    for row in results:
        print(row)

    # Generate visualization
    generate_chart(
        response,
        columns,
        results
    )

    return (
        f"Chart generated successfully. "
        f"Chart Type: {response.chart_type}. "
        f"Chart Path: outputs/charts/business_chart.png"
    )


# 6. Interactive Data Visualization Agent

if __name__ == "__main__":

    print("\nAI Data Visualization Agent is Ready!")
    print("Ask questions to generate business charts.")
    print("Type 'exit' to stop.\n")

    while True:

        user_question = input(
            "Ask your business question (or type 'exit'): "
        )

        if user_question.lower().strip() == "exit":

            print("\nVisualization Agent Stopped!")
            break

        try:

            result = run_visualization_agent(
                user_question
            )

            print("\n", result)

        except Exception as e:

            print("\nError:", str(e))

        print("\n" + "-" * 50)

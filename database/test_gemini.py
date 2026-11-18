
import os
import sqlite3

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# Define SQL response structure
class SQLQuery(BaseModel):
    query: str = Field(
        description="A valid SQLite SELECT query"
    )


# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

structured_llm = llm.with_structured_output(SQLQuery)


# Generate SQL query
user_question = "What is the total revenue by each region?"

response = structured_llm.invoke(
    f"""
    Generate a SQLite SELECT query for this question:
    {user_question}

    Table: sales
    Columns: Order_ID, Date, Product, Category,
             Region, Quantity, Price, Profit

    Revenue = Quantity * Price

    Return only a SELECT query.
    """
)

sql_query = response.query

print("\nGenerated SQL:")
print(sql_query)


# Execute SQL using read-only database connection
connection = sqlite3.connect(
    "file:data/sales.db?mode=ro",
    uri=True
)

connection.execute("PRAGMA query_only = ON")

try:
    cursor = connection.cursor()
    cursor.execute(sql_query)

    results = cursor.fetchall()

    print("\nSQL Query Results:")
    for row in results:
        print(row)

finally:
    connection.close()

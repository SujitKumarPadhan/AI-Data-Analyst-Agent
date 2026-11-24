
import sqlite3
import json
import re

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


# 3. Parse LLM JSON response
def parse_sql_response(response):

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = str(content).strip()

    # Remove Markdown code fences
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    ).strip()

    try:
        data = json.loads(content)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                f"Invalid JSON response from Groq: {content}"
            )

        data = json.loads(match.group())

    return SQLQuery.model_validate(data)


# 4. Execute SQL safely on uploaded dataset
def execute_sql(query, db_path):

    query = query.strip().rstrip(";").strip()

    normalized_query = query.upper()

    # Allow SELECT and WITH queries only
    if not (
        normalized_query.startswith("SELECT")
        or normalized_query.startswith("WITH")
    ):
        raise ValueError(
            "Only SELECT and read-only WITH queries are allowed!"
        )

    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True
    )

    try:

        connection.execute("PRAGMA query_only = ON")

        write_operations = {
            getattr(sqlite3, name)
            for name in [
                "SQLITE_INSERT",
                "SQLITE_UPDATE",
                "SQLITE_DELETE",
                "SQLITE_CREATE_INDEX",
                "SQLITE_CREATE_TABLE",
                "SQLITE_CREATE_TRIGGER",
                "SQLITE_CREATE_VIEW",
                "SQLITE_DROP_INDEX",
                "SQLITE_DROP_TABLE",
                "SQLITE_DROP_TRIGGER",
                "SQLITE_DROP_VIEW",
                "SQLITE_ALTER_TABLE",
                "SQLITE_ATTACH",
                "SQLITE_DETACH",
                "SQLITE_PRAGMA",
                "SQLITE_TRANSACTION",
                "SQLITE_SAVEPOINT",
                "SQLITE_REINDEX",
                "SQLITE_ANALYZE"
            ]
            if hasattr(sqlite3, name)
        }

        def sql_authorizer(
            action,
            arg1,
            arg2,
            db_name,
            source
        ):

            if action in write_operations:
                return sqlite3.SQLITE_DENY

            if action == sqlite3.SQLITE_READ:
                if arg1 != "dataset":
                    return sqlite3.SQLITE_DENY

            if action == sqlite3.SQLITE_FUNCTION:
                if str(arg2).lower() == "load_extension":
                    return sqlite3.SQLITE_DENY

            return sqlite3.SQLITE_OK

        connection.set_authorizer(sql_authorizer)

        cursor = connection.cursor()
        cursor.execute(query)

        if cursor.description is None:
            raise ValueError(
                "SQL query did not return any columns."
            )

        columns = [
            description[0]
            for description in cursor.description
        ]

        results = cursor.fetchall()

        return columns, results

    finally:
        connection.close()


# 5. Generate SQL query
def generate_sql(
    user_question,
    columns,
    previous_query=None,
    error_message=None
):

    column_details = ", ".join(
        str(column) for column in columns
    )

    correction_context = ""

    if previous_query and error_message:

        correction_context = f"""
        PREVIOUS QUERY FAILED.

        Previous SQL:
        {previous_query}

        SQLite Error:
        {error_message}

        Correct the SQL query based on the error.
        Use only available columns.
        Do not repeat the same mistake.
        """

    prompt = f"""
    You are an expert SQLite Data Analyst.

    User Question:
    {user_question}

    Database Table:
    dataset

    Available Columns:
    {column_details}

    Instructions:

    1. Generate one valid SQLite SELECT or WITH query.
    2. Use only available columns.
    3. Use table name dataset.
    4. Never generate INSERT, UPDATE, DELETE or DROP.
    5. For department-wise comparisons, use GROUP BY.
    6. When comparing departments, return ALL departments.
    7. Do not remove departments using WHERE or HAVING
       when all departments are requested.
    8. Do not use LIMIT 1 for all-department comparisons.
    9. Calculate department-wise values using CTEs
       or subqueries when required.
    10. Handle ties correctly.
    11. Include all requested metrics.
    12. Generate SQL according to the actual question.
    13. UNION queries must have matching columns.
    14. Prefer GROUP BY instead of unnecessary UNION ALL.
    15. Verify every column name.
    16. Calculate overall averages separately if required.
    17. Never use unavailable columns.
    18. Return only valid JSON in this format:

    {{
        "query": "SELECT ... FROM dataset"
    }}

    Do not include Markdown or explanations.

    {correction_context}
    """

    response = llm.invoke(prompt)

    parsed_response = parse_sql_response(response)

    return parsed_response.query.strip()


# 6. Dynamic SQL Agent with automatic correction
def run_sql_agent(
    user_question,
    db_path,
    columns
):

    max_attempts = 2

    previous_query = None
    error_message = None

    for attempt in range(1, max_attempts + 1):

        sql_query = None

        try:

            print(
                f"\n[SQL AGENT] Attempt {attempt}/{max_attempts}"
            )

            sql_query = generate_sql(
                user_question,
                columns,
                previous_query,
                error_message
            )

            print("\nGenerated SQL:")
            print(sql_query)

            result = execute_sql(
                sql_query,
                db_path
            )

            print(
                "\n[SQL AGENT] Query Executed Successfully!"
            )

            return result

        except Exception as error:

            error_message = str(error)

            print(
                f"\n[SQL AGENT] Attempt {attempt} Failed:"
            )

            print(error_message)

            if attempt < max_attempts:

                print(
                    "\n[SQL AGENT] Generating Corrected SQL..."
                )

                previous_query = sql_query

            else:

                raise RuntimeError(
                    f"SQL Agent failed after {max_attempts} "
                    f"attempts. Last error: {error_message}"
                )

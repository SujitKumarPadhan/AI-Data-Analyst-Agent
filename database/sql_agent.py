
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

        # Enable read-only query mode
        connection.execute("PRAGMA query_only = ON")

        # Additional SQL security using SQLite authorizer
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

        def sql_authorizer(action, arg1, arg2, db_name, source):

            # Block all database write operations
            if action in write_operations:
                return sqlite3.SQLITE_DENY

            # Allow reading only from uploaded dataset table
            if action == sqlite3.SQLITE_READ:
                if arg1 != "dataset":
                    return sqlite3.SQLITE_DENY

            # Block potentially unsafe extension loading
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


# 4. Generate SQL query
def generate_sql(
    user_question,
    columns,
    previous_query=None,
    error_message=None
):

    column_details = ", ".join(columns)

    correction_context = ""

    if previous_query and error_message:

        correction_context = f"""
        PREVIOUS QUERY FAILED.

        Previous SQL:
        {previous_query}

        SQLite Error:
        {error_message}

        Correct the SQL query based on the error.
        Use only the available columns.
        Do not repeat the same mistake.
        """

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

        1. Generate one valid SQLite SELECT or WITH query.

        2. Use only the available columns.

        3. Use table name dataset.

        4. Never generate INSERT, UPDATE, DELETE or DROP.

        5. For department-wise comparisons, use GROUP BY Department.

        6. When comparing departments, return ALL departments.

        7. Do not remove departments using WHERE or HAVING
           when the user asks for an all-department comparison.

        8. Do not use LIMIT 1 for an all-department comparison.

        9. For highest or lowest comparisons, calculate
           department-wise values first using a CTE or subquery.

        10. Handle ties correctly.

        11. Include all requested metrics.

        12. Generate SQL according to the actual user question.

        13. If using UNION or UNION ALL, every SELECT must
            return the same number of columns in the same order.

        14. Corresponding UNION columns must have compatible types.

        15. Prefer GROUP BY instead of unnecessary UNION ALL.

        16. Before generating SQL, verify every column name
            against the available columns.

        17. For overall averages, calculate them in a separate
            CTE or subquery if the main query uses GROUP BY.

        18. Do not use columns that are unavailable in the dataset.

        19. Return only one SQL query in the JSON field "query".

        {correction_context}
        """
    )

    return response.query.strip()


# 5. Dynamic SQL Agent with automatic correction
def run_sql_agent(user_question, db_path, columns):

    max_attempts = 2

    previous_query = None
    error_message = None

    for attempt in range(1, max_attempts + 1):

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

                previous_query = sql_query if "sql_query" in locals() else None

            else:

                raise RuntimeError(
                    f"SQL Agent failed after {max_attempts} "
                    f"attempts. Last error: {error_message}"
                )

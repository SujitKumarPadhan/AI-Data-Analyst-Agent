
import os
import sys
from typing import TypedDict, Optional, Any, Literal

from dotenv import load_dotenv

# 1. Project path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# 2. Load environment variables
load_dotenv(
    dotenv_path=os.path.join(PROJECT_ROOT, ".env")
)

# 3. Check Groq API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

# 4. Import required libraries
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# 5. Import dynamic agents
from tools.dynamic_eda_tool import create_dynamic_eda_tools
from database.sql_agent import run_sql_agent
from visualization.chart_agent import run_visualization_agent
from reports.business_report_agent import run_report_agent


# 6. Define workflow state
class AgentState(TypedDict):

    user_question: str
    dataframe: Any
    db_path: str
    columns: list

    selected_agent: Optional[str]
    agent_result: Optional[str]
    final_response: Optional[str]


# 7. Router response structure
class RouterResponse(BaseModel):

    selected_agent: Literal[
        "eda",
        "sql",
        "visualization",
        "report"
    ]


# 8. EDA tool selection structure
class ToolSelection(BaseModel):

    tools: list[str]


# 9. Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

router_llm = llm.with_structured_output(
    RouterResponse
)

planner_llm = llm.with_structured_output(
    ToolSelection,
    method="json_mode"
)


# 10. Agent Router
def agent_router(state):

    question = state["user_question"]

    response = router_llm.invoke(
        f"""
        You are an AI Data Analyst Agent Router.

        Select exactly one agent for the user's question.

        Available Agents:

        1. eda:
           Dataset overview, missing values,
           statistics, correlation and categorical analysis.

        2. sql:
           Database queries, filtering, grouping,
           sorting and calculations.

        3. visualization:
           Generate bar, line or pie charts.

        4. report:
           Generate complete data analysis reports.

        User Question:
        {question}

        Return the correct agent.
        """
    )

    return {
        "selected_agent": response.selected_agent
    }


# 11. Dynamic EDA Agent
def eda_node(state):

    question = state["user_question"]
    df = state["dataframe"]

    dynamic_tools = create_dynamic_eda_tools(df)

    tool_map = {
        selected_tool.name: selected_tool
        for selected_tool in dynamic_tools
    }

    available_tools = "\n".join(
        f"- {selected_tool.name}: {selected_tool.description}"
        for selected_tool in dynamic_tools
    )

    # Select EDA tools using JSON output
    response = planner_llm.invoke(
        f"""
        You are an expert Data Analysis Planner.

        User Question:
        {question}

        Available EDA Tools:
        {available_tools}

        Select only the tools required to answer
        the user's question.

        Return a valid JSON object with a field named "tools".

        The tools field must contain a list of exact tool names.
        Do not select unavailable tools.

        Example JSON:
        {{"tools": ["get_dataset_overview", "get_missing_values"]}}
        """
    )

    selected_tools = response.tools

    # Execute selected tools
    results = []

    for tool_name in selected_tools:

        if tool_name not in tool_map:
            continue

        selected_tool = tool_map[tool_name]

        if tool_name == "get_top_values":

            column_name = next(
                (
                    column
                    for column in df.columns
                    if column.lower() in question.lower()
                ),
                None
            )

            if column_name is None:
                continue

            result = selected_tool.invoke({
                "column_name": column_name
            })

        else:

            result = selected_tool.invoke({})

        results.append(
            f"{tool_name}: {result}"
        )

    return {
        "agent_result": "\n".join(results)
        if results
        else "No suitable EDA tools were selected."
    }


# 12. Dynamic SQL Agent
def sql_node(state):

    columns, results = run_sql_agent(
        state["user_question"],
        state["db_path"],
        state["columns"]
    )

    return {
        "agent_result": (
            f"Columns: {columns}\n"
            f"Results: {results}"
        )
    }


# 13. Dynamic Visualization Agent
def visualization_node(state):

    result = run_visualization_agent(
        state["user_question"],
        state["db_path"],
        state["columns"]
    )

    return {
        "agent_result": result
    }


# 14. Dynamic Report Agent
def report_node(state):

    result = run_report_agent(
        state["user_question"],
        state["dataframe"]
    )

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    with open(
        "outputs/reports/business_report.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(result)

    return {
        "agent_result": result
    }



# 15. Final Response
def final_response_node(state):

    response = llm.invoke(
        f"""
        You are a professional AI Data Analyst.

        User Question:
        {state["user_question"]}

        Selected Agent:
        {state["selected_agent"]}

        Actual Analysis Results:
        {state["agent_result"]}

        STRICT DATA ACCURACY RULES:

        1. Use only information supported by actual analysis results.

        2. Never invent, assume, or modify numerical values.

        3. Never assume currency.
           If the dataset does not specify currency,
           use the column name Monthly_Salary without
           mentioning USD, INR, dollars, or rupees.

        4. Never assume the meaning of a categorical value.
           For example, Attrition = Yes must be described
           as "Attrition = Yes records".
           Never automatically interpret it as employees
           who left the company.

        5. Never describe attrition as low, high, moderate,
           or healthy without a verified external benchmark.

        6. Do not invent industry standards or benchmarks.

        7. Do not claim causation or correlation
           unless the analysis explicitly supports it.

        8. Do not claim that job satisfaction affects
           attrition or that performance causes attrition.

        9. Clearly distinguish actual observations
           from recommendations.

        10. If information is unavailable,
            explicitly mention that it is unavailable.

        11. Do not create conclusions from assumptions.

        12. Keep the response clear and professional.

        RESPONSE FORMAT:

        - Use simple professional English.
        - Use Markdown headings and bullet points.
        - Present numerical findings accurately.
        - Explain important observations.
        - Give practical recommendations only
          when supported by the available data.

        Your priority is factual accuracy, not making
        the report sound impressive.
        """
    )

    return {
        "final_response": response.content
    }

# 16. Build LangGraph Workflow
workflow = StateGraph(AgentState)

workflow.add_node("router", agent_router)
workflow.add_node("eda", eda_node)
workflow.add_node("sql", sql_node)
workflow.add_node("visualization", visualization_node)
workflow.add_node("report", report_node)
workflow.add_node("final_response", final_response_node)


# 17. Workflow connections
workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["selected_agent"],
    {
        "eda": "eda",
        "sql": "sql",
        "visualization": "visualization",
        "report": "report"
    }
)

workflow.add_edge("eda", "final_response")
workflow.add_edge("sql", "final_response")
workflow.add_edge("visualization", "final_response")
workflow.add_edge("report", "final_response")

workflow.add_edge("final_response", END)


# 18. Compile workflow
app = workflow.compile()


if __name__ == "__main__":

    print("Dynamic AI Data Analyst Workflow is ready!")

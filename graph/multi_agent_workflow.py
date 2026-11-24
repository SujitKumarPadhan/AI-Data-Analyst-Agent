
import os
import sys
import operator

from typing import TypedDict, Optional, Any, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# 1. Project Path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# 2. Load Environment Variables
load_dotenv(
    dotenv_path=os.path.join(PROJECT_ROOT, ".env")
)


# 3. Check Groq API Key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )


# 4. Import Agents
from tools.dynamic_eda_tool import create_dynamic_eda_tools
from database.sql_agent import run_sql_agent
from visualization.chart_agent import run_visualization_agent
from reports.business_report_agent import run_report_agent


# 5. Multi-Agent Workflow State
class AgentState(TypedDict):

    user_question: str
    dataframe: Any
    db_path: str
    columns: list

    selected_agents: list[str]

    agent_results: Annotated[
        dict[str, str],
        operator.or_
    ]

    final_response: Optional[str]


# 6. Individual Agent Task State
class AgentTask(TypedDict):

    user_question: str
    dataframe: Any
    db_path: str
    columns: list
    agent_name: str


# 7. Planner Response Structure
class AgentSelection(BaseModel):

    selected_agents: list[
        Literal[
            "eda",
            "sql",
            "visualization",
            "report"
        ]
    ]


# 8. EDA Tool Selection Structure
class ToolSelection(BaseModel):

    tools: list[str]


# 9. Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

agent_planner_llm = llm.with_structured_output(
    AgentSelection
)

planner_llm = llm.with_structured_output(
    ToolSelection,
    method="json_mode"
)


# 10. Multi-Agent Planner Node
def agent_planner_node(state: AgentState):

    question = state["user_question"]

    response = agent_planner_llm.invoke(
        f"""
        You are an expert AI Data Analyst Agent Planner.

        Select the required agents for the user's request.

        AVAILABLE AGENTS:

        1. eda:
           Dataset overview, missing values, numerical
           statistics, categorical analysis and correlation.

        2. sql:
           SQL queries, filtering, grouping, sorting
           and calculations.

        3. visualization:
           Bar charts, line charts, pie charts,
           scatter plots and trend lines.

        4. report:
           Business reports, key findings and recommendations.

        SELECTION RULES:

        - Select only the agents required.
        - For complete dataset analysis, select all four agents.
        - If the user requests a chart, include visualization.
        - If the user requests a business report, include report.
        - For department-wise calculations, include SQL when appropriate.
        - For dataset statistics or correlation, include EDA.
        - Avoid unnecessary agents.
        - Return each agent only once.

        USER QUESTION:
        {question}
        """
    )

    valid_agents = [
        "eda",
        "sql",
        "visualization",
        "report"
    ]

    selected_agents = [
        agent
        for agent in valid_agents
        if agent in response.selected_agents
    ]

    if not selected_agents:
        selected_agents = ["eda"]

    print("\nSelected Agents:", selected_agents)

    return {
        "selected_agents": selected_agents,
        "agent_results": {}
    }


# 11. Dynamic EDA Agent
def eda_node(state: AgentTask):

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

        Example:
        {{"tools": ["get_dataset_overview", "get_missing_values"]}}
        """
    )

    selected_tools = response.tools

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
        "agent_results": {
            "eda": "\n".join(results)
            if results
            else "No suitable EDA tools were selected."
        }
    }


# 12. Dynamic SQL Agent
def sql_node(state: AgentTask):

    columns, results = run_sql_agent(
        state["user_question"],
        state["db_path"],
        state["columns"]
    )

    return {
        "agent_results": {
            "sql": (
                f"Columns: {columns}\n"
                f"Results: {results}"
            )
        }
    }


# 13. Dynamic Visualization Agent
def visualization_node(state: AgentTask):

    result = run_visualization_agent(
        state["user_question"],
        state["db_path"],
        state["columns"]
    )

    return {
        "agent_results": {
            "visualization": str(result)
        }
    }


# 14. Dynamic Report Agent
def report_node(state: AgentState):

    print("\nGenerating Business Report...")

    # Collect completed agent results
    agent_results = state.get("agent_results", {})

    # Get SQL Agent results
    sql_results = agent_results.get(
        "sql",
        "SQL Agent results are not available."
    )

    # Generate report using dataset and SQL results
    result = run_report_agent(
        state["user_question"],
        state["dataframe"],
        sql_results
    )

    # Create report output directory
    report_directory = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "reports"
    )

    os.makedirs(
        report_directory,
        exist_ok=True
    )

    # Save business report
    report_path = os.path.join(
        report_directory,
        "business_report.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(str(result))

    print(
        "\nBusiness Report Saved:",
        report_path
    )

    return {
        "agent_results": {
            "report": str(result)
        }
    }


# 15. Parallel Agent Executor with Retry
def agent_executor_node(task: AgentTask):

    agent_name = task["agent_name"]

    agent_functions = {
        "eda": eda_node,
        "sql": sql_node,
        "visualization": visualization_node,
        "report": report_node
    }

    max_attempts = 2

    print(
        f"\n[{agent_name.upper()}] Agent Started..."
    )

    for attempt in range(1, max_attempts + 1):

        try:

            print(
                f"[{agent_name.upper()}] Attempt {attempt}/{max_attempts}"
            )

            agent_function = agent_functions[agent_name]

            result = agent_function(task)

            print(
                f"[{agent_name.upper()}] Agent Completed Successfully!"
            )

            return result

        except Exception as error:

            print(
                f"\n[{agent_name.upper()}] Attempt {attempt} Failed: {error}"
            )

            if attempt < max_attempts:

                print(
                    f"[{agent_name.upper()}] Retrying Agent..."
                )

            else:

                error_message = (
                    f"{agent_name.upper()} Agent Failed "
                    f"after {max_attempts} attempts: {str(error)}"
                )

                print(
                    f"\n[ERROR] {error_message}"
                )

                return {
                    "agent_results": {
                        agent_name: error_message
                    }
                }


# 16. Parallel Agent Dispatcher
def dispatch_agents(state: AgentState):

    selected_agents = state["selected_agents"]

    print(
        "\n========== PARALLEL EXECUTION STARTED =========="
    )

    # Exclude Report Agent from parallel execution
    parallel_agents = [
        agent
        for agent in selected_agents
        if agent != "report"
    ]

    # If only Report Agent is selected,
    # execute it through the executor
    if not parallel_agents and "report" in selected_agents:

        parallel_agents = ["report"]

    return [
        Send(
            "agent_executor",
            {
                "user_question": state["user_question"],
                "dataframe": state["dataframe"],
                "db_path": state["db_path"],
                "columns": state["columns"],
                "agent_name": agent_name
            }
        )
        for agent_name in parallel_agents
    ]


# 16.1 Report Agent Routing
def dispatch_report(state: AgentState):

    selected_agents = state["selected_agents"]

    completed_results = state.get(
        "agent_results",
        {}
    )

    # Execute Report Agent only after
    # parallel agents have completed
    if (
        "report" in selected_agents
        and "report" not in completed_results
    ):

        print(
            "\n========== REPORT AGENT EXECUTION STARTED =========="
        )

        return "report_executor"

    return "final_response"


# 17. Final Response Node
def final_response_node(state: AgentState):

    response = llm.invoke(
        f"""
        You are a professional AI Data Analyst.

        USER QUESTION:
        {state["user_question"]}

        SELECTED AGENTS:
        {state["selected_agents"]}

        ACTUAL RESULTS FROM EACH AGENT:
        {state["agent_results"]}

        Combine the results from all executed agents
        into one clear response.

        STRICT DATA ACCURACY RULES:

        1. Use only information supported by actual results.
        2. Never invent or modify numerical values.
        3. Never assume currency.
        4. Use Monthly_Salary without assuming a currency.
        5. Attrition = Yes must be described as
           "Attrition = Yes records".
        6. Never automatically interpret Attrition = Yes
           as employees who left the company.
        7. Do not invent industry benchmarks.
        8. Do not claim causation or correlation unless
           analysis explicitly supports it.
        9. Distinguish observations from recommendations.
        10. If information is unavailable, mention it.
        11. Do not claim an agent completed a task
            unless its result is present.
        12. If visualization results contain a chart path,
            mention that the chart was generated.
        13. Never invent chart results or file paths.
        14. If an agent result contains "Agent Failed",
            clearly mention that agent's failure.
        15. Do not invent or estimate missing agent results.
        16. Use successful agents' results even if
            another agent failed.
        17. Use SQL Agent results when available.
        18. Do not contradict the actual calculations
            returned by SQL Agent or EDA Agent.

        RESPONSE FORMAT:

        - Use simple professional English.
        - Use Markdown headings and bullet points.
        - Present numerical findings accurately.
        - Combine relevant results from different agents.
        - Explain important observations.
        - Give practical recommendations only when supported.
        - Mention agent failures when present.
        - Avoid unnecessary repetition.

        Your priority is factual accuracy.
        """
    )

    print(
        "\n========== FINAL RESPONSE GENERATED =========="
    )

    return {
        "final_response": response.content
    }


# 18. Build Multi-Agent LangGraph
workflow = StateGraph(AgentState)

workflow.add_node(
    "planner",
    agent_planner_node
)

workflow.add_node(
    "agent_executor",
    agent_executor_node
)

workflow.add_node(
    "report_executor",
    report_node
)

workflow.add_node(
    "final_response",
    final_response_node
)


# 19. Workflow Connections
workflow.add_edge(
    START,
    "planner"
)

# Planner -> Parallel Agents
workflow.add_conditional_edges(
    "planner",
    dispatch_agents,
    ["agent_executor"]
)

# Parallel Agents -> Report Agent or Final Response
workflow.add_conditional_edges(
    "agent_executor",
    dispatch_report,
    [
        "report_executor",
        "final_response"
    ]
)

# Report Agent -> Final Response
workflow.add_edge(
    "report_executor",
    "final_response"
)

# Final Response -> END
workflow.add_edge(
    "final_response",
    END
)


# 20. Compile Workflow
app = workflow.compile()


if __name__ == "__main__":

    print(
        "Parallel Multi-Agent AI Data Analyst Workflow is ready!"
    )

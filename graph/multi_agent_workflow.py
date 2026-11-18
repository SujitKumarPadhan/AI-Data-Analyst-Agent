
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from typing import TypedDict, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END

from tools.eda_tool import eda_tools
from database.sql_agent import run_sql_agent
from visualization.chart_agent import run_visualization_agent
from reports.business_report_agent import run_report_agent


load_dotenv()


# 1. Define shared workflow state
class AgentState(TypedDict):

    user_question: str

    selected_agent: Optional[str]

    agent_result: Optional[str]

    final_response: Optional[str]


# 2. Define router response
class RouterResponse(BaseModel):

    selected_agent: Literal[
        "eda",
        "sql",
        "visualization",
        "report"
    ] = Field(
        description="Select the most suitable agent"
    )


# 3. Define EDA planner response
class ToolSelection(BaseModel):

    tools: list[str] = Field(
        description="List of all required EDA tool names"
    )


# 4. Initialize Groq LLM
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


# 5. Prepare EDA tools
tool_map = {
    tool.name: tool
    for tool in eda_tools
}

available_tools = "\n".join(
    f"- {tool.name}: {tool.description}"
    for tool in eda_tools
)


# 6. Create AI Agent Router
def agent_router(state: AgentState):

    question = state["user_question"]

    response = router_llm.invoke(
        f"""
        You are an AI Data Analyst Agent Router.

        Select the most suitable agent.

        Available Agents:

        eda: Data analysis, statistics and insights.

        sql: SQL queries and database calculations.

        visualization: Bar, line and pie charts.

        report: Business reports and summaries.

        User Question:
        {question}

        Select exactly one agent.
        """
    )

    selected_agent = response.selected_agent

    print("\nSelected Agent:", selected_agent.upper())

    return {
        "selected_agent": selected_agent
    }


# 7. EDA Agent Node
def eda_node(state: AgentState):

    print("\nEDA Agent Executing...")

    question = state["user_question"]

    # Select all required tools
    response = planner_llm.invoke(
        f"""
        You are an expert Data Analysis Planner.

        User Question:
        {question}

        Available EDA Tools:
        {available_tools}

        Instructions:
        1. Select all tools required to answer the question.
        2. Select exact tool names from the available tools.
        3. Do not select unnecessary tools.
        4. If multiple metrics are requested, select all relevant tools.
        5. Return an empty list if no tool is required.

        Return the response in JSON format.
        The JSON must contain one field named "tools".
        The "tools" field must contain a list of exact tool names.
        """
    )

    selected_tools = response.tools

    print("\nAI Planner Selected Tools:")

    for tool_name in selected_tools:
        print("-", tool_name)

    tool_results = []

    # Execute selected tools
    for tool_name in selected_tools:

        if tool_name not in tool_map:
            print("\nUnknown Tool:", tool_name)
            continue

        selected_tool = tool_map[tool_name]

        result = selected_tool.invoke({})

        tool_results.append(
            f"{tool_name}: {result}"
        )

        print("\nTool Selected:", tool_name)
        print("Tool Result:", result)

    if tool_results:

        agent_result = "\n".join(tool_results)

    else:

        agent_result = "No relevant EDA tools were selected."

    return {
        "agent_result": agent_result
    }


# 8. SQL Agent Node
def sql_node(state: AgentState):

    print("\nSQL Agent Executing...")

    columns, results = run_sql_agent(
        state["user_question"]
    )

    print("\nSQL Query Results:")

    for row in results:
        print(row)

    agent_result = (
        f"Columns: {columns}\n"
        f"Results: {results}"
    )

    return {
        "agent_result": agent_result
    }


# 9. Visualization Agent Node
def visualization_node(state: AgentState):

    print("\nVisualization Agent Executing...")

    result = run_visualization_agent(
        state["user_question"]
    )

    return {
        "agent_result": result
    }


# 10. Report Agent Node
def report_node(state: AgentState):

    print("\nReport Agent Executing...")

    result = run_report_agent(
        state["user_question"]
    )

    return {
        "agent_result": result
    }


# 11. Final Response Node
def final_response_node(state: AgentState):

    print("\nGenerating Final Response...")

    response = llm.invoke(
        f"""
        You are a professional AI Data Analyst Assistant.

        User Question:
        {state["user_question"]}

        Selected Agent:
        {state["selected_agent"]}

        Agent Result:
        {state["agent_result"]}

        Instructions:
        1. Give a clear and simple answer.
        2. Use actual information from the agent result.
        3. Do not invent numbers or facts.
        4. Use bullet points wherever useful.
        5. Keep the response professional.
        6. All monetary values must use INR (₹), never USD ($).
        7. Do not change the numerical values.
        """
    )

    final_response = response.text

    print("\nFinal Answer:")
    print(final_response)

    return {
        "final_response": final_response
    }


# 12. Build LangGraph Workflow
workflow = StateGraph(AgentState)


# Add nodes
workflow.add_node("router", agent_router)
workflow.add_node("eda", eda_node)
workflow.add_node("sql", sql_node)
workflow.add_node("visualization", visualization_node)
workflow.add_node("report", report_node)
workflow.add_node("final_response", final_response_node)


# Connect START to router
workflow.add_edge(START, "router")


# Conditional routing
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


# Connect all agents to Final Response Node
workflow.add_edge("eda", "final_response")
workflow.add_edge("sql", "final_response")
workflow.add_edge("visualization", "final_response")
workflow.add_edge("report", "final_response")


# Connect Final Response Node to END
workflow.add_edge("final_response", END)


# 13. Compile Workflow
app = workflow.compile()


# 14. Test Workflow
if __name__ == "__main__":

    initial_state = {
        "user_question": "What is the total revenue and total profit using SQL?",
        "selected_agent": None,
        "agent_result": None,
        "final_response": None
    }

    result = app.invoke(initial_state)

    print("\nFinal Workflow Result:")
    print(result)

    print("\nFinal AI Response:")
    print(result["final_response"])

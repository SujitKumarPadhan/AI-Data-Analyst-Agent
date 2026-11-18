
import os
import sqlite3
import subprocess
import sys

import streamlit as st

# Create database automatically if it does not exist
if not os.path.exists("data/sales.db"):
    subprocess.run(
        [sys.executable, "database/create_database.py"],
        check=True
    )

from graph.multi_agent_workflow import app as agent_workflow


st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst Agent")
st.write("Your intelligent business data analysis assistant.")

st.divider()

st.subheader("Ask Your Business Question")

user_question = st.text_input(
    "Enter your question:",
    placeholder="Example: Generate a complete business report."
)

if st.button("Analyze"):

    if user_question:

        initial_state = {
            "user_question": user_question,
            "selected_agent": None,
            "agent_result": None,
            "final_response": None
        }

        with st.spinner("AI Agent is analyzing your question..."):

            result = agent_workflow.invoke(initial_state)

        st.success("Analysis Completed!")

        st.subheader("AI Agent Response")
        st.markdown(result["final_response"])

        selected_agent = result["selected_agent"]

        st.subheader("Selected Agent")
        st.write(selected_agent.upper())

        if selected_agent == "visualization":

            chart_path = "outputs/charts/business_chart.png"

            if os.path.exists(chart_path):
                st.subheader("📈 Business Visualization")
                st.image(chart_path, use_container_width=True)
            else:
                st.warning("Chart file not found.")

        if selected_agent == "report":

            report_path = "outputs/reports/business_report.md"

            if os.path.exists(report_path):

                st.subheader("📑 Business Report")

                with open(report_path, "r", encoding="utf-8") as file:
                    report_content = file.read()

                st.markdown(report_content)

            else:
                st.warning("Business report file not found.")

    else:
        st.warning("Please enter a question.")

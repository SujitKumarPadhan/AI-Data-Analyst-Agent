
import os
import uuid
import streamlit as st
import pandas as pd

from tools.dataset_manager import (
    load_uploaded_dataset,
    save_dataframe
)

from graph.multi_agent_workflow import app as agent_workflow


# 1. Streamlit configuration
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst Agent")

st.write(
    "Upload your CSV or Excel dataset and analyze it "
    "using AI-powered multi-agents."
)


# 2. Create unique session ID
if "session_id" not in st.session_state:

    st.session_state.session_id = uuid.uuid4().hex


# 3. Dataset Upload
st.subheader("📁 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)


# 4. Load uploaded dataset or sample dataset
dataset_info = None

if uploaded_file is not None:

    try:

        dataset_info = load_uploaded_dataset(
            uploaded_file,
            st.session_state.session_id
        )

        st.success(
            f"Uploaded successfully: {dataset_info['file_name']}"
        )

    except Exception as e:

        st.error(f"Dataset loading error: {str(e)}")

else:

    st.info(
        "No file uploaded. Using the default sample sales dataset."
    )

    try:

        sample_df = pd.read_csv(
            "data/raw/sales_data.csv"
        )

        dataset_info = save_dataframe(
            sample_df,
            st.session_state.session_id,
            "sales_data.csv"
        )

    except Exception as e:

        st.error(f"Sample dataset loading error: {str(e)}")


# 5. Display Dataset Information
if dataset_info is not None:

    st.divider()

    st.subheader("📊 Dataset Preview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Rows",
        dataset_info["rows"]
    )

    col2.metric(
        "Total Columns",
        len(dataset_info["columns"])
    )

    col3.metric(
        "Total Missing Values",
        int(dataset_info["dataframe"].isnull().sum().sum())
    )

    st.dataframe(
        dataset_info["dataframe"].head(10),
        width="stretch"
    )

    with st.expander("View Dataset Columns"):

        st.write(dataset_info["columns"])


# 6. Ask Questions
st.divider()

st.subheader("🤖 Ask Your Data Analyst Agent")

user_question = st.text_input(
    "Enter your question:",
    placeholder="Example: Analyze the dataset and generate a business report."
)


# 7. Run Multi-Agent Workflow
if st.button("Analyze Dataset", type="primary"):

    if dataset_info is None:

        st.error("Please upload a valid dataset.")

    elif not user_question.strip():

        st.warning("Please enter a question.")

    else:

        initial_state = {
            "user_question": user_question,
            "dataframe": dataset_info["dataframe"],
            "db_path": dataset_info["db_path"],
            "columns": dataset_info["columns"],
            "selected_agents": [],
            "agent_results": {},
            "final_response": None
        }

        try:

            with st.spinner(
                "AI Multi-Agent System is analyzing your dataset..."
            ):

                result = agent_workflow.invoke(
                    initial_state
                )

             

            st.success("Multi-Agent Analysis Completed!")

            # Final AI response
            st.subheader("🤖 AI Agent Response")

            st.markdown(
                result["final_response"]
            )

            # Display selected agents
            selected_agents = result["selected_agents"]

            st.divider()

            st.subheader("🤖 Executed Agents")

            for agent_name in selected_agents:

                st.success(
                    f"✓ {agent_name.upper()} Agent Executed"
                )

            # Display individual agent results
            with st.expander("View Individual Agent Results"):

                for agent_name, agent_result in result["agent_results"].items():

                    st.markdown(
                        f"### {agent_name.upper()} Agent"
                    )

                    st.markdown(
                        str(agent_result)
                    )

                    st.divider()

            # Display session-specific chart
            if "visualization" in selected_agents:

                session_folder = os.path.dirname(
                    os.path.abspath(dataset_info["db_path"])
                )

                chart_path = os.path.join(
                    session_folder,
                    "charts",
                    "business_chart.png"
                )

                if os.path.exists(chart_path):

                    st.subheader("📈 Data Visualization")

                    st.image(
                        chart_path,
                        width="stretch"
                    )

                else:

                    st.warning(
                        "Visualization agent executed, "
                        "but chart file was not found."
                    )

            # Display business report
            if "report" in selected_agents:

                st.subheader("📑 Data Analysis Report")

                report_result = result["agent_results"].get(
                    "report"
                )

                if report_result:

                    st.markdown(
                        report_result
                    )

                else:

                    st.warning("Report result is unavailable.")

        except Exception as e:

            st.error(
                f"Analysis failed: {str(e)}"
            )

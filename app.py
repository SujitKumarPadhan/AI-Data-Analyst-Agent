
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
    "using AI-powered agents."
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
        use_container_width=True
    )

    with st.expander("View Dataset Columns"):

        st.write(dataset_info["columns"])


# 6. Ask Questions
st.divider()

st.subheader("🤖 Ask Your Data Analyst Agent")

user_question = st.text_input(
    "Enter your question:",
    placeholder="Example: Show dataset overview or generate a chart."
)


# 7. Run Agent Workflow
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
            "selected_agent": None,
            "agent_result": None,
            "final_response": None
        }

        try:

            with st.spinner(
                "AI Agent is analyzing your dataset..."
            ):

                result = agent_workflow.invoke(
                    initial_state
                )

            st.success("Analysis Completed!")

            st.subheader("🤖 AI Agent Response")

            st.markdown(
                result["final_response"]
            )

            selected_agent = result["selected_agent"]

            st.subheader("Selected Agent")

            st.write(selected_agent.upper())

            # Display session-specific chart
            if selected_agent == "visualization":

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
                        use_container_width=True
                    )

                else:

                    st.warning("Chart file not found.")

            # Display report
            if selected_agent == "report":

                st.subheader("📑 Data Analysis Report")

                st.markdown(
                    result["agent_result"]
                )

        except Exception as e:

            st.error(
                f"Analysis failed: {str(e)}"
            )

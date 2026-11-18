# 📊 AI Data Analyst Agent

An intelligent multi-agent AI Data Analyst built using **Python, LangGraph, Groq, SQL, and Streamlit**. It understands business questions in natural language and automatically selects the right AI agent to perform data analysis, SQL queries, visualizations, and business report generation.

## 🚀 Project Overview

The AI Data Analyst Agent helps users analyze business sales data without manually writing Python or SQL queries.

Users can ask questions in natural language, and the system intelligently routes requests to specialized agents to generate meaningful business insights.

## ✨ Key Features

- **EDA Agent:** Performs business data analysis using 32 custom analytical tools.
- **SQL Agent:** Converts natural language questions into SQL queries and executes them on SQLite.
- **Visualization Agent:** Generates Bar, Line, and Pie charts.
- **Business Report Agent:** Automatically generates business reports in Markdown format.
- **Multi-Agent Workflow:** Uses LangGraph to route questions to the appropriate agent.
- **Interactive Web Application:** Provides a user-friendly interface using Streamlit.
- **LLM Integration:** Uses Groq with the `openai/gpt-oss-120b` model.
- **Monitoring:** Uses LangSmith for LLM tracing and workflow monitoring.

## 🏗️ System Architecture

```text
                 User Question
                       |
                       v
                Streamlit UI
                       |
                       v
                 LangGraph Router
                       |
          +------------+------------+
          |            |            |
          v            v            v
       EDA Agent    SQL Agent   Visualization
          |            |         Agent
          |            |            |
          |            |            |
          +------------+------------+
                       |
                       v
                Report Agent
                       |
                       v
               Final AI Response
                       |
                       v
                 Streamlit UI
```

*The router selects one suitable agent for each user question. Each agent processes its task and returns results to the final response node.*

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| LangChain | LLM and tool integration |
| LangGraph | Multi-agent orchestration |
| Groq | LLM inference |
| SQLite | Database management |
| Pandas | Data analysis |
| Matplotlib | Data visualization |
| Streamlit | Web application |
| Pydantic | Structured output |
| LangSmith | Monitoring and tracing |
| Git & GitHub | Version control |

## 📂 Project Structure

```text
AI-Data-Analyst-Agent/
│
├── agents/
│
├── data/
│   └── raw/
│       └── sales_data.csv
│
├── database/
│   ├── create_database.py
│   ├── sql_agent.py
│   ├── test_database.py
│   └── test_gemini.py
│
├── graph/
│   └── multi_agent_workflow.py
│
├── outputs/
│   ├── charts/
│   └── reports/
│
├── reports/
│   └── business_report_agent.py
│
├── tools/
│   ├── data_loader.py
│   └── eda_tool.py
│
├── visualization/
│   └── chart_agent.py
│
├── app.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 📊 Dataset Information

The project uses a sample sales dataset containing 10 records and 8 original columns.

**Dataset columns:**

- Order_ID
- Date
- Product
- Category
- Region
- Quantity
- Price
- Profit

The EDA Agent calculates revenue using:

```python
Revenue = Quantity * Price
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SujitKumarPadhan/AI-Data-Analyst-Agent.git
```

### 2. Navigate to the Project

```bash
cd AI-Data-Analyst-Agent
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows PowerShell:

```powershell
venv\Scripts\Activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the project root directory.

Add your API credentials:

```env
GROQ_API_KEY=your_groq_api_key

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=AI-Data-Analyst-Agent
```

Get your Groq API key from [Groq Console](https://console.groq.com/).

Get your LangSmith API key from [LangSmith](https://smith.langchain.com/).

**Important:** Never upload your `.env` file or expose your API keys publicly.

### 7. Create the SQLite Database

```bash
python database/create_database.py
```

### 8. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## 💡 Example Business Questions

**EDA Agent**
```text
What is the total revenue and total profit?
```

**SQL Agent**
```text
What is the total revenue and total profit using SQL?
```

**Visualization Agent**
```text
Show revenue by region using a bar chart.
```

**Business Report Agent**
```text
Generate a complete business report.
```

## 📈 Sample Business Insights

Based on the included sales dataset:

| Business Metric | Result |
|---|---:|
| Total Revenue | ₹4,96,000 |
| Total Profit | ₹53,000 |
| Total Orders | 10 |
| Profit Margin | 10.69% |
| Average Order Value | ₹49,600 |
| Best-Selling Product | Mobile |
| Highest Revenue Product | Laptop |
| Best Revenue Region | East |

## 📊 Sample Output

### Business Visualization

The Visualization Agent generates charts and saves them in:

```text
outputs/charts/business_chart.png
```

### Business Report

The Business Report Agent generates reports and saves them in:

```text
outputs/reports/business_report.md
```

## 🔍 LangSmith Monitoring

LangSmith is integrated to monitor LLM and LangGraph execution.

It helps inspect:

- LLM execution traces
- Agent workflow execution
- Model responses
- Execution errors and latency

## 🎯 Project Objectives

- Build a practical multi-agent AI application.
- Automate business data analysis using natural language.
- Integrate LLMs with Python tools and SQL databases.
- Implement intelligent agent routing using LangGraph.
- Generate business insights, charts, and reports.

## 👨‍💻 Author

**Sujit Kumar Padhan**

B.Tech Computer Science and Engineering

**GitHub:** [SujitKumarPadhan](https://github.com/SujitKumarPadhan)

**Project Repository:** [AI Data Analyst Agent](https://github.com/SujitKumarPadhan/AI-Data-Analyst-Agent)

---

⭐ If you find this project interesting, consider giving it a star on GitHub.
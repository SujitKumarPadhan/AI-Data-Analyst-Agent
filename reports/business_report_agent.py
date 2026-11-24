
import json
import pandas as pd
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def run_report_agent(
    user_question,
    df,
    sql_results=None
):

    print("\nGenerating Data Analysis Report...")

    dataset = df.copy()

    # 1. Dataset Overview
    overview = {
        "total_rows": len(dataset),
        "total_columns": len(dataset.columns),
        "column_names": dataset.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in dataset.dtypes.items()
        },
        "duplicate_rows": int(dataset.duplicated().sum())
    }

    # 2. Missing Values
    missing_values = {
        column: int(value)
        for column, value in dataset.isnull().sum().items()
    }

    # 3. Numerical Analysis
    numeric_df = dataset.select_dtypes(include="number")

    numeric_summary = (
        numeric_df.describe()
        .round(2)
        .to_dict()
        if not numeric_df.empty else {}
    )

    # 4. Categorical Analysis
    categorical_df = dataset.select_dtypes(
        include=["object", "category", "bool"]
    )

    categorical_summary = {}

    for column in categorical_df.columns:

        categorical_summary[column] = {
            "unique_values": int(
                categorical_df[column].nunique()
            ),
            "top_values": {
                str(key): int(value)
                for key, value in (
                    categorical_df[column]
                    .value_counts()
                    .head(10)
                    .items()
                )
            }
        }

    # 5. Department-wise Salary Analysis
    department_analysis = {}

    if (
        "Department" in dataset.columns
        and "Monthly_Salary" in dataset.columns
        and pd.api.types.is_numeric_dtype(
            dataset["Monthly_Salary"]
        )
    ):

        salary_analysis = (
            dataset.groupby("Department")["Monthly_Salary"]
            .agg(["count", "mean", "min", "max"])
            .assign(
                salary_range=lambda x: x["max"] - x["min"]
            )
            .round(2)
            .sort_values("mean", ascending=False)
        )

        salary_ranking = {
            str(key): float(value)
            for key, value in salary_analysis["mean"].items()
        }

        overall_average = round(
            float(dataset["Monthly_Salary"].mean()), 2
        )

        salary_insights = {}

        if not salary_analysis.empty:

            highest_department = salary_analysis["mean"].idxmax()
            lowest_department = salary_analysis["mean"].idxmin()

            highest_salary = float(
                salary_analysis.loc[highest_department, "mean"]
            )

            lowest_salary = float(
                salary_analysis.loc[lowest_department, "mean"]
            )

            highest_department_min = float(
                salary_analysis.loc[highest_department, "min"]
            )

            highest_department_max = float(
                salary_analysis.loc[highest_department, "max"]
            )

            salary_insights = {
                "overall_average_salary": overall_average,

                "highest_average_salary_department": str(
                    highest_department
                ),

                "highest_average_salary": round(
                    highest_salary, 2
                ),

                "lowest_average_salary_department": str(
                    lowest_department
                ),

                "lowest_average_salary": round(
                    lowest_salary, 2
                ),

                "difference_between_highest_and_lowest_department_averages": round(
                    highest_salary - lowest_salary, 2
                ),

                "highest_department_difference_from_overall_average": round(
                    highest_salary - overall_average, 2
                ),

                "highest_department_salary_range": round(
                    highest_department_max - highest_department_min, 2
                )
            }

        department_analysis["salary"] = salary_analysis.to_dict()
        department_analysis["salary_ranking"] = salary_ranking
        department_analysis["salary_insights"] = salary_insights

    # 6. Department-wise Attrition Analysis
    if (
        "Department" in dataset.columns
        and "Attrition" in dataset.columns
    ):

        attrition_table = pd.crosstab(
            dataset["Department"],
            dataset["Attrition"]
        )

        department_analysis["attrition"] = (
            attrition_table.to_dict()
        )

        department_analysis["attrition_percentage"] = {}

        for department in attrition_table.index:

            total = int(
                attrition_table.loc[department].sum()
            )

            yes_count = int(
                attrition_table.loc[department].get("Yes", 0)
            )

            department_analysis["attrition_percentage"][
                str(department)
            ] = round(
                yes_count / total * 100, 2
            ) if total else 0

    # 7. Overall Attrition Analysis
    attrition_analysis = {}

    if "Attrition" in dataset.columns:

        attrition_counts = dataset["Attrition"].value_counts()

        total_records = len(dataset)

        attrition_analysis = {
            "counts": {
                str(key): int(value)
                for key, value in attrition_counts.items()
            },

            "total_records": total_records,

            "percentage": {
                str(key): round(
                    value / total_records * 100, 2
                )
                for key, value in attrition_counts.items()
            } if total_records else {}
        }

    # 8. Prepare Dataset Information
    dataset_information = {
        "overview": overview,
        "missing_values": missing_values,
        "numeric_analysis": numeric_summary,
        "categorical_analysis": categorical_summary,
        "department_analysis": department_analysis,
        "attrition_analysis": attrition_analysis,

        # SQL Agent calculated results
        "sql_analysis_results": (
            sql_results
            if sql_results is not None
            else "SQL Agent results are not available."
        )
    }

    # 9. Generate AI Report
    response = llm.invoke(
        f"""
        You are a professional Data Analyst.

        Generate a factual and professional business
        analysis report using only the supplied calculations.

        User Request:
        {user_question or "Generate a complete data analysis report."}

        ACTUAL CALCULATED DATA:
        {json.dumps(
            dataset_information,
            indent=2,
            default=str
        )}

        REPORT SECTIONS:

        1. Executive Summary
        2. Dataset Overview
        3. Data Quality Analysis
        4. Numerical Analysis
        5. Categorical Analysis
        6. Department-wise Salary and Experience Analysis
        7. Employee Performance Insights
        8. Department-wise Attrition Analysis
        9. Key Business Insights
        10. Actionable Recommendations

        STRICT DATA ACCURACY RULES:

        1. Treat supplied Python calculations as the only
           source of numerical truth.

        2. Never invent, modify, estimate, or assume
           numerical values.

        3. Never assume salary currency.
           Refer to the column as Monthly_Salary.
           Never mention INR, USD, rupees, or dollars.

        4. Use the supplied salary ranking to identify
           the highest and lowest average salary departments.

        5. Clearly distinguish these three calculations:
           - Difference between department average salaries.
           - Difference from overall average salary.
           - Salary range within a department (maximum minus minimum).

        6. Never confuse the department salary range
           with the difference between department averages.

        7. Never describe Attrition = Yes records as employees
           who left the company, confirmed departures, or turnover.

        8. When discussing this column, use the phrase
           "Attrition = Yes records".

        9. Refer to counts as records, not confirmed employees
           who left.

        10. Do not describe attrition as low, high, moderate,
            healthy, or unhealthy without a verified benchmark.

        11. Do not assume the scale or range of
            Job_Satisfaction or Performance_Score.

        12. Report their numerical statistics without calling
            them good, poor, high, or low.

        13. Do not assume that any category is expected,
            correct, incorrect, or abnormal.

        14. Do not claim Employee_ID or Employee_Name
            values are unique unless uniqueness was
            explicitly calculated and supplied.

        15. In Data Quality Analysis, report only checks
            actually supplied, such as missing values
            and duplicate rows.

        16. Do not claim that all data is valid,
            internally consistent, or error-free.

        17. Do not claim correlation or causation unless
            the corresponding analysis is supplied.

        18. Clearly distinguish actual observations
            from recommendations.

        19. Present recommendations as possible follow-up
            actions, not proven solutions.

        20. If information is unavailable, state that
            it is not available in the dataset or supplied analysis.

        21. Avoid unsupported descriptive words such as
            modest, significant, excellent, poor, healthy,
            unhealthy, or concerning.

        22. Use simple professional English and Markdown.

        23. SQL Agent results are additional calculated data
            and must be considered when available.

        24. Use SQL Agent results for requested calculations,
            including department-wise average experience.

        25. Never claim a calculation is unavailable
            if it is present in SQL Agent results.

        26. If Python calculations and SQL results appear
            inconsistent, identify the inconsistency clearly.
            Do not silently choose one result.

        27. Include relevant SQL Agent findings in the
            Department-wise Salary and Experience Analysis
            section.

        28. Identify all tied departments when the supplied
            results show a tie for the highest or lowest value.

        29. Do not treat SQL Agent results as verified if
            they contain an error message or failed execution.

        30. Do not invent experience statistics if SQL results
            do not provide them.

        Your priority is factual accuracy.
        Never make the report sound impressive by adding
        unsupported claims.
        """
    )

    content = response.content

    if isinstance(content, str):
        report = content

    else:
        report = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    print("\nData Analysis Report Generated Successfully!")

    return report


if __name__ == "__main__":

    df = pd.read_csv("data/raw/sales_data.csv")

    report = run_report_agent(
        "Generate a complete data analysis report.",
        df
    )

    print(report)

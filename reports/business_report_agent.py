
import os
import json
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


# 1. Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# 2. Reusable Report Agent Function
def run_report_agent(user_question=None):

    print("\nGenerating Business Report...")

    # Load sales dataset
    df = pd.read_csv("data/raw/sales_data.csv")

    df["Revenue"] = df["Quantity"] * df["Price"]

    # Calculate business metrics
    total_revenue = int(df["Revenue"].sum())
    total_profit = int(df["Profit"].sum())
    total_quantity = int(df["Quantity"].sum())
    total_orders = int(df["Order_ID"].nunique())

    profit_margin = round(
        (total_profit / total_revenue) * 100, 2
    )

    # Product performance
    product_performance = (
        df.groupby("Product")
        .agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    best_product = product_performance.loc[
        product_performance["Revenue"].idxmax(),
        "Product"
    ]

    # Region performance
    region_performance = (
        df.groupby("Region")["Revenue"]
        .sum()
        .to_dict()
    )

    best_region = max(
        region_performance,
        key=region_performance.get
    )

    # Monthly performance
    df["Date"] = pd.to_datetime(df["Date"])

    df["Month"] = df["Date"].dt.strftime("%Y-%m")

    monthly_performance = (
        df.groupby("Month")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    monthly_data = monthly_performance.to_dict(
        orient="records"
    )

    # Prepare business data
    business_data = {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "total_quantity": total_quantity,
        "total_orders": total_orders,
        "profit_margin": profit_margin,
        "best_product": best_product,
        "best_region": best_region,
        "region_performance": region_performance,
        "product_performance": product_performance.to_dict(
            orient="records"
        ),
        "monthly_performance": monthly_data
    }

    # Generate business report
    response = llm.invoke(
        f"""
        You are a professional Business Data Analyst.

        User Request:
        {user_question or "Generate a complete business report."}

        Analyze the following actual sales data and
        generate a professional business report.

        Business Data:
        {json.dumps(business_data, indent=2)}

        Include these sections:

        1. Executive Summary
        2. Overall Business Performance
        3. Product Performance Analysis
        4. Regional Sales Analysis
        5. Monthly Sales Trend
        6. Key Business Insights
        7. Business Recommendations

        Instructions:
        - Use simple professional English.
        - Use the exact values from the supplied data.
        - Mention currency as INR (Rs.).
        - Do not invent data or statistics.
        - Recommendations must be based on actual results.
        - Format the report using Markdown.
        """
    )

    # Extract AI response
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

    # Save report
    os.makedirs("outputs/reports", exist_ok=True)

    report_path = "outputs/reports/business_report.md"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # Display summary
    print("\nBusiness Report Generated Successfully!")

    print("\nBusiness Summary:")
    print("Total Revenue:", f"Rs. {total_revenue:,}")
    print("Total Profit:", f"Rs. {total_profit:,}")
    print("Total Orders:", total_orders)
    print("Profit Margin:", f"{profit_margin}%")
    print("Best Product:", best_product)
    print("Best Region:", best_region)

    print("\nReport Saved At:", report_path)

    return (
        f"Business report generated successfully. "
        f"Report Path: {report_path}"
    )


# 3. Run independently
if __name__ == "__main__":

    run_report_agent()

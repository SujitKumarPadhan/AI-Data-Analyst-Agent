
import ast

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from tools.eda_tool import eda_tools


# 1. Load API Key
load_dotenv()


# 2. Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# 3. Define Tool Selection Structure
class ToolSelection(BaseModel):
    tools: list[str] = Field(
        description="List of all required EDA tool names"
    )


# 4. Initialize AI Tool Planner
planner_llm = llm.with_structured_output(
    ToolSelection,
    method="json_mode"
)


# 5. Create Tool Dictionary
tool_map = {
    tool.name: tool
    for tool in eda_tools
}


# 6. Prepare Available Tool Descriptions
available_tools = "\n".join(
    f"- {tool.name}: {tool.description}"
    for tool in eda_tools
)


# 7. Multiple Question Loop
while True:

    user_question = input(
        "\nAsk your business question (or type 'exit'): "
    )

    if user_question.lower() == "exit":

        print("\nThank you for using AI Data Analyst Agent!")
        break

    print("\nUser Question:", user_question)

    try:

        # 8. AI Planner Selects Required Tools
        selection = planner_llm.invoke(
            f"""
            You are an expert AI Data Analyst Tool Planner.

            Select ALL tools required to answer the user's question.

            IMPORTANT RULES:
            - Return valid JSON.
            - JSON must contain one field named "tools".
            - "tools" must be a list of exact tool names.
            - Select multiple tools if multiple analyses are requested.
            - Never skip any requested analysis.
            - Select only tools from the available tools list.
            - Do not provide explanations.

            Available Tools:
            {available_tools}

            User Question:
            {user_question}
            """
        )

        # 9. Get Selected Tool Names
        selected_tool_names = list(
            dict.fromkeys(selection.tools)
        )

        selected_tool_names = [
            name for name in selected_tool_names
            if name in tool_map
        ]

        print("\nAI Planner Selected Tools:")

        for name in selected_tool_names:
            print("-", name)

        tool_results = []

        # 10. Execute All Selected Tools
        for tool_name in selected_tool_names:

            selected_tool = tool_map[tool_name]

            result = selected_tool.invoke({})

            tool_results.append(
                (tool_name, result)
            )

            print("\nTool Selected:", tool_name)
            print("Tool Result:", result)

        # 11. Generate Business Insights
        if tool_results:

            results = dict(tool_results)

            print("\nFinal AI Answer:")

            # Total Revenue
            if "get_total_revenue" in results:

                revenue = int(results["get_total_revenue"])

                print(f"Total Revenue: ₹{revenue:,}")

            # Total Profit
            if "get_total_profit" in results:

                profit = int(results["get_total_profit"])

                print(f"Total Profit: ₹{profit:,}")

            # Profit Margin
            if (
                "get_total_revenue" in results
                and "get_total_profit" in results
            ):

                revenue = int(results["get_total_revenue"])
                profit = int(results["get_total_profit"])

                if revenue != 0:

                    profit_margin = (profit / revenue) * 100

                    print(f"Profit Margin: {profit_margin:.2f}%")

            # Total Quantity
            if "get_total_quantity" in results:

                quantity = int(results["get_total_quantity"])

                print(f"Total Quantity Sold: {quantity} units")

            # Best-Selling Product
            if "get_best_selling_product" in results:

                product = results["get_best_selling_product"]

                print(f"Best-Selling Product: {product}")

            # Average Order Value
            if "get_average_order_value" in results:

                average_order_value = float(
                    results["get_average_order_value"]
                )

                print(
                    f"Average Order Value: ₹{average_order_value:,.2f}"
                )

            # Revenue by Region
            if "get_revenue_by_region" in results:

                region_data = results["get_revenue_by_region"]

                print("\nRevenue by Region:")

                for region, revenue in region_data.items():

                    print(f"{region}: ₹{revenue:,}")

            # Revenue by Product
            if "get_revenue_by_product" in results:

                product_data = results["get_revenue_by_product"]

                print("\nRevenue by Product:")

                for product, revenue in product_data.items():

                    print(f"{product}: ₹{revenue:,}")

            # Profit by Product
            if "get_profit_by_product" in results:

                product_profit = results["get_profit_by_product"]

                print("\nProfit by Product:")

                for product, profit in product_profit.items():

                    print(f"{product}: ₹{profit:,}")

            # Monthly Revenue
            if "get_monthly_revenue" in results:

                monthly_data = results["get_monthly_revenue"]

                print("\nMonthly Sales Revenue:")

                for month, revenue in monthly_data.items():

                    print(f"{month}: ₹{revenue:,}")

            # Top 5 Products
            if "get_top_5_products" in results:

                top_products = results["get_top_5_products"]

                print("\nTop Best-Selling Products:")

                for product, quantity in top_products.items():

                    print(f"{product}: {quantity} units")

            # Monthly Sales Growth
            if "get_monthly_sales_growth" in results:

                growth_data = results["get_monthly_sales_growth"]

                print("\nMonthly Sales Growth:")

                for month, growth in growth_data.items():

                    if growth >= 0:
                        print(f"{month}: Growth of {growth}%")
                    else:
                        print(f"{month}: Decrease of {abs(growth)}%")

            # Revenue by Category
            if "get_revenue_by_category" in results:

                category_data = results["get_revenue_by_category"]

                print("\nRevenue by Category:")

                for category, revenue in category_data.items():

                    print(f"{category}: ₹{revenue:,}")

            # Profit by Category
            if "get_profit_by_category" in results:

                category_profit = results["get_profit_by_category"]

                print("\nProfit by Category:")

                for category, profit in category_profit.items():

                    print(f"{category}: ₹{profit:,}")

            # Orders by Region
            if "get_orders_by_region" in results:

                region_orders = results["get_orders_by_region"]

                print("\nTotal Orders by Region:")

                for region, orders in region_orders.items():

                    print(f"{region}: {orders} orders")

            # Orders by Product
            if "get_orders_by_product" in results:

                product_orders = results["get_orders_by_product"]

                print("\nTotal Orders by Product:")

                for product, orders in product_orders.items():

                    print(f"{product}: {orders} orders")

            # Product Performance
            if "get_product_performance" in results:

                performance_data = results["get_product_performance"]

                print("\nProduct Performance Analysis:")

                for product, details in performance_data.items():

                    print(f"\nProduct: {product}")
                    print(
                        f"Total Quantity: {details['Total_Quantity']} units"
                    )
                    print(
                        f"Total Revenue: ₹{details['Total_Revenue']:,}"
                    )
                    print(
                        f"Total Profit: ₹{details['Total_Profit']:,}"
                    )

            # Category Performance
            if "get_category_performance" in results:

                category_performance = results["get_category_performance"]

                print("\nCategory Performance Analysis:")

                for category, details in category_performance.items():

                    print(f"\nCategory: {category}")
                    print(
                        f"Total Quantity: {details['Total_Quantity']} units"
                    )
                    print(
                        f"Total Revenue: ₹{details['Total_Revenue']:,}"
                    )
                    print(
                        f"Total Profit: ₹{details['Total_Profit']:,}"
                    )

            # Profit Margin by Category
            if "get_profit_margin_by_category" in results:

                margin_data = results["get_profit_margin_by_category"]

                print("\nProfit Margin by Category:")

                for category, margin in margin_data.items():

                    print(f"{category}: {margin:.2f}%")

            # Profit Margin by Product
            if "get_profit_margin_by_product" in results:

                margin_data = results["get_profit_margin_by_product"]

                print("\nProfit Margin by Product:")

                for product, margin in margin_data.items():

                    print(f"{product}: {margin:.2f}%")

            # Monthly Profit
            if "get_monthly_profit" in results:

                monthly_profit = results["get_monthly_profit"]

                print("\nMonthly Profit Analysis:")

                for month, profit in monthly_profit.items():

                    print(f"{month}: ₹{profit:,}")

            # Monthly Profit Growth
            if "get_monthly_profit_growth" in results:

                growth_data = results["get_monthly_profit_growth"]

                print("\nMonthly Profit Growth:")

                for month, growth in growth_data.items():

                    if growth >= 0:
                        print(f"{month}: Growth of {growth}%")
                    else:
                        print(f"{month}: Decrease of {abs(growth)}%")

            # Revenue Contribution by Region
            if "get_revenue_contribution_by_region" in results:

                contribution_data = results["get_revenue_contribution_by_region"]

                print("\nRevenue Contribution by Region:")

                for region, contribution in contribution_data.items():

                    print(f"{region}: {contribution:.2f}%")

            # Revenue Contribution by Product
            if "get_revenue_contribution_by_product" in results:

                contribution_data = results["get_revenue_contribution_by_product"]

                print("\nRevenue Contribution by Product:")

                for product, contribution in contribution_data.items():

                    print(f"{product}: {contribution:.2f}%")

            # Profit Contribution by Region
            if "get_profit_contribution_by_region" in results:

                contribution_data = results["get_profit_contribution_by_region"]

                print("\nProfit Contribution by Region:")

                for region, contribution in contribution_data.items():

                    print(f"{region}: {contribution:.2f}%")

            # Profit Contribution by Product
            if "get_profit_contribution_by_product" in results:

                contribution_data = results["get_profit_contribution_by_product"]

                print("\nProfit Contribution by Product:")

                for product, contribution in contribution_data.items():

                    print(f"{product}: {contribution:.2f}%")

            # Profit Contribution by Category
            if "get_profit_contribution_by_category" in results:

                contribution_data = results["get_profit_contribution_by_category"]

                print("\nProfit Contribution by Category:")

                for category, contribution in contribution_data.items():

                    print(f"{category}: {contribution:.2f}%")

            # Revenue Growth by Region
            if "get_revenue_growth_by_region" in results:

                growth_data = results["get_revenue_growth_by_region"]

                print("\nRevenue Growth by Region:")

                for region, monthly_growth in growth_data.items():

                    print(f"\nRegion: {region}")

                    if not monthly_growth:
                        print("Not enough monthly data to calculate growth.")
                        continue

                    for month, growth in monthly_growth.items():

                        if growth >= 0:
                            print(f"{month}: Growth of {growth:.2f}%")
                        else:
                            print(f"{month}: Decrease of {abs(growth):.2f}%")

            # Profit Growth by Region
            if "get_profit_growth_by_region" in results:

                growth_data = results["get_profit_growth_by_region"]

                print("\nProfit Growth by Region:")

                for region, monthly_growth in growth_data.items():

                    print(f"\nRegion: {region}")

                    if not monthly_growth:
                        print("Not enough monthly data to calculate growth.")
                        continue

                    for month, growth in monthly_growth.items():

                        if growth >= 0:
                            print(f"{month}: Growth of {growth:.2f}%")
                        else:
                            print(f"{month}: Decrease of {abs(growth):.2f}%")

            # Revenue Growth by Product
            if "get_revenue_growth_by_product" in results:

                growth_data = results["get_revenue_growth_by_product"]

                print("\nRevenue Growth by Product:")

                for product, monthly_growth in growth_data.items():

                    print(f"\nProduct: {product}")

                    if not monthly_growth:
                        print("Not enough monthly data to calculate growth.")
                        continue

                    for month, growth in monthly_growth.items():

                        if growth >= 0:
                            print(f"{month}: Growth of {growth:.2f}%")
                        else:
                            print(f"{month}: Decrease of {abs(growth):.2f}%")

            # Profit Growth by Product
            if "get_profit_growth_by_product" in results:

                growth_data = results["get_profit_growth_by_product"]

                print("\nProfit Growth by Product:")

                for product, monthly_growth in growth_data.items():

                    print(f"\nProduct: {product}")

                    if not monthly_growth:
                        print("Not enough monthly data to calculate growth.")
                        continue

                    for month, growth in monthly_growth.items():

                        if growth >= 0:
                            print(f"{month}: Growth of {growth:.2f}%")
                        else:
                            print(f"{month}: Decrease of {abs(growth):.2f}%")

            # Revenue Growth by Category
            if "get_revenue_growth_by_category" in results:

                growth_data = results["get_revenue_growth_by_category"]

                print("\nRevenue Growth by Category:")

                for category, monthly_growth in growth_data.items():

                    print(f"\nCategory: {category}")

                    if not monthly_growth:
                        print("Not enough monthly data to calculate growth.")
                        continue

                    for month, growth in monthly_growth.items():

                        if growth >= 0:
                            print(f"{month}: Growth of {growth:.2f}%")
                        else:
                            print(f"{month}: Decrease of {abs(growth):.2f}%")

            # Profit Growth by Category
            if "get_profit_growth_by_category" in results:

                growth_data = results["get_profit_growth_by_category"]

                print("\nProfit Growth by Category:")

                for category, monthly_growth in growth_data.items():

                    print(f"\nCategory: {category}")

                    if not monthly_growth:
                        print("Not enough monthly data to calculate growth.")
                        continue

                    for month, growth in monthly_growth.items():

                        if growth >= 0:
                            print(f"{month}: Growth of {growth:.2f}%")
                        else:
                            print(f"{month}: Decrease of {abs(growth):.2f}%")

        else:

            print("\nNo relevant EDA tool was selected.")

            response = llm.invoke(
                f"""
                You are an AI Data Analyst.
                Answer the user's question clearly.

                User Question: {user_question}
                """
            )

            print("\nGroq Answer:")
            print(response.content)

    except Exception as e:

        print("\nError:", str(e))

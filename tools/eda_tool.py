
import pandas as pd
from langchain.tools import tool

# Load dataset
df = pd.read_csv("data/raw/sales_data.csv")

# Calculate revenue
df["Revenue"] = df["Quantity"] * df["Price"]


@tool
def get_total_revenue():
    """Calculate the total revenue from all sales."""
    return int(df["Revenue"].sum())


@tool
def get_total_profit():
    """Calculate the total profit from all sales."""
    return int(df["Profit"].sum())


@tool
def get_total_quantity():
    """Calculate the total quantity of products sold."""
    return int(df["Quantity"].sum())


@tool
def get_best_selling_product():
    """Find the product with the highest total quantity sold."""
    return df.groupby("Product")["Quantity"].sum().idxmax()


@tool
def get_revenue_by_region():
    """Calculate total revenue for each sales region."""
    region_revenue = df.groupby("Region")["Revenue"].sum()
    return region_revenue.to_dict()


@tool
def get_revenue_by_product():
    """Calculate total revenue for each product."""
    product_revenue = df.groupby("Product")["Revenue"].sum()
    return product_revenue.to_dict()


@tool
def get_profit_by_product():
    """Calculate total profit for each product."""
    product_profit = df.groupby("Product")["Profit"].sum()
    return product_profit.to_dict()


@tool
def get_monthly_revenue():
    """Calculate total revenue for each month."""
    df["Date"] = pd.to_datetime(df["Date"])

    monthly_revenue = df.groupby(
        df["Date"].dt.strftime("%B")
    )["Revenue"].sum()

    return monthly_revenue.to_dict()


@tool
def get_top_5_products():
    """Find the top 5 products based on total quantity sold."""
    top_products = (
        df.groupby("Product")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    return top_products.to_dict()


@tool
def get_monthly_sales_growth():
    """Calculate month-over-month sales growth percentage."""

    df["Date"] = pd.to_datetime(df["Date"])

    monthly_sales = df.groupby(
        df["Date"].dt.to_period("M")
    )["Revenue"].sum().sort_index()

    growth = monthly_sales.pct_change() * 100

    return {
        str(month): round(value, 2)
        for month, value in growth.items()
        if pd.notna(value)
    }


@tool
def get_revenue_by_category():
    """Calculate total revenue for each product category."""

    category_revenue = df.groupby("Category")["Revenue"].sum()

    return category_revenue.to_dict()


@tool
def get_profit_by_category():
    """Calculate total profit for each product category."""

    category_profit = df.groupby("Category")["Profit"].sum()

    return category_profit.to_dict()


@tool
def get_average_order_value():
    """Calculate the average revenue generated per order."""

    total_revenue = df["Revenue"].sum()
    total_orders = df["Order_ID"].nunique()

    average_order_value = total_revenue / total_orders

    return round(average_order_value, 2)


@tool
def get_orders_by_region():
    """Calculate the total number of unique orders in each region."""

    region_orders = df.groupby("Region")["Order_ID"].nunique()

    return region_orders.to_dict()


@tool
def get_orders_by_product():
    """Calculate the total number of unique orders for each product."""

    product_orders = df.groupby("Product")["Order_ID"].nunique()

    return product_orders.to_dict()


@tool
def get_product_performance():
    """Calculate total quantity, revenue, and profit for each product."""

    product_performance = df.groupby("Product").agg(
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum")
    )

    return product_performance.to_dict(orient="index")


@tool
def get_category_performance():
    """Calculate total quantity, revenue, and profit for each category."""

    category_performance = df.groupby("Category").agg(
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum")
    )

    return category_performance.to_dict(orient="index")


@tool
def get_profit_margin_by_category():
    """Calculate profit margin percentage for each category."""

    category_data = df.groupby("Category").agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum")
    )

    category_data["Profit_Margin"] = (
        category_data["Total_Profit"] /
        category_data["Total_Revenue"]
    ) * 100

    return category_data["Profit_Margin"].round(2).to_dict()


@tool
def get_profit_margin_by_product():
    """Calculate profit margin percentage for each product."""

    product_data = df.groupby("Product").agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Profit=("Profit", "sum")
    )

    product_data["Profit_Margin"] = (
        product_data["Total_Profit"] /
        product_data["Total_Revenue"]
    ) * 100

    return product_data["Profit_Margin"].round(2).to_dict()


@tool
def get_monthly_profit():
    """Calculate total profit for each month."""

    df["Date"] = pd.to_datetime(df["Date"])

    monthly_profit = df.groupby(
        df["Date"].dt.strftime("%B")
    )["Profit"].sum()

    return monthly_profit.to_dict()


@tool
def get_monthly_profit_growth():
    """Calculate month-over-month profit growth percentage."""

    df["Date"] = pd.to_datetime(df["Date"])

    monthly_profit = df.groupby(
        df["Date"].dt.to_period("M")
    )["Profit"].sum().sort_index()

    growth = monthly_profit.pct_change() * 100

    return {
        str(month): round(value, 2)
        for month, value in growth.items()
        if pd.notna(value)
    }


@tool
def get_revenue_contribution_by_region():
    """Calculate each region's percentage contribution to total revenue."""

    region_revenue = df.groupby("Region")["Revenue"].sum()

    total_revenue = df["Revenue"].sum()

    contribution = (region_revenue / total_revenue) * 100

    return {
        region: round(value, 2)
        for region, value in contribution.items()
    }


@tool
def get_revenue_contribution_by_product():
    """Calculate each product's percentage contribution to total revenue."""

    product_revenue = df.groupby("Product")["Revenue"].sum()

    total_revenue = df["Revenue"].sum()

    contribution = (product_revenue / total_revenue) * 100

    return {
        product: round(value, 2)
        for product, value in contribution.items()
    }


@tool
def get_profit_contribution_by_region():
    """Calculate each region's percentage contribution to total profit."""

    region_profit = df.groupby("Region")["Profit"].sum()

    total_profit = df["Profit"].sum()

    contribution = (region_profit / total_profit) * 100

    return {
        region: round(value, 2)
        for region, value in contribution.items()
    }


# STEP 25: Profit Contribution by Product
@tool
def get_profit_contribution_by_product():
    """Calculate each product's percentage contribution to total profit."""

    product_profit = df.groupby("Product")["Profit"].sum()

    total_profit = df["Profit"].sum()

    contribution = (product_profit / total_profit) * 100

    return {
        product: round(value, 2)
        for product, value in contribution.items()
    }


@tool
def get_profit_contribution_by_category():
    """Calculate each category's percentage contribution to total profit."""

    category_profit = df.groupby("Category")["Profit"].sum()

    total_profit = df["Profit"].sum()

    contribution = (category_profit / total_profit) * 100

    return {
        category: round(value, 2)
        for category, value in contribution.items()
    }


# STEP 27: Revenue Growth by Region

@tool
def get_revenue_growth_by_region():
    """Calculate month-over-month revenue growth for each region."""

    monthly_revenue = df.copy()

    monthly_revenue["Date"] = pd.to_datetime(monthly_revenue["Date"])
    monthly_revenue["Month"] = monthly_revenue["Date"].dt.to_period("M")

    monthly_revenue["Revenue"] = (
        monthly_revenue["Quantity"] * monthly_revenue["Price"]
    )

    result = {}

    for region in monthly_revenue["Region"].unique():

        region_data = monthly_revenue[
            monthly_revenue["Region"] == region
        ]

        monthly = (
            region_data.groupby("Month")["Revenue"]
            .sum()
            .sort_index()
        )

        growth = monthly.pct_change() * 100
        growth = growth.dropna().round(2)

        result[region] = {
            str(month): float(value)
            for month, value in growth.items()
        }

    return result


# STEP 28: Profit Growth by Region

@tool
def get_profit_growth_by_region():
    """Calculate month-over-month profit growth for each region."""

    monthly_profit = df.copy()

    monthly_profit["Date"] = pd.to_datetime(monthly_profit["Date"])
    monthly_profit["Month"] = monthly_profit["Date"].dt.to_period("M")

    result = {}

    for region in monthly_profit["Region"].unique():

        region_data = monthly_profit[
            monthly_profit["Region"] == region
        ]

        monthly = (
            region_data.groupby("Month")["Profit"]
            .sum()
            .sort_index()
        )

        growth = monthly.pct_change() * 100
        growth = growth.replace([float("inf"), -float("inf")], float("nan"))
        growth = growth.dropna().round(2)

        result[region] = {
            str(month): float(value)
            for month, value in growth.items()
        }

    return result

# STEP 29: Revenue Growth by Product

@tool
def get_revenue_growth_by_product():
    """Calculate month-over-month revenue growth for each product."""

    product_data = df.copy()

    product_data["Date"] = pd.to_datetime(product_data["Date"])
    product_data["Month"] = product_data["Date"].dt.to_period("M")

    product_data["Revenue"] = (
        product_data["Quantity"] * product_data["Price"]
    )

    result = {}

    for product in product_data["Product"].unique():

        product_sales = product_data[
            product_data["Product"] == product
        ]

        monthly = (
            product_sales.groupby("Month")["Revenue"]
            .sum()
            .sort_index()
        )

        growth = monthly.pct_change() * 100

        growth = growth.replace(
            [float("inf"), -float("inf")],
            float("nan")
        )

        growth = growth.dropna().round(2)

        result[product] = {
            str(month): float(value)
            for month, value in growth.items()
        }

    return result


# STEP 30: Profit Growth by Product

@tool
def get_profit_growth_by_product():
    """Calculate month-over-month profit growth for each product."""

    product_data = df.copy()

    product_data["Date"] = pd.to_datetime(product_data["Date"])
    product_data["Month"] = product_data["Date"].dt.to_period("M")

    result = {}

    for product in product_data["Product"].unique():

        product_sales = product_data[
            product_data["Product"] == product
        ]

        monthly = (
            product_sales.groupby("Month")["Profit"]
            .sum()
            .sort_index()
        )

        growth = monthly.pct_change() * 100

        growth = growth.replace(
            [float("inf"), -float("inf")],
            float("nan")
        )

        growth = growth.dropna().round(2)

        result[product] = {
            str(month): float(value)
            for month, value in growth.items()
        }

    return result


# STEP 31: Revenue Growth by Category

@tool
def get_revenue_growth_by_category():
    """Calculate month-over-month revenue growth for each category."""

    category_data = df.copy()

    category_data["Date"] = pd.to_datetime(category_data["Date"])
    category_data["Month"] = category_data["Date"].dt.to_period("M")

    category_data["Revenue"] = (
        category_data["Quantity"] * category_data["Price"]
    )

    result = {}

    for category in category_data["Category"].unique():

        category_sales = category_data[
            category_data["Category"] == category
        ]

        monthly = (
            category_sales.groupby("Month")["Revenue"]
            .sum()
            .sort_index()
        )

        growth = monthly.pct_change() * 100

        growth = growth.replace(
            [float("inf"), -float("inf")],
            float("nan")
        )

        growth = growth.dropna().round(2)

        result[category] = {
            str(month): float(value)
            for month, value in growth.items()
        }

    return result


# STEP 32: Profit Growth by Category

@tool
def get_profit_growth_by_category():
    """Calculate month-over-month profit growth for each category."""

    category_data = df.copy()

    category_data["Date"] = pd.to_datetime(category_data["Date"])
    category_data["Month"] = category_data["Date"].dt.to_period("M")

    result = {}

    for category in category_data["Category"].unique():

        category_sales = category_data[
            category_data["Category"] == category
        ]

        monthly = (
            category_sales.groupby("Month")["Profit"]
            .sum()
            .sort_index()
        )

        growth = monthly.pct_change() * 100

        growth = growth.replace(
            [float("inf"), -float("inf")],
            float("nan")
        )

        growth = growth.dropna().round(2)

        result[category] = {
            str(month): float(value)
            for month, value in growth.items()
        }

    return result



# List of AI tools
eda_tools = [
    get_total_revenue,
    get_total_profit,
    get_total_quantity,
    get_best_selling_product,
    get_revenue_by_region,
    get_revenue_by_product,
    get_profit_by_product,
    get_monthly_revenue,
    get_top_5_products,
    get_monthly_sales_growth,
    get_revenue_by_category,
    get_profit_by_category,
    get_average_order_value,
    get_orders_by_region,
    get_orders_by_product,
    get_product_performance,
    get_category_performance,
    get_profit_margin_by_category,
    get_profit_margin_by_product,
    get_monthly_profit,
    get_monthly_profit_growth,
    get_revenue_contribution_by_region,
    get_revenue_contribution_by_product,
    get_profit_contribution_by_region,
    get_profit_contribution_by_product,
    get_profit_contribution_by_category,
    get_revenue_growth_by_region,
    get_profit_growth_by_region,
    get_revenue_growth_by_product,
    get_profit_growth_by_product,
    get_revenue_growth_by_category,
    get_profit_growth_by_category
]


if __name__ == "__main__":
    print("Available AI Tools:")

    for tool in eda_tools:
        print(tool.name)

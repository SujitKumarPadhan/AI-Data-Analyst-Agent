
import pandas as pd

# Load the sales dataset
df = pd.read_csv("data/raw/sales_data.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())


# Dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())


# Calculate Revenue
df["Revenue"] = df["Quantity"] * df["Price"]

# Total Revenue
total_revenue = df["Revenue"].sum()

# Total Profit
total_profit = df["Profit"].sum()

# Total Products Sold
total_quantity = df["Quantity"].sum()

# Best Selling Product
best_selling_product = (
    df.groupby("Product")["Quantity"]
    .sum()
    .idxmax()
)

print("\n--- Business Analysis ---")
print("Total Revenue:", total_revenue)
print("Total Profit:", total_profit)
print("Total Products Sold:", total_quantity)
print("Best Selling Product:", best_selling_product)


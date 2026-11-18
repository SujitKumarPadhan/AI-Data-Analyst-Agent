
import pandas as pd
import sqlite3
import os

# CSV file path
csv_path = "data/raw/sales_data.csv"

# Database folder
os.makedirs("data", exist_ok=True)

# Load CSV dataset
df = pd.read_csv(csv_path)

# Connect to SQLite database
connection = sqlite3.connect("data/sales.db")

# Create sales table
df.to_sql(
    "sales",
    connection,
    if_exists="replace",
    index=False
)

# Check total records
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM sales")

total_records = cursor.fetchone()[0]

print("Database created successfully!")
print("Table Name: sales")
print("Total Records:", total_records)

connection.close()

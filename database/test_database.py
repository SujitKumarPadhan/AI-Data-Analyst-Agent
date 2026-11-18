
import sqlite3

# Connect to database
connection = sqlite3.connect("data/sales.db")

cursor = connection.cursor()

# 1. Show all records
cursor.execute("SELECT * FROM sales")

records = cursor.fetchall()

print("\nAll Sales Records:")

for record in records:
    print(record)

# 2. Calculate total revenue
cursor.execute("""
    SELECT SUM(Quantity * Price)
    FROM sales
""")

total_revenue = cursor.fetchone()[0]

print("\nTotal Revenue:", total_revenue)

# 3. Revenue by region
cursor.execute("""
    SELECT Region, SUM(Quantity * Price)
    FROM sales
    GROUP BY Region
""")

region_data = cursor.fetchall()

print("\nRevenue by Region:")

for region, revenue in region_data:
    print(region, revenue)

connection.close()

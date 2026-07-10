import os
import pandas as pd

# ==========================================================
# CUSTOMER PURCHASE SEQUENCE PREDICTION
# Week 3 - Data Preprocessing
# ==========================================================

print("=" * 60)
print("LOADING ONLINE RETAIL DATASET")
print("=" * 60)

# -----------------------------
# Load Dataset
# -----------------------------

file_path = "../../data/online_retail_II.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Year 2010-2011"
)

print("\nDataset Loaded Successfully")
print(f"Rows : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# -----------------------------
# Missing Values
# -----------------------------

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

print("\nCleaning Dataset...")

# Remove rows where Customer ID is missing
df = df.dropna(subset=["Customer ID"])

# Remove rows where Description is missing
df = df.dropna(subset=["Description"])

# Remove cancelled invoices
df = df[~df["Invoice"].astype(str).str.startswith("C")]

# Remove negative quantity
df = df[df["Quantity"] > 0]

# Remove zero or negative price
df = df[df["Price"] > 0]

# Convert Customer ID to integer
df["Customer ID"] = df["Customer ID"].astype(int)

# Convert InvoiceDate
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Sort customer purchase history
df = df.sort_values(
    by=["Customer ID", "InvoiceDate"]
)

# -----------------------------
# Create Total Amount
# -----------------------------

df["TotalAmount"] = df["Quantity"] * df["Price"]

# -----------------------------
# Save Clean Dataset
# -----------------------------

os.makedirs("../../data/processed", exist_ok=True)

output_path = "../../data/processed/clean_online_retail.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nDataset Cleaned Successfully!")

print("\nFinal Shape")
print(df.shape)

print("\nFirst 5 Rows")
print(df.head())

print("\nClean Dataset Saved At:")
print(output_path)

print("\nPreprocessing Completed Successfully!")
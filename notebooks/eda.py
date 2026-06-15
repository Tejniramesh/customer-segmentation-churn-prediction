import pandas as pd

# Load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# First 5 rows
print("FIRST 5 ROWS")
print(df.head())

print("\n")

# Dataset information
print("DATASET INFO")
print(df.info())

print("\n")

# Dataset shape
print("ROWS AND COLUMNS")
print(df.shape)

print("\n")

# Statistics
print("SUMMARY")
print(df.describe())
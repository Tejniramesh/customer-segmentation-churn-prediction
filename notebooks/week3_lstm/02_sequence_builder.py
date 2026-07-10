import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("BUILDING PURCHASE SEQUENCES")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("../../data/processed/clean_online_retail.csv")

# Keep only required columns
df = df[[
    "Customer ID",
    "InvoiceDate",
    "StockCode"
]]

# Sort purchases
df = df.sort_values(
    ["Customer ID", "InvoiceDate"]
)

# Encode product IDs
encoder = LabelEncoder()

df["ProductID"] = encoder.fit_transform(df["StockCode"])

# Create purchase history
customer_history = (
    df.groupby("Customer ID")["ProductID"]
      .apply(list)
)

sequence_length = 5

X = []
y = []

for purchases in customer_history:

    if len(purchases) <= sequence_length:
        continue

    for i in range(len(purchases) - sequence_length):

        X.append(
            purchases[i:i+sequence_length]
        )

        y.append(
            purchases[i+sequence_length]
        )

print("\nNumber of sequences:", len(X))

# Save

os.makedirs("../../data/lstm", exist_ok=True)

with open("../../data/lstm/X.pkl","wb") as f:
    pickle.dump(X,f)

with open("../../data/lstm/y.pkl","wb") as f:
    pickle.dump(y,f)

with open("../../data/lstm/label_encoder.pkl","wb") as f:
    pickle.dump(encoder,f)

print("\nSequence dataset saved successfully.")
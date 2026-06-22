import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Features for clustering
features = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

# Scale data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

# Segment names
segment_names = {
    0: "Premium Customers",
    1: "New Customers",
    2: "Regular Customers"
}

df["Segment_Name"] = df["Segment"].map(segment_names)

# Segment analysis
segment_analysis = df.groupby("Segment")[["tenure", "MonthlyCharges", "TotalCharges"]].mean()

print("\nSegment Analysis")
print(segment_analysis)

# Graph
plt.figure(figsize=(8,5))

segment_counts = df["Segment_Name"].value_counts()

plt.bar(segment_counts.index, segment_counts.values)

plt.title("Customer Segments")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("segment_distribution.png")

print("\nGraph saved as segment_distribution.png")
# Churn analysis by segment

print("\nChurn Rate by Segment")

churn_analysis = pd.crosstab(
    df['Segment_Name'],
    df['Churn'],
    normalize='index'
) * 100

print(churn_analysis)
import matplotlib.pyplot as plt

churn_percent = pd.crosstab(
    df['Segment_Name'],
    df['Churn'],
    normalize='index'
) * 100

churn_percent['Yes'].plot(
    kind='bar',
    title='Churn Rate by Customer Segment'
)

plt.ylabel("Churn Percentage")
plt.tight_layout()
plt.savefig("churn_rate_by_segment.png")

print("\nGraph saved as churn_rate_by_segment.png")
# Save segmented customers

df.to_csv("customer_segments.csv", index=False)

print("\nCustomer segments saved as customer_segments.csv")
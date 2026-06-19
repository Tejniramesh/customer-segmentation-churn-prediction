import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully!")
print("Rows and Columns:", df.shape)

# Select features for K-Means
features = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

print("\nSelected Features:")
print(features.head())

print("\nData Types:")
print(features.dtypes)
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove empty values
df = df.dropna()

print("\nAfter Cleaning:")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].dtypes)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("Reached KMeans section")

# Select clean features
features = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("\nScaling Completed!")
print("Scaled Data Shape:", scaled_features.shape)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

print("\nSegments Created Successfully!")

print("\nSegment Distribution:")
print(df['Segment'].value_counts())
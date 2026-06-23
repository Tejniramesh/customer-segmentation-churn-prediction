import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Target
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Feature Engineering
df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})

df['OnlineSecurity'] = df['OnlineSecurity'].map({
    'No': 0,
    'Yes': 1,
    'No internet service': 0
})

df['TechSupport'] = df['TechSupport'].map({
    'No': 0,
    'Yes': 1,
    'No internet service': 0
})

df['Contract'] = df['Contract'].map({
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
})

# Features
X = df[
    [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'Partner',
        'Dependents',
        'PaperlessBilling',
        'OnlineSecurity',
        'TechSupport',
        'Contract'
    ]
]

# Target
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

# Graph
plt.figure(figsize=(8,5))

plt.bar(
    importance['Feature'],
    importance['Importance']
)

plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")

print("\nGraph saved as feature_importance.png")
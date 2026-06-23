import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Cleaning
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

y = df['Churn']

# Optimized Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# 5-Fold Cross Validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("Cross Validation Scores:")
print(scores)

print("\nAverage Accuracy:")
print(round(scores.mean() * 100, 2), "%")
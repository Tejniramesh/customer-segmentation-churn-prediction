import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Optimized Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Optimized Random Forest Accuracy:")
print(round(accuracy * 100, 2), "%")
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})

X = df[
    [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'Partner',
        'Dependents',
        'PaperlessBilling'
    ]
]

y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print("Precision:", round(precision_score(y_test, pred)*100, 2), "%")
print("Recall:", round(recall_score(y_test, pred)*100, 2), "%")
print("F1 Score:", round(f1_score(y_test, pred)*100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, prob)*100, 2), "%")
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Convert Churn
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# ==========================
# FEATURE ENGINEERING
# ==========================

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

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("===================================")
print("MODEL COMPARISON")
print("===================================")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:")
print(round(lr_accuracy * 100, 2), "%")

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:")
print(round(dt_accuracy * 100, 2), "%")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:")
print(round(rf_accuracy * 100, 2), "%")

# Best Model
accuracies = {
    "Logistic Regression": lr_accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy
}

best_model = max(accuracies, key=accuracies.get)

print("\n===================================")
print("BEST MODEL")
print("===================================")
print(best_model)
print("Accuracy:", round(accuracies[best_model] * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, lr_predictions)

print("\nConfusion Matrix")
print(cm)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 60)
print("BERT MODEL EVALUATION")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("../../data/bert/clean_reviews.csv")

# Use a small evaluation sample
df = df.sample(n=1000, random_state=42)

dataset = Dataset.from_pandas(df)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("../../models/bert_final")

def tokenize(batch):
    return tokenizer(
        batch["Review Text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.rename_column("Sentiment", "labels")

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# Load trained model
model = BertForSequenceClassification.from_pretrained("../../models/bert_final")

trainer = Trainer(model=model)

predictions = trainer.predict(dataset)

preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="weighted"
)

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

cm = confusion_matrix(labels, preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive", "Neutral"]
)

disp.plot()

plt.savefig("../../outputs/bert_confusion_matrix.png")

print("\nConfusion Matrix Saved!")
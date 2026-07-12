import os
import numpy as np
import pandas as pd

from datasets import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("=" * 60)
print("TRAINING BERT MODEL")
print("=" * 60)

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("../../data/bert/clean_reviews.csv")

print("Dataset Shape :", df.shape)

# -----------------------------------
# Train Test Split
# -----------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Sentiment"]
)

# -----------------------------------
# Use Smaller Dataset (Faster Training)
# -----------------------------------

train_df = train_df.sample(
    n=5000,
    random_state=42
)

test_df = test_df.sample(
    n=1000,
    random_state=42
)

print("Training :", len(train_df))
print("Testing :", len(test_df))

# -----------------------------------
# Tokenizer
# -----------------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["Review Text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("Sentiment", "labels")
test_dataset = test_dataset.rename_column("Sentiment", "labels")

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

# -----------------------------------
# Model
# -----------------------------------

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

# -----------------------------------
# Metrics
# -----------------------------------

def compute_metrics(pred):

    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------------------
# Training Arguments
# -----------------------------------

training_args = TrainingArguments(

    output_dir="../../models/bert",

    eval_strategy="epoch",

    save_strategy="epoch",

    num_train_epochs=1,

    per_device_train_batch_size=32,

    per_device_eval_batch_size=32,

    logging_steps=25,

    load_best_model_at_end=True,

    report_to="none"
)

# -----------------------------------
# Trainer
# -----------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)

# -----------------------------------
# Train
# -----------------------------------

trainer.train()

# -----------------------------------
# Save Model
# -----------------------------------

os.makedirs("../../models/bert_final", exist_ok=True)

trainer.save_model("../../models/bert_final")

tokenizer.save_pretrained("../../models/bert_final")

print("\nModel Saved Successfully!")
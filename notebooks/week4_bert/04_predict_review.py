import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

print("=" * 60)
print("CUSTOMER REVIEW SENTIMENT PREDICTION")
print("=" * 60)

tokenizer = BertTokenizer.from_pretrained("../../models/bert_final")

model = BertForSequenceClassification.from_pretrained("../../models/bert_final")

model.eval()

labels = {
    0: "Negative 😞",
    1: "Positive 😊",
    2: "Neutral 😐"
}

while True:

    review = input("\nEnter Customer Review (type exit to quit): ")

    if review.lower() == "exit":
        break

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()

    print("\nPrediction :", labels[prediction])
    print("Confidence :", round(confidence * 100, 2), "%")
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("TRAINING LSTM MODEL")
print("=" * 60)

# -----------------------------
# Load Sequences
# -----------------------------

with open("../../data/lstm/X.pkl", "rb") as f:
    X = pickle.load(f)

with open("../../data/lstm/y.pkl", "rb") as f:
    y = pickle.load(f)

X = np.array(X)
y = np.array(y)

print("Total Samples :", len(X))

# -----------------------------
# Check Vocabulary Size
# -----------------------------

max_token = max(np.max(X), np.max(y))
vocab_size = int(max_token) + 1

print("Maximum Token :", max_token)
print("Vocabulary Size :", vocab_size)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# -----------------------------
# Build Model
# -----------------------------

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=64
    )
)

model.add(LSTM(64))

model.add(
    Dense(
        vocab_size,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    callbacks=[early_stop]
)

# -----------------------------
# Save Model
# -----------------------------

os.makedirs("../../models", exist_ok=True)

model.save("../../models/lstm_purchase_model.keras")

print("\nModel Saved Successfully!")

# -----------------------------
# Plot Loss
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

os.makedirs("../../outputs", exist_ok=True)

plt.savefig("../../outputs/lstm_loss.png")

print("Loss graph saved.")

# -----------------------------
# Test Accuracy
# -----------------------------

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Accuracy :", round(acc * 100, 2), "%")
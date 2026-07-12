import os
import pandas as pd

print("=" * 60)
print("WEEK 4 - DATA PREPROCESSING")
print("=" * 60)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

file_path = "../../data/Womens Clothing E-Commerce Reviews.csv"

df = pd.read_csv(file_path)

print("\nDataset Loaded Successfully!")
print("Shape :", df.shape)

# ----------------------------------------------------
# Keep Required Columns
# ----------------------------------------------------

df = df[
    [
        "Review Text",
        "Rating",
        "Recommended IND"
    ]
]

# ----------------------------------------------------
# Remove Missing Reviews
# ----------------------------------------------------

df = df.dropna(subset=["Review Text"])

# ----------------------------------------------------
# Create Sentiment Labels
# ----------------------------------------------------

def sentiment_label(rating):

    if rating >= 4:
        return 1      # Positive

    elif rating == 3:
        return 2      # Neutral

    else:
        return 0      # Negative

df["Sentiment"] = df["Rating"].apply(sentiment_label)

print("\nSentiment Distribution\n")
print(df["Sentiment"].value_counts())

# ----------------------------------------------------
# Save Clean Dataset
# ----------------------------------------------------

os.makedirs("../../data/bert", exist_ok=True)

df.to_csv(
    "../../data/bert/clean_reviews.csv",
    index=False
)

print("\nClean dataset saved successfully!")
print(df.head())

print("\nPreprocessing Completed!")
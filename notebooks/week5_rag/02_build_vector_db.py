import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

print("=" * 60)
print("BUILDING VECTOR DATABASE")
print("=" * 60)

# -------------------------------
# Load Chunks
# -------------------------------

chunks = []

with open("../../data/rag/chunks.txt", "r", encoding="utf-8") as file:

    text = file.read()

for chunk in text.split("CHUNK"):

    chunk = chunk.strip()

    if len(chunk) == 0:
        continue

    lines = chunk.split("\n")

    content = " ".join(lines[1:])

    chunks.append(content)

print("Total Chunks :", len(chunks))

# -------------------------------
# Load Embedding Model
# -------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nGenerating Embeddings...")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True
)

print("Embedding Shape :", embeddings.shape)

# -------------------------------
# Create FAISS Index
# -------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# -------------------------------
# Save
# -------------------------------

os.makedirs("../../models/rag", exist_ok=True)

faiss.write_index(
    index,
    "../../models/rag/vector_index.faiss"
)

with open("../../models/rag/chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("\nVector Database Saved Successfully!")

print("=" * 60)
print("RAG DATABASE READY")
print("=" * 60)
import os

print("=" * 60)
print("PREPARING KNOWLEDGE BASE")
print("=" * 60)

# -------------------------------
# Load Knowledge Base
# -------------------------------

file_path = "../../data/rag/knowledge_base.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("\nKnowledge Base Loaded Successfully!")

print("\nTotal Characters :", len(text))

# -------------------------------
# Split into Chunks
# -------------------------------

chunks = []

current_chunk = ""

for line in text.split("\n"):

    line = line.strip()

    if line == "":
        continue

    current_chunk += line + " "

    if len(current_chunk) > 400:

        chunks.append(current_chunk.strip())

        current_chunk = ""

if current_chunk:
    chunks.append(current_chunk.strip())

print("\nTotal Chunks Created :", len(chunks))

# -------------------------------
# Save Chunks
# -------------------------------

os.makedirs("../../data/rag", exist_ok=True)

with open("../../data/rag/chunks.txt", "w", encoding="utf-8") as file:

    for i, chunk in enumerate(chunks):

        file.write(f"CHUNK {i+1}\n")

        file.write(chunk)

        file.write("\n\n")

print("\nChunks Saved Successfully!")

print("=" * 60)
print("KNOWLEDGE BASE READY")
print("=" * 60)
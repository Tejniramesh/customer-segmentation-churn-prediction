import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

print("=" * 60)
print("AI CUSTOMER INSIGHTS ASSISTANT")
print("=" * 60)

# -----------------------------------
# Load Embedding Model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# Load Vector Database
# -----------------------------------

index = faiss.read_index("../../models/rag/vector_index.faiss")

with open("../../models/rag/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Knowledge Base Loaded Successfully!")

# -----------------------------------
# Chat Loop
# -----------------------------------

while True:

    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    print("\nSearching Knowledge Base...")

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(question_embedding, 3)

    print("\n" + "=" * 60)
    print("TOP KNOWLEDGE RETRIEVED")
    print("=" * 60)

    retrieved_text = ""

    for i in indices[0]:

        print("\n•", chunks[i])

        retrieved_text += chunks[i] + "\n"

    print("\n" + "=" * 60)
    print("AI RECOMMENDATION")
    print("=" * 60)

    if "churn" in question.lower():

        print("""
• Focus on customers with month-to-month contracts.

• Offer loyalty rewards.

• Improve customer support.

• Provide personalized discounts.
""")

    elif "sentiment" in question.lower() or "review" in question.lower():

        print("""
• Monitor negative reviews regularly.

• Improve products receiving repeated complaints.

• Respond quickly to customer feedback.
""")

    elif "segment" in question.lower():

        print("""
• Reward premium customers.

• Recommend products to regular customers.

• Send offers to occasional customers.
""")

    elif "purchase" in question.lower():

        print("""
• Recommend frequently purchased products.

• Cross-sell related products.

• Use purchase history for personalization.
""")

    else:

        print("""
Based on the retrieved knowledge,
customer analytics should combine segmentation,
purchase prediction,
sentiment analysis
and churn prediction
to improve retention and business growth.
""")

    print("=" * 60)
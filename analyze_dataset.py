from pathlib import Path
import json
import statistics

# Path to the processed dataset
file_path = Path("law-rag/data/civil-law.json")

# Load the JSON dataset
with file_path.open("r", encoding="utf-8") as f:
    documents = json.load(f)

# Calculate the character length of each document
lengths = [len(doc["text"]) for doc in documents]

print("Number of documents:", len(documents))
print()

print("Minimum document length:", min(lengths))
print("Maximum document length:", max(lengths))
print("Average document length:", round(statistics.mean(lengths), 2))
print("Median document length:", statistics.median(lengths))
print()

# Find the 10 longest documents
longest_documents = sorted(
    documents,
    key=lambda doc: len(doc["text"]),
    reverse=True
)[:10]

print("=" * 70)
print("10 longest documents")
print("=" * 70)

for doc in longest_documents:
    print(
        f"Article: {doc['article']} | "
        f"Version: {doc['version']} | "
        f"Characters: {len(doc['text'])}"
    )
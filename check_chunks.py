from pathlib import Path
import json

file_path = Path("law-rag/data/civil-law-chunks.json")

with file_path.open("r", encoding="utf-8") as f:
    chunks = json.load(f)


print("Number of chunks:", len(chunks))
print()


# Show the longest chunks
longest_chunks = sorted(
    chunks,
    key=lambda chunk: len(chunk["text"]),
    reverse=True
)[:10]


print("=" * 70)
print("10 longest chunks")
print("=" * 70)

for chunk in longest_chunks:

    print(
        f"Article: {chunk['article']} | "
        f"Version: {chunk['version']} | "
        f"Chunk: {chunk['chunk_index']} | "
        f"Characters: {len(chunk['text'])}"
    )


print()
print("=" * 70)
print("Sample chunks")
print("=" * 70)


for chunk in chunks[:5]:

    print()
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Article: {chunk['article']}")
    print(f"Version: {chunk['version']}")
    print(f"Chunk index: {chunk['chunk_index']}")
    print(f"Characters: {len(chunk['text'])}")
    print()
    print(chunk["text"])
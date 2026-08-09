import json
import chromadb
import ollama

# Path to the chunked dataset
JSON_FILE = "law-rag/data/civil-law-chunks.json"

# Chroma database path
CHROMA_PATH = "law-rag/chroma_db"

# Chroma collection name
COLLECTION_NAME = "civil_law_v2"

# Embedding model
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


# Load chunked dataset
with open(JSON_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


# Connect to Chroma
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


print("Number of chunks:", len(documents))
print("Starting embedding...")


# Process chunks
for i, document in enumerate(documents):

    text = document["text"]

    # Generate embedding
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    embedding = response["embedding"]

    # Store in Chroma
    collection.upsert(
        ids=[document["chunk_id"]],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "article": document["article"],
            "version": document["version"] or "current",
            "chunk_index": document["chunk_index"],
            "source": document["source"]
        }]
    )

    print(
        f"[{i + 1}/{len(documents)}] "
        f"Article {document['article']} "
        f"Chunk {document['chunk_index']}"
    )


print()
print("Embedding completed.")
print("Documents in Chroma:", collection.count())
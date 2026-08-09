import json
import os
import ollama
import chromadb


# =========================
# Configuration
# =========================

JSON_FILE = "law-rag/data/civil-law.json"
DB_DIR = "chroma_db"
COLLECTION_NAME = "laws"


# =========================
# Connect to ChromaDB
# =========================

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================
# Load JSON data
# =========================

with open("JSON_FILE", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total JSON records: {len(data)}")


# =========================
# Prepare documents
# =========================

documents = []
metadatas = []
ids = []

for i, item in enumerate(data):

    article = str(item.get("article", "")).strip()
    version = item.get("version")
    text = str(item.get("text", "")).strip()

    if not text:
        continue

    # Create the document used for embedding
    document = f"ماده {article}\n{text}"

    # Handle records without a version
    version_text = str(version) if version is not None else "No version"

    documents.append(document)

    metadatas.append({
        "article": article,
        "version": version_text
    })

    ids.append(f"law_{i}")


print(f"Records ready for processing: {len(documents)}")


# =========================
# Generate embeddings
# =========================

print("Generating embeddings...")

embeddings = []

for i, document in enumerate(documents):

    response = ollama.embed(
        model="qwen3-embedding:0.6b",
        input=document
    )

    embedding = response["embeddings"][0]
    embeddings.append(embedding)

    if (i + 1) % 100 == 0:
        print(f"Embedding: {i + 1}/{len(documents)}")


# =========================
# Store data in ChromaDB
# =========================

print("Saving data to ChromaDB...")

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


print()
print("=" * 50)
print("RAG database created successfully!")
print("=" * 50)
print(f"Total records in database: {collection.count()}")
print(f"Database path: {os.path.abspath(DB_DIR)}")

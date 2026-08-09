import chromadb
import ollama


# Chroma database path
CHROMA_PATH = "law-rag/chroma_db"

# Collection name
COLLECTION_NAME = "civil_law_v2"

# Embedding model
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


# Connect to Chroma
client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# Get user query
query = input("Enter your legal question: ")


# Create embedding for the query
response = ollama.embeddings(
    model=EMBEDDING_MODEL,
    prompt=query
)

query_embedding = response["embedding"]


# Search Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)


print()
print("=" * 70)
print("Top 10 Retrieved Results")
print("=" * 70)


for i in range(10):

    print()
    print(f"Result #{i + 1}")
    print("-" * 70)

    metadata = results["metadatas"][0][i]
    document = results["documents"][0][i]
    distance = results["distances"][0][i]

    print(f"Article: {metadata['article']}")
    print(f"Version: {metadata['version']}")
    print(f"Chunk: {metadata['chunk_index']}")
    print(f"Distance: {distance}")

    print()
    print("Text:")
    print(document)

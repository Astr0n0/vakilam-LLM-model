import chromadb
import ollama

# =========================
# Configuration
# =========================

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# The three failed queries
FAILED_CASES = [
    {
        "question": "رضایت زوجین در عقد نکاح چه اهمیتی دارد؟",
        "expected_articles": ["۱۰۷۰"],
    },
    {
        "question": "چه کسانی به دلیل قرابت نمی‌توانند با یکدیگر ازدواج کنند؟",
        "expected_articles": ["۱۰۴۵"],
    },
    {
        "question": "عده چیست و چه مفهومی دارد؟",
        "expected_articles": ["۱۱۵۰"],
    },
]


# =========================
# Connect to ChromaDB
# =========================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("=" * 70)
print("FAILURE INSPECTION")
print("=" * 70)

print()
print(f"Collection: {COLLECTION_NAME}")
print(
    f"Number of documents: "
    f"{collection.count()}"
)


# =========================
# Inspect each failure
# =========================

for case in FAILED_CASES:

    question = case["question"]
    expected_articles = case["expected_articles"]

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    print()
    print("Expected articles:")
    print(", ".join(expected_articles))

    # ---------------------------------
    # Create query embedding
    # ---------------------------------

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    query_embedding = response["embedding"]

    # ---------------------------------
    # Search
    # ---------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20
    )

    print()
    print("=" * 70)
    print("TOP 20 SEMANTIC RESULTS")
    print("=" * 70)

    for rank in range(
        len(results["ids"][0])
    ):

        article = results["metadatas"][0][rank].get(
            "article"
        )

        version = results["metadatas"][0][rank].get(
            "version"
        )

        distance = results["distances"][0][rank]

        document = results["documents"][0][rank]

        marker = ""

        if article in expected_articles:
            marker = "  <-- EXPECTED ARTICLE"

        print()
        print(
            f"Rank: {rank + 1}"
        )

        print(
            f"Article: {article}{marker}"
        )

        print(
            f"Version: {version}"
        )

        print(
            f"Distance: {distance}"
        )

        print(
            f"Text: {document}"
        )
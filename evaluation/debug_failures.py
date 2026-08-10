import chromadb
import ollama

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Collection:", COLLECTION_NAME)
print("Documents:", collection.count())


QUESTIONS = [
    {
        "question": "رضایت زوجین در عقد نکاح چه اهمیتی دارد؟",
        "expected": "۱۰۷۰"
    },
    {
        "question": "چه کسانی به دلیل قرابت نمی‌توانند با یکدیگر ازدواج کنند؟",
        "expected": "۱۰۴۵"
    },
    {
        "question": "عده چیست و چه مفهومی دارد؟",
        "expected": "۱۱۵۰"
    }
]


for item in QUESTIONS:

    question = item["question"]
    expected = item["expected"]

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print()
    print("EXPECTED ARTICLE:", expected)

    # -----------------------------------------
    # 1. Check exact article existence
    # -----------------------------------------

    exact = collection.get(
        where={
            "article": expected
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    print()
    print("EXACT ARTICLE CHECK")
    print("-" * 80)
    print("Found:", len(exact["ids"]))

    for i in range(len(exact["ids"])):

        print()
        print("ID:", exact["ids"][i])
        print("Metadata:", exact["metadatas"][i])
        print("Document:")
        print(exact["documents"][i][:1000])

    # -----------------------------------------
    # 2. Semantic search
    # -----------------------------------------

    print()
    print("SEMANTIC SEARCH")
    print("-" * 80)

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    query_embedding = response["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20
    )

    for i in range(len(results["ids"][0])):

        article = results["metadatas"][0][i].get(
            "article"
        )

        distance = results["distances"][0][i]

        print(
            f"{i + 1:2}. "
            f"Article={article} | "
            f"Distance={distance:.6f}"
        )

    # -----------------------------------------
    # 3. Check whether expected article
    #    appears in top 20
    # -----------------------------------------

    retrieved_articles = results["metadatas"][0]

    ranks = []

    for i, metadata in enumerate(retrieved_articles):

        if metadata.get("article") == expected:
            ranks.append(i + 1)

    print()
    print("EXPECTED ARTICLE RANK")

    if ranks:
        print("Found at rank:", ranks)
    else:
        print("NOT FOUND IN TOP 20")
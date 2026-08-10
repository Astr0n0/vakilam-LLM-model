import json
import os
import re

import chromadb
import ollama


# ============================================================
# Configuration
# ============================================================

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

SEMANTIC_TOP_K = 20
FINAL_TOP_K = 10

QUESTIONS_FILE = "evaluation/legal_questions.json"
RESULTS_FILE = "evaluation/evaluation_results.json"


# ============================================================
# Persian / Arabic / English digits
# ============================================================

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"


def normalize_digits(text):
    translation_table = str.maketrans(
        PERSIAN_DIGITS + ARABIC_DIGITS,
        ENGLISH_DIGITS + ENGLISH_DIGITS
    )

    return text.translate(translation_table)


def to_persian_digits(text):
    english_to_persian = str.maketrans(
        ENGLISH_DIGITS,
        PERSIAN_DIGITS
    )

    arabic_to_persian = str.maketrans(
        ARABIC_DIGITS,
        PERSIAN_DIGITS
    )

    text = text.translate(english_to_persian)
    text = text.translate(arabic_to_persian)

    return text


# ============================================================
# Article number extraction
# ============================================================

def extract_article_number(query):
    normalized_query = normalize_digits(query)

    pattern = r"(?:ماده|ماده‌ی|مادهٔ|article)\s*[-‌]?\s*(\d+)"

    match = re.search(
        pattern,
        normalized_query,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# ChromaDB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Collection loaded: {COLLECTION_NAME}")
print(f"Number of documents in collection: {collection.count()}")
print()


# ============================================================
# Exact article retrieval
# ============================================================

def exact_article_search(article_number):

    article_number_persian = to_persian_digits(
        article_number
    )

    result = collection.get(
        where={
            "article": article_number_persian
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    results = []

    for i in range(len(result["ids"])):

        results.append({
            "id": result["ids"][i],
            "article": result["metadatas"][i].get("article"),
            "document": result["documents"][i],
            "metadata": result["metadatas"][i],
            "distance": 0.0,
            "retrieval_type": "exact"
        })

    return results


# ============================================================
# Semantic retrieval
# ============================================================

def semantic_search(query):

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )

    query_embedding = response["embedding"]

    semantic = collection.query(
        query_embeddings=[query_embedding],
        n_results=SEMANTIC_TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    results = []

    for i in range(len(semantic["documents"][0])):

        metadata = semantic["metadatas"][0][i]

        results.append({
            "id": semantic["ids"][0][i],
            "article": metadata.get("article"),
            "document": semantic["documents"][0][i],
            "metadata": metadata,
            "distance": semantic["distances"][0][i],
            "retrieval_type": "semantic"
        })

    return results


# ============================================================
# Hybrid retrieval
# ============================================================

def hybrid_search(query):

    article_number = extract_article_number(query)

    exact_results = []
    semantic_results = []

    # --------------------------------------------------------
    # Exact search
    # --------------------------------------------------------

    if article_number is not None:

        exact_results = exact_article_search(
            article_number
        )

    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    semantic_results = semantic_search(query)

    # --------------------------------------------------------
    # Merge
    #
    # Exact results always get priority.
    # --------------------------------------------------------

    merged = []
    seen_ids = set()

    for result in exact_results:

        if result["id"] not in seen_ids:

            merged.append(result)
            seen_ids.add(result["id"])

    for result in semantic_results:

        if result["id"] not in seen_ids:

            merged.append(result)
            seen_ids.add(result["id"])

    return merged[:FINAL_TOP_K]


# ============================================================
# Evaluation metrics
# ============================================================

def calculate_first_relevant_rank(
    retrieved_articles,
    expected_articles
):

    expected_set = set(expected_articles)

    for rank, article in enumerate(
        retrieved_articles,
        start=1
    ):

        if article in expected_set:
            return rank

    return None


def hit_at_k(
    retrieved_articles,
    expected_articles,
    k
):

    expected_set = set(expected_articles)

    return any(
        article in expected_set
        for article in retrieved_articles[:k]
    )


# ============================================================
# Load evaluation questions
# ============================================================

if not os.path.exists(QUESTIONS_FILE):

    raise FileNotFoundError(
        f"Evaluation file not found: {QUESTIONS_FILE}"
    )


with open(
    QUESTIONS_FILE,
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


print(
    f"Number of evaluation questions: "
    f"{len(questions)}"
)

print()


# ============================================================
# Run evaluation
# ============================================================

evaluation_results = []

hit1_count = 0
hit3_count = 0
hit5_count = 0
hit10_count = 0

reciprocal_ranks = []


for index, item in enumerate(
    questions,
    start=1
):

    question = item["question"]

    expected_articles = [
        to_persian_digits(str(article))
        for article in item["expected_articles"]
    ]

    print("=" * 70)

    print(question)

    print()

    print("Expected articles:")
    print(", ".join(expected_articles))

    print()

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = hybrid_search(question)

    retrieved_articles = [
        result["article"]
        for result in results
    ]

    print("Retrieved articles:")

    for rank, article in enumerate(
        retrieved_articles,
        start=1
    ):

        print(
            f"{rank}. Article {article}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    first_rank = calculate_first_relevant_rank(
        retrieved_articles,
        expected_articles
    )

    hit1 = hit_at_k(
        retrieved_articles,
        expected_articles,
        1
    )

    hit3 = hit_at_k(
        retrieved_articles,
        expected_articles,
        3
    )

    hit5 = hit_at_k(
        retrieved_articles,
        expected_articles,
        5
    )

    hit10 = hit_at_k(
        retrieved_articles,
        expected_articles,
        10
    )

    if hit1:
        hit1_count += 1

    if hit3:
        hit3_count += 1

    if hit5:
        hit5_count += 1

    if hit10:
        hit10_count += 1

    if first_rank is not None:
        reciprocal_ranks.append(
            1 / first_rank
        )
    else:
        reciprocal_ranks.append(0)

    print()

    if first_rank is None:
        print(
            "First relevant result: NOT FOUND"
        )
    else:
        print(
            f"First relevant result: "
            f"Rank {first_rank}"
        )

    print(
        f"Hit@1: {hit1}"
    )

    print(
        f"Hit@3: {hit3}"
    )

    print(
        f"Hit@5: {hit5}"
    )

    print(
        f"Hit@10: {hit10}"
    )

    # --------------------------------------------------------
    # Save detailed result
    # --------------------------------------------------------

    evaluation_results.append({
        "question": question,
        "expected_articles": expected_articles,
        "retrieved_articles": retrieved_articles,
        "first_relevant_rank": first_rank,
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "retrieval_results": [
            {
                "article": result["article"],
                "retrieval_type": result["retrieval_type"],
                "distance": result["distance"]
            }
            for result in results
        ]
    })


# ============================================================
# Final metrics
# ============================================================

total = len(questions)

hit1 = hit1_count / total
hit3 = hit3_count / total
hit5 = hit5_count / total
hit10 = hit10_count / total

mrr = sum(reciprocal_ranks) / total


print()
print("=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

print(
    f"Hit@1: {hit1_count}/{total} "
    f"({hit1 * 100:.2f}%)"
)

print(
    f"Hit@3: {hit3_count}/{total} "
    f"({hit3 * 100:.2f}%)"
)

print(
    f"Hit@5: {hit5_count}/{total} "
    f"({hit5 * 100:.2f}%)"
)

print(
    f"Hit@10: {hit10_count}/{total} "
    f"({hit10 * 100:.2f}%)"
)

print(
    f"MRR: {mrr:.4f}"
)


# ============================================================
# Save results
# ============================================================

output = {
    "collection": COLLECTION_NAME,
    "embedding_model": EMBEDDING_MODEL,
    "semantic_top_k": SEMANTIC_TOP_K,
    "final_top_k": FINAL_TOP_K,
    "total_questions": total,
    "metrics": {
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "mrr": mrr
    },
    "questions": evaluation_results
}


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print(
    f"Results saved to: {RESULTS_FILE}"
)
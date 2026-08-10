from pathlib import Path
import json
import re
import math

import chromadb
import ollama
from rank_bm25 import BM25Okapi


# ============================================================
# Configuration
# ============================================================

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

SEMANTIC_TOP_K = 30
KEYWORD_TOP_K = 30

RRF_K = 60

EVALUATION_FILE = Path("evaluation/legal_questions.json")
OUTPUT_FILE = Path("evaluation/hybrid_evaluation_results.json")


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
    translation_table = str.maketrans(
        ENGLISH_DIGITS + ARABIC_DIGITS,
        PERSIAN_DIGITS + PERSIAN_DIGITS
    )

    return text.translate(translation_table)


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """
    Basic Persian text normalization for BM25.
    """

    text = str(text)

    # Arabic Yeh -> Persian Yeh
    text = text.replace("ي", "ی")

    # Arabic Kaf -> Persian Kaf
    text = text.replace("ك", "ک")

    # Remove zero-width non-joiner
    text = text.replace("\u200c", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    """
    Tokenize normalized Persian text.
    """

    text = normalize_text(text)

    return re.findall(
        r"[\w\u0600-\u06FF]+",
        text,
        flags=re.UNICODE
    )


# ============================================================
# ChromaDB
# ============================================================

print("=" * 80)
print("Loading ChromaDB")
print("=" * 80)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Collection:", COLLECTION_NAME)
print("Documents:", collection.count())
print()


# ============================================================
# Load all documents
# ============================================================

print("Loading documents from ChromaDB...")

all_documents = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

documents = all_documents["documents"]
metadatas = all_documents["metadatas"]
ids = all_documents["ids"]

print("Loaded documents:", len(documents))
print()


# ============================================================
# Build BM25 index
# ============================================================

print("Building BM25 index...")

tokenized_documents = [
    tokenize(document)
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

print("BM25 index ready.")
print()


# ============================================================
# Semantic Search
# ============================================================

def semantic_search(query, top_k=SEMANTIC_TOP_K):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    output = []

    for i in range(len(results["ids"][0])):

        output.append({
            "id": results["ids"][0][i],
            "article": results["metadatas"][0][i].get("article"),
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "rank": i + 1
        })

    return output


# ============================================================
# BM25 Search
# ============================================================

def bm25_search(query, top_k=KEYWORD_TOP_K):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    output = []

    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        output.append({
            "id": ids[index],
            "article": metadatas[index].get("article"),
            "document": documents[index],
            "metadata": metadatas[index],
            "score": float(scores[index]),
            "rank": rank
        })

    return output


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=RRF_K
):

    scores = {}
    records = {}

    # -----------------------------
    # Semantic results
    # -----------------------------

    for result in semantic_results:

        doc_id = result["id"]
        rank = result["rank"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        records[doc_id] = result

    # -----------------------------
    # BM25 results
    # -----------------------------

    for result in keyword_results:

        doc_id = result["id"]
        rank = result["rank"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        if doc_id not in records:
            records[doc_id] = result

    # -----------------------------
    # Sort by RRF score
    # -----------------------------

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    output = []

    for rank, (doc_id, score) in enumerate(
        ranked,
        start=1
    ):

        result = records[doc_id].copy()

        result["rrf_score"] = score
        result["rank"] = rank

        output.append(result)

    return output


# ============================================================
# Article helpers
# ============================================================

def article_variants(article):
    """
    Return possible representations of an article number.
    """

    normalized = normalize_digits(str(article))

    return {
        normalized,
        to_persian_digits(normalized)
    }


def is_expected_article(result_article, expected_article):

    if result_article is None:
        return False

    return bool(
        article_variants(result_article)
        &
        article_variants(expected_article)
    )


# ============================================================
# Find first relevant rank
# ============================================================

def first_relevant_rank(results, expected_articles):

    for rank, result in enumerate(
        results,
        start=1
    ):

        for expected in expected_articles:

            if is_expected_article(
                result.get("article"),
                expected
            ):
                return rank

    return None


# ============================================================
# Metrics
# ============================================================

def hit_at_k(rank, k):

    if rank is None:
        return False

    return rank <= k


def reciprocal_rank(rank):

    if rank is None:
        return 0.0

    return 1.0 / rank


# ============================================================
# Evaluation
# ============================================================

print("=" * 80)
print("Loading evaluation dataset")
print("=" * 80)

with EVALUATION_FILE.open(
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)

print("Questions:", len(questions))
print()


results = []


for question_index, item in enumerate(
    questions,
    start=1
):

    question = item["question"]
    expected_articles = item["expected_articles"]

    print("=" * 80)
    print(
        f"QUESTION {question_index}/{len(questions)}"
    )
    print("=" * 80)

    print("Question:")
    print(question)

    print()
    print("Expected articles:")
    print(expected_articles)

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    semantic_results = semantic_search(
        question
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    keyword_results = bm25_search(
        question
    )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results
    )

    # --------------------------------------------------------
    # Ranks
    # --------------------------------------------------------

    semantic_rank = first_relevant_rank(
        semantic_results,
        expected_articles
    )

    keyword_rank = first_relevant_rank(
        keyword_results,
        expected_articles
    )

    hybrid_rank = first_relevant_rank(
        hybrid_results,
        expected_articles
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("First relevant rank:")

    print(
        f"  Semantic : {semantic_rank}"
    )

    print(
        f"  BM25     : {keyword_rank}"
    )

    print(
        f"  RRF      : {hybrid_rank}"
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "question": question,
        "expected_articles": expected_articles,

        "semantic": {
            "first_relevant_rank": semantic_rank
        },

        "bm25": {
            "first_relevant_rank": keyword_rank
        },

        "rrf": {
            "first_relevant_rank": hybrid_rank
        }
    })

    print()


# ============================================================
# Calculate aggregate metrics
# ============================================================

def calculate_metrics(method):

    ranks = [
        item[method]["first_relevant_rank"]
        for item in results
    ]

    total = len(ranks)

    metrics = {}

    for k in [1, 3, 5, 10]:

        hits = sum(
            1
            for rank in ranks
            if rank is not None and rank <= k
        )

        metrics[f"hit_at_{k}"] = (
            hits / total
            if total > 0
            else 0
        )

    reciprocal_ranks = [
        reciprocal_rank(rank)
        for rank in ranks
    ]

    metrics["mrr"] = (
        sum(reciprocal_ranks) / total
        if total > 0
        else 0
    )

    return metrics


# ============================================================
# Final metrics
# ============================================================

semantic_metrics = calculate_metrics(
    "semantic"
)

bm25_metrics = calculate_metrics(
    "bm25"
)

rrf_metrics = calculate_metrics(
    "rrf"
)


# ============================================================
# Print comparison
# ============================================================

print()
print()
print("=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

print()

print(
    f"{'Method':<15}"
    f"{'Hit@1':<12}"
    f"{'Hit@3':<12}"
    f"{'Hit@5':<12}"
    f"{'Hit@10':<12}"
    f"{'MRR':<12}"
)

print("-" * 80)


for name, metrics in [
    ("Semantic", semantic_metrics),
    ("BM25", bm25_metrics),
    ("RRF", rrf_metrics)
]:

    print(
        f"{name:<15}"
        f"{metrics['hit_at_1']:<12.3f}"
        f"{metrics['hit_at_3']:<12.3f}"
        f"{metrics['hit_at_5']:<12.3f}"
        f"{metrics['hit_at_10']:<12.3f}"
        f"{metrics['mrr']:<12.3f}"
    )


# ============================================================
# Save results
# ============================================================

output = {
    "collection": COLLECTION_NAME,
    "embedding_model": EMBEDDING_MODEL,
    "total_questions": len(results),

    "metrics": {
        "semantic": semantic_metrics,
        "bm25": bm25_metrics,
        "rrf": rrf_metrics
    },

    "questions": results
}


with OUTPUT_FILE.open(
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
print("=" * 80)
print("Evaluation completed.")
print("=" * 80)

print(
    f"Results saved to: {OUTPUT_FILE}"
)
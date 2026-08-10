from pathlib import Path
import re

import chromadb
import ollama
from rank_bm25 import BM25Okapi


# ============================================================
# Configuration
# ============================================================

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

TOP_K = 20


# ============================================================
# Questions to debug
# ============================================================

QUESTIONS = [
    {
        "question": "چه کسانی به دلیل قرابت نمی‌توانند با یکدیگر ازدواج کنند؟",
        "expected": "۱۰۴۵",
    },
    {
        "question": "چه ازدواج‌هایی به دلیل مصاهره ممنوع هستند؟",
        "expected": "۱۰۴۷",
    },
    {
        "question": "چه شروطی را می‌توان ضمن عقد ازدواج قرار داد؟",
        "expected": "۱۱۱۹",
    },
    {
        "question": "عده چیست و چه مفهومی دارد؟",
        "expected": "۱۱۵۰",
    },
]


# ============================================================
# Normalization
# ============================================================

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"


def normalize_digits(text):

    table = str.maketrans(
        PERSIAN_DIGITS + ARABIC_DIGITS,
        ENGLISH_DIGITS + ENGLISH_DIGITS
    )

    return str(text).translate(table)


def normalize_text(text):

    text = str(text)

    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("\u200c", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):

    text = normalize_text(text)

    return re.findall(
        r"[\w\u0600-\u06FF]+",
        text,
        flags=re.UNICODE
    )


# ============================================================
# ChromaDB
# ============================================================

print("Loading ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    "Documents:",
    collection.count()
)


# ============================================================
# Load all documents
# ============================================================

data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

ids = data["ids"]
documents = data["documents"]
metadatas = data["metadatas"]


# ============================================================
# BM25
# ============================================================

print("Building BM25...")

tokenized_documents = [
    tokenize(doc)
    for doc in documents
]

bm25 = BM25Okapi(
    tokenized_documents
)


# ============================================================
# Semantic search
# ============================================================

def semantic_search(query):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    embedding = response["embeddings"][0]

    result = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    output = []

    for i in range(
        len(result["ids"][0])
    ):

        output.append({
            "rank": i + 1,
            "id": result["ids"][0][i],
            "article": result["metadatas"][0][i].get(
                "article"
            ),
            "distance": result["distances"][0][i],
            "document": result["documents"][0][i],
        })

    return output


# ============================================================
# BM25 search
# ============================================================

def bm25_search(query):

    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    output = []

    for rank, index in enumerate(
        ranked_indices[:TOP_K],
        start=1
    ):

        output.append({
            "rank": rank,
            "id": ids[index],
            "article": metadatas[index].get(
                "article"
            ),
            "score": float(scores[index]),
            "document": documents[index],
        })

    return output


# ============================================================
# RRF
# ============================================================

def rrf(
    semantic_results,
    bm25_results,
    k=60
):

    scores = {}
    records = {}

    for result in semantic_results:

        doc_id = result["id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + result["rank"])
        )

        records[doc_id] = result

    for result in bm25_results:

        doc_id = result["id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + result["rank"])
        )

        if doc_id not in records:
            records[doc_id] = result

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    output = []

    for rank, (doc_id, score) in enumerate(
        ranked,
        start=1
    ):

        result = records[doc_id].copy()

        result["rank"] = rank
        result["rrf_score"] = score

        output.append(result)

    return output


# ============================================================
# Print results
# ============================================================

def print_results(
    title,
    results,
    expected
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for result in results:

        article = result.get("article")

        marker = ""

        if (
            article is not None
            and normalize_digits(article)
            == normalize_digits(expected)
        ):
            marker = "  <<< EXPECTED"

        print(
            f"{result['rank']:>3}. "
            f"Article={article!s:<8} "
            f"ID={result['id']}"
            f"{marker}"
        )

        document = (
            result.get("document", "")
            .replace("\n", " ")
        )

        print(
            "     "
            + document[:250]
        )


# ============================================================
# Main
# ============================================================

for item in QUESTIONS:

    question = item["question"]
    expected = item["expected"]

    print()
    print()
    print("#" * 100)
    print("QUESTION")
    print("#" * 100)

    print(question)

    print()
    print(
        "EXPECTED ARTICLE:",
        expected
    )

    semantic_results = semantic_search(
        question
    )

    bm25_results = bm25_search(
        question
    )

    rrf_results = rrf(
        semantic_results,
        bm25_results
    )

    print_results(
        "SEMANTIC TOP 20",
        semantic_results,
        expected
    )

    print_results(
        "BM25 TOP 20",
        bm25_results,
        expected
    )

    print_results(
        "RRF TOP 20",
        rrf_results[:20],
        expected
    )
import re

import chromadb
import ollama
from rank_bm25 import BM25Okapi


# =========================
# Configuration
# =========================

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

SEMANTIC_TOP_K = 30
KEYWORD_TOP_K = 30
FINAL_TOP_K = 20


# =========================
# Persian / English digits
# =========================

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"


def normalize_digits(text):
    """
    Convert Persian and Arabic digits to English digits.
    """

    translation_table = str.maketrans(
        PERSIAN_DIGITS + ARABIC_DIGITS,
        ENGLISH_DIGITS + ENGLISH_DIGITS
    )

    return text.translate(translation_table)


def to_persian_digits(text):
    """
    Convert English and Arabic digits to Persian digits.
    """

    english_to_persian = str.maketrans(
        "0123456789",
        "۰۱۲۳۴۵۶۷۸۹"
    )

    arabic_to_persian = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩",
        "۰۱۲۳۴۵۶۷۸۹"
    )

    text = text.translate(english_to_persian)
    text = text.translate(arabic_to_persian)

    return text


# =========================
# Extract article number
# =========================

def extract_article_number(query):
    """
    Detect article numbers such as:

    ماده ۱۰۶۷
    ماده‌ی ۱۰۶۷
    ماده 1067
    Article 1067
    article 1067
    """

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


# =========================
# Connect to ChromaDB
# =========================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# =========================
# Legal version helpers
# =========================

def is_deleted_version(version):
    """
    Check whether a legal version is marked as deleted.
    """

    if not version:
        return False

    return (
        version.startswith("حذفی")
        or version.startswith("حذف")
    )


def version_date(version):
    """
    Extract the date from a version string.

    Examples:
        اصلاحی ۱۳۸۱/۴/۱
        مصوب ۱۳۰۷/۲/۱۸

    Returns:
        tuple(year, month, day)
        or None
    """

    if not version:
        return None

    normalized = normalize_digits(version)

    match = re.search(
        r"(\d{3,4})/(\d{1,2})/(\d{1,2})",
        normalized
    )

    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))

    return year, month, day


def select_current_version(results):
    """
    Select the legally relevant version of an article.

    Priority:
    1. current
    2. latest dated version

    Deleted versions are ignored.
    """

    if not results:
        return []

    # Remove deleted versions
    valid_results = []

    for result in results:
        version = result["metadata"].get(
            "version",
            ""
        )

        if is_deleted_version(version):
            continue

        valid_results.append(result)

    if not valid_results:
        return []

    # Prefer current version
    current_results = [
        result
        for result in valid_results
        if result["metadata"].get("version") == "current"
    ]

    if current_results:
        return current_results

    # No current version:
    # select latest dated version
    dated_results = []

    for result in valid_results:
        version = result["metadata"].get(
            "version",
            ""
        )

        date = version_date(version)

        if date is not None:
            dated_results.append(
                (date, result)
            )

    if not dated_results:
        return []

    dated_results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [dated_results[0][1]]


# =========================
# Persian text normalization
# =========================

def normalize_text(text):
    """
    Normalize Persian/Arabic text for keyword matching.
    """

    if not text:
        return ""

    text = normalize_digits(text)

    # Arabic Yeh -> Persian Yeh
    text = text.replace("ي", "ی")

    # Arabic Kaf -> Persian Kaf
    text = text.replace("ك", "ک")

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text
    )

    # Normalize half-space
    text = text.replace("\u200c", " ")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize_persian(text):
    """
    Simple tokenization for Persian legal text.
    """

    text = normalize_text(text)

    return re.findall(
        r"[\w\u0600-\u06FF]+",
        text
    )


# =========================
# Exact article search
# =========================

def exact_article_search(article_number):
    """
    Retrieve an exact legal article by article number.
    """

    article_number_persian = to_persian_digits(
        article_number
    )

    exact = collection.get(
        where={
            "article": article_number_persian
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    exact_results = []

    if not exact["ids"]:
        return []

    for i in range(len(exact["ids"])):
        exact_results.append({
            "id": exact["ids"][i],
            "document": exact["documents"][i],
            "metadata": exact["metadatas"][i],
            "distance": 0.0,
            "retrieval_type": "exact"
        })

    return select_current_version(
        exact_results
    )


# =========================
# Semantic Search
# =========================

def semantic_search(query, top_k=SEMANTIC_TOP_K):
    """
    Semantic search using Qwen embeddings and ChromaDB.
    """

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )

    query_embedding = response["embedding"]

    semantic = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    raw_results = []

    for i in range(len(semantic["documents"][0])):
        raw_results.append({
            "id": semantic["ids"][0][i],
            "document": semantic["documents"][0][i],
            "metadata": semantic["metadatas"][0][i],
            "distance": semantic["distances"][0][i],
            "retrieval_type": "semantic"
        })

    # Group by article
    results_by_article = {}

    for result in raw_results:
        article = result["metadata"].get(
            "article"
        )

        if article is None:
            continue

        if article not in results_by_article:
            results_by_article[article] = []

        results_by_article[article].append(
            result
        )

    # Resolve current legal version
    final_results = []

    for _, results in results_by_article.items():
        selected = select_current_version(
            results
        )

        final_results.extend(
            selected
        )

    return final_results


# =========================
# Keyword Search
# =========================

def keyword_search(query, top_k=KEYWORD_TOP_K):
    """
    BM25 keyword search over legal articles.
    """

    all_data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]

    if not documents:
        return []

    # Build BM25 corpus
    tokenized_corpus = [
        tokenize_persian(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    query_tokens = tokenize_persian(
        query
    )

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    raw_results = []

    for index in ranked_indices[:top_k]:
        raw_results.append({
            "id": ids[index],
            "document": documents[index],
            "metadata": metadatas[index],
            "keyword_score": float(
                scores[index]
            ),
            "distance": None,
            "retrieval_type": "keyword"
        })

    # Group by article
    results_by_article = {}

    for result in raw_results:
        article = result["metadata"].get(
            "article"
        )

        if article is None:
            continue

        if article not in results_by_article:
            results_by_article[article] = []

        results_by_article[article].append(
            result
        )

    # Resolve current legal version
    final_results = []

    for _, results in results_by_article.items():
        selected = select_current_version(
            results
        )

        final_results.extend(
            selected
        )

    final_results.sort(
        key=lambda x: x["keyword_score"],
        reverse=True
    )

    return final_results[:top_k]


# =========================
# Reciprocal Rank Fusion
# =========================

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    exact_results=None,
    k=60
):
    """
    Combine semantic and keyword rankings using
    Reciprocal Rank Fusion (RRF).

    RRF score:
        1 / (k + rank)

    Exact article matches always receive highest priority.
    """

    if exact_results is None:
        exact_results = []

    fused = {}

    # Exact results
    for result in exact_results:
        result_id = result["id"]

        if result_id not in fused:
            fused[result_id] = {
                "result": result,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "keyword_rank": None,
                "exact": True
            }

    # Semantic ranking
    for rank, result in enumerate(
        semantic_results,
        start=1
    ):
        result_id = result["id"]

        if result_id not in fused:
            fused[result_id] = {
                "result": result,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "keyword_rank": None,
                "exact": False
            }

        fused[result_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

        fused[result_id]["semantic_rank"] = rank

    # Keyword ranking
    for rank, result in enumerate(
        keyword_results,
        start=1
    ):
        result_id = result["id"]

        if result_id not in fused:
            fused[result_id] = {
                "result": result,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "keyword_rank": None,
                "exact": False
            }

        fused[result_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

        fused[result_id]["keyword_rank"] = rank

    ranked = sorted(
        fused.values(),
        key=lambda x: (
            x["exact"],
            x["rrf_score"]
        ),
        reverse=True
    )

    final_results = []

    for item in ranked:
        result = item["result"].copy()

        result["rrf_score"] = item[
            "rrf_score"
        ]

        result["semantic_rank"] = item[
            "semantic_rank"
        ]

        result["keyword_rank"] = item[
            "keyword_rank"
        ]

        final_results.append(
            result
        )

    return final_results[:FINAL_TOP_K]


# =========================
# Main Retrieval Function
# =========================

def retrieve(query):
    """
    Main retrieval pipeline.

    Flow:

    Exact article query:
        Exact lookup

    Normal query:
        Semantic search
        +
        BM25 keyword search
        +
        Reciprocal Rank Fusion
    """

    if not query:
        return []

    query = query.strip()

    if not query:
        return []

    article_number = extract_article_number(
        query
    )

    exact_results = []
    semantic_results = []
    keyword_results = []

    # Exact article lookup
    if article_number is not None:
        exact_results = exact_article_search(
            article_number
        )

    # Normal hybrid search
    else:
        semantic_results = semantic_search(
            query,
            top_k=SEMANTIC_TOP_K
        )

        keyword_results = keyword_search(
            query,
            top_k=KEYWORD_TOP_K
        )

    merged_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        exact_results=exact_results
    )

    return merged_results


# =========================
# Diagnostics
# =========================

def print_results(query, results):
    """
    Print retrieval results for manual testing.
    """

    print()
    print("=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print()
    print("=" * 70)
    print("FINAL RETRIEVED RESULTS")
    print("=" * 70)

    if not results:
        print("No relevant results found.")
        return

    for i, result in enumerate(
        results,
        start=1
    ):
        metadata = result["metadata"]

        print()
        print(f"Result #{i}")
        print("-" * 70)

        print(
            f"Article: "
            f"{metadata.get('article')}"
        )

        print(
            f"Version: "
            f"{metadata.get('version')}"
        )

        print(
            f"Chunk: "
            f"{metadata.get('chunk_index')}"
        )

        print(
            f"Retrieval type: "
            f"{result.get('retrieval_type')}"
        )

        print(
            f"Semantic rank: "
            f"{result.get('semantic_rank')}"
        )

        print(
            f"Keyword rank: "
            f"{result.get('keyword_rank')}"
        )

        print(
            f"RRF score: "
            f"{result.get('rrf_score', 0):.6f}"
        )

        print(
            f"Distance: "
            f"{result.get('distance')}"
        )

        print()
        print("Text:")
        print(
            result["document"]
        )


# =========================
# Command-line Test
# =========================

if __name__ == "__main__":

    query = input(
        "Enter your legal question: "
    ).strip()

    if not query:
        print(
            "Question cannot be empty."
        )
        raise SystemExit

    results = retrieve(
        query
    )

    print_results(
        query,
        results
    )
    
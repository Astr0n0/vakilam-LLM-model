import re

import chromadb
import ollama


# =========================
# Configuration
# =========================

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

SEMANTIC_TOP_K = 10
FINAL_TOP_K = 10


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
# Get user question
# =========================

query = input("Enter your legal question: ").strip()

if not query:
    print("Question cannot be empty.")
    raise SystemExit


print()
print("=" * 70)
print("QUERY")
print("=" * 70)
print(query)


# =========================
# Exact Article Lookup
# =========================

article_number = extract_article_number(query)

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

    return (year, month, day)


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

    # ---------------------------------
    # Remove deleted versions
    # ---------------------------------

    valid_results = []

    for result in results:

        version = result["metadata"].get(
            "version", ""
        )

        if is_deleted_version(version):
            continue

        valid_results.append(result)

    if not valid_results:
        return []

    # ---------------------------------
    # Prefer current version
    # ---------------------------------

    current_results = [
        result
        for result in valid_results
        if result["metadata"].get("version") == "current"
    ]

    if current_results:
        return current_results

    # ---------------------------------
    # No current version:
    # select latest dated version
    # ---------------------------------

    dated_results = []

    for result in valid_results:

        version = result["metadata"].get(
            "version", ""
        )

        date = version_date(version)

        if date is not None:
            dated_results.append(
                (date, result)
            )

    if not dated_results:
        return []

    # Latest date wins
    dated_results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [dated_results[0][1]]


exact_results = []


if article_number is not None:

    print()
    print("=" * 70)
    print("EXACT ARTICLE SEARCH")
    print("=" * 70)

    print(f"Detected article: {article_number}")

    # Convert English digits to Persian digits
    # because ChromaDB stores article numbers in Persian digits.
    article_number_persian = to_persian_digits(
        article_number
    )

    print(
        f"Normalized article: {article_number_persian}"
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

    if exact["ids"]:

        for i in range(len(exact["ids"])):

            exact_results.append({
                "id": exact["ids"][i],
                "document": exact["documents"][i],
                "metadata": exact["metadatas"][i],
                "distance": 0.0,
                "retrieval_type": "exact"
            })

        print(
            f"Exact matches found: {len(exact_results)}"
        )

        # ---------------------------------
        # Resolve legal version
        # ---------------------------------

        exact_results = select_current_version(
            exact_results
        )

        print(
            f"Exact matches after version filtering: "
            f"{len(exact_results)}"
        )

    else:

        print("No exact article match found.")


# =========================
# Semantic Search
# =========================

semantic_results = []


if article_number is None:

    print()
    print("=" * 70)
    print("SEMANTIC SEARCH")
    print("=" * 70)

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )

    query_embedding = response["embedding"]

    semantic = collection.query(
        query_embeddings=[query_embedding],
        n_results=SEMANTIC_TOP_K
    )

    # ---------------------------------
    # Build raw semantic results
    # ---------------------------------

    raw_semantic_results = []

    for i in range(len(semantic["documents"][0])):

        raw_semantic_results.append({
            "id": semantic["ids"][0][i],
            "document": semantic["documents"][0][i],
            "metadata": semantic["metadatas"][0][i],
            "distance": semantic["distances"][0][i],
            "retrieval_type": "semantic"
        })

    print(
        f"Semantic matches found: "
        f"{len(raw_semantic_results)}"
    )

    # ---------------------------------
    # Group results by article
    # ---------------------------------

    results_by_article = {}

    for result in raw_semantic_results:

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

    # ---------------------------------
    # Resolve legal version
    # ---------------------------------

    for article, results in results_by_article.items():

        selected = select_current_version(
            results
        )

        semantic_results.extend(
            selected
        )

    print(
        f"Semantic results after "
        f"version filtering: "
        f"{len(semantic_results)}"
    )


else:

    print()
    print("=" * 70)
    print("SEMANTIC SEARCH SKIPPED")
    print("=" * 70)

    print(
        "Exact article query detected. "
        "Semantic search is not needed."
    )


# =========================
# Merge Results
# =========================

merged_results = []

seen_ids = set()

for result in exact_results:

    if result["id"] not in seen_ids:

        merged_results.append(result)
        seen_ids.add(result["id"])


for result in semantic_results:

    if result["id"] not in seen_ids:

        merged_results.append(result)
        seen_ids.add(result["id"])


# Limit final results
merged_results = merged_results[:FINAL_TOP_K]


# =========================
# Display Final Results
# =========================

print()
print("=" * 70)
print("FINAL RETRIEVED RESULTS")
print("=" * 70)


if not merged_results:

    print("No relevant results found.")
    raise SystemExit


for i, result in enumerate(merged_results):

    metadata = result["metadata"]

    print()
    print(f"Result #{i + 1}")
    print("-" * 70)

    print(
        f"Article: {metadata['article']}"
    )

    print(
        f"Version: {metadata['version']}"
    )

    print(
        f"Chunk: {metadata['chunk_index']}"
    )

    print(
        f"Retrieval type: {result['retrieval_type']}"
    )

    print(
        f"Distance: {result['distance']}"
    )

    print()
    print("Text:")
    print(result["document"])
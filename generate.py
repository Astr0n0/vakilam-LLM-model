import ollama

from retrieve import retrieve


# =========================
# Configuration
# =========================

LLM_MODEL = "qwen3:8b"

CONTEXT_TOP_K = 10


# =========================
# System prompt
# =========================

SYSTEM_PROMPT = """
شما یک دستیار حقوقی فارسی هستید.

وظیفه شما پاسخ دادن به پرسش‌های حقوقی فقط بر اساس
منابع قانونی ارائه‌شده در CONTEXT است.

قواعد مهم:

1. فقط بر اساس اطلاعات موجود در CONTEXT پاسخ دهید.

2. اگر اطلاعات کافی برای پاسخ وجود ندارد، بدون هیچ توضیح اضافه‌ای بگویید:
«اطلاعات کافی در منابع بازیابی‌شده برای پاسخ دقیق وجود ندارد.»

3. هیچ ماده قانونی، شماره ماده، حکم یا واقعیت حقوقی را از خودتان ایجاد نکنید.

4. در پاسخ، شماره مواد قانونی مرتبط را در صورت وجود ذکر کنید.

5. پاسخ را به زبان فارسی و واضح ارائه کنید.

6. ابتدا پاسخ مستقیم به سؤال را بدهید.

7. سپس مستندات قانونی مرتبط را ذکر کنید.

8. بین متن صریح قانون و استنباط یا توضیح خودتان تفاوت قائل شوید.

9. اگر CONTEXT با سؤال ارتباط کافی ندارد، از دانش عمومی یا دانش داخلی مدل
برای تکمیل پاسخ استفاده نکنید.
"""


# =========================
# Build context
# =========================

def build_context(results, top_k=CONTEXT_TOP_K):
    """
    Convert retrieval results into a context
    that can be given to the language model.
    """

    context_parts = []

    for i, result in enumerate(
        results[:top_k],
        start=1
    ):
        metadata = result["metadata"]

        article = metadata.get(
            "article",
            "unknown"
        )

        version = metadata.get(
            "version",
            "unknown"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "unknown"
        )

        retrieval_type = result.get(
            "retrieval_type",
            "unknown"
        )

        semantic_rank = result.get(
            "semantic_rank"
        )

        keyword_rank = result.get(
            "keyword_rank"
        )

        rrf_score = result.get(
            "rrf_score",
            0
        )

        document = result.get(
            "document",
            ""
        )

        context_parts.append(
            f"""
[Source {i}]

Article: {article}
Version: {version}
Chunk: {chunk_index}
Retrieval type: {retrieval_type}
Semantic rank: {semantic_rank}
Keyword rank: {keyword_rank}
RRF score: {rrf_score}

Text:
{document}
"""
        )

    return "\n".join(
        context_parts
    )


# =========================
# Generate legal answer
# =========================

def generate_answer(query):
    """
    Retrieve legal sources and generate
    an answer using Qwen.
    """

    if not query:
        return {
            "answer": None,
            "sources": []
        }

    query = query.strip()

    if not query:
        return {
            "answer": None,
            "sources": []
        }

    # -------------------------
    # Retrieval
    # -------------------------

    results = retrieve(
        query
    )

    if not results:
        return {
            "answer": (
                "اطلاعات کافی در منابع بازیابی‌شده "
                "برای پاسخ دقیق وجود ندارد."
            ),
            "sources": []
        }

    # -------------------------
    # Context
    # -------------------------

    context = build_context(
        results
    )

    # -------------------------
    # User prompt
    # -------------------------

    user_prompt = f"""
پرسش کاربر:

{query}


CONTEXT:

{context}


فقط بر اساس منابع بالا به پرسش پاسخ بده.
"""

    # -------------------------
    # LLM generation
    # -------------------------

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    answer = response[
        "message"
    ][
        "content"
    ]

    return {
        "answer": answer,
        "sources": results[:CONTEXT_TOP_K]
    }


# =========================
# Display sources
# =========================

def print_sources(sources):
    """
    Display retrieved legal sources
    for command-line testing.
    """

    print()
    print("=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    if not sources:
        print(
            "No sources retrieved."
        )
        return

    for i, result in enumerate(
        sources,
        start=1
    ):
        metadata = result[
            "metadata"
        ]

        print(
            f"{i}. "
            f"Article "
            f"{metadata.get('article')} "
            f"| Version: "
            f"{metadata.get('version')} "
            f"| Type: "
            f"{result.get('retrieval_type')} "
            f"| Semantic rank: "
            f"{result.get('semantic_rank')} "
            f"| Keyword rank: "
            f"{result.get('keyword_rank')} "
            f"| RRF: "
            f"{result.get('rrf_score', 0):.6f}"
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

    print()
    print(
        "Searching legal database..."
    )

    result = generate_answer(
        query
    )

    print()
    print("=" * 70)
    print("LEGAL ANSWER")
    print("=" * 70)

    print(
        result["answer"]
    )

    print_sources(
        result["sources"]
    )

import ollama

from answerability import classify_answerability
from citation_validator import validate_citations
from retrieve import retrieve
from scope_guard import classify_scope


# =========================
# Configuration
# =========================

LLM_MODEL = "qwen3:8b"

CONTEXT_TOP_K = 6

MAX_GENERATION_ATTEMPTS = 2


# =========================
# System Prompt
# =========================

SYSTEM_PROMPT = """
شما یک دستیار حقوقی فارسی هستید.

وظیفه شما پاسخ دادن به پرسش کاربر فقط و فقط بر اساس
منابع قانونی ارائه‌شده در CONTEXT است.

قواعد الزامی:

1. فقط از اطلاعاتی استفاده کنید که صراحتاً در CONTEXT وجود دارد.

2. از دانش داخلی، اطلاعات عمومی یا دانش حقوقی خارج از CONTEXT
برای تکمیل پاسخ استفاده نکنید.

3. هیچ ماده قانونی، شماره ماده، حکم، شرط یا واقعیت حقوقی را
از خودتان ایجاد نکنید.

4. فقط شماره موادی را ذکر کنید که دقیقاً در CONTEXT وجود دارند.

5. شماره ماده را دقیقاً همان‌طور که در فیلد Article منبع آمده است
ذکر کنید و هرگز آن را حدس نزنید یا تغییر ندهید.

6. اگر یک نتیجه‌گیری مستقیماً از متن منابع قابل اثبات نیست،
آن را به عنوان حکم قطعی بیان نکنید.

7. از تعمیم دادن حکم یک ماده به موضوع دیگر خودداری کنید،
مگر اینکه خود CONTEXT صراحتاً چنین ارتباطی را بیان کرده باشد.

8. اگر اطلاعات کافی برای پاسخ دقیق وجود ندارد، فقط بگویید:
«اطلاعات کافی در منابع بازیابی‌شده برای پاسخ دقیق وجود ندارد.»

9. ابتدا پاسخ مستقیم و مختصر به سؤال بدهید.

10. سپس فقط مواد قانونی‌ای را که واقعاً برای پاسخ استفاده کرده‌اید
در بخش «مستندات قانونی» ذکر کنید.

11. اگر بخشی از پاسخ توضیح یا برداشت از متن قانون است،
آن را صریحاً با عبارت «بر اساس متن منابع می‌توان گفت» مشخص کنید.

12. پاسخ را به زبان فارسی، روشن و بدون اضافه‌گویی ارائه کنید.

13. هر ادعای حقوقی باید مستقیماً به همان ماده‌ای متصل باشد
که متن آن ادعا را بیان کرده است.

14. اطلاعات یک ماده را به ماده دیگری نسبت ندهید.

15. اگر ماده‌ای فقط درباره «معامله» صحبت می‌کند، آن را به نکاح
تعمیم ندهید مگر اینکه خود CONTEXT این ارتباط را صراحتاً بیان کند.

16. برای هر مورد پاسخ، ابتدا متن یا مفهوم صریح منبع را بررسی کنید
و سپس فقط شماره همان ماده را ذکر کنید.

17. اگر درباره ارتباط یک ماده با پرسش مطمئن نیستید،
آن ماده را در پاسخ استفاده نکنید.
"""


# =========================
# Build Context
# =========================

def build_context(
    results,
    top_k=CONTEXT_TOP_K
):
    """
    Convert retrieval results into context
    for the language model.
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
# Build User Prompt
# =========================

def build_user_prompt(
    query,
    context,
    retry=False
):
    """
    Build the prompt used for answer generation.
    """

    retry_instruction = ""

    if retry:
        retry_instruction = """
پاسخ قبلی دارای ارجاع نامعتبر به ماده قانونی بوده است.

این بار بسیار سخت‌گیرانه عمل کن:

- فقط شماره موادی را ذکر کن که در CONTEXT وجود دارند.
- شماره هیچ ماده‌ای را حدس نزن.
- هیچ ماده جدیدی ایجاد نکن.
- فقط ادعاهایی را بیان کن که مستقیماً از متن منابع پشتیبانی می‌شوند.
- اگر ارتباط یک ماده با سؤال روشن نیست، آن ماده را استفاده نکن.
"""

    return f"""
پرسش کاربر:

{query}


CONTEXT:

{context}


فقط بر اساس منابع بالا به پرسش پاسخ بده.

شماره هیچ ماده‌ای را که در CONTEXT وجود ندارد ذکر نکن.
شماره مواد را عیناً از فیلد Article منابع بردار.
هیچ حکم حقوقی خارج از متن منابع اضافه نکن.

برای هر ادعا فقط از همان منبعی استفاده کن که آن ادعا صراحتاً
در متن آن آمده است.

مثال:
اگر یک منبع درباره «توالی ایجاب و قبول» صحبت می‌کند،
نباید «رضایت طرفین» را به همان ماده نسبت بدهی.

از تعمیم مواد عمومی به موضوع سؤال خودداری کن مگر اینکه
ارتباط آن در CONTEXT صراحتاً ذکر شده باشد.

{retry_instruction}
"""


# =========================
# Generate One Answer
# =========================

def generate_once(
    query,
    context,
    retry=False
):
    """
    Generate one candidate answer.
    """

    user_prompt = build_user_prompt(
        query=query,
        context=context,
        retry=retry
    )

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
        ],
        options={
            "temperature": 0
        }
    )

    return response[
        "message"
    ][
        "content"
    ]


# =========================
# Generate Legal Answer
# =========================

def generate_answer(query):
    """
    Full Vakilam AI pipeline.

    Flow:
        input validation
        scope classification
        retrieval
        answerability
        generation
        citation validation
    """

    # -------------------------
    # Input Validation
    # -------------------------

    if not query:
        return {
            "answer": None,
            "sources": [],
            "status": "invalid_input"
        }

    query = query.strip()

    if not query:
        return {
            "answer": None,
            "sources": [],
            "status": "invalid_input"
        }

    # -------------------------
    # Scope Guard
    # -------------------------

    scope = classify_scope(
        query
    )

    if scope != "ALLOWED":
        return {
            "answer": (
                "این دستیار فقط به پرسش‌های مرتبط "
                "با حوزه حقوقی پاسخ می‌دهد."
            ),
            "sources": [],
            "status": "out_of_scope"
        }

    # -------------------------
    # Retrieval
    # -------------------------

    results = retrieve(
        query
    )

    # -------------------------
    # Answerability Gate
    # -------------------------

    answerability = (
        classify_answerability(
            results
        )
    )

    if answerability != "ANSWERABLE":
        return {
            "answer": (
                "اطلاعات کافی در منابع موجود "
                "برای پاسخ دقیق به این پرسش وجود ندارد."
            ),
            "sources": [],
            "status": "insufficient_context"
        }

    # -------------------------
    # Context
    # -------------------------

    sources = results[
        :CONTEXT_TOP_K
    ]

    context = build_context(
        sources
    )

    # -------------------------
    # Generation + Citation Validation
    # -------------------------

    for attempt in range(
        MAX_GENERATION_ATTEMPTS
    ):
        retry = attempt > 0

        answer = generate_once(
            query=query,
            context=context,
            retry=retry
        )

        citation_result = validate_citations(
            answer,
            sources
        )

        if citation_result["valid"]:
            return {
                "answer": answer,
                "sources": sources,
                "status": "answered"
            }

    # -------------------------
    # Citation Validation Failed
    # -------------------------

    return {
        "answer": (
            "اطلاعات کافی در منابع موجود "
            "برای ارائه پاسخ دقیق و قابل استناد "
            "وجود ندارد."
        ),
        "sources": [],
        "status": "insufficient_context"
    }


# =========================
# Display Sources
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
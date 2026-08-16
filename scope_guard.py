import ollama


# =========================
# Configuration
# =========================

CLASSIFIER_MODEL = "qwen3:8b"


# =========================
# Scope Prompt
# =========================

SCOPE_PROMPT = """
شما فقط یک طبقه‌بند هستید و نباید به سؤال کاربر پاسخ دهید.

هدف شما تشخیص این است که آیا سؤال کاربر در حوزه حقوقی
و قابل بررسی توسط دستیار حقوقی وکیلم است یا خیر.

سؤال ALLOWED است اگر درباره موضوعات حقوقی باشد، مانند:

- قانون
- قوانین مدنی
- مواد قانونی
- قراردادها
- معاملات
- مالکیت
- تعهدات
- مسئولیت
- نکاح و ازدواج
- طلاق
- خانواده
- ارث
- وصیت
- اموال
- حقوق و تکالیف اشخاص
- مسائل و پرسش‌های حقوقی مشابه
- شکایت و طرح دعوا
- مسئولیت مدنی و کیفری
- تخلفات حرفه‌ای
- شکایت از پزشک، وکیل، کارفرما یا اشخاص و نهادها
- جرایم رایانه‌ای و شکایت‌های مرتبط

سؤال DENIED است اگر مربوط به حوزه حقوقی نباشد، مانند:

- آشپزی
- ورزش
- برنامه‌نویسی
- تکنولوژی
- پزشکی
- سرگرمی
- آب و هوا
- مسائل عمومی غیرحقوقی
- سؤال‌های روزمره نامرتبط

قواعد:

1. به سؤال پاسخ نده.
2. هیچ توضیحی ارائه نکن.
3. فقط یکی از این دو عبارت را خروجی بده:

ALLOWED

یا

DENIED

4. اگر سؤال درباره اقدام قانونی، شکایت، جرم، مسئولیت، حق، تعهد،
مطالبه، دادگاه یا پیگیری حقوقی باشد، حتی اگر موضوع اصلی آن
پزشکی، فناوری، شغل یا حوزه دیگری باشد، سؤال را ALLOWED در نظر بگیر.
"""


# =========================
# Scope Classification
# =========================

def classify_scope(query):
    """
    Classify whether a user question
    belongs to the allowed legal domain.
    """

    if not query:
        return "DENIED"

    query = query.strip()

    if not query:
        return "DENIED"

    response = ollama.chat(
        model=CLASSIFIER_MODEL,
        messages=[
            {
                "role": "system",
                "content": SCOPE_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ],
        options={
            "temperature": 0
        }
    )

    decision = response[
        "message"
    ][
        "content"
    ].strip().upper()

    if "ALLOWED" in decision:
        return "ALLOWED"

    return "DENIED"


def is_in_scope(query):
    """
    Return True if the question
    is inside the allowed legal scope.
    """

    return classify_scope(query) == "ALLOWED"


# =========================
# Command-line Test
# =========================

if __name__ == "__main__":

    query = input(
        "Enter a question: "
    ).strip()

    decision = classify_scope(
        query
    )

    print()
    print("=" * 70)
    print("SCOPE DECISION")
    print("=" * 70)
    print(decision)
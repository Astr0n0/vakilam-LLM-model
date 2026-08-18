import json
import re

import ollama

from retrieve import normalize_digits


# =========================
# Configuration
# =========================

VALIDATOR_MODEL = "qwen3:8b"


# =========================
# Helpers
# =========================

def extract_article_claims(answer):
    """
    Extract lines or bullet points that cite
    a legal article.

    Returns:
        [
            {
                "article": "1067",
                "claim": "..."
            }
        ]
    """

    if not answer:
        return []

    claims = []

    lines = answer.splitlines()

    for line in lines:
        normalized_line = normalize_digits(
            line
        )

        matches = re.findall(
            r"ماده\s+(\d+)",
            normalized_line
        )

        for article in matches:
            claims.append({
                "article": article,
                "claim": line.strip()
            })

    return claims


def build_source_map(sources):
    """
    Map article number to its retrieved text.
    """

    source_map = {}

    for source in sources:
        metadata = source.get(
            "metadata",
            {}
        )

        article = metadata.get(
            "article"
        )

        if article is None:
            continue

        normalized_article = normalize_digits(
            str(article)
        )

        document = source.get(
            "document",
            ""
        )

        source_map[
            normalized_article
        ] = document

    return source_map


# =========================
# Claim validation
# =========================

def validate_claim(
    claim,
    article,
    source_text
):
    """
    Ask the validator model whether the claim
    is directly supported by the cited source.
    """

    prompt = f"""
شما فقط یک ارزیاب هستید.

وظیفه شما این است که بررسی کنید آیا ادعای زیر
واقعاً و مستقیماً توسط متن ماده قانونی ارائه‌شده
پشتیبانی می‌شود یا خیر.

شماره ماده:
{article}

ادعا:
{claim}

متن منبع:
{source_text}

قواعد:

1. فقط بر اساس متن منبع تصمیم بگیر.
2. هیچ دانش حقوقی خارج از متن منبع استفاده نکن.
3. اگر ادعا شامل تعمیم، برداشت اضافی یا نتیجه‌ای باشد
که مستقیماً در متن منبع وجود ندارد، آن را UNSUPPORTED بدان.
4. اگر ادعا محتوای منبع را به ماده دیگری نسبت داده باشد،
آن را UNSUPPORTED بدان.
5. فقط یکی از دو عبارت زیر را خروجی بده:

SUPPORTED

یا

UNSUPPORTED
"""

    response = ollama.chat(
        model=VALIDATOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
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

    if "SUPPORTED" == decision:
        return True

    return False


# =========================
# Grounding validation
# =========================

def validate_grounding(
    answer,
    sources
):
    """
    Validate claims that cite legal articles.

    Returns:
        {
            "valid": bool,
            "claims": [...],
            "unsupported_claims": [...]
        }
    """

    claims = extract_article_claims(
        answer
    )

    source_map = build_source_map(
        sources
    )

    checked_claims = []
    unsupported_claims = []

    for item in claims:
        article = item[
            "article"
        ]

        claim = item[
            "claim"
        ]

        source_text = source_map.get(
            article
        )

        if source_text is None:
            result = {
                "article": article,
                "claim": claim,
                "supported": False,
                "reason": "source_not_found"
            }

            checked_claims.append(
                result
            )

            unsupported_claims.append(
                result
            )

            continue

        supported = validate_claim(
            claim=claim,
            article=article,
            source_text=source_text
        )

        result = {
            "article": article,
            "claim": claim,"supported": supported
        }

        checked_claims.append(
            result
        )

        if not supported:
            unsupported_claims.append(
                result
            )

    return {
        "valid": (
            len(unsupported_claims) == 0
        ),
        "claims": checked_claims,
        "unsupported_claims": (
            unsupported_claims
        )
    }
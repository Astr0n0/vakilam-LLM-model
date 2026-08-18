import json
import os
import re
import sys


# =========================
# Project root
# =========================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from generate import generate_answer
from retrieve import normalize_digits


# =========================
# Configuration
# =========================

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "generation_grounding_questions.json"
)

RUNS_PER_QUESTION = 2


# =========================
# Helpers
# =========================

def normalize_article(article):
    if article is None:
        return None

    return normalize_digits(
        str(article)
    )


def extract_cited_articles(answer):
    """
    Extract article numbers explicitly cited
    in expressions such as:

    ماده ۱۰۶۷
    ماده 1067
    """

    if not answer:
        return set()

    normalized_answer = normalize_digits(
        answer
    )

    matches = re.findall(
    r"(?:ماده|article)\s+(\d+)",
    normalized_answer,
    flags=re.IGNORECASE
    )

    return set(matches)


def extract_source_articles(sources):
    articles = set()

    for source in sources:
        metadata = source.get(
            "metadata",
            {}
        )

        article = metadata.get(
            "article"
        )

        if article is not None:
            articles.add(
                normalize_article(article)
            )

    return articles


# =========================
# Load tests
# =========================

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    tests = json.load(file)


# =========================
# Evaluation
# =========================

total_runs = 0
passed_runs = 0

failures = []


for test_index, test in enumerate(
    tests,
    start=1
):
    question = test["question"]

    expected_status = test[
        "expected_status"
    ]

    allowed_articles = {
        normalize_article(article)
        for article in test.get(
            "allowed_articles",
            []
        )
    }

    forbidden_articles = {
        normalize_article(article)
        for article in test.get(
            "forbidden_articles",
            []
        )
    }

    for run_number in range(
        1,
        RUNS_PER_QUESTION + 1
    ):
        total_runs += 1

        print()
        print("=" * 70)
        print(
            f"TEST {test_index}/{len(tests)} "
            f"- RUN {run_number}/{RUNS_PER_QUESTION}"
        )
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        result = generate_answer(
            question
        )

        actual_status = result.get(
            "status"
        )

        answer = result.get(
            "answer"
        ) or ""

        sources = result.get(
            "sources",
            []
        )

        cited_articles = (
            extract_cited_articles(
                answer
            )
        )

        source_articles = (
            extract_source_articles(
                sources
            )
        )

        # Articles cited by the model that
        # were not present in retrieved sources.
        hallucinated_articles = (
            cited_articles
            - source_articles
        )

        # Articles specifically forbidden
        # for this benchmark question.
        used_forbidden_articles = (
            cited_articles
            & forbidden_articles
        )

        # For answered benchmark questions,
        # citations must stay inside the
        # approved article set.
        unexpected_articles = set()

        if (
            expected_status == "answered"
            and allowed_articles
        ):
            unexpected_articles = (
                cited_articles
                - allowed_articles
            )

        errors = []

        if (
        expected_status == "answered"
        and allowed_articles
        and not cited_articles
        ):
            errors.append(
                "answered response contains no detectable article citations"
            )

        if actual_status != expected_status:
            errors.append(
                (
                    "status mismatch: "
                    f"expected={expected_status}, "
                    f"actual={actual_status}"
                )
            )

        if hallucinated_articles:
            errors.append(
                (
                    "citations outside retrieved "
                    f"sources: "
                    f"{sorted(hallucinated_articles)}"
                )
            )

        if used_forbidden_articles:
            errors.append(
                (
                    "forbidden articles used: "
                    f"{sorted(used_forbidden_articles)}"
                )
            )

        if unexpected_articles:
            errors.append(
                (
                    "unexpected benchmark articles: "
                    f"{sorted(unexpected_articles)}"
                )
            )

        print(
            f"Status: {actual_status}"
        )

        print(
            f"Cited articles: "
            f"{sorted(cited_articles)}"
        )

        print(
            f"Retrieved articles: "
            f"{sorted(source_articles)}"
        )

        if errors:
            print("Result: FAIL")

            for error in errors:
                print(
                    f"  - {error}"
                )

            failures.append({
                "question": question,
                "run": run_number,
                "errors": errors,
                "answer": answer
            })

        else:
            passed_runs += 1

            print("Result: PASS")


# =========================
# Final Report
# =========================

accuracy = (
    passed_runs
    / total_runs
    * 100
    if total_runs
    else 0
)

print()
print()
print("=" * 70)
print("GENERATION GROUNDING RESULTS")
print("=" * 70)

print(
    f"Total runs: {total_runs}"
)

print(
    f"Passed: {passed_runs}"
)

print(
    f"Failed: {total_runs - passed_runs}"
)

print(
    f"Pass rate: {accuracy:.2f}%"
)


if failures:
    print()
    print("=" * 70)
    print("FAILED RUNS")
    print("=" * 70)

    for failure in failures:
        print()

        print(
            f"Question: "
            f"{failure['question']}"
        )

        print(
            f"Run: "
            f"{failure['run']}"
        )

        for error in failure[
            "errors"
        ]:
            print(
                f"- {error}"
            )

        print()
        print("Answer:")
        print(
            failure["answer"]
        )

else:
    print()
    print(
        "All generation grounding tests passed."
    )
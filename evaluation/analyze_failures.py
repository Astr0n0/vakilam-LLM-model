import json

# =========================
# Configuration
# =========================

RESULTS_FILE = "evaluation/evaluation_results.json"


# =========================
# Load evaluation results
# =========================

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)


# =========================
# Analyze failures
# =========================

failures = []

for item in results:

    expected_articles = set(
        item.get("expected_articles", [])
    )

    retrieved_articles = item.get(
        "retrieved_articles",
        []
    )

    retrieved_articles = set(retrieved_articles)

    # Failure means NONE of the expected articles
    # appeared in the Top-10 results.

    if not expected_articles.intersection(
        retrieved_articles
    ):
        failures.append(item)


# =========================
# Display summary
# =========================

print("=" * 70)
print("RETRIEVAL FAILURE ANALYSIS")
print("=" * 70)

print()
print(f"Total evaluation questions: {len(results)}")
print(f"Failed questions: {len(failures)}")
print()


# =========================
# Display failures
# =========================

for index, item in enumerate(failures, start=1):

    print("=" * 70)
    print(f"FAILURE #{index}")
    print("=" * 70)

    print()
    print("Question:")
    print(item.get("question"))

    print()
    print("Expected articles:")
    print(
        ", ".join(item.get("expected_articles", []))
    )

    print()
    print("Retrieved articles:")

    retrieved = item.get(
        "retrieved_articles",
        []
    )

    if not retrieved:
        print("None")
    else:
        for rank, article in enumerate(
            retrieved,
            start=1
        ):
            print(
                f"{rank}. Article {article}"
            )

    print()
    print("First relevant rank:")

    first_rank = item.get(
        "first_relevant_rank"
    )

    if first_rank is None:
        print("NOT FOUND")
    else:
        print(first_rank)

    print()


# =========================
# Detailed diagnosis
# =========================

print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)

print()

if not failures:

    print(
        "No complete retrieval failures found."
    )

else:

    for item in failures:

        print(
            f"- {item.get('question')}"
        )
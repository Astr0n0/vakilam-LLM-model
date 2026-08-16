import json
import os
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


from retrieve import retrieve


# =========================
# Configuration
# =========================

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "answerability_questions.json"
)


# =========================
# Helpers
# =========================

def get_best_semantic_distance(results):
    distances = [
        result.get("distance")
        for result in results
        if result.get("distance") is not None
    ]

    if not distances:
        return None

    return min(distances)


def get_dual_match_count(results, top_k=5):
    """
    Count results that appeared in both
    semantic and keyword rankings.
    """

    count = 0

    for result in results[:top_k]:
        if (
            result.get("semantic_rank") is not None
            and result.get("keyword_rank") is not None
        ):
            count += 1

    return count


# =========================
# Load questions
# =========================

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    tests = json.load(file)


# =========================
# Inspection
# =========================

for index, test in enumerate(
    tests,
    start=1
):
    question = test["question"]
    expected = test["expected"]

    print()
    print("=" * 70)
    print(
        f"TEST {index}/{len(tests)}"
    )
    print("=" * 70)

    print(
        f"Question: {question}"
    )

    print(
        f"Expected: {expected}"
    )

    results = retrieve(
        question
    )

    if not results:
        print("Results: 0")
        print("Best semantic distance: None")
        print("Top RRF score: None")
        print("Dual matches in top 5: 0")
        continue

    best_distance = get_best_semantic_distance(
        results
    )

    top_rrf = results[0].get(
        "rrf_score",
        0
    )

    dual_matches = get_dual_match_count(
        results
    )

    top_article = results[0][
        "metadata"
    ].get(
        "article"
    )

    print(
        f"Results: {len(results)}"
    )

    print(
        f"Top article: {top_article}"
    )

    print(
        f"Best semantic distance: "
        f"{best_distance}"
    )

    print(
        f"Top RRF score: "
        f"{top_rrf:.6f}"
    )

    print(
        f"Dual matches in top 5: "
        f"{dual_matches}"
    )
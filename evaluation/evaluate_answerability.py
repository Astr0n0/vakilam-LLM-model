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
from answerability import (
    classify_answerability,
    get_best_semantic_distance
)


# =========================
# Configuration
# =========================

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "answerability_questions.json"
)


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
# Evaluation
# =========================

total = len(tests)
correct = 0

answerable_total = 0
answerable_correct = 0

insufficient_total = 0
insufficient_correct = 0

failures = []


for index, test in enumerate(
    tests,
    start=1
):
    question = test["question"]
    expected = test["expected"]

    print()
    print("=" * 70)
    print(
        f"TEST {index}/{total}"
    )
    print("=" * 70)

    print(
        f"Question: {question}"
    )

    results = retrieve(
        question
    )

    actual = classify_answerability(
        results
    )

    best_distance = get_best_semantic_distance(
        results
    )

    print(
        f"Expected: {expected}"
    )

    print(
        f"Actual:   {actual}"
    )

    print(
        f"Best distance: {best_distance}"
    )

    if expected == "ANSWERABLE":
        answerable_total += 1

    if expected == "INSUFFICIENT":
        insufficient_total += 1

    if actual == expected:

        correct += 1

        if expected == "ANSWERABLE":
            answerable_correct += 1

        if expected == "INSUFFICIENT":
            insufficient_correct += 1

        print("Result:   PASS")

    else:

        failures.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "distance": best_distance
        })

        print("Result:   FAIL")


# =========================
# Metrics
# =========================

accuracy = (
    correct / total * 100
    if total
    else 0
)

answerable_accuracy = (
    answerable_correct
    / answerable_total
    * 100
    if answerable_total
    else 0
)

insufficient_accuracy = (
    insufficient_correct
    / insufficient_total
    * 100
    if insufficient_total
    else 0
)


# =========================
# Report
# =========================

print()
print()
print("=" * 70)
print("ANSWERABILITY EVALUATION RESULTS")
print("=" * 70)

print(
    f"Total: {total}"
)

print(
    f"Correct: {correct}"
)

print(
    f"Wrong: {total - correct}"
)

print(
    f"Overall accuracy: "
    f"{accuracy:.2f}%"
)

print()

print(
    f"ANSWERABLE accuracy: "
    f"{answerable_correct}/{answerable_total} "
    f"({answerable_accuracy:.2f}%)"
)

print(
    f"INSUFFICIENT accuracy: "
    f"{insufficient_correct}/{insufficient_total} "
    f"({insufficient_accuracy:.2f}%)"
)


# =========================
# Failures
# =========================

if failures:

    print()
    print("=" * 70)
    print("FAILED QUESTIONS")
    print("=" * 70)

    for failure in failures:

        print()

        print(
            f"Question: "
            f"{failure['question']}"
        )

        print(
            f"Expected: "
            f"{failure['expected']}"
        )

        print(
            f"Actual: "
            f"{failure['actual']}"
        )

        print(
            f"Distance: "
            f"{failure['distance']}"
        )

else:

    print()
    print(
        "All answerability tests passed."
    )
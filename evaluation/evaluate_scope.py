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


from scope_guard import classify_scope


# =========================
# Configuration
# =========================

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "scope_questions.json"
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

allowed_correct = 0
allowed_total = 0

denied_correct = 0
denied_total = 0

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

    print(
        f"Expected: {expected}"
    )

    actual = classify_scope(
        question
    )

    print(
        f"Actual:   {actual}"
    )

    if expected == "ALLOWED":
        allowed_total += 1

    elif expected == "DENIED":
        denied_total += 1

    if actual == expected:

        correct += 1

        if expected == "ALLOWED":
            allowed_correct += 1

        elif expected == "DENIED":
            denied_correct += 1

        print("Result:   PASS")

    else:

        failures.append({
            "question": question,
            "expected": expected,
            "actual": actual
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

allowed_accuracy = (
    allowed_correct / allowed_total * 100
    if allowed_total
    else 0
)

denied_accuracy = (
    denied_correct / denied_total * 100
    if denied_total
    else 0
)


# =========================
# Final report
# =========================

print()
print()
print("=" * 70)
print("SCOPE EVALUATION RESULTS")
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
    f"ALLOWED accuracy: "
    f"{allowed_correct}/{allowed_total} "
    f"({allowed_accuracy:.2f}%)"
)

print(
    f"DENIED accuracy: "
    f"{denied_correct}/{denied_total} "
    f"({denied_accuracy:.2f}%)"
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

else:

    print()
    print(
        "All scope tests passed."
    )
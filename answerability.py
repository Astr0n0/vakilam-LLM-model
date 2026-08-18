from config import MAX_SEMANTIC_DISTANCE


def get_best_semantic_distance(results):
    distances = [
        result.get("distance")
        for result in results
        if result.get("distance") is not None
    ]

    if not distances:
        return None

    return min(distances)


def has_exact_match(results):
    return any(
        result.get("retrieval_type") == "exact"
        for result in results
    )


def classify_answerability(results):
    if not results:
        return "INSUFFICIENT"

    if has_exact_match(results):
        return "ANSWERABLE"

    best_distance = get_best_semantic_distance(results)

    if best_distance is None:
        return "INSUFFICIENT"

    if best_distance <= MAX_SEMANTIC_DISTANCE:
        return "ANSWERABLE"

    return "INSUFFICIENT"


def is_answerable(results):
    return classify_answerability(results) == "ANSWERABLE"
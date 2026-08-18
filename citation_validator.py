import re

from retrieve import normalize_digits


def extract_cited_articles(answer):
    """
    Extract article numbers explicitly cited
    in the generated answer.
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
    """
    Extract article numbers from retrieved sources.
    """

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
                normalize_digits(
                    str(article)
                )
            )

    return articles


def validate_citations(answer, sources):
    """
    Check whether every cited article exists
    in the retrieved sources.
    """

    cited_articles = extract_cited_articles(
        answer
    )

    source_articles = extract_source_articles(
        sources
    )

    invalid_articles = (
        cited_articles
        - source_articles
    )

    return {
        "valid": not invalid_articles,
        "cited_articles": cited_articles,
        "source_articles": source_articles,
        "invalid_articles": invalid_articles
    }
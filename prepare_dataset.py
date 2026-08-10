import json
import re
from collections import defaultdict
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("law-rag/data/civil-law.json")
OUTPUT_PATH = Path("law-rag/data/civil-law-canonical.json")


# ============================================================
# Persian / Arabic digits
# ============================================================

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"

DIGIT_TRANSLATION = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    ENGLISH_DIGITS + ENGLISH_DIGITS,
)


# ============================================================
# Text normalization
# ============================================================

def normalize_digits(text: str) -> str:
    return text.translate(DIGIT_TRANSLATION)


def normalize_article(article: str) -> str:
    """
    Convert Persian/Arabic digits to English digits.

    Example:
        '۲۱۸' -> '218'
    """
    return normalize_digits(str(article)).strip()


def normalize_text(text: str) -> str:
    """
    Conservative normalization.

    We intentionally do NOT aggressively modify the legal text.
    The original text must remain untouched.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\u200c", "\u200c")  # preserve ZWNJ
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize Arabic characters commonly found in Persian text.
    text = text.replace("ي", "ی")
    text = text.replace("ى", "ی")
    text = text.replace("ك", "ک")

    # Normalize repeated spaces, but preserve line structure.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Version parsing
# ============================================================

VERSION_PATTERN = re.compile(
    r"^(?P<type>اصلاحی|الحاقی|حذف|حذفی|مصوب)"
    r"(?:\s+)?"
    r"(?P<date>\d{4}/\d{1,2}/\d{1,2})?$"
)


def parse_version(version):
    """
    Convert the raw version string into structured metadata.

    Examples:

        اصلاحی ۱۳۷۰/۸/۱۴
        -> {
            "version_type": "amendment",
            "version_date": "1370/08/14"
        }

        حذفی ۱۳۶۱/۱۰/۸
        -> {
            "version_type": "repealed",
            "version_date": "1361/10/08"
        }

        مصوب ۱۳۰۷/۲/۱۸
        -> {
            "version_type": "original",
            "version_date": "1307/02/18"
        }

        None
        -> {
            "version_type": "unspecified",
            "version_date": None
        }
    """

    if version is None:
        return {
            "version_type": "unspecified",
            "version_date": None,
        }

    version = str(version).strip()

    match = VERSION_PATTERN.match(version)

    if not match:
        return {
            "version_type": "unknown",
            "version_date": None,
        }

    raw_type = match.group("type")
    raw_date = match.group("date")

    type_mapping = {
        "اصلاحی": "amendment",
        "الحاقی": "addition",
        "حذف": "repealed",
        "حذفی": "repealed",
        "مصوب": "original",
    }

    version_type = type_mapping[raw_type]

    version_date = None

    if raw_date:
        parts = normalize_digits(raw_date).split("/")

        year = parts[0]
        month = parts[1].zfill(2)
        day = parts[2].zfill(2)

        version_date = f"{year}/{month}/{day}"

    return {
        "version_type": version_type,
        "version_date": version_date,
    }


# ============================================================
# Determine legal status
# ============================================================

def determine_status(records):
    """
    Determine the status of each version within one article.

    Rules:

    1. If there is a repealed version and no later amendment,
       the article is repealed.

    2. If there are amendment/addition versions,
       the latest dated amendment/addition is current.

    3. If there are only original/unspecified records,
       the latest available record is current.

    4. Unspecified records are NOT automatically current when
       a dated amendment/repeal exists.
    """

    parsed = []

    for index, record in enumerate(records):
        metadata = parse_version(record.get("version"))

        parsed.append({
            "index": index,
            "record": record,
            **metadata,
        })

    # --------------------------------------------------------
    # Repealed versions
    # --------------------------------------------------------

    repealed = [
        x for x in parsed
        if x["version_type"] == "repealed"
    ]

    amendments = [
        x for x in parsed
        if x["version_type"] in {"amendment", "addition"}
    ]

    # --------------------------------------------------------
    # Helper: sortable date
    # --------------------------------------------------------

    def date_key(item):
        date = item["version_date"]

        if not date:
            return (-1, -1, -1)

        y, m, d = map(int, date.split("/"))
        return (y, m, d)

    # --------------------------------------------------------
    # Find current record
    # --------------------------------------------------------

    current_index = None

    if amendments:
        # Latest amendment/addition is current.
        latest = max(amendments, key=date_key)
        current_index = latest["index"]

    elif not repealed:
        # No amendment and no repeal.
        # Use latest dated record.
        dated = [
            x for x in parsed
            if x["version_date"] is not None
        ]

        if dated:
            latest = max(dated, key=date_key)
            current_index = latest["index"]

        elif parsed:
            # Only unspecified records.
            current_index = parsed[-1]["index"]

    # --------------------------------------------------------
    # Assign status
    # --------------------------------------------------------

    result = []

    for item in parsed:
        status = "historical"

        if item["version_type"] == "repealed":
            status = "repealed"

        if item["index"] == current_index:
            status = "current"

        result.append({
            "index": item["index"],
            "status": status,
            "version_type": item["version_type"],
            "version_date": item["version_date"],
        })

    return result


# ============================================================
# Build canonical dataset
# ============================================================

def build_dataset(data):

    grouped = defaultdict(list)

    for record in data:
        article = normalize_article(record["article"])
        grouped[article].append(record)

    canonical = []

    for article, records in grouped.items():

        statuses = determine_status(records)

        for record, status_info in zip(records, statuses):

            version = record.get("version")

            item = {
                "id": (
                    f"civil_law_"
                    f"{article}_"
                    f"{status_info['status']}_"
                    f"{status_info['index']}"
                ),

                "law_id": "civil_law",

                "law_name": "قانون مدنی",

                "article": article,

                "article_persian": str(record["article"]),

                "version": version,

                "version_type": status_info["version_type"],

                "version_date": status_info["version_date"],

                "status": status_info["status"],

                "source": "civil-law",

                "text": record["text"],

                "text_normalized": normalize_text(record["text"]),
            }

            canonical.append(item)

    # Sort by article number and then version date.
    canonical.sort(
        key=lambda x: (
            int(x["article"]),
            x["version_date"] or "0000/00/00",
        )
    )

    return canonical


# ============================================================
# Validation
# ============================================================

def validate_dataset(dataset):

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    total = len(dataset)

    statuses = defaultdict(int)
    version_types = defaultdict(int)
    articles = defaultdict(list)

    for item in dataset:
        statuses[item["status"]] += 1
        version_types[item["version_type"]] += 1
        articles[item["article"]].append(item)

    print(f"Total records: {total}")
    print(f"Unique articles: {len(articles)}")

    print()
    print("Statuses:")

    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")

    print()
    print("Version types:")

    for version_type, count in sorted(version_types.items()):
        print(f"  {version_type}: {count}")

    # --------------------------------------------------------
    # Current count
    # --------------------------------------------------------

    articles_with_current = 0
    articles_without_current = []

    for article, records in articles.items():

        current_records = [
            x for x in records
            if x["status"] == "current"
        ]

        if current_records:
            articles_with_current += 1
        else:
            articles_without_current.append(article)

    print()
    print(f"Articles with current version: {articles_with_current}")

    if articles_without_current:
        print()
        print("Articles WITHOUT current version:")
        print(
            "  ",
            ", ".join(articles_without_current)
        )

    # --------------------------------------------------------
    # Multiple current records
    # --------------------------------------------------------

    multiple_current = []

    for article, records in articles.items():

        current_count = sum(
            x["status"] == "current"
            for x in records
        )

        if current_count > 1:
            multiple_current.append(article)

    print()

    if multiple_current:
        print("WARNING: Multiple current records:")
        print("  ", ", ".join(multiple_current))
    else:
        print("No articles have multiple current records.")

    # --------------------------------------------------------
    # Unknown version types
    # --------------------------------------------------------

    unknown_versions = [
        x for x in dataset
        if x["version_type"] == "unknown"
    ]

    print()

    if unknown_versions:
        print(
            f"WARNING: Unknown version formats: "
            f"{len(unknown_versions)}"
        )

        for item in unknown_versions[:20]:
            print(
                f"  Article {item['article']}: "
                f"{item['version']!r}"
            )

    else:
        print("No unknown version formats.")

    print()
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("Loading dataset...")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Input records: {len(data)}")

    dataset = build_dataset(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Canonical dataset written to:")
    print(OUTPUT_PATH)

    validate_dataset(dataset)


if __name__ == "__main__":
    main()
import json

PATH = "law-rag/data/civil-law-canonical.json"

TARGET_ARTICLES = [
    "1",
    "2",
    "3",
    "218",
    "653",
    "989",
    "1041",
    "1130",
    "1210",
    "1306",
]

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for article in TARGET_ARTICLES:

    records = [
        item
        for item in data
        if item["article"] == article
    ]

    print("=" * 80)
    print(f"ARTICLE {article}")
    print("=" * 80)

    for item in records:
        print(
            f"status={item['status']:<10} | "
            f"type={item['version_type']:<12} | "
            f"date={str(item['version_date']):<12} | "
            f"version={item['version']!r}"
        )

    print()
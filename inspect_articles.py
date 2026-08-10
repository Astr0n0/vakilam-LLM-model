import json

DATA_PATH = "law-rag/data/civil-law.json"

TARGET_ARTICLES = {"۲۱۸", "۹۸۹"}

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if item["article"] in TARGET_ARTICLES:
        print("=" * 80)
        print(f"ARTICLE: {item['article']}")
        print(f"VERSION: {item['version']!r}")
        print("-" * 80)
        print(item["text"])
        print()
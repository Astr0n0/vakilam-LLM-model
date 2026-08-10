import json
import collections

DATA_PATH = "law-rag/data/civil-law.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

by_article = collections.defaultdict(list)

for item in data:
    by_article[item["article"]].append(item)

multi_version = {
    article: versions
    for article, versions in by_article.items()
    if len(versions) > 1
}

print(f"Total records: {len(data)}")
print(f"Unique articles: {len(by_article)}")
print(f"Multi-version articles: {len(multi_version)}")
print()

for article, versions in multi_version.items():
    print(f"Article {article}:")
    
    for item in versions:
        print(f"  version = {item['version']!r}")

    print()
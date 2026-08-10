import json
import re

PATH = "law-rag/data/civil-law.json"

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


patterns = {
    "article": r"ماده\s+[۰-۹0-9]+",
    "note": r"تبصره",
    "chapter": r"فصل",
    "section": r"مبحث",
    "book": r"کتاب",
    "part": r"باب",
}


counts = {key: 0 for key in patterns}

examples = {key: [] for key in patterns}


for item in data:

    text = item.get("text", "")

    for name, pattern in patterns.items():

        if re.search(pattern, text):

            counts[name] += 1

            if len(examples[name]) < 5:
                examples[name].append({
                    "article": item["article"],
                    "version": item["version"],
                    "text": text[:500]
                })


print("=" * 70)
print("STRUCTURE ANALYSIS")
print("=" * 70)

for name in patterns:

    print()
    print(f"{name}: {counts[name]} records")

    for example in examples[name]:

        print("-" * 50)
        print(
            f"Article: {example['article']} | "
            f"Version: {example['version']}"
        )
        print(example["text"])
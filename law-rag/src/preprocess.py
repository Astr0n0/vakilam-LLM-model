from pathlib import Path
import re
import json

# Path to the law text file
input_path = Path("data/civil-law.txt")

# Path for the processed JSON dataset
output_path = Path("data/civil-law.json")

# Read the source file
text = input_path.read_text(encoding="utf-8")

# Find the beginning of each article
pattern = r"ماده\s+([۰-۹0-9]+)\s*[-–—]\s*(?:\[([^\]]+)\])?"

matches = list(re.finditer(pattern, text))

documents = []

# Extract each article
for i, match in enumerate(matches):

    article_number = match.group(1)
    version_info = match.group(2)

    # Start position of the current article
    start = match.start()

    # End position = beginning of the next article
    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        end = len(text)

    # Extract and clean the article text
    article_text = text[start:end].strip()

    # Create a structured document
    document = {
        "article": article_number,
        "version": version_info,
        "text": article_text
    }

    documents.append(document)


# Save the dataset as JSON
with output_path.open("w", encoding="utf-8") as f:
    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )


print("Number of extracted documents:", len(documents))
print(f"Dataset saved to: {output_path}")
from pathlib import Path
import json
import re

# Input and output paths
input_path = Path("law-rag/data/civil-law.json")
output_path = Path("law-rag/data/civil-law-chunks.json")

# Maximum size of a chunk in characters
MAX_CHARS = 1000


def create_version_id(version):
    """
    Convert the version information into a safe identifier.
    """

    if version is None:
        return "current"

    # Remove spaces and special characters
    version_id = re.sub(r"[^\w]+", "_", version)

    return version_id.strip("_")


def split_long_text(text, max_chars):
    """
    Split long text using natural boundaries when possible.
    """

    chunks = []

    remaining = text.strip()

    while len(remaining) > max_chars:

        # Look for the last paragraph break before the limit
        split_position = remaining.rfind("\n", 0, max_chars)

        # If there is no paragraph break,
        # look for the last sentence-ending punctuation.
        if split_position < max_chars * 0.5:
            punctuation_positions = [
                remaining.rfind("،", 0, max_chars),
                remaining.rfind(".", 0, max_chars),
                remaining.rfind("؛", 0, max_chars),
                remaining.rfind("؟", 0, max_chars),
                remaining.rfind("!", 0, max_chars),
            ]

            split_position = max(punctuation_positions)

        # If no suitable boundary exists,
        # fall back to a hard character split.
        if split_position < max_chars * 0.5:
            split_position = max_chars

        chunk = remaining[:split_position].strip()

        chunks.append(chunk)

        remaining = remaining[split_position:].strip()

    # Add the final part
    if remaining:
        chunks.append(remaining)

    return chunks


# Load the original dataset
with input_path.open("r", encoding="utf-8") as f:
    documents = json.load(f)


chunks = []

for document in documents:

    article = document["article"]
    version = document["version"]
    text = document["text"]

    version_id = create_version_id(version)

    # Keep short articles as a single chunk
    if len(text) <= MAX_CHARS:

        chunk_texts = [text]

    else:

        chunk_texts = split_long_text(
            text,
            MAX_CHARS
        )

    # Create metadata for every chunk
    for chunk_index, chunk_text in enumerate(chunk_texts):

        chunk_id = (
            f"article_{article}_"
            f"{version_id}_"
            f"{chunk_index}"
        )

        chunks.append({
            "chunk_id": chunk_id,
            "article": article,
            "version": version,
            "chunk_index": chunk_index,
            "source": "civil-law",
            "text": chunk_text
        })


# Save the final chunked dataset
with output_path.open("w", encoding="utf-8") as f:

    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


print("Original documents:", len(documents))
print("Generated chunks:", len(chunks))
print(f"Chunked dataset saved to: {output_path}")
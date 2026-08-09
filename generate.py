import chromadb
import ollama


# =========================
# Configuration
# =========================

CHROMA_PATH = "law-rag/chroma_db"
COLLECTION_NAME = "civil_law_v2"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:8b"

TOP_K = 10


# =========================
# Connect to ChromaDB
# =========================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# =========================
# Get user question
# =========================

query = input("Enter your legal question: ").strip()

if not query:
    print("Question cannot be empty.")
    raise SystemExit


# =========================
# Create query embedding
# =========================

print("\nSearching legal database...")

response = ollama.embeddings(
    model=EMBEDDING_MODEL,
    prompt=query
)

query_embedding = response["embedding"]


# =========================
# Retrieve relevant chunks
# =========================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K
)


# =========================
# Build context
# =========================

context_parts = []

for i in range(len(results["documents"][0])):

    document = results["documents"][0][i]
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    context_parts.append(
        f"""
[Source {i + 1}]
Article: {metadata["article"]}
Version: {metadata["version"]}
Chunk: {metadata["chunk_index"]}
Distance: {distance}

Text:
{document}
"""
    )


context = "\n".join(context_parts)


# =========================
# System prompt
# =========================

system_prompt = """
شما یک دستیار حقوقی فارسی هستید.

وظیفه شما پاسخ دادن به پرسش‌های حقوقی بر اساس
منابع قانونی ارائه‌شده در CONTEXT است.

قواعد مهم:

1. فقط بر اساس اطلاعات موجود در CONTEXT پاسخ دهید.
2. اگر اطلاعات کافی برای پاسخ وجود ندارد، صریحاً بگویید:
   «اطلاعات کافی در منابع بازیابی‌شده برای پاسخ دقیق وجود ندارد.»
3. هیچ ماده قانونی، شماره ماده، حکم یا واقعیت حقوقی را از خودتان ایجاد نکنید.
4. در پاسخ، شماره ماده قانونی مرتبط را در صورت وجود ذکر کنید.
5. پاسخ را به زبان فارسی و واضح ارائه کنید.
6. ابتدا پاسخ مستقیم به سؤال را بدهید.
7. سپس مستندات قانونی مرتبط را ذکر کنید.
8. بین متن قانون و استنباط خودتان تفاوت قائل شوید.
"""


# =========================
# User prompt
# =========================

user_prompt = f"""
پرسش کاربر:

{query}


CONTEXT:
{context}


بر اساس منابع بالا به پرسش پاسخ بده.
"""


# =========================
# Generate answer
# =========================

print("\nGenerating legal answer...\n")

response = ollama.chat(
    model=LLM_MODEL,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)


answer = response["message"]["content"]


# =========================
# Display answer
# =========================

print("=" * 70)
print("LEGAL ANSWER")
print("=" * 70)

print(answer)

print()
print("=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)

for i in range(len(results["documents"][0])):

    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print(
        f"{i + 1}. Article {metadata['article']} "
        f"| Version: {metadata['version']} "
        f"| Distance: {distance:.4f}"
    )
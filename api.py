from fastapi import FastAPI
from pydantic import BaseModel

from generate import generate_answer


app = FastAPI(
    title="Vakilam AI API",
    version="0.1.0"
)


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


def serialize_sources(sources):
    serialized = []

    for source in sources:
        metadata = source.get("metadata", {})

        serialized.append({
            "article": metadata.get("article"),
            "version": metadata.get("version"),
            "text": source.get("document")
        })

    return serialized


@app.post("/chat")
def chat(request: ChatRequest):

    result = generate_answer(
        request.question
    )

    return {
        "status": result.get("status"),
        "answer": result.get("answer"),
        "sources": serialize_sources(
            result.get("sources", [])
        )
    }

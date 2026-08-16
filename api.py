from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generate import generate_answer


# =========================
# Application
# =========================

app = FastAPI(
    title="Vakilam AI API",
    version="0.1.0"
)


# =========================
# Schemas
# =========================

class ChatRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    article: Optional[str] = None
    version: Optional[str] = None
    text: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    sources: List[SourceResponse] = Field(
        default_factory=list
    )


# =========================
# Helpers
# =========================

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


# =========================
# Health Check
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# =========================
# Chat
# =========================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    try:
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

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal AI service error."
        ) from exc
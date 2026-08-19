import uuid
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from security import verify_api_key
from generate import generate_answer


# =========================
# Application
# =========================

app = FastAPI(
    title="Vakilam AI API",
    version="0.1.0"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]


class OpenAIChoiceMessage(BaseModel):
    role: str
    content: str


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIChoiceMessage
    finish_reason: str


class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    model: str
    choices: List[OpenAIChoice]


# =========================
# Helpers
# =========================

def serialize_sources(sources):
    serialized = []

    for source in sources:
        metadata = source.get(
            "metadata",
            {}
        )

        serialized.append({
            "article": metadata.get(
                "article"
            ),
            "version": metadata.get(
                "version"
            ),
            "text": source.get(
                "document"
            )
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
# Native Vakilam Chat Endpoint
# =========================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
    request: ChatRequest,
    _: bool = Depends(verify_api_key)
):
    try:
        result = generate_answer(
            request.question
        )

        return {
            "status": result.get(
                "status"
            ),
            "answer": result.get(
                "answer"
            ),
            "sources": serialize_sources(
                result.get(
                    "sources",
                    []
                )
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal AI service error."
        ) from exc


# =========================
# OpenAI-Compatible Endpoint
# =========================

@app.post(
    "/v1/chat/completions",
    response_model=OpenAIChatResponse
)
def openai_chat_completion(
    request: OpenAIChatRequest,
    _: bool = Depends(verify_api_key)
):
    try:
        user_messages = [
            message.content
            for message in request.messages
            if message.role == "user"
        ]

        if not user_messages:
            raise HTTPException(
                status_code=400,
                detail="No user message provided."
            )

        question = user_messages[-1]

        result = generate_answer(
            question
        )

        answer = (
            result.get(
                "answer"
            )
            or ""
        )

        return {
            "id": (
                f"chatcmpl-"
                f"{uuid.uuid4().hex}"
            ),
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal AI service error."
        ) from exc
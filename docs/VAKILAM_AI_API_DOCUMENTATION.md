# Vakilam AI API Documentation

## 1. Overview

Vakilam AI Service is the artificial intelligence assistant backend service for the Vakilam legal platform.

The service provides legal question answering based on:

- Legal document retrieval
- Hybrid search (semantic + keyword retrieval)
- Vector database search
- Qwen3:8B language model generation
- Answerability validation
- Citation validation
- Grounding validation


The service is exposed through a FastAPI HTTP API and is designed to be consumed by the Laravel backend.

---

# 2. Architecture

```
Laravel Backend
        |
        | HTTP Request + API Key
        |
        v
FastAPI AI Service
        |
        +----------------------+
        |                      |
        v                      v
 Scope Guard            Retrieval System
                               |
                               v
                         ChromaDB
                               |
                               v
                         Qwen3:8B
                               |
                               v
                    Response Validation
```

## Processing Flow

1. User sends a legal question from Vakilam application.
2. Laravel backend sends the request to AI API.
3. API validates the API Key.
4. Scope Guard checks whether the question is legal-related.
5. Retrieval system searches available legal sources.
6. Qwen3:8B generates the response based on retrieved context.
7. Validation layers check response reliability.
8. Final response is returned to Laravel.


---

# 3. Authentication

All AI endpoints except health check require API Key authentication.

The API Key must be sent through HTTP Header:

```
X-API-Key: your-secret-key
```

The API Key is configured using:

```
API_KEY
```

environment variable.


Example:

```
X-API-Key: vakilam-production-key
```


---

# 4. API Endpoints


# 4.1 Health Check

Used for service availability monitoring.

## Request

```
GET /health
```


## Response

```json
{
  "status": "ok"
}
```


---

# 4.2 Native Vakilam Chat Endpoint

This endpoint provides the native Vakilam AI response format.


## Request

```
POST /chat
```


## Headers

```
Content-Type: application/json

X-API-Key: your-secret-key
```


## Body

```json
{
  "question": "شرایط صحت عقد نکاح چیست؟"
}
```


## Response

```json
{
  "status": "answered",
  "answer": "پاسخ حقوقی...",
  "sources": [
    {
      "article": "1067",
      "version": "current",
      "text": "..."
    }
  ]
}
```


---

# 4.3 OpenAI Compatible Endpoint

This endpoint follows OpenAI Chat Completion format.

It can be used by systems expecting an OpenAI-compatible API structure.


## Request

```
POST /v1/chat/completions
```


## Headers

```
Content-Type: application/json

X-API-Key: your-secret-key
```


## Body

```json
{
  "model": "qwen3:8b",
  "messages": [
    {
      "role": "user",
      "content": "شرایط صحت عقد نکاح چیست؟"
    }
  ]
}
```


## Response

```json
{
  "id": "chatcmpl-example",
  "object": "chat.completion",
  "model": "qwen3:8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "پاسخ حقوقی..."
      },
      "finish_reason": "stop"
    }
  ]
}
```


---

# 5. Response Statuses


## answered

The AI generated an answer using available legal sources.


Example:

```json
{
  "status": "answered"
}
```


---

## out_of_scope

The question is outside the legal assistant scope.


Example:

```
طرز تهیه قرمه سبزی چیست؟
```


Response:

```json
{
  "status": "out_of_scope",
  "answer": "این دستیار فقط به پرسش‌های مرتبط با حوزه حقوقی پاسخ می‌دهد."
}
```


---

## insufficient_context

Available legal sources are not sufficient for generating a reliable answer.


Example:

```json
{
  "status": "insufficient_context",
  "answer": "اطلاعات کافی در منابع موجود برای پاسخ دقیق به این پرسش وجود ندارد."
}
```


---

# 6. Environment Configuration

Example `.env` configuration:


```env
# API Security

API_KEY=change-this-secret-key


# AI Model Configuration

LLM_MODEL=qwen3:8b


# Vector Database

CHROMA_DB_PATH=law-rag/chroma_db


# Retrieval Configuration

MAX_SEMANTIC_DISTANCE=1.2
```


---

# 7. Laravel Integration Example


Example using Laravel HTTP Client:


```php
$response = Http::withHeaders([
    'X-API-Key' => config('services.vakilam.ai_key'),
])->post(
    'http://ai-service/v1/chat/completions',
    [
        'model' => 'qwen3:8b',
        'messages' => [
            [
                'role' => 'user',
                'content' => $question
            ]
        ]
    ]
);
```


The Laravel backend is responsible for:

- User authentication
- Conversation management
- User-specific chat history
- Permission handling
- Saving AI responses


The AI service is responsible for:

- Legal retrieval
- AI generation
- Source-based answering
- Response validation


---

# 8. Deployment Requirements


Required components:

- Python environment
- Python dependencies from requirements.txt
- Ollama runtime
- Qwen3:8B model
- ChromaDB database


Install dependencies:

```bash
pip install -r requirements.txt
```


Run service:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```


---

# 9. AI Pipeline


The AI processing pipeline:


```
User Question

        |
        v

Scope Validation

        |
        v

Question Answerability Check

        |
        v

Hybrid Retrieval

(Semantic + Keyword)

        |
        v

Legal Context Assembly

        |
        v

Qwen3:8B Generation

        |
        v

Citation Validation

        |
        v

Grounding Validation

        |
        v

Final Response
```


---

# 10. AI Limitations


This AI assistant:

- Does not replace a lawyer.
- Does not guarantee legal outcomes.
- Does not make final legal decisions.
- Generates responses based on available legal sources.
- Requires human legal review for professional legal decisions.

---

# 11. Version Information


Service:

```
Vakilam AI API
```

API Version:

```
0.1.0
```

Model:

```
Qwen3:8B
```

Framework:

```
FastAPI
```

---

# 12. Integration Contract


## Laravel Backend Responsibilities

Laravel backend is responsible for:

- User authentication
- User authorization
- Creating and managing conversations
- Storing user chat history
- Associating conversations with users
- Rate limiting users
- Handling API errors from AI service


## AI Service Responsibilities

Vakilam AI service is responsible for:

- Receiving legal questions
- Validating request scope
- Retrieving relevant legal sources
- Generating AI responses
- Returning source references
- Returning response status


## User Conversation Isolation

The AI API is stateless regarding user identity.

Each user's conversation history must be managed by Laravel backend.

Laravel should send only the required conversation context when multi-turn conversations are needed.

The AI service does not mix user conversations because it does not store user sessions internally.
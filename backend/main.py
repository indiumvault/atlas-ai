import os
from collections import defaultdict, deque
from time import monotonic

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

APP_ORIGIN = os.getenv("APP_ORIGIN", "https://indiumvault.github.io").rstrip("/")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
AI_MODEL = os.getenv("AI_MODEL", "").strip()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()

app = FastAPI(title="Atlas AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[APP_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

SYSTEM_PROMPT = (
    "You are Atlas AI, a helpful assistant for learning, coding, writing, and "
    "research. Be clear and honest. If uncertain, say so. Never claim to have "
    "performed actions you have not performed."
)

# Lightweight per-process throttle for the prototype. This is not a substitute
# for account-based quotas or a distributed rate limiter in a public production app.
REQUESTS_PER_MINUTE = 20
request_times = defaultdict(deque)


class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def valid_role(cls, value: str) -> str:
        if value not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        return value

    @field_validator("content")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content cannot be blank")
        return value.strip()


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=12)


class ChatResponse(BaseModel):
    reply: str
    model: str


@app.get("/")
def root():
    return {"name": "Atlas AI API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "provider_configured": bool(AI_API_KEY and AI_MODEL)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    # Keep this API unavailable until a real provider key and model are configured.
    if not AI_API_KEY or not AI_MODEL:
        raise HTTPException(
            status_code=503,
            detail="AI provider is not configured yet. Add AI_API_KEY and AI_MODEL in Render Environment settings.",
        )

    client_ip = request.client.host if request.client else "unknown"
    now = monotonic()
    history = request_times[client_ip]
    while history and now - history[0] > 60:
        history.popleft()
    if len(history) >= REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")
    history.append(now)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": m.role, "content": m.content} for m in payload.messages)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            response = await client.post(
                f"{AI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {AI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": AI_MODEL,
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            )
        if response.status_code >= 400:
            # Do not return provider response bodies; they can contain internal details.
            raise HTTPException(status_code=502, detail="The AI provider rejected the request. Check the provider settings.")
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("Empty response")
        return ChatResponse(reply=reply.strip(), model=AI_MODEL)
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail="The AI provider could not complete the request. Please try again.")

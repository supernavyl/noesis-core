"""FastAPI server — OpenAI-compatible chat endpoint + admin routes."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from noesis.core.config import constitution, settings
from noesis.core.orchestrator import Depth, Orchestrator, QueryRequest
from noesis.knowledge.vector_store import VectorStore


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    max_depth: int | None = None
    require_sandbox: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    depth_used: int
    models_invoked: list[str]
    sources: list[dict[str, Any]]
    confidence: float


app = FastAPI(title="NOESIS", version="0.0.1")
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "env": settings().noesis_env}


@app.get("/v1/constitution")
async def get_constitution() -> dict[str, Any]:
    c = constitution()
    return {
        "version": c.version,
        "last_modified": c.last_modified,
        "principles": c.principles,
        "prohibitions": c.prohibitions,
    }


@app.get("/v1/corpus/stats")
async def corpus_stats() -> dict[str, int]:
    vs = VectorStore()
    return {"chunks": await vs.count()}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    orch = get_orchestrator()
    request = QueryRequest(
        question=req.question,
        session_id=req.session_id or str(uuid.uuid4()),
        max_depth=Depth(req.max_depth) if req.max_depth else Depth.CONSTITUTIONAL,
        require_sandbox=req.require_sandbox,
    )
    try:
        result = await orch.run(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return ChatResponse(
        session_id=request.session_id,
        answer=result.answer,
        depth_used=int(result.depth_used),
        models_invoked=result.models_invoked,
        sources=result.sources,
        confidence=result.confidence,
    )

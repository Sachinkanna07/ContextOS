"""Retrieval and compilation API routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from contextos.api.server import get_service
from contextos.core.models import (
    CompiledContext,
    CompilationConfig,
    RetrievalConfig,
    RetrievalResult,
)

router = APIRouter(tags=["retrieval"])


class RetrieveRequest(BaseModel):
    """Request body for retrieval."""
    query: str = Field(min_length=1)
    config: RetrievalConfig | None = None


class CompileRequest(BaseModel):
    """Request body for compilation."""
    query: str = Field(min_length=1)
    config: CompilationConfig | None = None
    retrieval_config: RetrievalConfig | None = None


@router.post("/retrieve", response_model=RetrievalResult)
async def retrieve(request: RetrieveRequest) -> RetrievalResult:
    """Retrieve relevant memories for a query."""
    retrieval_service = get_service("retrieval")
    return await retrieval_service.retrieve(request.query, request.config)


@router.post("/compile", response_model=CompiledContext)
async def compile_context(request: CompileRequest) -> CompiledContext:
    """Retrieve and compile context for a query."""
    retrieval_service = get_service("retrieval")
    compilation_service = get_service("compilation")

    # First retrieve
    retrieval_result = await retrieval_service.retrieve(
        request.query, request.retrieval_config
    )

    # Then compile
    return await compilation_service.compile(
        query=request.query,
        memories=retrieval_result.memories,
        config=request.config,
    )

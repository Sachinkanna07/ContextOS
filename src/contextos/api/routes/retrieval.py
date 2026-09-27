"""Retrieval and compilation API routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from contextos.api.server import get_service
from contextos.core.models import (
    CompiledContext,
    CompilationConfig,
    ContextBudget,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
)

router = APIRouter(tags=["retrieval"])


class RetrieveRequest(BaseModel):
    """Request body for retrieval."""

    query: str | RetrievalQuery
    config: RetrievalConfig | None = None

    @field_validator("query")
    @classmethod
    def nonblank_string_query(cls, value: str | RetrievalQuery) -> str | RetrievalQuery:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Query cannot be blank")
        return value


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
    optimizer = get_service("optimizer")

    # First retrieve
    retrieval_result = await retrieval_service.retrieve(
        request.query, request.retrieval_config
    )

    compilation_config = request.config or CompilationConfig()
    selection = optimizer.optimize(
        request.query,
        retrieval_result.memories,
        ContextBudget(max_tokens=compilation_config.budget),
    )

    # Carry selected memories and eligible oversized rescue candidates explicitly.
    return await compilation_service.compile(
        query=request.query,
        memories=selection,
        config=compilation_config,
    )

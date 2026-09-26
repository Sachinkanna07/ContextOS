"""Ingestion API routes."""

from __future__ import annotations

from fastapi import APIRouter

from contextos.api.server import get_service
from contextos.core.models import IngestRequest, IngestResult

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResult)
async def ingest(request: IngestRequest) -> IngestResult:
    """Ingest content into ContextOS."""
    ingestion_service = get_service("ingestion")
    return await ingestion_service.ingest(request)

"""FastAPI server setup for ContextOS.

This is the HTTP API that the CLI and future MCP server communicate with.
All business logic lives in the service layer — this is a thin translation
layer between HTTP and service protocol calls.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from contextos import __version__
from contextos.core.exceptions import (
    ContextOSError,
    InvalidTransitionError,
    MemoryNotFoundError,
    SecretDetectedError,
)

logger = logging.getLogger(__name__)

# Service instances — set by daemon wiring before startup
_services: dict[str, Any] = {}


def set_services(services: dict[str, Any]) -> None:
    """Set service instances. Called by daemon wiring at startup."""
    _services.update(services)


def get_service(name: str) -> Any:
    """Get a service by name. Raises if not set."""
    if name not in _services:
        raise RuntimeError(f"Service '{name}' not initialized")
    return _services[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("ContextOS API server starting (v%s)", __version__)
    yield
    logger.info("ContextOS API server shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ContextOS",
        description="Local-first personal AI memory runtime",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # --- Middleware ---

    @app.middleware("http")
    async def request_timing(request: Request, call_next):
        """Log request timing."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(
            "%s %s — %d (%.1fms)",
            request.method, request.url.path, response.status_code, elapsed,
        )
        return response

    # --- Exception Handlers ---

    @app.exception_handler(MemoryNotFoundError)
    async def memory_not_found_handler(request: Request, exc: MemoryNotFoundError):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    @app.exception_handler(InvalidTransitionError)
    async def invalid_transition_handler(request: Request, exc: InvalidTransitionError):
        return JSONResponse(status_code=422, content={"error": str(exc)})

    @app.exception_handler(SecretDetectedError)
    async def secret_detected_handler(request: Request, exc: SecretDetectedError):
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "secret_types": exc.secret_types,
            },
        )

    @app.exception_handler(ContextOSError)
    async def contextos_error_handler(request: Request, exc: ContextOSError):
        return JSONResponse(status_code=500, content={"error": str(exc)})

    # --- Register Routes ---
    from contextos.api.routes.ingest import router as ingest_router
    from contextos.api.routes.memories import router as memories_router
    from contextos.api.routes.retrieval import router as retrieval_router
    from contextos.api.routes.system import router as system_router

    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(memories_router, prefix="/api/v1")
    app.include_router(retrieval_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")

    return app

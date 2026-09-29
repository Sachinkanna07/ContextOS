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
from fastapi.exceptions import RequestValidationError
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

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        """Return validation structure without echoing attacker-controlled values."""
        safe_errors = [
            {key: value for key, value in error.items() if key not in {"input", "ctx"}}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

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

    from contextos.core.exceptions import (
        ContextWindowExceededError,
        MalformedProviderResponseError,
        ModelUnavailableError,
        ProviderAuthenticationError,
        ProviderRateLimitError,
        ProviderTimeoutError,
        ProviderUnavailableError,
        RoutingFailureError,
    )

    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable_handler(request: Request, exc: ProviderUnavailableError):
        return JSONResponse(status_code=503, content={"error": str(exc), "provider_id": exc.provider_id})

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=404, content={"error": str(exc), "model_id": exc.model_id})

    @app.exception_handler(ContextWindowExceededError)
    async def context_window_handler(request: Request, exc: ContextWindowExceededError):
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "model_id": exc.model_id,
                "required_tokens": exc.required_tokens,
                "context_window": exc.context_window,
                "prompt_tokens": exc.prompt_tokens,
                "compiled_context_tokens": exc.compiled_context_tokens,
                "reserved_output_tokens": exc.reserved_output_tokens,
            },
        )

    @app.exception_handler(MalformedProviderResponseError)
    async def malformed_provider_handler(request: Request, exc: MalformedProviderResponseError):
        return JSONResponse(
            status_code=502,
            content={"error": str(exc), "provider_id": exc.provider_id},
        )

    @app.exception_handler(ProviderTimeoutError)
    async def provider_timeout_handler(request: Request, exc: ProviderTimeoutError):
        return JSONResponse(status_code=504, content={"error": str(exc)})

    @app.exception_handler(ProviderAuthenticationError)
    async def provider_auth_handler(request: Request, exc: ProviderAuthenticationError):
        return JSONResponse(status_code=401, content={"error": str(exc)})

    @app.exception_handler(ProviderRateLimitError)
    async def provider_rate_limit_handler(request: Request, exc: ProviderRateLimitError):
        headers = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(int(exc.retry_after))
        return JSONResponse(status_code=429, content={"error": str(exc)}, headers=headers)

    @app.exception_handler(RoutingFailureError)
    async def routing_failure_handler(request: Request, exc: RoutingFailureError):
        return JSONResponse(status_code=400, content={"error": str(exc), "policy": exc.policy})

    @app.exception_handler(ContextOSError)
    async def contextos_error_handler(request: Request, exc: ContextOSError):
        return JSONResponse(status_code=500, content={"error": str(exc)})

    # --- Register Routes ---
    from contextos.api.routes.ingest import router as ingest_router
    from contextos.api.routes.memories import router as memories_router
    from contextos.api.routes.models import router as models_router
    from contextos.api.routes.retrieval import router as retrieval_router
    from contextos.api.routes.system import router as system_router
    from contextos.api.routes.desktop import router as desktop_router

    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(memories_router, prefix="/api/v1")
    app.include_router(retrieval_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")
    app.include_router(desktop_router, prefix="/api/v1")

    return app

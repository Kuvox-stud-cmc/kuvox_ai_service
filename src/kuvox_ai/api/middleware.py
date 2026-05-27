"""HTTP middleware: request-ID propagation + structured access logging."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from kuvox_ai.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Bind a request ID to logs and emit one structured access line per request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid4().hex
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "http.request",
                status=status_code,
                duration_ms=round(duration_ms, 2),
            )
            structlog.contextvars.clear_contextvars()

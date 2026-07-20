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
from kuvox_ai.metrics import HTTP_LATENCY, HTTP_REQUESTS

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Bind a request ID to logs and emit one structured access line per request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid4().hex
        editor_correlation_id = request.headers.get("x-kuvox-editor-correlation-id") or request_id
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            editor_correlation_id=editor_correlation_id,
            method=request.method,
            path=request.url.path,
        )
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            response.headers["x-kuvox-editor-correlation-id"] = editor_correlation_id
            return response
        finally:
            duration_seconds = time.perf_counter() - start
            duration_ms = duration_seconds * 1000
            route = getattr(request.scope.get("route"), "path", "unmatched")
            HTTP_REQUESTS.labels("ai", request.method, route, str(status_code)).inc()
            HTTP_LATENCY.labels("ai", request.method, route).observe(duration_seconds)
            logger.info(
                "http.request",
                status=status_code,
                duration_ms=round(duration_ms, 2),
            )
            structlog.contextvars.clear_contextvars()

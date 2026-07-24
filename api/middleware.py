"""
Module: api/middleware.py

Purpose:
    Defines custom Starlette/FastAPI middleware for the Institutional
    Memory System API.

Responsibilities:
    - Log every incoming request and outgoing response with timing.
    - Attach a unique request ID to every request for traceability.
    - Measure and log request processing duration.
    - Provide structured, consistent logging across all API traffic.

Workflow:
    Phase 1 — Request arrives — generate request ID and start timer.
    Phase 2 — Request is processed by the route handler.
    Phase 3 — Response is captured — log status code and duration.
    Phase 4 — Request ID is attached to response headers.
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all incoming requests and outgoing responses.

    Attaches a unique request ID to each request for end-to-end tracing
    across logs, and measures total processing time for observability.

    Attributes:
        Inherits from BaseHTTPMiddleware — no additional state required.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Processes each request, logging entry, exit, and timing.

        Args:
            request: The incoming Starlette Request object.
            call_next: The next handler in the middleware chain.

        Returns:
            The Response object, with X-Request-ID and X-Process-Time
            headers attached.
        """
        request_id = uuid.uuid4().hex[:12]
        start_time = time.perf_counter()

        # Attach request_id to request state for use in route handlers
        request.state.request_id = request_id

        logger.info(
            "→ Request received | id='{}' | method='{}' | path='{}'",
            request_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "✗ Request failed | id='{}' | path='{}' | "
                "duration={}ms | error={}",
                request_id,
                request.url.path,
                duration_ms,
                exc,
            )
            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "← Response sent | id='{}' | path='{}' | status={} | "
            "duration={}ms",
            request_id,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        # Attach tracing headers to the response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)

        return response

"""
Module: orchestrators/master_orchestrator.py

Purpose:
    The top-level brain of the Institutional Memory System that receives
    all incoming requests and routes them to the correct sub-orchestrator.

Responsibilities:
    - Accept all incoming requests: queries, captures, and audits.
    - Classify request intent and route to the correct sub-orchestrator.
    - Maintain a unified request/response contract for the API layer.
    - Provide system-wide health status by aggregating sub-system checks.
    - Log all requests and responses for observability.

Workflow:
    Phase 1 — Receive a MasterRequest from the API layer.
    Phase 2 — Classify request type: QUERY, CAPTURE, or AUDIT.
    Phase 3 — Route to RetrievalOrchestrator, CaptureOrchestrator,
              or AuditOrchestrator.
    Phase 4 — Return a unified MasterResponse to the API layer.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings
from memory.memory_manager import MemoryManager, memory_manager
from orchestrators.retrieval_orchestrator import (
    RetrievalOrchestrator,
    RetrievalRequest,
)
from orchestrators.capture_orchestrator import (
    CaptureOrchestrator,
    CaptureRequest,
    CaptureResult,
)
from orchestrators.audit_orchestrator import (
    AuditOrchestrator,
    AuditRequest,
    AuditReport,
    AuditScope,
)
from schemas.agent_schema import AgentOutput, QueryCategory


# ── Request Type Enumeration ─────────────────────────────────────────────────

class RequestType(str, Enum):
    """Defines the top-level type of a master request."""

    QUERY = "query"       # User asking a question
    CAPTURE = "capture"   # Submitting content for knowledge capture
    AUDIT = "audit"       # Requesting an audit scan
    HEALTH = "health"     # System health check


# ── Master Request & Response Models ─────────────────────────────────────────

class MasterRequest(BaseModel):
    """Unified input model accepted by the Master Orchestrator.

    The API layer constructs this model and passes it directly
    to the Master Orchestrator for routing.
    """

    request_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique request identifier"
    )
    request_type: RequestType = Field(
        ..., description="Type of request: QUERY, CAPTURE, AUDIT, or HEALTH"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for multi-turn context"
    )

    # ── QUERY fields ──────────────────────────────────────────────────────────
    query: Optional[str] = Field(
        None, description="User query string (required for QUERY type)"
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of documents to retrieve"
    )
    category_hint: Optional[QueryCategory] = Field(
        None, description="Optional category hint to bypass router"
    )
    metadata_filter: Optional[dict] = Field(
        None, description="Optional ChromaDB metadata filter"
    )

    # ── CAPTURE fields ────────────────────────────────────────────────────────
    capture_request: Optional[CaptureRequest] = Field(
        None, description="Capture request (required for CAPTURE type)"
    )

    # ── AUDIT fields ──────────────────────────────────────────────────────────
    audit_scope: AuditScope = Field(
        default=AuditScope.FULL,
        description="Scope of audit scan (used for AUDIT type)"
    )
    generate_audit_summary: bool = Field(
        default=True,
        description="Whether to generate LLM summary for audit"
    )


class MasterResponse(BaseModel):
    """Unified response model returned by the Master Orchestrator."""

    request_id: str = Field(..., description="Echo of the incoming request ID")
    request_type: RequestType = Field(..., description="Type of request handled")
    session_id: Optional[str] = Field(None)
    success: bool = Field(..., description="Whether the request succeeded")
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(
        ..., description="End-to-end processing time in milliseconds"
    )

    # ── Type-specific payloads ────────────────────────────────────────────────
    query_result: Optional[AgentOutput] = Field(
        None, description="Populated for QUERY requests"
    )
    capture_result: Optional[CaptureResult] = Field(
        None, description="Populated for CAPTURE requests"
    )
    audit_report: Optional[AuditReport] = Field(
        None, description="Populated for AUDIT requests"
    )
    health_status: Optional[dict[str, Any]] = Field(
        None, description="Populated for HEALTH requests"
    )

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str] = Field(
        None, description="Error message if request failed"
    )


class MasterOrchestrator:
    """Top-level orchestrator routing all requests to sub-orchestrators.

    Acts as the single entry point for the entire Institutional Memory
    System. The API layer communicates exclusively with this class,
    which then delegates to the appropriate sub-orchestrator.

    Attributes:
        _retrieval: Handles all knowledge retrieval queries.
        _capture: Handles all knowledge capture operations.
        _audit: Handles all audit scan operations.
        _memory: Shared memory manager across all orchestrators.
    """

    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        """Initialises the MasterOrchestrator and all sub-orchestrators.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        self._memory = memory or memory_manager

        logger.info("Initialising MasterOrchestrator...")

        self._retrieval = RetrievalOrchestrator(memory=self._memory)
        self._capture = CaptureOrchestrator(memory=self._memory)
        self._audit = AuditOrchestrator(memory=self._memory)

        logger.info("MasterOrchestrator ready.")

    # ── Private Request Handlers ─────────────────────────────────────────────

    def _handle_query(
        self,
        request: MasterRequest,
    ) -> MasterResponse:
        """Handles QUERY type requests via the RetrievalOrchestrator.

        Args:
            request: The incoming MasterRequest with query fields.

        Returns:
            A MasterResponse with query_result populated.
        """
        import time

        if not request.query:
            return self._error_response(
                request=request,
                error="Query string is required for QUERY request type.",
                processing_time_ms=0.0,
            )

        start = time.perf_counter()

        try:
            retrieval_request = RetrievalRequest(
                query=request.query,
                session_id=request.session_id,
                top_k=request.top_k,
                category_hint=request.category_hint,
                metadata_filter=request.metadata_filter,
            )

            result = self._retrieval.retrieve(retrieval_request)
            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )

            logger.info(
                "MasterOrchestrator QUERY handled | request_id='{}' | "
                "agent='{}' | confidence={} | time={}ms",
                request.request_id,
                result.agent_name,
                result.confidence,
                processing_time_ms,
            )

            return MasterResponse(
                request_id=request.request_id,
                request_type=RequestType.QUERY,
                session_id=request.session_id,
                success=True,
                processing_time_ms=processing_time_ms,
                query_result=result,
            )

        except Exception as exc:
            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )
            logger.error(
                "MasterOrchestrator QUERY failed | request_id='{}': {}",
                request.request_id,
                exc,
            )
            return self._error_response(
                request=request,
                error=str(exc),
                processing_time_ms=processing_time_ms,
            )

    def _handle_capture(
        self,
        request: MasterRequest,
    ) -> MasterResponse:
        """Handles CAPTURE type requests via the CaptureOrchestrator.

        Args:
            request: The incoming MasterRequest with capture_request populated.

        Returns:
            A MasterResponse with capture_result populated.
        """
        import time

        if not request.capture_request:
            return self._error_response(
                request=request,
                error="capture_request is required for CAPTURE request type.",
                processing_time_ms=0.0,
            )

        start = time.perf_counter()

        try:
            result = self._capture.capture(request.capture_request)
            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )

            logger.info(
                "MasterOrchestrator CAPTURE handled | request_id='{}' | "
                "type='{}' | success={} | time={}ms",
                request.request_id,
                request.capture_request.content_type.value,
                result.success,
                processing_time_ms,
            )

            return MasterResponse(
                request_id=request.request_id,
                request_type=RequestType.CAPTURE,
                session_id=request.session_id,
                success=result.success,
                processing_time_ms=processing_time_ms,
                capture_result=result,
            )

        except Exception as exc:
            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )
            logger.error(
                "MasterOrchestrator CAPTURE failed | request_id='{}': {}",
                request.request_id,
                exc,
            )
            return self._error_response(
                request=request,
                error=str(exc),
                processing_time_ms=processing_time_ms,
            )

    def _handle_audit(
        self,
        request: MasterRequest,
    ) -> MasterResponse:
        """Handles AUDIT type requests via the AuditOrchestrator.

        Args:
            request: The incoming MasterRequest with audit fields.

        Returns:
            A MasterResponse with audit_report populated.
        """
        import time

        start = time.perf_counter()

        try:
            audit_request = AuditRequest(
                scope=request.audit_scope,
                session_id=request.session_id,
                generate_summary=request.generate_audit_summary,
            )

            report = self._audit.audit(audit_request)
            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )

            logger.info(
                "MasterOrchestrator AUDIT handled | request_id='{}' | "
                "health='{}' | findings={} | time={}ms",
                request.request_id,
                report.overall_health,
                report.total_findings,
                processing_time_ms,
            )

            return MasterResponse(
                request_id=request.request_id,
                request_type=RequestType.AUDIT,
                session_id=request.session_id,
                success=True,
                processing_time_ms=processing_time_ms,
                audit_report=report,
            )

        except Exception as exc:
            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )
            logger.error(
                "MasterOrchestrator AUDIT failed | request_id='{}': {}",
                request.request_id,
                exc,
            )
            return self._error_response(
                request=request,
                error=str(exc),
                processing_time_ms=processing_time_ms,
            )

    def _handle_health(
        self,
        request: MasterRequest,
    ) -> MasterResponse:
        """Handles HEALTH type requests by aggregating sub-system status.

        Args:
            request: The incoming MasterRequest.

        Returns:
            A MasterResponse with health_status populated.
        """
        import time

        start = time.perf_counter()

        try:
            health = self._memory.health_check()
            health["app_name"] = settings.app_name
            health["environment"] = settings.environment
            health["version"] = "1.0.0"

            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )

            logger.info(
                "MasterOrchestrator HEALTH check | status='{}' | time={}ms",
                health.get("status", "unknown"),
                processing_time_ms,
            )

            return MasterResponse(
                request_id=request.request_id,
                request_type=RequestType.HEALTH,
                session_id=request.session_id,
                success=True,
                processing_time_ms=processing_time_ms,
                health_status=health,
            )

        except Exception as exc:
            processing_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )
            logger.error(
                "MasterOrchestrator HEALTH check failed: {}", exc
            )
            return self._error_response(
                request=request,
                error=str(exc),
                processing_time_ms=processing_time_ms,
            )

    # ── Error Response Builder ────────────────────────────────────────────────

    def _error_response(
        self,
        request: MasterRequest,
        error: str,
        processing_time_ms: float,
    ) -> MasterResponse:
        """Builds a standardised error MasterResponse.

        Args:
            request: The original MasterRequest.
            error: Error message string.
            processing_time_ms: Time elapsed before failure.

        Returns:
            A MasterResponse with success=False and error populated.
        """
        return MasterResponse(
            request_id=request.request_id,
            request_type=request.request_type,
            session_id=request.session_id,
            success=False,
            processing_time_ms=processing_time_ms,
            error=error,
        )

    # ── Public Entry Point ───────────────────────────────────────────────────

    def process(self, request: MasterRequest) -> MasterResponse:
        """Processes any incoming request and routes to the correct handler.

        This is the single public entry point for the entire Institutional
        Memory System. The API layer calls only this method.

        Args:
            request: A validated MasterRequest from the API layer.

        Returns:
            A MasterResponse with the appropriate payload populated.
        """
        logger.info(
            "MasterOrchestrator received request | id='{}' | type='{}' | "
            "session='{}'",
            request.request_id,
            request.request_type.value,
            request.session_id or "none",
        )

        route_map = {
            RequestType.QUERY: self._handle_query,
            RequestType.CAPTURE: self._handle_capture,
            RequestType.AUDIT: self._handle_audit,
            RequestType.HEALTH: self._handle_health,
        }

        handler = route_map.get(request.request_type)

        if not handler:
            logger.error(
                "MasterOrchestrator: unknown request type '{}'",
                request.request_type,
            )
            return self._error_response(
                request=request,
                error=f"Unknown request type: '{request.request_type}'",
                processing_time_ms=0.0,
            )

        return handler(request)

    def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        category_hint: Optional[QueryCategory] = None,
    ) -> AgentOutput:
        """Convenience method for direct query execution.

        Wraps process() for simple programmatic query access without
        constructing a full MasterRequest.

        Args:
            query_text: The natural language query string.
            session_id: Optional session ID for context.
            top_k: Number of documents to retrieve.
            category_hint: Optional category to bypass router.

        Returns:
            The AgentOutput from the retrieval pipeline.

        Raises:
            RuntimeError: If the query request fails.
        """
        request = MasterRequest(
            request_type=RequestType.QUERY,
            query=query_text,
            session_id=session_id,
            top_k=top_k,
            category_hint=category_hint,
        )

        response = self.process(request)

        if not response.success or not response.query_result:
            raise RuntimeError(
                response.error or "Query failed with no error message."
            )

        return response.query_result

    def run_audit(
        self,
        scope: AuditScope = AuditScope.FULL,
    ) -> AuditReport:
        """Convenience method for direct audit execution.

        Args:
            scope: The audit scope to run.

        Returns:
            The AuditReport from the audit pipeline.

        Raises:
            RuntimeError: If the audit request fails.
        """
        request = MasterRequest(
            request_type=RequestType.AUDIT,
            audit_scope=scope,
        )

        response = self.process(request)

        if not response.success or not response.audit_report:
            raise RuntimeError(
                response.error or "Audit failed with no error message."
            )

        return response.audit_report


# ── Module-level singleton ────────────────────────────────────────────────────
# Imported by the FastAPI app and Streamlit UI.

master_orchestrator = MasterOrchestrator()

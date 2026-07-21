"""
Module: orchestrators/audit_orchestrator.py

Purpose:
    Orchestrates all audit agents to proactively scan the institutional
    memory for knowledge gaps, stale content, and single points of failure.

Responsibilities:
    - Coordinate GapDetectorAgent, StalenessAgent, and
      SinglePointOfFailureAgent in a unified audit scan.
    - Run agents sequentially or selectively based on audit scope.
    - Aggregate findings from all audit agents into a single AuditReport.
    - Provide a lightweight quick-scan mode for frequent checks.
    - Return a structured AuditReport for the dashboard and API.

Workflow:
    Phase 1 — Receive AuditRequest specifying scan scope.
    Phase 2 — Run selected audit agents based on scope.
    Phase 3 — Collect and aggregate all findings.
    Phase 4 — Generate a unified executive summary via LLM.
    Phase 5 — Return a structured AuditReport.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel, Field

from agents.audit.gap_detector_agent import GapDetectorAgent, GapDetectorOutput
from agents.audit.staleness_agent import StalenessAgent, StalenessScanOutput
from agents.audit.single_point_failure_agent import (
    SinglePointOfFailureAgent,
    SPFScanOutput,
)
from config.settings import settings
from memory.memory_manager import MemoryManager, memory_manager
from schemas.memory_schema import KnowledgeGap, KnowledgeGapSeverity


# ── Audit Scope Enumeration ──────────────────────────────────────────────────

class AuditScope(str, Enum):
    """Defines the scope of an audit scan."""

    FULL = "full"               # Run all three audit agents
    GAPS_ONLY = "gaps_only"     # Run only GapDetectorAgent
    STALENESS_ONLY = "staleness_only"  # Run only StalenessAgent
    SPF_ONLY = "spf_only"       # Run only SinglePointOfFailureAgent
    QUICK = "quick"             # Run gap detector only with reduced depth


# ── Audit Request & Report Models ────────────────────────────────────────────

class AuditRequest(BaseModel):
    """Input model for an audit scan request."""

    scope: AuditScope = Field(
        default=AuditScope.FULL,
        description="Scope of the audit scan"
    )
    session_id: Optional[str] = Field(
        None, description="Optional session ID for tracing"
    )
    generate_summary: bool = Field(
        default=True,
        description="Whether to generate LLM executive summary"
    )


class AuditFinding(BaseModel):
    """A single normalised finding from any audit agent."""

    finding_id: str = Field(..., description="Unique finding identifier")
    source_agent: str = Field(..., description="Agent that produced this finding")
    severity: str = Field(..., description="CRITICAL, HIGH, MEDIUM, or LOW")
    affected_area: str = Field(..., description="Area affected by this finding")
    description: str = Field(..., description="Description of the finding")
    recommended_action: str = Field(..., description="Action to address finding")
    category: str = Field(
        ..., description="GAP, STALENESS, or SINGLE_POINT_OF_FAILURE"
    )


class AuditReport(BaseModel):
    """Comprehensive audit report aggregating all agent findings."""

    report_id: str = Field(..., description="Unique report identifier")
    scope: AuditScope = Field(..., description="Scope of the audit performed")
    session_id: Optional[str] = Field(None)
    scanned_at: datetime = Field(default_factory=datetime.utcnow)

    # Agent-specific raw outputs
    gap_output: Optional[GapDetectorOutput] = Field(None)
    staleness_output: Optional[StalenessScanOutput] = Field(None)
    spf_output: Optional[SPFScanOutput] = Field(None)

    # Aggregated findings
    total_findings: int = Field(default=0)
    critical_findings: int = Field(default=0)
    high_findings: int = Field(default=0)
    findings: list[AuditFinding] = Field(default_factory=list)

    # Summary
    overall_health: str = Field(
        default="UNKNOWN",
        description="HEALTHY, DEGRADED, or CRITICAL"
    )
    executive_summary: str = Field(
        default="", description="LLM-generated executive summary"
    )
    top_actions: list[str] = Field(
        default_factory=list,
        description="Top 5 recommended actions across all findings"
    )


# ── Health Score Thresholds ──────────────────────────────────────────────────

_CRITICAL_HEALTH_THRESHOLD = 3    # ≥ this many critical findings → CRITICAL
_DEGRADED_HEALTH_THRESHOLD = 1    # ≥ this many critical findings → DEGRADED


class AuditOrchestrator:
    """Orchestrates all audit agents for institutional memory health checks.

    Coordinates gap detection, staleness analysis, and single point of
    failure identification into a unified audit workflow. Returns a
    comprehensive AuditReport consumed by the dashboard and API.

    Attributes:
        _gap_agent: Detects knowledge gaps in the memory system.
        _staleness_agent: Detects stale or outdated content.
        _spf_agent: Identifies single points of knowledge failure.
        _llm: Gemini LLM for executive summary generation.
    """

    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        """Initialises the AuditOrchestrator with all audit agents.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        mem = memory or memory_manager

        self._gap_agent = GapDetectorAgent(memory=mem)
        self._staleness_agent = StalenessAgent(memory=mem)
        self._spf_agent = SinglePointOfFailureAgent(memory=mem)

        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini.model,
            google_api_key=settings.gemini.api_key,
            temperature=0.1,
            max_output_tokens=1000,
        )

        logger.info("AuditOrchestrator initialised.")

    # ── Finding Normalisation ─────────────────────────────────────────────────

    def _normalise_gap_findings(
        self,
        output: GapDetectorOutput,
    ) -> list[AuditFinding]:
        """Converts GapDetectorOutput gaps into normalised AuditFinding objects.

        Args:
            output: Raw GapDetectorOutput from the GapDetectorAgent.

        Returns:
            List of normalised AuditFinding objects.
        """
        findings: list[AuditFinding] = []

        for gap in output.gaps:
            findings.append(
                AuditFinding(
                    finding_id=gap.gap_id,
                    source_agent="GapDetectorAgent",
                    severity=gap.severity.value,
                    affected_area=gap.affected_area,
                    description=gap.description,
                    recommended_action=gap.recommended_action,
                    category="GAP",
                )
            )

        return findings

    def _normalise_staleness_findings(
        self,
        output: StalenessScanOutput,
    ) -> list[AuditFinding]:
        """Converts StalenessScanOutput into normalised AuditFinding objects.

        Args:
            output: Raw StalenessScanOutput from the StalenessAgent.

        Returns:
            List of normalised AuditFinding objects.
        """
        findings: list[AuditFinding] = []

        for report in output.findings:
            findings.append(
                AuditFinding(
                    finding_id=report.report_id,
                    source_agent="StalenessAgent",
                    severity=report.severity,
                    affected_area=report.affected_area,
                    description=report.description,
                    recommended_action=report.recommended_action,
                    category="STALENESS",
                )
            )

        return findings

    def _normalise_spf_findings(
        self,
        output: SPFScanOutput,
    ) -> list[AuditFinding]:
        """Converts SPFScanOutput into normalised AuditFinding objects.

        Args:
            output: Raw SPFScanOutput from the SinglePointOfFailureAgent.

        Returns:
            List of normalised AuditFinding objects.
        """
        findings: list[AuditFinding] = []

        for spf in output.findings:
            findings.append(
                AuditFinding(
                    finding_id=spf.spf_id,
                    source_agent="SinglePointOfFailureAgent",
                    severity=spf.risk_level,
                    affected_area=spf.affected_area,
                    description=spf.description,
                    recommended_action=spf.recommended_action,
                    category="SINGLE_POINT_OF_FAILURE",
                )
            )

        return findings

    # ── Health Assessment ────────────────────────────────────────────────────

    def _assess_overall_health(
        self, critical_count: int
    ) -> str:
        """Determines overall system health based on critical finding count.

        Args:
            critical_count: Total number of CRITICAL findings across agents.

        Returns:
            Health status string: HEALTHY, DEGRADED, or CRITICAL.
        """
        if critical_count >= _CRITICAL_HEALTH_THRESHOLD:
            return "CRITICAL"
        if critical_count >= _DEGRADED_HEALTH_THRESHOLD:
            return "DEGRADED"
        return "HEALTHY"

    # ── Executive Summary ────────────────────────────────────────────────────

    def _generate_executive_summary(
        self,
        report: AuditReport,
    ) -> str:
        """Generates a unified executive summary across all audit findings.

        Args:
            report: The partially-populated AuditReport with all findings.

        Returns:
            A concise executive summary string from the LLM.
        """
        findings_text = "\n".join([
            f"- [{f.severity}] {f.category}: {f.description}"
            for f in report.findings[:15]
        ])

        prompt = f"""
You are the Chief Audit Intelligence Agent for a Corporate Institutional Memory System.
Produce a concise executive summary of the following institutional memory audit findings.

AUDIT SCOPE: {report.scope.value}
OVERALL HEALTH: {report.overall_health}
TOTAL FINDINGS: {report.total_findings}
CRITICAL FINDINGS: {report.critical_findings}

TOP FINDINGS:
{findings_text}

INSTRUCTIONS:
1. Summarise the overall health of the institutional memory system.
2. Highlight the 3 most critical risks and their business impact.
3. Provide 5 specific, actionable recommendations ordered by priority.
4. Keep the total summary under 400 words.
5. Use clear, executive-friendly language.

Write the executive summary now:
""".strip()

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as exc:
            logger.warning(
                "AuditOrchestrator LLM summary failed: {}", exc
            )
            return (
                f"Audit complete. Overall health: {report.overall_health}. "
                f"Found {report.total_findings} findings "
                f"({report.critical_findings} critical)."
            )

    # ── Top Actions Extraction ────────────────────────────────────────────────

    def _extract_top_actions(
        self,
        findings: list[AuditFinding],
        limit: int = 5,
    ) -> list[str]:
        """Extracts the top recommended actions from critical findings.

        Prioritises CRITICAL severity findings, then HIGH, then MEDIUM.

        Args:
            findings: All normalised AuditFinding objects.
            limit: Maximum number of actions to return.

        Returns:
            List of top recommended action strings.
        """
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_findings = sorted(
            findings,
            key=lambda f: severity_order.get(f.severity, 4),
        )

        seen: set[str] = set()
        actions: list[str] = []

        for finding in sorted_findings:
            action = finding.recommended_action
            if action not in seen:
                seen.add(action)
                actions.append(action)
            if len(actions) >= limit:
                break

        return actions

    # ── Agent Runners ────────────────────────────────────────────────────────

    def _run_gap_agent(self) -> Optional[GapDetectorOutput]:
        """Safely runs the GapDetectorAgent and returns its output.

        Returns:
            GapDetectorOutput or None if the agent fails.
        """
        try:
            logger.info("AuditOrchestrator: running GapDetectorAgent.")
            output = self._gap_agent.scan()
            logger.info(
                "GapDetectorAgent complete | gaps={}",
                output.gaps_found,
            )
            return output
        except Exception as exc:
            logger.error("GapDetectorAgent failed: {}", exc)
            return None

    def _run_staleness_agent(self) -> Optional[StalenessScanOutput]:
        """Safely runs the StalenessAgent and returns its output.

        Returns:
            StalenessScanOutput or None if the agent fails.
        """
        try:
            logger.info("AuditOrchestrator: running StalenessAgent.")
            output = self._staleness_agent.scan()
            logger.info(
                "StalenessAgent complete | findings={}",
                output.total_findings,
            )
            return output
        except Exception as exc:
            logger.error("StalenessAgent failed: {}", exc)
            return None

    def _run_spf_agent(self) -> Optional[SPFScanOutput]:
        """Safely runs the SinglePointOfFailureAgent and returns its output.

        Returns:
            SPFScanOutput or None if the agent fails.
        """
        try:
            logger.info(
                "AuditOrchestrator: running SinglePointOfFailureAgent."
            )
            output = self._spf_agent.scan()
            logger.info(
                "SinglePointOfFailureAgent complete | findings={}",
                output.spf_count,
            )
            return output
        except Exception as exc:
            logger.error("SinglePointOfFailureAgent failed: {}", exc)
            return None

    # ── Public Entry Point ───────────────────────────────────────────────────

    def audit(self, request: AuditRequest) -> AuditReport:
        """Runs a full or scoped institutional memory audit.

        This is the single public entry point for all audit operations.
        Coordinates agent execution, aggregates findings, assesses health,
        and generates an executive summary.

        Args:
            request: A validated AuditRequest specifying audit scope.

        Returns:
            A comprehensive AuditReport with all findings and summary.
        """
        import uuid

        report_id = f"audit_{uuid.uuid4().hex[:12]}"

        logger.info(
            "AuditOrchestrator starting | scope='{}' | report_id='{}'",
            request.scope.value,
            report_id,
        )

        all_findings: list[AuditFinding] = []
        gap_output: Optional[GapDetectorOutput] = None
        staleness_output: Optional[StalenessScanOutput] = None
        spf_output: Optional[SPFScanOutput] = None

        # ── Phase 2: Run agents based on scope ───────────────────────────────
        if request.scope in {AuditScope.FULL, AuditScope.GAPS_ONLY, AuditScope.QUICK}:
            gap_output = self._run_gap_agent()
            if gap_output:
                all_findings.extend(
                    self._normalise_gap_findings(gap_output)
                )

        if request.scope in {AuditScope.FULL, AuditScope.STALENESS_ONLY}:
            staleness_output = self._run_staleness_agent()
            if staleness_output:
                all_findings.extend(
                    self._normalise_staleness_findings(staleness_output)
                )

        if request.scope in {AuditScope.FULL, AuditScope.SPF_ONLY}:
            spf_output = self._run_spf_agent()
            if spf_output:
                all_findings.extend(
                    self._normalise_spf_findings(spf_output)
                )

        # ── Phase 3: Aggregate findings ───────────────────────────────────────
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_findings.sort(
            key=lambda f: severity_order.get(f.severity, 4)
        )

        critical_count = sum(
            1 for f in all_findings if f.severity == "CRITICAL"
        )
        high_count = sum(
            1 for f in all_findings if f.severity == "HIGH"
        )

        overall_health = self._assess_overall_health(critical_count)
        top_actions = self._extract_top_actions(all_findings)

        # ── Phase 4: Build partial report for summary generation ──────────────
        report = AuditReport(
            report_id=report_id,
            scope=request.scope,
            session_id=request.session_id,
            gap_output=gap_output,
            staleness_output=staleness_output,
            spf_output=spf_output,
            total_findings=len(all_findings),
            critical_findings=critical_count,
            high_findings=high_count,
            findings=all_findings,
            overall_health=overall_health,
            top_actions=top_actions,
        )

        # ── Phase 5: Generate executive summary ───────────────────────────────
        if request.generate_summary and all_findings:
            report.executive_summary = self._generate_executive_summary(report)
        elif not all_findings:
            report.executive_summary = (
                "Institutional memory audit complete. "
                "No significant issues detected. System health: HEALTHY."
            )

        logger.info(
            "AuditOrchestrator complete | report_id='{}' | health='{}' | "
            "total={} | critical={} | high={}",
            report_id,
            overall_health,
            len(all_findings),
            critical_count,
            high_count,
        )

        return report

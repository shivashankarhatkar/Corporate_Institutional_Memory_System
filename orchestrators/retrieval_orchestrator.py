"""
Module: orchestrators/retrieval_orchestrator.py

Purpose:
    Orchestrates all retrieval agents to answer user queries by routing
    classified queries to the correct specialist agent and returning
    a unified response.

Responsibilities:
    - Receive a classified query from the Master Orchestrator.
    - Route the query to the correct specialist retrieval agent based
      on QueryCategory from the Router Agent.
    - Execute the selected agent and return the AgentOutput.
    - Handle fallback routing when confidence is below threshold.
    - Support multi-agent retrieval for UNKNOWN category queries.
    - Cache results via MemoryManager for repeated queries.

Workflow:
    Phase 1 — Receive RetrievalRequest with query and category.
    Phase 2 — Select the appropriate specialist agent.
    Phase 3 — Execute the agent via execute() for timing and logging.
    Phase 4 — Apply fallback if confidence is below threshold.
    Phase 5 — Return the final AgentOutput.
"""

from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from agents.retrieval.router_agent import RouterAgent, RouterOutput
from agents.retrieval.decision_agent import DecisionAgent
from agents.retrieval.people_agent import PeopleAgent
from agents.retrieval.policy_agent import PolicyAgent
from agents.retrieval.project_agent import ProjectAgent
from agents.retrieval.competitive_agent import CompetitiveAgent
from memory.memory_manager import MemoryManager, memory_manager
from schemas.agent_schema import (
    AgentInput,
    AgentOutput,
    AgentStatus,
    QueryCategory,
)


# ── Retrieval Request Model ──────────────────────────────────────────────────

class RetrievalRequest(BaseModel):
    """Input model for a knowledge retrieval request.

    Attributes:
        query: The user's natural language question.
        session_id: Optional session ID for context tracking.
        top_k: Number of documents to retrieve per agent.
        category_hint: Optional category to bypass router classification.
        metadata_filter: Optional ChromaDB filter for targeted retrieval.
    """

    query: str = Field(
        ..., min_length=3,
        description="User's natural language query"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for context tracking"
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Documents to retrieve per agent call"
    )
    category_hint: Optional[QueryCategory] = Field(
        None, description="Optional category hint to bypass router"
    )
    metadata_filter: Optional[dict] = Field(
        None, description="Optional ChromaDB metadata filter"
    )


# ── Confidence Threshold ─────────────────────────────────────────────────────

# Below this confidence, fallback to a secondary agent is attempted
_FALLBACK_CONFIDENCE_THRESHOLD = 0.3


class RetrievalOrchestrator:
    """Orchestrates all retrieval agents for query answering.

    Receives classified queries from the Master Orchestrator, routes
    them to specialist agents, and applies fallback strategies when
    primary agent confidence is insufficient.

    Attributes:
        _router: RouterAgent for query classification.
        _decision_agent: Handles DECISION category queries.
        _people_agent: Handles PEOPLE category queries.
        _policy_agent: Handles POLICY category queries.
        _project_agent: Handles PROJECT category queries.
        _competitive_agent: Handles competitive intelligence queries.
    """

    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        """Initialises the RetrievalOrchestrator with all retrieval agents.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        mem = memory or memory_manager

        self._router = RouterAgent(memory=mem)
        self._decision_agent = DecisionAgent(memory=mem)
        self._people_agent = PeopleAgent(memory=mem)
        self._policy_agent = PolicyAgent(memory=mem)
        self._project_agent = ProjectAgent(memory=mem)
        self._competitive_agent = CompetitiveAgent(memory=mem)

        # Category → agent dispatch map
        self._agent_map = {
            QueryCategory.DECISION: self._decision_agent,
            QueryCategory.PEOPLE: self._people_agent,
            QueryCategory.POLICY: self._policy_agent,
            QueryCategory.PROJECT: self._project_agent,
        }

        logger.info("RetrievalOrchestrator initialised.")

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _build_agent_input(
        self,
        request: RetrievalRequest,
        category: QueryCategory,
    ) -> AgentInput:
        """Constructs a standardised AgentInput for a retrieval agent.

        Args:
            request: The incoming RetrievalRequest.
            category: The classified QueryCategory.

        Returns:
            A populated AgentInput ready for agent execution.
        """
        return AgentInput(
            query=request.query,
            category=category,
            session_id=request.session_id,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

    def _is_competitive_query(self, query: str) -> bool:
        """Detects whether a query is competitive intelligence-related.

        Args:
            query: The user's query string.

        Returns:
            True if the query contains competitive intelligence signals.
        """
        competitive_signals = [
            "competitor", "competition", "versus", " vs ",
            "rival", "market share", "industry", "benchmark",
            "compared to", "better than", "worse than",
        ]
        query_lower = query.lower()
        return any(signal in query_lower for signal in competitive_signals)

    def _execute_primary_agent(
        self,
        request: RetrievalRequest,
        router_output: RouterOutput,
    ) -> AgentOutput:
        """Executes the primary specialist agent for the classified category.

        Args:
            request: The incoming RetrievalRequest.
            router_output: Classification output from the Router Agent.

        Returns:
            AgentOutput from the primary specialist agent.
        """
        category = router_output.category

        # Check for competitive intelligence signals first
        if self._is_competitive_query(request.query):
            logger.info(
                "RetrievalOrchestrator: competitive query detected — "
                "routing to CompetitiveAgent."
            )
            agent_input = self._build_agent_input(
                request, QueryCategory.DECISION
            )
            return self._competitive_agent.execute(agent_input)

        # Route to specialist agent by category
        agent = self._agent_map.get(category)

        if agent:
            agent_input = self._build_agent_input(request, category)
            return agent.execute(agent_input)

        # UNKNOWN category — run all agents and pick best result
        logger.info(
            "RetrievalOrchestrator: UNKNOWN category — "
            "running multi-agent retrieval."
        )
        return self._multi_agent_retrieval(request)

    def _multi_agent_retrieval(
        self, request: RetrievalRequest
    ) -> AgentOutput:
        """Runs all specialist agents and returns the highest-confidence answer.

        Used when the Router Agent classifies a query as UNKNOWN or when
        no single category is clearly dominant.

        Args:
            request: The incoming RetrievalRequest.

        Returns:
            The AgentOutput with the highest confidence score.
        """
        logger.info("Running multi-agent retrieval for UNKNOWN query.")

        results: list[AgentOutput] = []

        for category, agent in self._agent_map.items():
            try:
                agent_input = self._build_agent_input(request, category)
                output = agent.execute(agent_input)
                if output.status != AgentStatus.FAILED:
                    results.append(output)
            except Exception as exc:
                logger.warning(
                    "Multi-agent retrieval: agent '{}' failed: {}",
                    agent.agent_name,
                    exc,
                )

        if not results:
            return AgentOutput(
                agent_name="RetrievalOrchestrator",
                query=request.query,
                answer=(
                    "No relevant information found across all knowledge "
                    "categories. Please try rephrasing your query."
                ),
                sources=[],
                status=AgentStatus.PARTIAL,
                confidence=0.0,
            )

        # Return the result with highest confidence
        best = max(
            results,
            key=lambda r: r.confidence or 0.0,
        )

        logger.info(
            "Multi-agent retrieval complete | best_agent='{}' | confidence={}",
            best.agent_name,
            best.confidence,
        )

        return best

    def _apply_fallback(
        self,
        request: RetrievalRequest,
        primary_output: AgentOutput,
        router_output: RouterOutput,
    ) -> AgentOutput:
        """Attempts a fallback agent when primary confidence is too low.

        If the primary agent returns low confidence and the router
        suggested a fallback category, runs the fallback agent and
        returns whichever result has higher confidence.

        Args:
            request: The incoming RetrievalRequest.
            primary_output: The low-confidence primary agent output.
            router_output: Router output containing fallback category.

        Returns:
            The higher-confidence result between primary and fallback.
        """
        fallback_category = router_output.fallback_category

        if not fallback_category or fallback_category == QueryCategory.UNKNOWN:
            logger.debug(
                "No valid fallback category available — returning primary."
            )
            return primary_output

        fallback_agent = self._agent_map.get(fallback_category)

        if not fallback_agent:
            return primary_output

        logger.info(
            "Applying fallback | primary='{}' (confidence={}) → fallback='{}'",
            router_output.category.value,
            primary_output.confidence,
            fallback_category.value,
        )

        try:
            agent_input = self._build_agent_input(request, fallback_category)
            fallback_output = fallback_agent.execute(agent_input)

            # Return whichever has higher confidence
            if (fallback_output.confidence or 0.0) > (
                primary_output.confidence or 0.0
            ):
                logger.info(
                    "Fallback agent outperformed primary | "
                    "fallback_confidence={}",
                    fallback_output.confidence,
                )
                return fallback_output

        except Exception as exc:
            logger.warning("Fallback agent execution failed: {}", exc)

        return primary_output

    # ── Public Entry Point ───────────────────────────────────────────────────

    def retrieve(self, request: RetrievalRequest) -> AgentOutput:
        """Orchestrates retrieval for a user query end-to-end.

        This is the single public entry point called by the Master
        Orchestrator for all retrieval operations. Handles routing,
        primary execution, and fallback in a single call.

        Args:
            request: A validated RetrievalRequest with query and options.

        Returns:
            The final AgentOutput with answer, sources, and metadata.
        """
        logger.info(
            "RetrievalOrchestrator received query | query='{}'",
            request.query[:80],
        )

        # ── Phase 1: Classify query (or use hint) ────────────────────────────
        if request.category_hint:
            logger.info(
                "Using category hint: '{}'",
                request.category_hint.value,
            )
            router_output = RouterOutput(
                query=request.query,
                category=request.category_hint,
                confidence=1.0,
                reasoning="Category provided directly by caller.",
                fallback_category=None,
            )
        else:
            router_output = self._router.classify(request.query)

        logger.info(
            "Query classified | category='{}' | confidence={}",
            router_output.category.value,
            router_output.confidence,
        )

        # ── Phase 2: Execute primary agent ───────────────────────────────────
        primary_output = self._execute_primary_agent(request, router_output)

        # ── Phase 3: Apply fallback if confidence is too low ─────────────────
        if (primary_output.confidence or 0.0) < _FALLBACK_CONFIDENCE_THRESHOLD:
            final_output = self._apply_fallback(
                request, primary_output, router_output
            )
        else:
            final_output = primary_output

        logger.info(
            "RetrievalOrchestrator complete | agent='{}' | "
            "confidence={} | sources={}",
            final_output.agent_name,
            final_output.confidence,
            len(final_output.sources),
        )

        return final_output

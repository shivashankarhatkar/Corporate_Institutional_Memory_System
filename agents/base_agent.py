"""
Module: agents/base_agent.py

Purpose:
    Defines the abstract base class that every agent in the Institutional
    Memory System must inherit from and implement. every agent will use this agent.

Responsibilities:
    - Enforce a consistent interface across all agents via abstract methods.
    - Initialise the shared Gemini LLM and MemoryManager for all agents.
    - Provide common utility methods: LLM invocation, context building,
      source extraction, and response formatting.
    - Implement cache-aware execution flow shared by all agents.
    - Provide structured logging for every agent run.

Workflow:
    Phase 1 — Subclass inherits BaseAgent and implements run().
    Phase 2 — Agent calls self._retrieve() to get relevant chunks.
    Phase 3 — Agent calls self._build_context() to format chunks.
    Phase 4 — Agent calls self._invoke_llm() with prompt + context.
    Phase 5 — Agent calls self._build_output() to return AgentOutput.
"""

from abc import ABC, abstractmethod
from typing import Optional
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from loguru import logger

from config.settings import settings
from memory.memory_manager import memory_manager, MemoryManager
from schemas.agent_schema import (
    AgentInput,
    AgentOutput,
    AgentStatus,
    Source,
    QueryCategory,
)
from schemas.memory_schema import VectorSearchResult


class BaseAgent(ABC):
    """Abstract base class for all agents in the Institutional Memory System.

    Every agent in the system inherits from BaseAgent and must implement
    the run() method. Common LLM initialisation, retrieval, context
    building, and output formatting are provided here to eliminate
    duplication across agent implementations.

    Attributes:
        agent_name: Human-readable name identifying this agent.
        category: The QueryCategory this agent specialises in.
        _llm: The shared ChatGoogleGenerativeAI instance.
        _memory: The shared MemoryManager instance.
    """

    def __init__(
        self,
        agent_name: str,
        category: QueryCategory,
        memory: Optional[MemoryManager] = None,
    ) -> None:
        """Initialises the BaseAgent with LLM and memory dependencies.

        Args:
            agent_name: Human-readable name for this agent instance.
            category: The QueryCategory this agent handles.
            memory: Optional MemoryManager instance for dependency
                injection. Uses the module singleton if not provided.
        """
        self.agent_name = agent_name
        self.category = category
        self._memory = memory or memory_manager

        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini.model,
            google_api_key=settings.gemini.api_key,
            temperature=settings.gemini.temperature,
            max_output_tokens=settings.gemini.max_tokens,
        )

        logger.info(
            "Agent initialised | name='{}' | category='{}' | model='{}'",
            self.agent_name,
            self.category.value,
            settings.gemini.model,
        )

    # ── Abstract Interface ───────────────────────────────────────────────────

    @abstractmethod
    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Executes the agent's core logic for a given input.

        Every subclass must implement this method. It is the single
        entry point called by orchestrators to invoke an agent.

        Args:
            agent_input: The standardised AgentInput containing the
                query, category, filters, and session context.

        Returns:
            A fully populated AgentOutput with answer and sources.
        """

    @abstractmethod
    def _build_prompt(self, query: str, context: str) -> str:
        """Builds the agent-specific prompt string.

        Each agent defines its own prompt template tuned for its
        specialisation (decision, people, policy, project).

        Args:
            query: The user's original query string.
            context: The formatted retrieval context string.

        Returns:
            The complete prompt string to send to the LLM.
        """

    # ── Shared Utility Methods ───────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """Retrieves relevant chunks from memory for the given query.

        Delegates to MemoryManager which handles cache-first retrieval
        and vector store fallback automatically.

        Args:
            query: The natural language query to retrieve context for.
            top_k: Number of results to retrieve.
            metadata_filter: Optional ChromaDB metadata filter.

        Returns:
            List of VectorSearchResult objects ordered by relevance.
        """
        results = self._memory.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            agent_context=self.agent_name,
        )

        logger.debug(
            "Retrieved {} chunks | agent='{}' | query='{}'",
            len(results),
            self.agent_name,
            query[:60],
        )

        return results

    def _build_context(self, results: list[VectorSearchResult]) -> str:
        """Formats retrieved chunks into a numbered context string for the LLM.

        Each chunk is formatted with its index, sender, date, and text
        so the LLM can reference specific sources in its answer.

        Args:
            results: List of VectorSearchResult objects to format.

        Returns:
            A formatted multi-line string ready for prompt injection.
            Returns a fallback message if results list is empty.
        """
        if not results:
            return "No relevant documents found in institutional memory."

        context_parts: list[str] = []

        for i, result in enumerate(results, start=1):
            sender = result.metadata.get("sender", "unknown")
            date = result.metadata.get("date", "unknown date")
            subject = result.metadata.get("subject", "no subject")

            context_parts.append(
                f"[Source {i}]\n"
                f"Sender : {sender}\n"
                f"Date   : {date}\n"
                f"Subject: {subject}\n"
                f"Content: {result.text}\n"
            )

        return "\n---\n".join(context_parts)

    def _invoke_llm(self, prompt: str) -> str:
        """Sends a prompt to the Gemini LLM and returns the response text.

        Args:
            prompt: The complete prompt string to send to the LLM.

        Returns:
            The LLM's response as a plain string.

        Raises:
            RuntimeError: If the LLM call fails after retries.
        """
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        except Exception as exc:
            logger.error(
                "LLM invocation failed | agent='{}': {}",
                self.agent_name,
                exc,
            )
            raise RuntimeError(
                f"LLM call failed in agent '{self.agent_name}': {exc}"
            ) from exc

    def _extract_sources(
        self, results: list[VectorSearchResult]
    ) -> list[Source]:
        """Converts VectorSearchResult objects into Source models.

        Extracts the metadata fields required by the AgentOutput.sources
        field from each retrieved chunk.

        Args:
            results: List of VectorSearchResult objects to convert.

        Returns:
            List of Source objects ready for inclusion in AgentOutput.
        """
        sources: list[Source] = []

        for result in results:
            meta = result.metadata

            sources.append(
                Source(
                    document_id=result.chunk_id,
                    message_id=meta.get("message_id", "unknown"),
                    sender=meta.get("sender", "unknown"),
                    date=meta.get("date", ""),
                    subject=meta.get("subject", ""),
                    excerpt=result.text[:300],
                    relevance_score=result.relevance_score,
                )
            )

        return sources

    def _build_output(
        self,
        query: str,
        answer: str,
        sources: list[Source],
        status: AgentStatus = AgentStatus.SUCCESS,
        confidence: Optional[float] = None,
        follow_up_questions: Optional[list[str]] = None,
    ) -> AgentOutput:
        """Constructs a standardised AgentOutput from agent results.

        Args:
            query: The original user query string.
            answer: The LLM-generated answer string.
            sources: List of Source objects used to generate the answer.
            status: Execution status. Defaults to SUCCESS.
            confidence: Optional self-assessed confidence score (0-1).
            follow_up_questions: Optional list of suggested follow-ups.

        Returns:
            A fully populated AgentOutput instance.
        """
        return AgentOutput(
            agent_name=self.agent_name,
            query=query,
            answer=answer,
            sources=sources,
            category=self.category,
            status=status,
            confidence=confidence,
            follow_up_questions=follow_up_questions or [],
        )

    def _handle_empty_results(self, query: str) -> AgentOutput:
        """Returns a graceful AgentOutput when no relevant chunks are found.

        Args:
            query: The original user query string.

        Returns:
            An AgentOutput with PARTIAL status and an explanatory answer.
        """
        logger.warning(
            "No results found | agent='{}' | query='{}'",
            self.agent_name,
            query[:60],
        )

        return self._build_output(
            query=query,
            answer=(
                "I could not find relevant information in the institutional "
                "memory for your query. The knowledge base may not contain "
                "data on this topic, or try rephrasing your question."
            ),
            sources=[],
            status=AgentStatus.PARTIAL,
            confidence=0.0,
        )

    def _timed_run(self, agent_input: AgentInput) -> tuple[AgentOutput, float]:
        """Executes run() and measures wall-clock execution time.

        Args:
            agent_input: The standardised AgentInput to process.

        Returns:
            A tuple of (AgentOutput, duration_seconds).
        """
        start = time.perf_counter()
        output = self.run(agent_input)
        duration = round(time.perf_counter() - start, 3)
        return output, duration

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Public entry point that wraps run() with logging and timing.

        Orchestrators should call execute() rather than run() directly
        to ensure consistent logging and error handling across all agents.

        Args:
            agent_input: The standardised AgentInput to process.

        Returns:
            AgentOutput from the agent's run() implementation.
        """
        logger.info(
            "Agent executing | name='{}' | query='{}'",
            self.agent_name,
            agent_input.query[:60],
        )

        try:
            output, duration = self._timed_run(agent_input)

            logger.info(
                "Agent complete | name='{}' | status='{}' | "
                "sources={} | duration={}s",
                self.agent_name,
                output.status.value,
                len(output.sources),
                duration,
            )

            return output

        except Exception as exc:
            logger.error(
                "Agent failed | name='{}' | query='{}' | error={}",
                self.agent_name,
                agent_input.query[:60],
                exc,
            )

            return self._build_output(
                query=agent_input.query,
                answer=f"Agent '{self.agent_name}' encountered an error: {exc}",
                sources=[],
                status=AgentStatus.FAILED,
                confidence=0.0,
            )

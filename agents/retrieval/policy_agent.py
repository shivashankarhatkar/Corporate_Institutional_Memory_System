"""
Module: agents/retrieval/policy_agent.py

Purpose:
    Retrieves and synthesises answers to policy-related queries from
    the institutional memory using RAG over ChromaDB vector store.

Responsibilities:
    - Handle queries classified as QueryCategory.POLICY by the router.
    - Retrieve relevant email chunks related to policies and procedures.
    - Retrieve synthetic policy documents from ChromaDB if available.
    - Synthesise grounded answers about company rules, procedures,
      compliance requirements, and standard operating practices.
    - Always include policy reasoning — not just the rule itself.
    - Return structured sources and follow-up questions.

Workflow:
    Phase 1 — Receive AgentInput with a POLICY category query.
    Phase 2 — Retrieve relevant chunks from ChromaDB.
    Phase 3 — Detect policy topic and apply metadata filters if possible.
    Phase 4 — Build policy-focused context from retrieved chunks.
    Phase 5 — Invoke Gemini with policy-specific prompt.
    Phase 6 — Parse follow-up questions from LLM response.
    Phase 7 — Return AgentOutput with answer, sources, and follow-ups.
"""

import re
from typing import Optional

from loguru import logger

from agents.base_agent import BaseAgent
from schemas.agent_schema import (
    AgentInput,
    AgentOutput,
    AgentStatus,
    QueryCategory,
)
from schemas.memory_schema import VectorSearchResult


# ── Policy Topic Keywords ────────────────────────────────────────────────────
# Maps policy topic labels to associated keywords for context enrichment.

_POLICY_TOPIC_MAP: dict[str, list[str]] = {
    "travel": ["travel", "trip", "flight", "hotel", "expense", "reimbursement"],
    "hr": ["leave", "vacation", "sick", "absence", "attendance", "remote", "wfh"],
    "finance": ["budget", "expense", "payment", "invoice", "approval", "spend"],
    "compliance": ["compliance", "regulation", "legal", "audit", "policy", "rule"],
    "security": ["password", "access", "security", "confidential", "data", "privacy"],
    "communication": ["email", "communication", "disclosure", "press", "media"],
    "trading": ["trading", "position", "risk", "limit", "hedge", "market"],
}


class PolicyAgent(BaseAgent):
    """Specialist agent for policy-related institutional memory queries.

    Answers questions about company rules, procedures, compliance
    requirements, and standard operating practices. Focuses on
    explaining not just WHAT the policy is, but WHY it exists
    and WHO is responsible for it.

    Inherits all retrieval, LLM invocation, and output formatting
    utilities from BaseAgent.
    """

    def __init__(self, memory=None) -> None:
        """Initialises the PolicyAgent.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        super().__init__(
            agent_name="PolicyAgent",
            category=QueryCategory.POLICY,
            memory=memory,
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """Builds the policy-specific RAG prompt for Gemini.

        Instructs the LLM to focus on policy reasoning, ownership,
        history of changes, and compliance implications.

        Args:
            query: The user's original policy-related query.
            context: Formatted string of retrieved email chunks.

        Returns:
            The complete prompt string to send to Gemini.
        """
        return f"""
You are the Policy Intelligence Agent for a Corporate Institutional Memory System.
Your role is to answer questions about company policies, procedures, compliance
requirements, and standard operating practices using email evidence.

RETRIEVED POLICY EVIDENCE:
{context}

USER QUERY:
{query}

INSTRUCTIONS:
1. Answer using ONLY the evidence provided in the retrieved emails above.
2. Always explain WHAT the policy or procedure is.
3. Always explain WHY this policy exists if the reasoning is evident.
4. Identify WHO owns or enforces this policy if mentioned.
5. Note WHEN this policy was established or last changed if available.
6. Highlight any exceptions, edge cases, or special circumstances mentioned.
7. Cite sources using [Source N] inline in your answer.
8. If the policy is not found in the evidence, clearly state this and
   suggest what type of document might contain the answer.

After your main answer, provide exactly 3 follow-up questions:

FOLLOW_UP_1: <question>
FOLLOW_UP_2: <question>
FOLLOW_UP_3: <question>

Begin your answer now:
""".strip()

    def _detect_policy_topic(self, query: str) -> Optional[str]:
        """Detects the policy topic from the query using keyword matching.

        Maps the query to a known policy topic category from the
        _POLICY_TOPIC_MAP for potential metadata filtering.

        Args:
            query: The user query string to analyse.

        Returns:
            The matched policy topic label, or None if no match found.
        """
        query_lower = query.lower()

        for topic, keywords in _POLICY_TOPIC_MAP.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.debug(
                    "PolicyAgent detected topic: '{}'", topic
                )
                return topic

        return None

    def _build_enriched_query(self, query: str, topic: Optional[str]) -> str:
        """Enriches the query with policy-specific terminology.

        Appends topic keywords to the query to improve ChromaDB
        semantic retrieval for policy-related chunks.

        Args:
            query: The original user query string.
            topic: The detected policy topic label.

        Returns:
            An enriched query string for better semantic retrieval.
        """
        if not topic:
            return f"{query} policy procedure rule guideline compliance"

        topic_keywords = " ".join(_POLICY_TOPIC_MAP.get(topic, []))
        return f"{query} {topic_keywords} policy procedure"

    def _retrieve_with_fallback(
        self,
        query: str,
        enriched_query: str,
        top_k: int,
        metadata_filter: Optional[dict],
    ) -> list[VectorSearchResult]:
        """Retrieves chunks with progressive fallback strategy.

        Attempts retrieval in order:
        1. Enriched query with metadata filter
        2. Enriched query without filter
        3. Original query without filter

        Args:
            query: The original user query string.
            enriched_query: The topic-enriched query string.
            top_k: Number of results to retrieve.
            metadata_filter: Optional ChromaDB metadata filter.

        Returns:
            List of VectorSearchResult objects from the first
            successful retrieval attempt.
        """
        # Attempt 1 — enriched query with filter
        if metadata_filter:
            results = self._retrieve(
                query=enriched_query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            if results:
                logger.debug(
                    "PolicyAgent: retrieved {} results with filter.",
                    len(results),
                )
                return results

        # Attempt 2 — enriched query without filter
        results = self._retrieve(
            query=enriched_query,
            top_k=top_k,
        )
        if results:
            logger.debug(
                "PolicyAgent: retrieved {} results with enriched query.",
                len(results),
            )
            return results

        # Attempt 3 — original query without filter
        results = self._retrieve(
            query=query,
            top_k=top_k,
        )
        logger.debug(
            "PolicyAgent: retrieved {} results with original query.",
            len(results),
        )
        return results

    def _build_policy_context(
        self,
        results: list[VectorSearchResult],
        topic: Optional[str],
    ) -> str:
        """Builds an enriched context string with policy topic header.

        Prepends a topic label to the standard context so the LLM
        understands the policy domain it is reasoning about.

        Args:
            results: Retrieved VectorSearchResult objects.
            topic: Detected policy topic label for context header.

        Returns:
            Formatted context string with optional topic header.
        """
        base_context = self._build_context(results)

        if topic:
            topic_header = (
                f"[POLICY DOMAIN: {topic.upper()}]\n"
                f"The following evidence relates to {topic} policies "
                f"and procedures.\n\n"
            )
            return f"{topic_header}{base_context}"

        return base_context

    def _parse_follow_up_questions(
        self, llm_response: str
    ) -> tuple[str, list[str]]:
        """Separates the main answer from follow-up questions.

        Args:
            llm_response: The raw LLM response string.

        Returns:
            A tuple of (clean_answer, list_of_follow_up_questions).
        """
        pattern = re.compile(r"FOLLOW_UP_\d+:\s*(.+)", re.IGNORECASE)
        follow_ups = [m.strip() for m in pattern.findall(llm_response)]
        clean_answer = pattern.sub("", llm_response).strip()
        return clean_answer, follow_ups

    def _assess_confidence(
        self,
        results: list[VectorSearchResult],
        answer: str,
    ) -> float:
        """Estimates answer confidence from retrieval quality.

        Args:
            results: Retrieved VectorSearchResult objects.
            answer: The synthesised LLM answer string.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not results:
            return 0.0

        avg_relevance = sum(
            r.relevance_score for r in results
        ) / len(results)

        insufficient_signals = [
            "not found", "no policy", "cannot find", "no information",
            "not mentioned", "no evidence", "no record",
        ]
        penalty = 0.2 if any(
            s in answer.lower() for s in insufficient_signals
        ) else 0.0

        return max(0.0, round(min(1.0, avg_relevance - penalty), 2))

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Executes the Policy Agent's retrieval and synthesis pipeline.

        Args:
            agent_input: Standard AgentInput containing the query,
                top_k setting, and optional metadata filters.

        Returns:
            AgentOutput with a grounded policy-focused answer,
            sources, confidence score, and follow-up questions.
        """
        query = agent_input.query

        # ── Phase 3: Detect policy topic ─────────────────────────────────────
        topic = self._detect_policy_topic(query)
        enriched_query = self._build_enriched_query(query, topic)

        logger.debug(
            "PolicyAgent | topic='{}' | enriched_query='{}'",
            topic,
            enriched_query[:80],
        )

        # ── Phase 2 & 3: Retrieve with fallback strategy ─────────────────────
        results = self._retrieve_with_fallback(
            query=query,
            enriched_query=enriched_query,
            top_k=agent_input.top_k,
            metadata_filter=agent_input.metadata_filter,
        )

        if not results:
            return self._handle_empty_results(query)

        # ── Phase 4: Build policy context ────────────────────────────────────
        full_context = self._build_policy_context(results, topic)

        # ── Phase 5: Invoke LLM ───────────────────────────────────────────────
        prompt = self._build_prompt(query, full_context)

        try:
            raw_response = self._invoke_llm(prompt)
        except RuntimeError as exc:
            return self._build_output(
                query=query,
                answer=str(exc),
                sources=[],
                status=AgentStatus.FAILED,
                confidence=0.0,
            )

        # ── Phase 6 & 7: Parse and return output ─────────────────────────────
        clean_answer, follow_ups = self._parse_follow_up_questions(raw_response)
        sources = self._extract_sources(results)
        confidence = self._assess_confidence(results, clean_answer)

        logger.info(
            "PolicyAgent answered | topic='{}' | sources={} | confidence={}",
            topic or "general",
            len(sources),
            confidence,
        )

        return self._build_output(
            query=query,
            answer=clean_answer,
            sources=sources,
            status=AgentStatus.SUCCESS,
            confidence=confidence,
            follow_up_questions=follow_ups,
        )

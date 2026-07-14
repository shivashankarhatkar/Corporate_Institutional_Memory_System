"""
Module: agents/retrieval/decision_agent.py

Purpose:
    Retrieves and synthesises answers to decision-related queries from
    the institutional memory using RAG over the ChromaDB vector store.

Responsibilities:
    - Handle queries classified as QueryCategory.DECISION by the router.
    - Retrieve the most relevant email chunks from ChromaDB.
    - Query the Neo4j graph for decision nodes linked to the topic.
    - Synthesise a grounded answer using Gemini with source attribution.
    - Extract and return structured sources with every answer.
    - Generate relevant follow-up questions for the user.

Workflow:
    Phase 1 — Receive AgentInput with a DECISION category query.
    Phase 2 — Retrieve top-k relevant chunks from ChromaDB via memory.
    Phase 3 — Query Neo4j graph for related decision nodes.
    Phase 4 — Build structured context from retrieved chunks + graph data.
    Phase 5 — Invoke Gemini LLM to synthesise a grounded answer.
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

class DecisionAgent(BaseAgent):
    """Specialist agent for decision-related institutional memory queries.

    Answers questions about why decisions were made, who made them,
    what alternatives were considered, and the context surrounding
    key organisational choices captured in email communications.

    Inherits all retrieval, LLM invocation, and output formatting
    utilities from BaseAgent.
    """

    def __init__(self, memory=None) -> None:
        """Initialises the DecisionAgent.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        super().__init__(
            agent_name="DecisionAgent",
            category=QueryCategory.DECISION,
            memory=memory,
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """Builds the decision-specific RAG prompt for Gemini.

        The prompt instructs the LLM to focus on extracting decision
        reasoning, ownership, and context from the provided sources.

        Args:
            query: The user's original decision-related query.
            context: Formatted string of retrieved email chunks.

        Returns:
            The complete prompt string to send to Gemini.
        """
        return f"""
            You are the Decision Archaeology Agent for a Corporate Institutional Memory System.
            Your role is to reconstruct and explain organisational decisions using only the
            evidence provided in the email communications below.

            RETRIEVED EMAIL EVIDENCE:
            {context}

            USER QUERY:
            {query}

            INSTRUCTIONS:
            1. Answer the query using ONLY the information in the retrieved emails above.
            2. Always identify WHO made or was involved in the decision.
            3. Always explain WHY the decision was made based on evidence.
            4. Always state WHEN the decision was made if the date is available.
            5. If the evidence is insufficient, clearly state what is missing.
            6. Cite sources by referencing [Source N] inline in your answer.
            7. Do not fabricate or infer beyond what the evidence supports.

            After your main answer, provide exactly 3 follow-up questions the user
            might want to ask next, formatted as:
            FOLLOW_UP_1: <question>
            FOLLOW_UP_2: <question>
            FOLLOW_UP_3: <question>

            Begin your answer now:
            """.strip()

    def _build_graph_context(self, query: str) -> str:
        """Queries Neo4j for decision nodes related to the query topic.

        Extracts keywords from the query and searches the graph store
        for matching Decision nodes to enrich the LLM context.

        Args:
            query: The user's query string used for keyword extraction.

        Returns:
            A formatted string of graph-based decision data, or an
            empty string if graph is unavailable or no matches found.
        """
        if not self._memory._graph_store:
            return ""

        # Extract meaningful keywords — skip common stop words
        stop_words = {
            "what", "why", "who", "how", "when", "where", "did",
            "was", "were", "the", "a", "an", "is", "are", "about",
            "tell", "me", "our", "we", "they", "it", "this", "that",
        }

        keywords = [
            word.lower()
            for word in re.findall(r"\b[a-zA-Z]{3,}\b", query)
            if word.lower() not in stop_words
        ]

        graph_results: list[dict] = []

        for keyword in keywords[:3]:  # Limit to top 3 keywords
            try:
                decisions = self._memory.search_decisions_graph(keyword)
                graph_results.extend(decisions)
            except Exception as exc:
                logger.debug(
                    "Graph search failed for keyword '{}': {}", keyword, exc
                )

        if not graph_results:
            return ""

        # Deduplicate by node_id
        seen: set[str] = set()
        unique_decisions: list[dict] = []
        for decision in graph_results:
            node_id = decision.get("node_id", "")
            if node_id not in seen:
                seen.add(node_id)
                unique_decisions.append(decision)

        # Format graph decisions into readable context
        graph_lines = ["\n[GRAPH KNOWLEDGE - Known Decisions]\n"]
        for decision in unique_decisions[:5]:  # Cap at 5 graph results
            graph_lines.append(
                f"- Decision: {decision.get('summary', 'N/A')}\n"
                f"  Date    : {decision.get('date', 'unknown')}\n"
                f"  Dept    : {decision.get('department', 'unknown')}\n"
            )

        return "\n".join(graph_lines)

    def _parse_follow_up_questions(self, llm_response: str) -> tuple[str, list[str]]:
        """Separates the main answer from follow-up questions in LLM response.

        The LLM is prompted to append follow-up questions in a structured
        format at the end of its response. This method splits and extracts them.

        Args:
            llm_response: The raw LLM response containing answer + follow-ups.

        Returns:
            A tuple of (clean_answer, list_of_follow_up_questions).
        """
        follow_ups: list[str] = []

        pattern = re.compile(
            r"FOLLOW_UP_\d+:\s*(.+)", re.IGNORECASE
        )
        matches = pattern.findall(llm_response)

        for match in matches:
            question = match.strip()
            if question:
                follow_ups.append(question)

        # Remove follow-up lines from the main answer
        clean_answer = pattern.sub("", llm_response).strip()

        return clean_answer, follow_ups

    def _assess_confidence(
        self,
        results: list[VectorSearchResult],
        answer: str,
    ) -> float:
        """Estimates the agent's confidence in its answer.

        Confidence is derived from the number and relevance of retrieved
        sources, and whether the answer indicates insufficient evidence.

        Args:
            results: The retrieved VectorSearchResult objects.
            answer: The synthesised answer string from the LLM.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        if not results:
            return 0.0

        # Base confidence on average relevance score of top results
        avg_relevance = sum(
            r.relevance_score for r in results
        ) / len(results)

        # Penalise if LLM signals insufficient evidence
        insufficient_signals = [
            "cannot find", "no information", "not available",
            "insufficient", "unclear", "no evidence",
        ]
        answer_lower = answer.lower()
        penalty = 0.2 if any(
            signal in answer_lower for signal in insufficient_signals
        ) else 0.0

        confidence = round(min(1.0, avg_relevance - penalty), 2)
        return max(0.0, confidence)

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Executes the Decision Agent's retrieval and synthesis pipeline.

        Args:
            agent_input: Standard AgentInput containing the query,
                top_k setting, and optional metadata filters.

        Returns:
            AgentOutput with a grounded answer, sources, confidence
            score, and follow-up question suggestions.
        """
        query = agent_input.query

        # ── Phase 2: Retrieve from ChromaDB ─────────────────────────────────
        results = self._retrieve(
            query=query,
            top_k=agent_input.top_k,
            metadata_filter=agent_input.metadata_filter,
        )

        if not results:
            return self._handle_empty_results(query)

        # ── Phase 3: Enrich with graph knowledge ─────────────────────────────
        graph_context = self._build_graph_context(query)

        # ── Phase 4: Build combined context ──────────────────────────────────
        vector_context = self._build_context(results)
        full_context = (
            f"{vector_context}\n\n{graph_context}"
            if graph_context
            else vector_context
        )

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

        # ── Phase 6: Parse follow-up questions ───────────────────────────────
        clean_answer, follow_ups = self._parse_follow_up_questions(raw_response)

        # ── Phase 7: Build and return output ─────────────────────────────────
        sources = self._extract_sources(results)
        confidence = self._assess_confidence(results, clean_answer)

        logger.info(
            "DecisionAgent answered | sources={} | confidence={} | follow_ups={}",
            len(sources),
            confidence,
            len(follow_ups),
        )

        return self._build_output(
            query=query,
            answer=clean_answer,
            sources=sources,
            status=AgentStatus.SUCCESS,
            confidence=confidence,
            follow_up_questions=follow_ups,
        )

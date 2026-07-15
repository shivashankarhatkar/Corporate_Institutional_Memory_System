"""
Module: agents/retrieval/people_agent.py

Purpose:
    Retrieves and synthesises answers to people-related queries from
    the institutional memory using RAG and Neo4j graph traversal.

Responsibilities:
    - Handle queries classified as QueryCategory.PEOPLE by the router.
    - Retrieve relevant email chunks filtered by person where possible.
    - Query Neo4j graph for person nodes, relationships, and networks.
    - Synthesise a grounded answer about people, roles and relationships.
    - Return structured sources and follow-up questions.

Workflow:
    Phase 1 — Receive AgentInput with a PEOPLE category query.
    Phase 2 — Extract person identifiers from the query.
    Phase 3 — Retrieve relevant chunks from ChromaDB.
    Phase 4 — Query Neo4j for person node and communication network.
    Phase 5 — Build combined vector + graph context.
    Phase 6 — Invoke Gemini to synthesise a grounded people-focused answer.
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


class PeopleAgent(BaseAgent):
    """Specialist agent for people-related institutional memory queries.

    Answers questions about specific individuals — their roles,
    responsibilities, communication patterns, decisions they were
    involved in, and their relationships with other people in the
    organisation.

    Combines ChromaDB semantic search with Neo4j graph traversal
    to provide rich, relationship-aware answers.
    """

    def __init__(self, memory=None) -> None:
        """Initialises the PeopleAgent.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        super().__init__(
            agent_name="PeopleAgent",
            category=QueryCategory.PEOPLE,
            memory=memory,
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """Builds the people-specific RAG prompt for Gemini.

        Args:
            query: The user's original people-related query.
            context: Formatted string of retrieved email chunks
                and graph relationship data.

        Returns:
            The complete prompt string to send to Gemini.
        """
        return f"""
You are the People Intelligence Agent for a Corporate Institutional Memory System.
Your role is to answer questions about people — their roles, responsibilities,
relationships, communication patterns, and involvement in decisions and projects.

RETRIEVED EVIDENCE (Emails + Graph Relationships):
{context}

USER QUERY:
{query}

INSTRUCTIONS:
1. Answer using ONLY the evidence provided above.
2. Identify the person's role and department if evident from the emails.
3. Describe their key relationships and who they communicated with most.
4. Highlight any decisions or projects they were involved in.
5. Note the time period covered by the available evidence.
6. Cite sources using [Source N] inline in your answer.
7. If the person is not found in the evidence, clearly state this.

After your main answer, provide exactly 3 follow-up questions:

FOLLOW_UP_1: <question>
FOLLOW_UP_2: <question>
FOLLOW_UP_3: <question>

Begin your answer now:
""".strip()

    def _extract_email_from_query(self, query: str) -> Optional[str]:
        """Extracts an email address directly mentioned in the query.

        Args:
            query: The user query string to search for email addresses.

        Returns:
            The first email address found in the query, or None.
        """
        pattern = re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        )
        match = pattern.search(query)
        return match.group(0).lower() if match else None

    def _extract_person_name_from_query(self, query: str) -> Optional[str]:
        """Extracts a probable person name from the query string.

        Uses a simple heuristic: looks for sequences of two or more
        capitalised words that are likely a person's name.

        Args:
            query: The user query string to analyse.

        Returns:
            A probable full name string, or None if not found.
        """
        # Match two or more consecutive capitalised words
        pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        match = pattern.search(query)
        return match.group(1) if match else None

    def _build_graph_context(
        self,
        email: Optional[str],
        name: Optional[str],
    ) -> str:
        """Queries Neo4j for person node data and communication network.

        Args:
            email: Email address of the person to look up in the graph.
            name: Full name as fallback if email is not available.

        Returns:
            Formatted string of graph-based person data, or empty
            string if graph is unavailable or person not found.
        """
        if not self._memory._graph_store:
            return ""

        if not email:
            logger.debug(
                "PeopleAgent: no email extracted — skipping graph lookup."
            )
            return ""

        graph_lines: list[str] = []

        # ── Person node ──────────────────────────────────────────────────────
        try:
            person = self._memory.get_person(email)
            if person:
                graph_lines.append("\n[GRAPH KNOWLEDGE - Person Profile]\n")
                graph_lines.append(
                    f"Name      : {person.get('name', 'N/A')}\n"
                    f"Email     : {person.get('email', 'N/A')}\n"
                    f"Department: {person.get('department', 'N/A')}\n"
                    f"Role      : {person.get('role', 'N/A')}\n"
                )
        except Exception as exc:
            logger.debug("Graph person lookup failed: {}", exc)

        # ── Decisions made by this person ────────────────────────────────────
        try:
            decisions = self._memory.get_decisions_by_person(email)
            if decisions:
                graph_lines.append("\n[GRAPH KNOWLEDGE - Decisions Involved In]\n")
                for decision in decisions[:5]:
                    graph_lines.append(
                        f"- {decision.get('summary', 'N/A')} "
                        f"({decision.get('date', 'unknown date')})\n"
                    )
        except Exception as exc:
            logger.debug("Graph decisions lookup failed: {}", exc)

        # ── Communication network ─────────────────────────────────────────────
        try:
            network = self._memory.get_communication_network(email)
            if network:
                graph_lines.append(
                    "\n[GRAPH KNOWLEDGE - Communication Network]\n"
                )
                contacts = [
                    p.get("email", "unknown") for p in network[:8]
                ]
                graph_lines.append(
                    f"Communicated with: {', '.join(contacts)}\n"
                )
        except Exception as exc:
            logger.debug("Graph network lookup failed: {}", exc)

        return "".join(graph_lines) if graph_lines else ""

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
            "not found", "no information", "cannot find",
            "not mentioned", "no evidence", "unclear",
        ]
        penalty = 0.2 if any(
            s in answer.lower() for s in insufficient_signals
        ) else 0.0

        return max(0.0, round(min(1.0, avg_relevance - penalty), 2))

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Executes the People Agent's retrieval and synthesis pipeline.

        Args:
            agent_input: Standard AgentInput containing the query,
                top_k setting, and optional metadata filters.

        Returns:
            AgentOutput with a grounded people-focused answer,
            sources, confidence score, and follow-up questions.
        """
        query = agent_input.query

        # ── Phase 2: Extract person identifiers ──────────────────────────────
        extracted_email = self._extract_email_from_query(query)
        extracted_name = self._extract_person_name_from_query(query)

        logger.debug(
            "PeopleAgent identifiers | email='{}' | name='{}'",
            extracted_email,
            extracted_name,
        )

        # ── Phase 3: Retrieve from ChromaDB ──────────────────────────────────
        # Filter by sender if a specific email was extracted from the query
        metadata_filter = (
            {"sender": extracted_email}
            if extracted_email
            else agent_input.metadata_filter
        )

        results = self._retrieve(
            query=query,
            top_k=agent_input.top_k,
            metadata_filter=metadata_filter,
        )

        # Fallback — retry without filter if filtered search returned nothing
        if not results and metadata_filter:
            logger.debug(
                "PeopleAgent: filtered search returned no results — "
                "retrying without filter."
            )
            results = self._retrieve(
                query=query,
                top_k=agent_input.top_k,
            )

        if not results:
            return self._handle_empty_results(query)

        # ── Phase 4: Query Neo4j graph ────────────────────────────────────────
        graph_context = self._build_graph_context(
            email=extracted_email,
            name=extracted_name,
        )

        # ── Phase 5: Build combined context ───────────────────────────────────
        vector_context = self._build_context(results)
        full_context = (
            f"{vector_context}\n\n{graph_context}"
            if graph_context
            else vector_context
        )

        # ── Phase 6: Invoke LLM ───────────────────────────────────────────────
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

        # ── Phase 7: Parse and return output ──────────────────────────────────
        clean_answer, follow_ups = self._parse_follow_up_questions(raw_response)
        sources = self._extract_sources(results)
        confidence = self._assess_confidence(results, clean_answer)

        logger.info(
            "PeopleAgent answered | sources={} | confidence={} | "
            "email='{}' | follow_ups={}",
            len(sources),
            confidence,
            extracted_email or "none",
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

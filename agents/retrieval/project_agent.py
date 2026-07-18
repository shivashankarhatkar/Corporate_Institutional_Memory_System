"""
Module: agents/retrieval/project_agent.py

Purpose:
    Retrieves and synthesises answers to project-related queries from
    the institutional memory using RAG and Neo4j graph traversal.

Responsibilities:
    - Handle queries classified as QueryCategory.PROJECT by the router.
    - Retrieve relevant email chunks related to projects and initiatives.
    - Query Neo4j for Project nodes, team members, and project decisions.
    - Synthesise grounded answers about project history, status, outcomes,
      and the people involved.
    - Return structured sources and follow-up questions.

Workflow:
    Phase 1 — Receive AgentInput with a PROJECT category query.
    Phase 2 — Extract project name or identifier from the query.
    Phase 3 — Retrieve relevant chunks from ChromaDB.
    Phase 4 — Query Neo4j for related Project nodes and team members.
    Phase 5 — Build combined vector + graph context.
    Phase 6 — Invoke Gemini to synthesise a project-focused answer.
    Phase 7 — Parse follow-up questions from LLM response.
    Phase 8 — Return AgentOutput with answer, sources, and follow-ups.
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


class ProjectAgent(BaseAgent):
    """Specialist agent for project-related institutional memory queries.

    Answers questions about specific projects, initiatives, and programs —
    their history, current status, outcomes, key decisions, and the
    people involved — using email evidence and graph relationships.

    Combines ChromaDB semantic search with Neo4j Project node traversal
    to provide comprehensive project intelligence answers.
    """

    def __init__(self, memory=None) -> None:
        """Initialises the ProjectAgent.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        super().__init__(
            agent_name="ProjectAgent",
            category=QueryCategory.PROJECT,
            memory=memory,
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """Builds the project-specific RAG prompt for Gemini.

        Instructs the LLM to focus on project history, status,
        team composition, decisions, and outcomes.

        Args:
            query: The user's original project-related query.
            context: Formatted string of retrieved email chunks
                and Neo4j Project node data.

        Returns:
            The complete prompt string to send to Gemini.
        """
        return f"""
You are the Project Intelligence Agent for a Corporate Institutional Memory System.
Your role is to answer questions about projects, initiatives, and programs using
only the evidence provided in the email communications and project records below.

RETRIEVED EVIDENCE (Emails + Project Records):
{context}

USER QUERY:
{query}

INSTRUCTIONS:
1. Answer using ONLY the evidence provided above.
2. Identify the PROJECT NAME and its purpose if evident.
3. Describe the project STATUS (active, completed, cancelled) if known.
4. List KEY PEOPLE involved — their roles and contributions.
5. Highlight important DECISIONS made during the project.
6. Note the PROJECT TIMELINE — start date, milestones, end date if available.
7. Summarise the OUTCOME or current state of the project.
8. Cite sources using [Source N] inline in your answer.
9. If the project is not clearly identified in the evidence, state this.

After your main answer, provide exactly 3 follow-up questions:

FOLLOW_UP_1: <question>
FOLLOW_UP_2: <question>
FOLLOW_UP_3: <question>

Begin your answer now:
""".strip()

    def _extract_project_name(self, query: str) -> Optional[str]:
        """Extracts a probable project name from the query string.

        Looks for explicit project name patterns such as
        "Project Alpha" or "the Sagewood initiative".

        Args:
            query: The user's query string to analyse.

        Returns:
            The extracted project name string, or None if not found.
        """
        # Pattern: "project X" or "initiative X" or "program X"
        explicit_pattern = re.compile(
            r"\b(?:project|initiative|program|programme)\s+([A-Z][A-Za-z0-9\s\-]{1,30})",
            re.IGNORECASE,
        )
        match = explicit_pattern.search(query)
        if match:
            return match.group(1).strip().title()

        # Pattern: quoted project name e.g. "the 'Sagewood' project"
        quoted_pattern = re.compile(r"['\"]([A-Za-z0-9\s\-]{2,30})['\"]")
        match = quoted_pattern.search(query)
        if match:
            return match.group(1).strip().title()

        return None

    def _extract_project_keywords(self, query: str) -> list[str]:
        """Extracts meaningful keywords from the query for graph search.

        Args:
            query: The user's project query string.

        Returns:
            A deduplicated list of keyword strings.
        """
        stop_words = {
            "what", "why", "who", "how", "when", "where", "is",
            "are", "the", "a", "an", "our", "we", "tell", "me",
            "about", "project", "initiative", "program", "did",
            "was", "were", "happened", "status", "outcome", "result",
        }

        keywords = [
            word.lower()
            for word in re.findall(r"\b[a-zA-Z]{3,}\b", query)
            if word.lower() not in stop_words
        ]

        return list(dict.fromkeys(keywords))

    def _build_graph_context(
        self,
        project_name: Optional[str],
        keywords: list[str],
    ) -> str:
        """Queries Neo4j for Project nodes related to the query.

        Searches by explicit project name first, then falls back to
        keyword-based search across Project node properties.

        Args:
            project_name: Extracted project name from the query.
            keywords: Fallback keywords for graph search.

        Returns:
            Formatted string of graph-based project data, or empty
            string if graph is unavailable or no projects found.
        """
        if not self._memory._graph_store:
            return ""

        graph_lines: list[str] = []
        found_projects: list[dict] = []

        try:
            with self._memory._graph_store._session() as session:

                # ── Search by explicit project name first ────────────────────
                if project_name:
                    cypher = """
                        MATCH (pr:Project)
                        WHERE toLower(pr.name) CONTAINS toLower($name)
                        RETURN pr
                        LIMIT 3
                    """
                    result = session.run(cypher, name=project_name)
                    found_projects.extend(
                        [dict(record["pr"]) for record in result]
                    )

                # ── Fallback: keyword search ──────────────────────────────────
                if not found_projects:
                    for keyword in keywords[:3]:
                        cypher = """
                            MATCH (pr:Project)
                            WHERE toLower(pr.name) CONTAINS toLower($keyword)
                            RETURN pr
                            LIMIT 2
                        """
                        result = session.run(cypher, keyword=keyword)
                        found_projects.extend(
                            [dict(record["pr"]) for record in result]
                        )

                if found_projects:
                    graph_lines.append(
                        "\n[GRAPH KNOWLEDGE - Project Records]\n"
                    )
                    for project in found_projects[:5]:
                        graph_lines.append(
                            f"Project   : {project.get('name', 'N/A')}\n"
                            f"Status    : {project.get('status', 'unknown')}\n"
                            f"Start Date: {project.get('start_date', 'N/A')}\n"
                            f"End Date  : {project.get('end_date', 'N/A')}\n"
                        )

                    # ── Fetch team members for found projects ─────────────────
                    for project in found_projects[:2]:
                        project_id = project.get("node_id", "")
                        if not project_id:
                            continue

                        cypher = """
                            MATCH (p:Person)-[:INVOLVED_IN]->(pr:Project {node_id: $project_id})
                            RETURN p.email AS email, p.name AS name, p.role AS role
                            LIMIT 10
                        """
                        result = session.run(cypher, project_id=project_id)
                        members = [dict(record) for record in result]

                        if members:
                            graph_lines.append(
                                f"\nTeam Members for '{project.get('name')}':\n"
                            )
                            for member in members:
                                graph_lines.append(
                                    f"  - {member.get('name', 'N/A')} "
                                    f"({member.get('email', 'N/A')}) "
                                    f"| Role: {member.get('role', 'N/A')}\n"
                                )

        except Exception as exc:
            logger.debug("ProjectAgent graph query failed: {}", exc)
            return ""

        return "".join(graph_lines) if graph_lines else ""

    def _build_timeline_context(self, results: list[VectorSearchResult]) -> str:
        """Extracts and formats a chronological timeline from retrieved chunks.

        Sorts retrieved email chunks by date to help the LLM understand
        the temporal progression of the project.

        Args:
            results: List of VectorSearchResult objects to sort by date.

        Returns:
            A formatted timeline string, or empty string if dates
            are unavailable in the retrieved chunks.
        """
        dated_results = [
            (r.metadata.get("date", ""), r)
            for r in results
            if r.metadata.get("date")
        ]

        if not dated_results:
            return ""

        # Sort chronologically by ISO date string
        dated_results.sort(key=lambda x: x[0])

        timeline_lines = ["\n[PROJECT TIMELINE - Chronological Email Evidence]\n"]

        for date_str, result in dated_results[:5]:
            sender = result.metadata.get("sender", "unknown")
            subject = result.metadata.get("subject", "no subject")
            timeline_lines.append(
                f"{date_str[:10]} | {sender} | {subject}\n"
                f"  {result.text[:150]}...\n"
            )

        return "\n".join(timeline_lines)

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
        project_name: Optional[str],
    ) -> float:
        """Estimates answer confidence from retrieval quality.

        Applies a bonus when a specific project name was successfully
        extracted and matched in the results.

        Args:
            results: Retrieved VectorSearchResult objects.
            answer: The synthesised LLM answer string.
            project_name: Extracted project name, if any.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not results:
            return 0.0

        avg_relevance = sum(
            r.relevance_score for r in results
        ) / len(results)

        insufficient_signals = [
            "not clearly identified", "no information", "cannot find",
            "not found", "no evidence", "not mentioned",
        ]
        penalty = 0.2 if any(
            s in answer.lower() for s in insufficient_signals
        ) else 0.0

        # Bonus when specific project name was matched
        bonus = 0.05 if project_name else 0.0

        return max(
            0.0, round(min(1.0, avg_relevance - penalty + bonus), 2)
        )

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Executes the Project Agent's retrieval and synthesis pipeline.

        Args:
            agent_input: Standard AgentInput containing the query,
                top_k setting, and optional metadata filters.

        Returns:
            AgentOutput with a grounded project-focused answer,
            sources, confidence score, and follow-up questions.
        """
        query = agent_input.query

        # ── Phase 2: Extract project identifiers ─────────────────────────────
        project_name = self._extract_project_name(query)
        keywords = self._extract_project_keywords(query)

        logger.debug(
            "ProjectAgent identifiers | project_name='{}' | keywords={}",
            project_name,
            keywords[:5],
        )

        # ── Phase 3: Retrieve from ChromaDB ──────────────────────────────────
        # Enrich query with project name for better semantic matching
        enriched_query = (
            f"{query} {project_name}" if project_name else query
        )

        results = self._retrieve(
            query=enriched_query,
            top_k=agent_input.top_k,
            metadata_filter=agent_input.metadata_filter,
        )

        if not results:
            return self._handle_empty_results(query)

        # ── Phase 4: Query Neo4j for Project nodes ────────────────────────────
        graph_context = self._build_graph_context(
            project_name=project_name,
            keywords=keywords,
        )

        # ── Phase 5: Build combined context ───────────────────────────────────
        vector_context = self._build_context(results)
        timeline_context = self._build_timeline_context(results)

        full_context_parts = [vector_context]
        if timeline_context:
            full_context_parts.append(timeline_context)
        if graph_context:
            full_context_parts.append(graph_context)

        full_context = "\n\n".join(full_context_parts)

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

        # ── Phase 7: Parse follow-up questions ───────────────────────────────
        clean_answer, follow_ups = self._parse_follow_up_questions(raw_response)

        # ── Phase 8: Build and return output ─────────────────────────────────
        sources = self._extract_sources(results)
        confidence = self._assess_confidence(results, clean_answer, project_name)

        logger.info(
            "ProjectAgent answered | project='{}' | sources={} | "
            "confidence={} | follow_ups={}",
            project_name or "unknown",
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

"""
Module: agents/capture/tribal_knowledge_agent.py

Purpose:
    Captures undocumented institutional knowledge from subject matter
    experts through structured conversational interviews, storing
    extracted insights permanently in the institutional memory.

Responsibilities:
    - Conduct structured multi-turn knowledge extraction interviews.
    - Generate targeted questions based on the expert's domain and role.
    - Extract key insights, processes, relationships, and decisions
      from interview responses.
    - Store extracted knowledge as Decision and Person nodes in Neo4j.
    - Store interview content in ChromaDB for semantic retrieval.
    - Produce a TribalKnowledgeReport summarising captured knowledge.

Workflow:
    Phase 1 — Receive expert profile and domain context.
    Phase 2 — Generate targeted interview questions via Gemini.
    Phase 3 — Process interview responses and extract insights.
    Phase 4 — Store extracted knowledge in Neo4j and ChromaDB.
    Phase 5 — Return TribalKnowledgeReport with all captured insights.
"""

import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from config.settings import settings
from schemas.agent_schema import (
    AgentInput,
    AgentOutput,
    AgentStatus,
    QueryCategory,
)
from schemas.memory_schema import (
    DecisionNode,
    PersonNode,
    GraphRelationship,
    RelationshipType,
)


# ── Tribal Knowledge Models ──────────────────────────────────────────────────

class ExpertProfile(BaseModel):
    """Profile of a subject matter expert being interviewed.

    Attributes:
        name: Full name of the expert.
        email: Email address (primary identifier).
        role: Current or former job role/title.
        department: Department the expert belongs to.
        domain: Primary knowledge domain to extract.
        years_experience: Years of experience in this domain.
    """

    name: str = Field(..., description="Expert's full name")
    email: str = Field(..., description="Expert's email address")
    role: str = Field(default="", description="Job role or title")
    department: str = Field(default="", description="Department")
    domain: str = Field(
        ..., description="Primary knowledge domain to capture"
    )
    years_experience: int = Field(
        default=0, description="Years of experience in domain"
    )


class KnowledgeInsight(BaseModel):
    """A single insight extracted from a tribal knowledge interview."""

    insight_id: str = Field(..., description="Unique insight identifier")
    category: str = Field(
        ...,
        description="PROCESS, RELATIONSHIP, DECISION, RISK, or CONTEXT"
    )
    insight: str = Field(..., description="The captured knowledge insight")
    importance: str = Field(
        default="MEDIUM",
        description="Importance level: CRITICAL, HIGH, MEDIUM, or LOW"
    )
    context: str = Field(
        default="", description="Additional context for this insight"
    )


class TribalKnowledgeReport(BaseModel):
    """Structured report produced by the Tribal Knowledge Agent."""

    interview_id: str = Field(
        ..., description="Unique interview session identifier"
    )
    expert_email: str = Field(..., description="Email of the expert interviewed")
    expert_name: str = Field(..., description="Name of the expert")
    domain: str = Field(..., description="Knowledge domain captured")
    interview_date: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of the interview"
    )
    questions_asked: list[str] = Field(
        default_factory=list,
        description="Questions generated and asked during interview"
    )
    insights_captured: list[KnowledgeInsight] = Field(
        default_factory=list,
        description="All insights extracted from the interview"
    )
    critical_insights: int = Field(
        default=0,
        description="Number of CRITICAL importance insights captured"
    )
    processes_documented: list[str] = Field(
        default_factory=list,
        description="Key processes documented during interview"
    )
    relationships_mapped: list[str] = Field(
        default_factory=list,
        description="Key relationships identified during interview"
    )
    summary: str = Field(
        default="", description="Executive summary of captured knowledge"
    )
    stored_in_graph: bool = Field(default=False)
    stored_in_vector: bool = Field(default=False)


# ── Question Templates by Domain ─────────────────────────────────────────────

_DOMAIN_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "finance": [
        "What financial approval processes do only you know about?",
        "Which vendors have special payment arrangements not in any document?",
        "What budget decisions have you made that no one else is aware of?",
        "Who should be consulted for financial decisions in your absence?",
        "What financial risks are you currently tracking informally?",
    ],
    "technology": [
        "What system dependencies exist that are not documented anywhere?",
        "Which parts of the infrastructure would fail if you weren't here?",
        "What workarounds have you built that are critical to operations?",
        "Who has the passwords or access credentials only you manage?",
        "What technical decisions were made verbally and never documented?",
    ],
    "operations": [
        "What operational processes exist only in your memory?",
        "Which vendor relationships depend entirely on your personal contact?",
        "What daily tasks would stop if you were unavailable for a week?",
        "What operational risks are you aware of that aren't formally tracked?",
        "Which escalation paths do you manage informally?",
    ],
    "legal": [
        "Which contract terms were negotiated verbally and never updated?",
        "What compliance obligations are you personally tracking?",
        "Which legal relationships or contacts are in your personal network?",
        "What legal risks are you aware of that aren't formally documented?",
        "Which past legal decisions should influence future contracts?",
    ],
    "hr": [
        "What employee agreements or accommodations exist informally?",
        "Which performance issues are being managed off the record?",
        "What compensation exceptions have been made that aren't documented?",
        "Who are the informal leaders or influencers in the organisation?",
        "What cultural knowledge should every new manager know?",
    ],
    "default": [
        "What knowledge do you have that exists nowhere else in the organisation?",
        "What would break first if you left tomorrow?",
        "Which relationships do you maintain that are critical to operations?",
        "What decisions have you made verbally that were never written down?",
        "What processes do you follow that no one else fully understands?",
        "What risks are you personally tracking that aren't formally documented?",
        "Who should be your successor for each critical responsibility?",
    ],
}


class TribalKnowledgeAgent(BaseAgent):
    """Capture agent that extracts undocumented tribal knowledge from experts.

    Conducts structured knowledge extraction interviews with subject
    matter experts, capturing insights, processes, relationships, and
    decisions that exist only in people's heads. Stores all captured
    knowledge permanently in both Neo4j and ChromaDB.
    """

    def __init__(self, memory=None) -> None:
        """Initialises the TribalKnowledgeAgent.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        super().__init__(
            agent_name="TribalKnowledgeAgent",
            category=QueryCategory.PEOPLE,
            memory=memory,
        )

    def _build_prompt(self, query: str, context: str = "") -> str:
        """Builds the knowledge extraction prompt for Gemini.

        Args:
            query: Combined interview content including questions
                and responses to extract insights from.
            context: Unused — kept for BaseAgent interface compliance.

        Returns:
            The complete extraction prompt string to send to Gemini.
        """
        return f"""
You are the Tribal Knowledge Extraction Agent for a Corporate Institutional
Memory System. Extract all valuable institutional knowledge from the following
expert interview content.

INTERVIEW CONTENT:
{query}

Extract and return in EXACT format:

SUMMARY: <2-3 sentence summary of key knowledge captured>

INSIGHTS:
INSIGHT_1: CATEGORY: <PROCESS|RELATIONSHIP|DECISION|RISK|CONTEXT> | IMPORTANCE: <CRITICAL|HIGH|MEDIUM|LOW> | INSIGHT: <the knowledge> | CONTEXT: <additional context>
INSIGHT_2: CATEGORY: <PROCESS|RELATIONSHIP|DECISION|RISK|CONTEXT> | IMPORTANCE: <CRITICAL|HIGH|MEDIUM|LOW> | INSIGHT: <the knowledge> | CONTEXT: <additional context>
(add more as needed or write NONE)

PROCESSES:
PROCESS_1: <key process documented>
PROCESS_2: <key process documented>
(add more as needed or write NONE)

RELATIONSHIPS:
REL_1: <key relationship or contact identified>
REL_2: <key relationship or contact identified>
(add more as needed or write NONE)

Respond ONLY in the format above. No extra text.
""".strip()

    def _generate_interview_questions(
        self, profile: ExpertProfile
    ) -> list[str]:
        """Generates targeted interview questions for an expert's domain.

        Selects domain-specific question templates and personalises them
        with the expert's role and department context.

        Args:
            profile: The ExpertProfile of the expert being interviewed.

        Returns:
            A list of targeted interview question strings.
        """
        domain_key = profile.domain.lower()

        # Find best matching domain template
        matched_key = next(
            (key for key in _DOMAIN_QUESTION_TEMPLATES if key in domain_key),
            "default",
        )

        base_questions = _DOMAIN_QUESTION_TEMPLATES[matched_key].copy()

        # Always append universal questions not in domain templates
        if matched_key != "default":
            base_questions.extend(
                _DOMAIN_QUESTION_TEMPLATES["default"][:3]
            )

        logger.debug(
            "Generated {} questions for domain='{}'",
            len(base_questions),
            profile.domain,
        )

        return base_questions

    def _parse_insights(self, response: str) -> list[KnowledgeInsight]:
        """Extracts structured KnowledgeInsight objects from LLM response.

        Args:
            response: The full LLM response string.

        Returns:
            List of KnowledgeInsight objects parsed from INSIGHT_N blocks.
        """
        import re

        pattern = re.compile(
            r"^INSIGHT_\d+:\s*CATEGORY:\s*(\w+)\s*\|\s*"
            r"IMPORTANCE:\s*(\w+)\s*\|\s*INSIGHT:\s*(.+?)\s*\|\s*"
            r"CONTEXT:\s*(.+)$",
            re.IGNORECASE | re.MULTILINE,
        )

        insights: list[KnowledgeInsight] = []
        valid_categories = {
            "PROCESS", "RELATIONSHIP", "DECISION", "RISK", "CONTEXT"
        }
        valid_importance = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

        for match in pattern.finditer(response):
            category = match.group(1).strip().upper()
            importance = match.group(2).strip().upper()
            insight_text = match.group(3).strip()
            context = match.group(4).strip()

            if insight_text.upper() == "NONE":
                continue

            if category not in valid_categories:
                category = "CONTEXT"
            if importance not in valid_importance:
                importance = "MEDIUM"

            insights.append(
                KnowledgeInsight(
                    insight_id=f"insight_{uuid.uuid4().hex[:8]}",
                    category=category,
                    insight=insight_text,
                    importance=importance,
                    context=context,
                )
            )

        return insights

    def _parse_list_items(self, response: str, prefix: str) -> list[str]:
        """Extracts a list of items matching a numbered prefix pattern.

        Args:
            response: The full LLM response string.
            prefix: The item prefix to match e.g. 'PROCESS', 'REL'.

        Returns:
            List of extracted string values for the given prefix.
        """
        import re

        pattern = re.compile(
            rf"^{prefix}_\d+:\s*(.+)$",
            re.IGNORECASE | re.MULTILINE,
        )
        return [
            m.strip()
            for m in pattern.findall(response)
            if m.strip().upper() != "NONE"
        ]

    def _extract_summary(self, response: str) -> str:
        """Extracts the SUMMARY field from the LLM response.

        Args:
            response: The full LLM response string.

        Returns:
            The summary string, or empty string if not found.
        """
        import re

        match = re.search(
            r"^SUMMARY:\s*(.+)$", response,
            re.IGNORECASE | re.MULTILINE
        )
        return match.group(1).strip() if match else ""

    def _store_in_graph(
        self,
        report: TribalKnowledgeReport,
        profile: ExpertProfile,
    ) -> bool:
        """Stores captured tribal knowledge in Neo4j graph.

        Creates a Person node for the expert, Decision nodes for each
        critical/high insight, and COMMUNICATED_WITH relationships for
        identified relationships.

        Args:
            report: The populated TribalKnowledgeReport.
            profile: The ExpertProfile of the interviewed expert.

        Returns:
            True if storage succeeded, False otherwise.
        """
        if not self._memory._graph_store:
            logger.warning(
                "TribalKnowledgeAgent: GraphStore unavailable."
            )
            return False

        try:
            # ── Upsert expert Person node ─────────────────────────────────────
            person = PersonNode(
                node_id=profile.email.lower(),
                name=profile.name,
                email=profile.email.lower(),
                department=profile.department,
                role=profile.role,
            )
            self._memory.upsert_person(person)

            # ── Store critical/high insights as Decision nodes ─────────────────
            for insight in report.insights_captured:
                if insight.importance not in {"CRITICAL", "HIGH"}:
                    continue

                decision = DecisionNode(
                    node_id=insight.insight_id,
                    summary=(
                        f"[TRIBAL:{insight.category}] {insight.insight}"
                    ),
                    source_message_id=report.interview_id,
                    department=profile.department,
                )
                self._memory.upsert_decision(decision)

                # Link expert to this knowledge node
                relationship = GraphRelationship(
                    from_node_id=profile.email.lower(),
                    to_node_id=insight.insight_id,
                    relationship_type=RelationshipType.MADE_DECISION,
                    properties={
                        "knowledge_type": "tribal",
                        "category": insight.category,
                        "interview_id": report.interview_id,
                    },
                )
                try:
                    self._memory.create_relationship(relationship)
                except Exception as exc:
                    logger.debug(
                        "Tribal knowledge relationship failed: {}", exc
                    )

            logger.info(
                "TribalKnowledgeAgent stored graph | expert='{}' | "
                "insights={}",
                profile.email,
                len(report.insights_captured),
            )
            return True

        except Exception as exc:
            logger.error(
                "TribalKnowledgeAgent graph storage failed: {}", exc
            )
            return False

    def _store_in_vector(
        self,
        report: TribalKnowledgeReport,
        profile: ExpertProfile,
        raw_content: str,
    ) -> bool:
        """Stores tribal knowledge interview content in ChromaDB.

        Builds a rich searchable document from the expert profile,
        insights, processes, and relationships for semantic retrieval.

        Args:
            report: The populated TribalKnowledgeReport.
            profile: The ExpertProfile of the expert.
            raw_content: Original interview content.

        Returns:
            True if storage succeeded, False otherwise.
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from sentence_transformers import SentenceTransformer

            client = chromadb.PersistentClient(
                path=settings.chromadb.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection(
                name=settings.chromadb.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Build rich searchable text
            searchable_text = (
                f"Expert: {profile.name} | Role: {profile.role} | "
                f"Department: {profile.department} | Domain: {profile.domain}\n"
                f"Summary: {report.summary}\n"
                f"Key Insights: "
                + " | ".join(
                    i.insight for i in report.insights_captured[:5]
                )
                + f"\nProcesses: {' | '.join(report.processes_documented[:3])}\n"
                f"Relationships: {' | '.join(report.relationships_mapped[:3])}"
            )

            model = SentenceTransformer(settings.chromadb.embedding_model)
            embedding = model.encode(
                searchable_text, convert_to_numpy=True
            ).tolist()

            collection.upsert(
                ids=[report.interview_id],
                documents=[searchable_text],
                embeddings=[embedding],
                metadatas=[{
                    "message_id": report.interview_id,
                    "chunk_index": 0,
                    "sender": profile.email.lower(),
                    "receiver": "tribal_knowledge_agent",
                    "subject": (
                        f"Tribal Knowledge: {profile.name} | {profile.domain}"
                    ),
                    "date": report.interview_date,
                    "word_count": len(searchable_text.split()),
                    "department": profile.department or "Unknown",
                }],
            )

            logger.info(
                "TribalKnowledgeAgent stored vector | expert='{}' | id='{}'",
                profile.email,
                report.interview_id,
            )
            return True

        except Exception as exc:
            logger.error(
                "TribalKnowledgeAgent vector storage failed: {}", exc
            )
            return False

    def conduct_interview(
        self,
        profile: ExpertProfile,
        interview_responses: str,
    ) -> TribalKnowledgeReport:
        """Conducts knowledge extraction from an expert's interview responses.

        This is the primary public method called by the Capture Orchestrator.
        Generates questions, processes responses, extracts insights, and
        stores all captured knowledge permanently.

        Args:
            profile: ExpertProfile of the subject matter expert.
            interview_responses: Raw text of the expert's responses to
                interview questions (Q&A format or free-form text).

        Returns:
            A fully populated TribalKnowledgeReport.
        """
        interview_id = f"tribal_{uuid.uuid4().hex[:12]}"

        logger.info(
            "TribalKnowledgeAgent interviewing | expert='{}' | domain='{}' | id='{}'",
            profile.email,
            profile.domain,
            interview_id,
        )

        # ── Phase 2: Generate questions ───────────────────────────────────────
        questions = self._generate_interview_questions(profile)

        # ── Phase 3: Extract insights from responses ──────────────────────────
        combined_content = (
            f"Expert: {profile.name} | Role: {profile.role} | "
            f"Department: {profile.department} | Domain: {profile.domain}\n\n"
            f"Interview Responses:\n{interview_responses}"
        )

        prompt = self._build_prompt(combined_content)

        try:
            raw_response = self._invoke_llm(prompt)
        except RuntimeError as exc:
            logger.error(
                "TribalKnowledgeAgent LLM extraction failed: {}", exc
            )
            return TribalKnowledgeReport(
                interview_id=interview_id,
                expert_email=profile.email,
                expert_name=profile.name,
                domain=profile.domain,
                summary=f"Extraction failed: {exc}",
            )

        # ── Parse extracted knowledge ─────────────────────────────────────────
        insights = self._parse_insights(raw_response)
        processes = self._parse_list_items(raw_response, "PROCESS")
        relationships = self._parse_list_items(raw_response, "REL")
        summary = self._extract_summary(raw_response)

        critical_count = sum(
            1 for i in insights if i.importance == "CRITICAL"
        )

        report = TribalKnowledgeReport(
            interview_id=interview_id,
            expert_email=profile.email,
            expert_name=profile.name,
            domain=profile.domain,
            questions_asked=questions,
            insights_captured=insights,
            critical_insights=critical_count,
            processes_documented=processes,
            relationships_mapped=relationships,
            summary=summary,
        )

        # ── Phase 4: Store in Neo4j ───────────────────────────────────────────
        report.stored_in_graph = self._store_in_graph(report, profile)

        # ── Phase 5: Store in ChromaDB ────────────────────────────────────────
        report.stored_in_vector = self._store_in_vector(
            report, profile, interview_responses
        )

        logger.info(
            "TribalKnowledgeAgent complete | expert='{}' | insights={} | "
            "critical={} | graph={} | vector={}",
            profile.email,
            len(insights),
            critical_count,
            report.stored_in_graph,
            report.stored_in_vector,
        )

        return report

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Implements BaseAgent.run() for orchestrator compatibility.

        Expects agent_input.query to contain interview content in format:
        'EMAIL:<email>|NAME:<name>|ROLE:<role>|DEPT:<dept>|DOMAIN:<domain>
        RESPONSES:<interview responses>'

        Args:
            agent_input: Standard AgentInput with structured interview content.

        Returns:
            AgentOutput with tribal knowledge summary and top insights
            as follow-up questions.
        """
        import re

        query = agent_input.query or ""

        if len(query.strip()) < 20:
            return self._build_output(
                query=query,
                answer="Insufficient interview content provided.",
                sources=[],
                status=AgentStatus.PARTIAL,
                confidence=0.0,
            )

        # ── Parse expert profile from structured query ─────────────────────────
        def _extract(pattern: str) -> str:
            match = re.search(pattern, query, re.IGNORECASE)
            return match.group(1).strip() if match else ""

        profile = ExpertProfile(
            name=_extract(r"NAME:([^|]+)") or "Unknown Expert",
            email=_extract(r"EMAIL:([^|]+)") or "unknown@unknown.com",
            role=_extract(r"ROLE:([^|]+)"),
            department=_extract(r"DEPT:([^|]+)"),
            domain=_extract(r"DOMAIN:([^|]+)") or "general",
        )

        # Extract responses section
        responses_match = re.search(
            r"RESPONSES:\s*(.+)", query,
            re.IGNORECASE | re.DOTALL,
        )
        responses = (
            responses_match.group(1).strip()
            if responses_match
            else query
        )

        report = self.conduct_interview(profile, responses)

        follow_ups = [
            i.insight
            for i in report.insights_captured
            if i.importance in {"CRITICAL", "HIGH"}
        ][:3]

        answer = (
            f"{report.summary}\n\n"
            f"Expert         : {report.expert_name} ({report.expert_email})\n"
            f"Domain         : {report.domain}\n"
            f"Insights       : {len(report.insights_captured)}\n"
            f"Critical       : {report.critical_insights}\n"
            f"Processes      : {len(report.processes_documented)}\n"
            f"Relationships  : {len(report.relationships_mapped)}\n"
            f"Stored in Graph: {report.stored_in_graph}\n"
            f"Stored in Vector: {report.stored_in_vector}"
        )

        return self._build_output(
            query=query[:100],
            answer=answer,
            sources=[],
            status=AgentStatus.SUCCESS,
            confidence=(
                0.9 if report.critical_insights > 0 else 0.6
            ),
            follow_up_questions=follow_ups,
        )

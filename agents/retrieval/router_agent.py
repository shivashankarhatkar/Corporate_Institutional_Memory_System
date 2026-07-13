"""
Module: agents/retrieval/router_agent.py

Purpose:
    Classifies incoming user queries into one of the defined QueryCategory
    types and routes them to the appropriate specialist retrieval agent.

Responsibilities:
    - Accept a raw user query string.
    - Use the Gemini LLM to classify the query into a QueryCategory.
    - Provide a confidence score for the classification.
    - Suggest a fallback category when classification is ambiguous.
    - Return a RouterOutput containing category and routing metadata.

Workflow:
    Phase 1 — Receive raw query string from the Master Orchestrator.
    Phase 2 — Build classification prompt with category definitions.
    Phase 3 — Invoke Gemini LLM to classify the query.
    Phase 4 — Parse and validate the LLM classification response.
    Phase 5 — Return RouterOutput with category and confidence.
"""

import re
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from schemas.agent_schema import (
    AgentInput,
    AgentOutput,
    AgentStatus,
    QueryCategory,
)


# ── Router Output Model ─────────────────────────────────────────────────

class RouterOutput(BaseModel):
    """Structured output produced by the Router Agent.

    Contains the classified category, confidence score, and reasoning
    used by the Master Orchestrator to select the correct specialist agent.
    """

    query: str = Field(..., description="Original user query string")
    category: QueryCategory = Field(
        ..., description="Classified query category"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Router confidence in the classification (0.0 - 1.0)"
    )
    reasoning: str = Field(
        ..., description="Brief reasoning for the classification decision"
    )
    fallback_category: Optional[QueryCategory] = Field(
        None,
        description="Secondary category if confidence is below threshold"
    )


# ── Category Definitions for Prompt ─────────────────────────────────────────

_CATEGORY_DEFINITIONS = """
DECISION  : Questions about why a decision was made, what was decided,
            who approved something, or the reasoning behind a choice.
            Examples: "Why did we stop X?", "Who decided to change Y?",
            "What was agreed in the Z meeting?"

PEOPLE    : Questions about a specific person, their role, responsibilities,
            communication patterns, or relationships with others.
            Examples: "Who is responsible for X?", "What does Y do?",
            "Who did Z report to?", "Find emails from person X."

PROJECT   : Questions about a specific project, initiative, or program —
            its status, history, outcomes, or the people involved.
            Examples: "What happened to project X?", "Who worked on Y?",
            "What was the outcome of initiative Z?"

POLICY    : Questions about company rules, procedures, compliance,
            guidelines, or standard operating practices.
            Examples: "What is the policy on X?", "How do we handle Y?",
            "What are the rules for Z?"

UNKNOWN   : Query does not clearly fit any of the above categories,
            or requires general knowledge not specific to any category.
"""


# ── Router Agent ─────────────────────────────────────────────────────────────

class RouterAgent(BaseAgent):
    """Classifies user queries and routes them to specialist agents.

    The RouterAgent is the first node in the LangGraph state machine.
    It receives every incoming query and determines which specialist
    retrieval agent should handle it.

    Inherits from BaseAgent but does not perform retrieval — classification
    is done entirely via LLM reasoning on the query text alone.
    """

    # Confidence threshold below which a fallback category is suggested
    _CONFIDENCE_THRESHOLD = 0.75

    def __init__(self, memory=None) -> None:
        """Initialises the RouterAgent.

        Args:
            memory: Optional MemoryManager for dependency injection.
        """
        super().__init__(
            agent_name="RouterAgent",
            category=QueryCategory.UNKNOWN,
            memory=memory,
        )

    def _build_prompt(self, query: str, context: str = "") -> str:
        """Builds the classification prompt for the Router Agent.

        The context parameter is unused by the router since classification
        is based on query text alone, not retrieved documents.

        Args:
            query: The user query string to classify.
            context: Unused by router. Kept for BaseAgent interface compliance.

        Returns:
            The complete classification prompt string.
        """
        return f"""
You are a Query Router for a Corporate Institutional Memory System.
Your sole task is to classify the user query into exactly one category.

CATEGORY DEFINITIONS:
{_CATEGORY_DEFINITIONS}

USER QUERY:
{query}

INSTRUCTIONS:
1. Read the query carefully and identify its primary intent.
2. Select the single most appropriate category from:
   DECISION, PEOPLE, PROJECT, POLICY, UNKNOWN
3. Assign a confidence score between 0.0 and 1.0.
4. Write one sentence explaining your reasoning.
5. If confidence is below 0.75, suggest a fallback category.

Respond in this EXACT format — no extra text:
CATEGORY: <category>
CONFIDENCE: <score>
REASONING: <one sentence>
FALLBACK: <category or NONE>
""".strip()

    def _parse_llm_response(self, response: str, query: str) -> RouterOutput:
        """Parses the structured LLM classification response.

        Extracts CATEGORY, CONFIDENCE, REASONING, and FALLBACK fields
        from the LLM output using regex pattern matching.

        Args:
            response: The raw LLM response string to parse.
            query: The original query string for inclusion in output.

        Returns:
            A RouterOutput with all fields populated.
            Falls back to UNKNOWN category if parsing fails.
        """
        def _extract(pattern: str, text: str) -> Optional[str]:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else None

        # Extract each field from the structured response
        raw_category = _extract(r"^CATEGORY:\s*(\w+)", response)
        raw_confidence = _extract(r"^CONFIDENCE:\s*([\d.]+)", response)
        raw_reasoning = _extract(r"^REASONING:\s*(.+)", response)
        raw_fallback = _extract(r"^FALLBACK:\s*(\w+)", response)

        # Validate and coerce category
        try:
            category = QueryCategory(raw_category.upper()) if raw_category else QueryCategory.UNKNOWN
        except ValueError:
            logger.warning(
                "RouterAgent received unknown category '{}' — defaulting to UNKNOWN",
                raw_category,
            )
            category = QueryCategory.UNKNOWN

        # Validate and coerce confidence score
        try:
            confidence = float(raw_confidence) if raw_confidence else 0.5
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5

        # Validate fallback category
        fallback: Optional[QueryCategory] = None
        if raw_fallback and raw_fallback.upper() != "NONE":
            try:
                fallback = QueryCategory(raw_fallback.upper())
            except ValueError:
                fallback = None

        return RouterOutput(
            query=query,
            category=category,
            confidence=confidence,
            reasoning=raw_reasoning or "No reasoning provided.",
            fallback_category=fallback,
        )

    def classify(self, query: str) -> RouterOutput:
        """Classifies a raw query string into a QueryCategory.

        This is the primary public method of the RouterAgent, called
        directly by the Master Orchestrator before agent dispatch.

        Args:
            query: The raw user query string to classify.

        Returns:
            A RouterOutput containing the category, confidence,
            reasoning, and optional fallback category.
        """
        logger.info(
            "RouterAgent classifying | query='{}'", query[:80]
        )

        prompt = self._build_prompt(query)

        try:
            raw_response = self._invoke_llm(prompt)
            logger.debug("Router LLM response:\n{}", raw_response)
            router_output = self._parse_llm_response(raw_response, query)

        except Exception as exc:
            logger.error("RouterAgent classification failed: {}", exc)
            router_output = RouterOutput(
                query=query,
                category=QueryCategory.UNKNOWN,
                confidence=0.0,
                reasoning=f"Classification failed due to error: {exc}",
                fallback_category=None,
            )

        logger.info(
            "RouterAgent classified | category='{}' | confidence={} | query='{}'",
            router_output.category.value,
            router_output.confidence,
            query[:60],
        )

        return router_output

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Implements BaseAgent.run() for orchestrator compatibility.

        Wraps classify() in the standard AgentOutput format so the
        router can be used as a node in the LangGraph state machine.

        Args:
            agent_input: Standard AgentInput from the orchestrator.

        Returns:
            AgentOutput where the answer contains the classification
            result as a formatted string.
        """
        router_output = self.classify(agent_input.query)

        answer = (
            f"Query classified as: {router_output.category.value}\n"
            f"Confidence        : {router_output.confidence}\n"
            f"Reasoning         : {router_output.reasoning}\n"
            f"Fallback          : "
            f"{router_output.fallback_category.value if router_output.fallback_category else 'None'}"
        )

        return self._build_output(
            query=agent_input.query,
            answer=answer,
            sources=[],
            status=AgentStatus.SUCCESS,
            confidence=router_output.confidence,
        )

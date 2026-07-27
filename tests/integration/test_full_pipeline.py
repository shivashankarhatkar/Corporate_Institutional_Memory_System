"""
Module: tests/integration/test_full_pipeline.py

Purpose:
    Integration tests verifying that ingestion, memory storage, agent
    retrieval, and orchestration work correctly together end-to-end.

Responsibilities:
    - Exercise the real ChromaDB vector store (isolated to a temp
      directory per test) rather than mocking it, to catch wiring bugs
      that unit-level mocks would hide.
    - Mock only the Gemini LLM boundary (BaseAgent._invoke_llm) so tests
      run fast, deterministically, and without API cost or network
      dependency.
    - Verify: parse → chunk → embed → store → retrieve → synthesise
      answer, across DecisionAgent, RetrievalOrchestrator, and
      MasterOrchestrator layers.
    - Verify capture flow: MeetingAgent → ChromaDB storage.

Notes:
    Neo4j is intentionally left unavailable in these tests (graph_store
    is not constructed) to verify the system's graceful-degradation path
    remains fully functional end-to-end without a graph database.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import ingestion.embedder as embedder_module
from agents.base_agent import BaseAgent
from agents.capture.meeting_agent import MeetingAgent
from config.settings import settings
from ingestion.chunker import chunk_emails
from ingestion.email_parser import _to_clean_email, RawEmail
from memory.cache import QueryCache
from memory.memory_manager import MemoryManager
from memory.vector_store import VectorStore
from orchestrators.master_orchestrator import (
    MasterOrchestrator,
    MasterRequest,
    RequestType,
)
from orchestrators.retrieval_orchestrator import (
    RetrievalOrchestrator,
    RetrievalRequest,
)
from schemas.agent_schema import AgentStatus, QueryCategory


# ── Fixtures: isolated ChromaDB per test ─────────────────────────────────────

@pytest.fixture
def isolated_chroma_settings(tmp_path, monkeypatch):
    """Redirects ChromaDB to a unique temp directory and collection per test.

    Also resets ingestion.embedder's module-level singletons so a fresh
    client/collection is created pointing at the isolated temp directory,
    rather than reusing a stale cached client from a previous test.

    Args:
        tmp_path: pytest's built-in temporary directory fixture.
        monkeypatch: pytest's built-in monkeypatching fixture.
    """
    test_dir = str(tmp_path / "chroma_test")
    monkeypatch.setattr(settings.chromadb, "persist_directory", test_dir)
    monkeypatch.setattr(settings.chromadb, "collection_name", "test_collection")

    # Reset embedder module singletons so they rebuild against the new path
    monkeypatch.setattr(embedder_module, "_embedding_model", None)
    monkeypatch.setattr(embedder_module, "_chroma_client", None)
    monkeypatch.setattr(embedder_module, "_chroma_collection", None)

    yield test_dir


@pytest.fixture
def real_vector_store(isolated_chroma_settings) -> VectorStore:
    """Provides a real VectorStore instance pointed at the isolated temp dir.

    Args:
        isolated_chroma_settings: Ensures ChromaDB settings are isolated.

    Returns:
        A VectorStore instance backed by a temporary ChromaDB directory.
    """
    return VectorStore()


@pytest.fixture
def real_memory_manager(real_vector_store: VectorStore) -> MemoryManager:
    """Provides a MemoryManager wired to a real (isolated) VectorStore and
    no GraphStore, exercising the graceful-degradation path.

    Args:
        real_vector_store: The isolated VectorStore fixture.

    Returns:
        A MemoryManager instance with graph_store=None.
    """
    return MemoryManager(
        vector_store=real_vector_store,
        graph_store=None,
        cache=QueryCache(),
    )


@pytest.fixture(autouse=True)
def patch_llm_client():
    """Patches ChatGoogleGenerativeAI construction for every test in this module.

    Prevents real API client instantiation across all agents constructed
    during these integration tests.
    """
    with patch("agents.base_agent.ChatGoogleGenerativeAI") as mock_llm_class:
        mock_llm_class.return_value = MagicMock()
        yield mock_llm_class


# ── Helper: ingest a fixed set of test emails into real ChromaDB ────────────

def _ingest_test_emails(vector_store: VectorStore) -> None:
    """Parses, chunks, embeds, and stores a small fixed set of test emails.

    Bypasses the CSV-reading stage of email_parser.py and constructs
    RawEmail/CleanEmail objects directly, to keep the integration test
    self-contained and independent of any external dataset file.

    Args:
        vector_store: The real VectorStore instance to store chunks into.
    """
    raw_emails = [
        RawEmail(
            file_path="test/inbox/1",
            message_id="<budget001@enron.com>",
            date="Mon, 14 May 2001 16:39:00 -0700 (PDT)",
            sender="phillip.allen@enron.com",
            receiver="tim.belden@enron.com",
            subject="Q3 Budget Decision",
            body=(
                "We decided to cut marketing spend by fifteen percent this "
                "quarter due to the ongoing revenue shortfall. Tim approved "
                "this change after reviewing the numbers with finance."
            ),
        ),
        RawEmail(
            file_path="test/inbox/2",
            message_id="<travel002@enron.com>",
            date="Tue, 22 Aug 2000 07:44:00 -0700 (PDT)",
            sender="phillip.allen@enron.com",
            receiver="john.lavorato@enron.com",
            subject="Travel Policy Update",
            body=(
                "Going forward, all business trips over five hundred dollars "
                "require manager pre-approval before booking any travel."
            ),
        ),
    ]

    clean_emails = [
        clean for raw in raw_emails
        if (clean := _to_clean_email(raw)) is not None
    ]
    assert len(clean_emails) == 2, "Test fixture emails failed cleaning validation."

    chunks = chunk_emails(clean_emails)

    # Directly embed and store using the isolated VectorStore's underlying
    # collection, reusing the same embedding model configured in settings.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.chromadb.embedding_model)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    vector_store._collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "message_id": c.message_id,
                "chunk_index": c.chunk_index,
                "sender": c.sender,
                "receiver": c.receiver,
                "subject": c.subject or "",
                "date": c.date,
                "word_count": c.word_count,
                "department": c.department or "",
            }
            for c in chunks
        ],
    )


# ── Test: ingestion → memory retrieval (no LLM involved) ─────────────────────

class TestIngestionToMemoryRetrieval:
    """Verifies emails flow correctly from parsing through to retrieval."""

    def test_ingested_email_is_retrievable_by_semantic_search(
        self, real_vector_store: VectorStore
    ) -> None:
        """A stored email should be found by a semantically related query."""
        _ingest_test_emails(real_vector_store)

        results = real_vector_store.search("why was marketing spend reduced")

        assert len(results) > 0
        assert any(
            "marketing spend" in r.text.lower() for r in results
        )

    def test_metadata_filter_restricts_results_to_sender(
        self, real_vector_store: VectorStore
    ) -> None:
        """search_by_sender should only return chunks from that sender."""
        _ingest_test_emails(real_vector_store)

        results = real_vector_store.search_by_sender(
            query="policy",
            sender_email="phillip.allen@enron.com",
        )

        assert len(results) > 0
        assert all(
            r.metadata["sender"] == "phillip.allen@enron.com" for r in results
        )

    def test_memory_manager_search_matches_vector_store_directly(
        self, real_memory_manager: MemoryManager, real_vector_store: VectorStore
    ) -> None:
        """MemoryManager.search should return equivalent results to the raw store."""
        _ingest_test_emails(real_vector_store)

        results = real_memory_manager.search("travel approval requirements")

        assert len(results) > 0
        assert any("pre-approval" in r.text.lower() for r in results)


# ── Test: DecisionAgent end-to-end with real memory + mocked LLM ─────────────

class TestDecisionAgentEndToEnd:
    """Verifies DecisionAgent correctly retrieves real chunks and calls the LLM."""

    def test_run_produces_grounded_answer_from_real_retrieval(
        self, real_memory_manager: MemoryManager, real_vector_store: VectorStore
    ) -> None:
        """DecisionAgent.run() should retrieve real chunks and synthesise an answer."""
        from agents.retrieval.decision_agent import DecisionAgent
        from schemas.agent_schema import AgentInput

        _ingest_test_emails(real_vector_store)

        agent = DecisionAgent(memory=real_memory_manager)
        agent._invoke_llm = MagicMock(
            return_value=(
                "Marketing spend was cut by 15% due to a revenue shortfall, "
                "approved by Tim. [Source 1]\n\n"
                "FOLLOW_UP_1: What other budget lines were reviewed?\n"
                "FOLLOW_UP_2: When will spend be restored?\n"
                "FOLLOW_UP_3: Who else was consulted?"
            )
        )

        output = agent.run(
            AgentInput(query="Why did we cut the marketing budget?")
        )

        assert output.status == AgentStatus.SUCCESS
        assert len(output.sources) > 0
        assert "revenue shortfall" in output.answer
        assert len(output.follow_up_questions) == 3

        # Verify the LLM was actually given real retrieved context, not empty
        called_prompt = agent._invoke_llm.call_args[0][0]
        assert "marketing spend" in called_prompt.lower()


# ── Test: RetrievalOrchestrator end-to-end ────────────────────────────────────

class TestRetrievalOrchestratorEndToEnd:
    """Verifies the full retrieval orchestration flow with a category hint."""

    def test_retrieve_with_category_hint_bypasses_router_and_returns_answer(
        self, real_memory_manager: MemoryManager, real_vector_store: VectorStore
    ) -> None:
        """Providing category_hint should skip router classification entirely."""
        _ingest_test_emails(real_vector_store)

        orchestrator = RetrievalOrchestrator(memory=real_memory_manager)

        # Mock every agent's LLM call at the class level so any internally
        # constructed specialist agent produces a deterministic response.
        BaseAgent._invoke_llm = MagicMock(
            return_value="Travel over $500 requires manager pre-approval. [Source 1]"
        )

        request = RetrievalRequest(
            query="What is the travel approval policy?",
            category_hint=QueryCategory.POLICY,
        )
        output = orchestrator.retrieve(request)

        assert output.status in {AgentStatus.SUCCESS, AgentStatus.PARTIAL}
        assert output.category == QueryCategory.POLICY
        assert "pre-approval" in output.answer.lower()


# ── Test: MasterOrchestrator full QUERY request ───────────────────────────────

class TestMasterOrchestratorEndToEnd:
    """Verifies the Master Orchestrator correctly wires a QUERY request through."""

    def test_process_query_request_returns_success_response(
        self, real_memory_manager: MemoryManager, real_vector_store: VectorStore
    ) -> None:
        """A full QUERY MasterRequest should return a populated MasterResponse."""
        _ingest_test_emails(real_vector_store)

        BaseAgent._invoke_llm = MagicMock(
            return_value=(
                "CATEGORY: DECISION\nCONFIDENCE: 0.9\n"
                "REASONING: asks why a decision was made\nFALLBACK: NONE"
            )
        )

        master = MasterOrchestrator(memory=real_memory_manager)

        request = MasterRequest(
            request_type=RequestType.QUERY,
            query="Why was the marketing budget reduced?",
        )
        response = master.process(request)

        assert response.success is True
        assert response.query_result is not None
        assert response.request_type == RequestType.QUERY
        assert response.processing_time_ms >= 0

    def test_health_request_returns_populated_health_status(
        self, real_memory_manager: MemoryManager
    ) -> None:
        """A HEALTH request should return subsystem status without erroring."""
        master = MasterOrchestrator(memory=real_memory_manager)

        request = MasterRequest(request_type=RequestType.HEALTH)
        response = master.process(request)

        assert response.success is True
        assert response.health_status is not None
        assert "status" in response.health_status


# ── Test: Capture flow → real ChromaDB storage ────────────────────────────────

class TestMeetingCaptureEndToEnd:
    """Verifies MeetingAgent extraction is correctly stored in real ChromaDB."""

    def test_meeting_capture_stores_transcript_in_vector_store(
        self, real_memory_manager: MemoryManager, isolated_chroma_settings
    ) -> None:
        """A captured meeting transcript should become retrievable afterwards."""
        agent = MeetingAgent(memory=real_memory_manager)
        agent._invoke_llm = MagicMock(
            return_value=(
                "MEETING_DATE: 2001-06-01\n"
                "PARTICIPANTS: phillip.allen@enron.com, tim.belden@enron.com\n"
                "SUMMARY: Team agreed to delay the pipeline expansion project.\n\n"
                "DECISIONS:\n"
                "DECISION_1: Delay pipeline expansion by two quarters | "
                "CONTEXT: budget constraints | PARTICIPANTS: phillip.allen@enron.com\n\n"
                "ACTION_ITEMS:\n"
                "ACTION_1: Notify vendor of delay | OWNER: phillip.allen@enron.com | DEADLINE: 2001-06-15\n\n"
                "KEY_TOPICS: pipeline, budget, delay"
            )
        )

        extraction = agent.extract_from_content(
            "Meeting notes: discussed pipeline expansion timeline and budget "
            "constraints. Decided to delay the project by two quarters."
        )

        assert extraction.stored_in_vector is True
        assert len(extraction.decisions) == 1
        assert len(extraction.action_items) == 1

        # Verify it is now retrievable via the same memory manager
        results = real_memory_manager.search(
            "pipeline expansion delay", use_cache=False
        )
        assert any(
            "pipeline" in r.text.lower() for r in results
        )

"""Dependency wiring for ContextOS daemon.

This is where all protocol implementations are bound to their interfaces.
One function, called at startup, that assembles the entire dependency graph.
No magic. No DI framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from contextos.config.settings import Settings
from contextos.core.enums import SecretDetectionMode
from contextos.storage.database import Database

logger = logging.getLogger(__name__)


async def wire_services(settings: Settings) -> dict[str, Any]:
    """Create and wire all service instances.

    Returns a dict of {service_name: instance} ready for injection into
    the API server and CLI.
    """
    services: dict[str, Any] = {}

    # --- Database ---
    data_dir = settings.daemon.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    db = Database(data_dir / "contextos.db")
    await db.initialize()
    services["database"] = db

    conn = db.connection()

    # --- Repositories ---
    from contextos.storage.memory_repo import SqliteMemoryRepository
    from contextos.storage.event_repo import SqliteEventRepository

    memory_repo = SqliteMemoryRepository(conn)
    event_repo = SqliteEventRepository(conn)
    from contextos.storage.relation_repo import SqliteRelationRepository
    from contextos.storage.graph_repo import SqliteGraphRepository
    relation_repo = SqliteRelationRepository(conn)
    graph_repo = SqliteGraphRepository(conn)
    services["memory_repo"] = memory_repo
    services["event_repo"] = event_repo
    services["relation_repo"] = relation_repo
    services["graph_repo"] = graph_repo

    # --- Temporal resolution ---
    from contextos.services.temporal import TemporalMemoryService

    temporal = TemporalMemoryService(memory_repo)
    services["temporal"] = temporal

    from contextos.services.graph import MemoryGraphService

    graph = MemoryGraphService(
        memory_repo=memory_repo,
        relation_repo=relation_repo,
        graph_repo=graph_repo,
    )
    services["graph"] = graph

    # --- Token Counter ---
    from contextos.services.token_counter import TiktokenCounter

    token_counter = TiktokenCounter()
    services["token_counter"] = token_counter

    # --- Token-aware optimizer ---
    from contextos.services.optimization import MemoryContextOptimizer

    optimizer = MemoryContextOptimizer(token_counter=token_counter)
    services["optimizer"] = optimizer

    # --- Embedding Service ---
    if settings.embedding.model == "deterministic":
        # Explicit local/test configuration. This avoids a model download while
        # retaining the normal retrieval, indexing, graph, and SQLite services.
        from contextos.embedding.deterministic import DeterministicEmbedding
        embedding_service = DeterministicEmbedding(16)
        vector_dimension = 16
    else:
        from contextos.embedding.sentence_transformers import SentenceTransformerEmbedding
        embedding_service = SentenceTransformerEmbedding(
            model_name=settings.embedding.model,
            device=settings.embedding.device,
        )
        vector_dimension = 384
    services["embedding"] = embedding_service

    # --- Vector Store ---
    from contextos.storage.vector.in_memory import InMemoryVectorStore

    vector_store = InMemoryVectorStore(dimension=vector_dimension)
    services["vector_store"] = vector_store

    # --- BM25 Index ---
    from contextos.storage.lexical.bm25 import BM25Index

    bm25_index = BM25Index()
    services["bm25_index"] = bm25_index

    # --- Explicit retrieval index synchronization ---
    from contextos.services.retrieval_index import RetrievalIndexSynchronizer

    retrieval_index = RetrievalIndexSynchronizer(
        memory_repo=memory_repo,
        embedding_service=embedding_service,
        vector_store=vector_store,
        lexical_index=bm25_index,
    )
    services["retrieval_index"] = retrieval_index

    # --- Secret Scanner ---
    from contextos.services.secret_scanner import PatternSecretScanner

    scanner = PatternSecretScanner(
        entropy_threshold=settings.privacy.entropy_threshold,
    )
    services["secret_scanner"] = scanner

    # --- Memory Extractor ---
    from contextos.services.extraction import RuleBasedMemoryExtractor

    extractor = RuleBasedMemoryExtractor()
    services["memory_extractor"] = extractor

    # --- Ingestion Pipeline ---
    from contextos.services.ingestion import IngestionPipeline

    secret_mode = SecretDetectionMode(settings.privacy.secret_detection)
    ingestion = IngestionPipeline(
        secret_scanner=scanner,
        memory_extractor=extractor,
        memory_repo=memory_repo,
        event_repo=event_repo,
        embedding_service=embedding_service,
        vector_store=vector_store,
        lexical_index=bm25_index,
        token_counter=token_counter,
        secret_detection_mode=secret_mode,
    )
    services["ingestion"] = ingestion

    # --- Phase 11: connector state and bounded sync manager ---
    from contextos.storage.connector_repo import SqliteConnectorRepository
    from contextos.connectors.manager import ConnectorManager
    connector_repo = SqliteConnectorRepository(conn)
    services["connector_repo"] = connector_repo
    services["connectors"] = ConnectorManager(
        state_repo=connector_repo, ingestion=ingestion, temporal=temporal,
    )

    # --- Memory Manager ---
    from contextos.services.memory import MemoryManager

    memory_manager = MemoryManager(
        memory_repo=memory_repo,
        event_repo=event_repo,
        vector_store=vector_store,
        lexical_index=bm25_index,
        embedding_service=embedding_service,
    )
    services["memory"] = memory_manager

    # --- Retrieval Engine ---
    from contextos.services.retrieval import HybridRetrievalEngine

    base_retrieval = HybridRetrievalEngine(
        memory_repo=memory_repo,
        vector_store=vector_store,
        lexical_index=bm25_index,
        embedding_service=embedding_service,
        index_synchronizer=retrieval_index,
    )
    from contextos.services.graph_retrieval import GraphAugmentedRetrievalEngine

    retrieval = GraphAugmentedRetrievalEngine(
        base_engine=base_retrieval,
        graph_service=graph,
        memory_repo=memory_repo,
    )
    services["base_retrieval"] = base_retrieval
    services["retrieval"] = retrieval

    # --- Context Compiler ---
    from contextos.services.compilation import QueryAwareContextCompiler

    compiler = QueryAwareContextCompiler(token_counter=token_counter)
    services["compilation"] = compiler

    # --- Phase 9: Telemetry Repository & Query Service ---
    from contextos.storage.telemetry_repo import SqliteTelemetryRepository
    from contextos.services.telemetry_query import TelemetryQueryService

    telemetry_repo = SqliteTelemetryRepository(conn)
    telemetry_query = TelemetryQueryService(telemetry_repo)
    services["telemetry_repo"] = telemetry_repo
    services["telemetry_query"] = telemetry_query

    # --- Phase 9: Provider Adapters ---
    from contextos.providers.fake import DeterministicFakeProvider
    from contextos.providers.ollama import OllamaProvider
    from contextos.providers.openai_compatible import OpenAICompatibleProvider

    fake_provider = DeterministicFakeProvider()
    ollama_provider = OllamaProvider()
    openai_compatible_provider = OpenAICompatibleProvider()

    providers = {
        fake_provider.provider_id: fake_provider,
        ollama_provider.provider_id: ollama_provider,
        openai_compatible_provider.provider_id: openai_compatible_provider,
    }
    services["fake_provider"] = fake_provider
    services["ollama_provider"] = ollama_provider
    services["openai_compatible_provider"] = openai_compatible_provider
    services["providers"] = providers

    # --- Phase 9: Model Router ---
    from contextos.core.enums import RoutingPolicy
    from contextos.services.router import DeterministicModelRouter

    router = DeterministicModelRouter(
        default_provider_id="fake",
        default_model_id="fake-default",
        default_policy=RoutingPolicy.LOCAL_FIRST,
    )
    services["router"] = router

    # --- Phase 9: Unified ContextOS Model Service ---
    from contextos.services.model_service import ContextOSModelService

    model_service = ContextOSModelService(
        retrieval_service=retrieval,
        optimizer=optimizer,
        compilation_service=compiler,
        router=router,
        providers=providers,
        telemetry_repo=telemetry_repo,
        token_counter=token_counter,
    )
    services["model_service"] = model_service

    logger.info("All services wired successfully")
    return services

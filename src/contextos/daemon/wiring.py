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
    relation_repo = SqliteRelationRepository(conn)
    services["memory_repo"] = memory_repo
    services["event_repo"] = event_repo
    services["relation_repo"] = relation_repo

    # --- Temporal resolution ---
    from contextos.services.temporal import TemporalMemoryService

    temporal = TemporalMemoryService(memory_repo)
    services["temporal"] = temporal

    # --- Token Counter ---
    from contextos.services.token_counter import TiktokenCounter

    token_counter = TiktokenCounter()
    services["token_counter"] = token_counter

    # --- Token-aware optimizer ---
    from contextos.services.optimization import MemoryContextOptimizer

    optimizer = MemoryContextOptimizer(token_counter=token_counter)
    services["optimizer"] = optimizer

    # --- Embedding Service ---
    from contextos.embedding.sentence_transformers import SentenceTransformerEmbedding

    embedding_service = SentenceTransformerEmbedding(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
    )
    services["embedding"] = embedding_service

    # --- Vector Store ---
    from contextos.storage.vector.in_memory import InMemoryVectorStore

    # Default dimension for all-MiniLM-L6-v2 is 384
    vector_store = InMemoryVectorStore(dimension=384)
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

    retrieval = HybridRetrievalEngine(
        memory_repo=memory_repo,
        vector_store=vector_store,
        lexical_index=bm25_index,
        embedding_service=embedding_service,
        index_synchronizer=retrieval_index,
    )
    services["retrieval"] = retrieval

    # --- Context Compiler ---
    from contextos.services.compilation import QueryAwareContextCompiler

    compiler = QueryAwareContextCompiler(token_counter=token_counter)
    services["compilation"] = compiler

    logger.info("All services wired successfully")
    return services

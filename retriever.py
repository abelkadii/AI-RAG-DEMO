"""Retrieval facade used by the agent."""

from __future__ import annotations

import os
from pathlib import Path

from models import EvidenceChunk
from vector_store import Embedder, PersistentVectorStore


class Retriever:
    def __init__(self, index_dir: str | Path | None = None, embedder: Embedder | None = None):
        self.store = PersistentVectorStore(index_dir or os.getenv("INDEX_DIR", "data/index"), embedder)

    def search(self, query: str, k: int = 6) -> list[EvidenceChunk]:
        results = self.store.search(query, k)
        return [
            chunk.model_copy(update={"source": "AWS Well-Architected Framework"})
            if chunk.source == "AWS-WAF" else chunk
            for chunk in results
        ]

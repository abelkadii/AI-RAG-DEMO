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
        return self.store.search(query, k)


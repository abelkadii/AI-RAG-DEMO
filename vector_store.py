"""Tiny persistent cosine-similarity store with swappable embedding backends."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from models import EvidenceChunk

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "aws", "be", "by", "for", "framework",
    "from", "how", "in", "is", "it", "of", "on", "or", "should", "that", "the",
    "their", "this", "to", "well", "wellarchitected", "while", "with", "workload",
}


def _terms(text: str) -> list[str]:
    result = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOPWORDS or len(token) < 3:
            continue
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.append(token)
    return result


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Dependency-free feature hashing; useful locally and in tests."""

    name = "local-hash-v1"

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _terms(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimensions
                matrix[row, index] += 1.0 if value & 1 else -1.0
        return _normalize(matrix)


class OpenAIEmbedder:
    def __init__(self, model: str, base_url: str | None, api_key: str):
        from openai import OpenAI

        self.name = model
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            response = self.client.embeddings.create(
                model=self.model, input=texts[start : start + 64]
            )
            vectors.extend(item.embedding for item in response.data)
        return _normalize(np.asarray(vectors, dtype=np.float32))


def configured_embedder() -> Embedder:
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if model and key:
        return OpenAIEmbedder(model, os.getenv("OPENAI_BASE_URL"), key)
    return HashEmbedder()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class PersistentVectorStore:
    def __init__(self, directory: str | Path, embedder: Embedder | None = None):
        self.directory = Path(directory)
        self.embedder = embedder or configured_embedder()
        self.vectors: np.ndarray | None = None
        self.chunks: list[EvidenceChunk] = []

    def build(self, chunks: Iterable[EvidenceChunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Cannot build an index with no chunks")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vectors = self.embedder.embed([chunk.text for chunk in self.chunks])
        np.save(self.directory / "vectors.npy", self.vectors)
        with (self.directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(chunk.model_dump_json() + "\n")
        (self.directory / "config.json").write_text(
            json.dumps({"embedder": self.embedder.name, "count": len(self.chunks)}, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        config_path = self.directory / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No index at {self.directory}. Run: python ingest.py")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["embedder"] != self.embedder.name:
            raise ValueError(
                f"Index uses {config['embedder']!r}, configured embedder is "
                f"{self.embedder.name!r}. Re-run ingestion."
            )
        self.vectors = np.load(self.directory / "vectors.npy")
        with (self.directory / "chunks.jsonl").open(encoding="utf-8") as handle:
            self.chunks = [EvidenceChunk.model_validate_json(line) for line in handle if line.strip()]

    def search(self, query: str, k: int = 6) -> list[EvidenceChunk]:
        if self.vectors is None:
            self.load()
        assert self.vectors is not None
        query_vector = self.embedder.embed([query])[0]
        scores = self.vectors @ query_vector
        query_terms = set(_terms(query))
        if query_terms:
            lexical = np.asarray(
                [len(query_terms & set(_terms(chunk.text))) / len(query_terms) for chunk in self.chunks],
                dtype=np.float32,
            )
            scores = scores + 0.25 * lexical
        top = np.argsort(scores)[::-1][: min(k, len(self.chunks))]
        return [self.chunks[int(i)].model_copy(update={"score": float(scores[i])}) for i in top]

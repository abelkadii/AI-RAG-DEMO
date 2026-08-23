"""Session-oriented PDF ingestion for uploaded demo documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pymupdf

from ingest import clean_text, split_page
from models import EvidenceChunk
from vector_store import InMemoryVectorStore

MAX_UPLOAD_FILES = 5
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class UploadedPDF:
    name: str
    content: bytes


class UploadedRetriever:
    def __init__(self, chunks: list[EvidenceChunk]):
        self.store = InMemoryVectorStore(chunks)

    @property
    def chunks(self) -> list[EvidenceChunk]:
        return self.store.chunks

    def search(self, query: str, k: int = 6) -> list[EvidenceChunk]:
        candidates = self.store.search(query, max(20, k * 3))
        return diversify_results(candidates, k=k)


def corpus_hash(files: list[UploadedPDF]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda file: file.name.lower()):
        digest.update(item.name.encode("utf-8", errors="ignore"))
        digest.update(len(item.content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(item.content).digest())
    return digest.hexdigest()


def build_uploaded_retriever(files: list[UploadedPDF]) -> UploadedRetriever:
    chunks = extract_uploaded_chunks(files)
    return UploadedRetriever(chunks)


def extract_uploaded_chunks(files: list[UploadedPDF]) -> list[EvidenceChunk]:
    if not files:
        raise ValueError("Upload at least one PDF.")
    if len(files) > MAX_UPLOAD_FILES:
        raise ValueError(f"Upload at most {MAX_UPLOAD_FILES} PDFs.")
    chunks: list[EvidenceChunk] = []
    for file in files:
        if not file.name.lower().endswith(".pdf"):
            raise ValueError(f"{file.name} is not a PDF.")
        if len(file.content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{file.name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB demo limit.")
        chunks.extend(extract_pdf_bytes(file.content, safe_source_name(file.name)))
    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")
    return chunks


def extract_pdf_bytes(content: bytes, source: str) -> list[EvidenceChunk]:
    document = pymupdf.open(stream=content, filetype="pdf")
    chunks: list[EvidenceChunk] = []
    prefix = hashlib.sha1(content).hexdigest()[:12]
    try:
        for page_index, page in enumerate(document):
            text = clean_uploaded_text(page.get_text("text"), source)
            for chunk_index, chunk_text in enumerate(split_page(text)):
                if useful_chunk_text(chunk_text):
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"{prefix}-p{page_index + 1:03d}-c{chunk_index:02d}",
                            page=page_index + 1,
                            text=chunk_text,
                            section=source,
                            source=source,
                        )
                )
    finally:
        document.close()
    return chunks


def safe_source_name(name: str) -> str:
    base = re.sub(r"[/\\\[\]\n\r]+", " ", name).strip()
    return base[:80] or "Uploaded document"


def diversify_results(candidates: list[EvidenceChunk], k: int = 6, max_per_page: int = 2) -> list[EvidenceChunk]:
    """Select relevant chunks while reducing duplicate-page/topic collapse."""
    if not candidates:
        return []
    max_score = max(chunk.score for chunk in candidates)
    threshold = max_score * 0.35 if max_score > 0 else float("-inf")
    pool = [chunk for chunk in candidates if chunk.score >= threshold]
    selected: list[EvidenceChunk] = []
    page_counts: dict[tuple[str, int], int] = {}
    while pool and len(selected) < k:
        best_index = 0
        best_value = float("-inf")
        for index, chunk in enumerate(pool):
            page_key = (chunk.source, chunk.page)
            if page_counts.get(page_key, 0) >= max_per_page:
                continue
            diversity_penalty = max((term_jaccard(chunk.text, picked.text) for picked in selected), default=0.0)
            value = chunk.score - (0.18 * diversity_penalty)
            if value > best_value:
                best_index = index
                best_value = value
        if best_value == float("-inf"):
            break
        chosen = pool.pop(best_index)
        selected.append(chosen)
        page_key = (chosen.source, chosen.page)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
    return selected


def term_jaccard(first: str, second: str) -> float:
    first_terms = set(re.findall(r"[\w\u0600-\u06FF]{4,}", first.lower()))
    second_terms = set(re.findall(r"[\w\u0600-\u06FF]{4,}", second.lower()))
    if not first_terms or not second_terms:
        return 0.0
    return len(first_terms & second_terms) / len(first_terms | second_terms)


def clean_uploaded_text(text: str, source: str) -> str:
    """Clean uploaded PDF text without discarding non-Latin scripts."""
    lines = []
    source_stem = re.sub(r"\.pdf$", "", source, flags=re.I).replace("_", " ").replace("-", " ").strip().lower()
    for raw_line in text.splitlines():
        line = raw_line.translate(
            str.maketrans(
                {
                    "\u00ad": "",
                    "\xa0": " ",
                    "\ufb00": "ff",
                    "\ufb01": "fi",
                    "\ufb02": "fl",
                    "\ufb03": "ffi",
                    "\ufb04": "ffl",
                }
            )
        )
        line = re.sub(r"\s+", " ", line).strip()
        line = repair_hyphenation_artifacts(line)
        if not line:
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue
        if re.fullmatch(r"(page|slide)\s+\d+\s+(of|/)\s+\d+", line, flags=re.I):
            continue
        normalized = re.sub(r"[_\-]+", " ", line).lower()
        if source_stem and normalized == source_stem:
            continue
        if looks_like_footer_or_title_fragment(line):
            continue
        if looks_garbled(line):
            continue
        lines.append(line)
    return clean_text(" ".join(lines))


def repair_hyphenation_artifacts(text: str) -> str:
    text = re.sub(r"\b([A-Za-z]{2,})-\s+([a-z]{2,})\b", r"\1\2", text)
    text = re.sub(r"\b([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})\b", r"\1\2", text)
    return text


def looks_like_footer_or_title_fragment(line: str) -> bool:
    words = re.findall(r"[\w\u0600-\u06FF]+", line)
    if len(words) <= 3 and len(line) <= 48:
        if re.search(r"\b(author|copyright|confidential|draft|version|www\.|@)\b", line, flags=re.I):
            return True
    if re.fullmatch(r"[A-Z][A-Za-z .,&-]{2,45}", line) and len(words) <= 4:
        return True
    return False


def looks_garbled(text: str) -> bool:
    if "\ufffd" in text:
        return True
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return True
    symbol_ratio = sum(1 for char in visible if not (char.isalnum() or "\u0600" <= char <= "\u06FF" or char in ".,;:!?؟،٪%()-_/")) / len(visible)
    return symbol_ratio > 0.25


def useful_chunk_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 40:
        return False
    if re.search(r"\b\d+\s*/\s*\d+\b", cleaned):
        return False
    if looks_garbled(cleaned):
        return False
    return True

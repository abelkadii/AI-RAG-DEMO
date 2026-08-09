"""Download, extract, chunk, and index the AWS Well-Architected PDF."""

from __future__ import annotations

import argparse
import os
import re
import urllib.request
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

from models import EvidenceChunk
from vector_store import PersistentVectorStore

DEFAULT_URL = "https://docs.aws.amazon.com/pdfs/wellarchitected/2024-06-27/framework/wellarchitected-framework-2024-06-27.pdf"
PILLARS = (
    "Operational Excellence",
    "Security",
    "Reliability",
    "Performance Efficiency",
    "Cost Optimization",
    "Sustainability",
)
PILLAR_CODES = {
    "OPS": "Operational Excellence",
    "SEC": "Security",
    "REL": "Reliability",
    "PERF": "Performance Efficiency",
    "COST": "Cost Optimization",
    "SUS": "Sustainability",
}


def download_pdf(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "agentic-rag-poc/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        destination.write_bytes(response.read())


def clean_text(text: str) -> str:
    text = text.translate(
        str.maketrans({"\u00ad": "", "\xa0": " ", "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"})
    )
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^AWS Well-Architected Framework Framework\s*", "", text, flags=re.I)
    return text.strip()


def detect_section(text: str, previous: str | None) -> str | None:
    code = re.search(r"\b(OPS|SEC|REL|PERF|COST|SUS)\s*\d{1,2}\b", text[:2500], re.I)
    if code:
        return PILLAR_CODES[code.group(1).upper()]
    lower = text.lower()
    for pillar in PILLARS:
        if pillar.lower() in lower[:1500]:
            return pillar
    return previous


def split_page(text: str, target_words: int = 260, overlap: int = 45) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        if end < len(words):
            window = " ".join(words[start:end])
            boundary = max(window.rfind(". "), window.rfind(": "))
            if boundary > len(window) * 0.65:
                end = start + len(window[: boundary + 1].split())
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_chunks(pdf_path: Path) -> list[EvidenceChunk]:
    document = pymupdf.open(pdf_path)
    chunks: list[EvidenceChunk] = []
    section: str | None = None
    for page_index, page in enumerate(document):
        text = clean_text(page.get_text("text"))
        section = detect_section(text, section)
        for chunk_index, chunk_text in enumerate(split_page(text)):
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"aws-waf-p{page_index + 1:03d}-c{chunk_index:02d}",
                    page=page_index + 1,
                    text=chunk_text,
                    section=section,
                    source="AWS-WAF",
                )
            )
    document.close()
    return chunks


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=os.getenv("PDF_PATH", "data/aws-well-architected-framework.pdf"))
    parser.add_argument("--url", default=os.getenv("PDF_URL", DEFAULT_URL))
    parser.add_argument("--index", default=os.getenv("INDEX_DIR", "data/index"))
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if args.force_download or not pdf_path.exists():
        download_pdf(args.url, pdf_path)
    print(f"Extracting {pdf_path}")
    chunks = extract_chunks(pdf_path)
    print(f"Extracted {len(chunks)} chunks; building index")
    PersistentVectorStore(args.index).build(chunks)
    print(f"Index saved to {args.index}")


if __name__ == "__main__":
    main()

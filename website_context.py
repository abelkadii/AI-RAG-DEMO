"""Bounded, best-effort website context for the client-source side of the POC."""

from __future__ import annotations

import html
import re
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from models import EvidenceChunk


MAX_WEBSITE_PAGES = 5
MAX_WEBSITE_CHARS_PER_PAGE = 8000
ALLOWED_PATHS = ("/", "/about", "/about-us", "/collections", "/shop", "/shipping", "/policies")


def fetch_website_evidence(url: str, max_pages: int = MAX_WEBSITE_PAGES) -> tuple[list[EvidenceChunk], str | None]:
    """Fetch only a small set of same-domain pages and degrade gracefully."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme != "https" or not parsed.hostname:
        return [], "Website context requires an https URL."
    host = parsed.hostname.lower()
    base = f"https://{host}/"
    chunks: list[EvidenceChunk] = []
    errors: list[str] = []
    for index, path in enumerate(ALLOWED_PATHS[:max_pages]):
        page_url = urljoin(base, path)
        try:
            request = Request(page_url, headers={"User-Agent": "AI-Document-Studio/1.0"})
            with urlopen(request, timeout=6) as response:  # nosec B310 - validated https, same host
                raw = response.read(1_000_000).decode("utf-8", errors="ignore")
            text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 80:
                continue
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"website-{host}-{index}",
                    page=index + 1,
                    section=path,
                    source=host,
                    text=text[:MAX_WEBSITE_CHARS_PER_PAGE],
                )
            )
        except Exception as error:  # website context is optional and must never block generation
            errors.append(f"{path}: {error}")
    if chunks:
        return chunks, None
    return [], "Website context could not be loaded; continuing with supplied client material and brief."


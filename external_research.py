"""Small, bounded public-web research adapter for strategy sections.

This is intentionally session-only and opt-in.  When disabled, callers receive
an explicit limitation rather than silently treating client requirements or a
client website as market evidence.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from models import EvidenceChunk


MAX_EXTERNAL_QUERIES = 4
MAX_EXTERNAL_RESULTS = 3
MAX_EXTERNAL_CHARS = 5000


@dataclass
class ExternalResearchReport:
    queries: list[str] = field(default_factory=list)
    results: int = 0
    errors: list[str] = field(default_factory=list)
    enabled: bool = False

    @property
    def notice(self) -> str:
        if not self.enabled:
            return "External research unavailable — generating from the supplied client evidence and marking market limitations."
        if self.results:
            return f"External research: {self.results} public results collected across {len(self.queries)} bounded queries."
        return "External research unavailable — public search returned no usable results; market validation remains required."


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str, str]] = []
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._snippet: list[str] = []
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        if tag.lower() == "a" and "result__a" in classes:
            self._anchor_href = attrs_dict.get("href")
            self._anchor_text = []
        elif classes & {"result__snippet", "result-snippet"}:
            self._in_snippet = True
            self._snippet = []

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_text.append(data)
        if self._in_snippet:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._anchor_href is not None:
            title = re.sub(r"\s+", " ", "".join(self._anchor_text)).strip()
            href = self._anchor_href
            snippet = re.sub(r"\s+", " ", "".join(self._snippet)).strip()
            if title and href:
                self.results.append((title, href, snippet))
            self._anchor_href = None
            self._anchor_text = []
            self._snippet = []
        if self._in_snippet and tag.lower() in {"div", "td", "span"}:
            self._in_snippet = False


def external_research_enabled() -> bool:
    return os.getenv("ENABLE_EXTERNAL_RESEARCH", "").strip().lower() in {"1", "true", "yes", "on"}


def _result_url(href: str) -> str | None:
    parsed = urlparse(html.unescape(href))
    if parsed.netloc.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            return unquote(target)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        return href
    return None


def research_public_sources(
    queries: list[str],
    *,
    max_queries: int = MAX_EXTERNAL_QUERIES,
    max_results: int = MAX_EXTERNAL_RESULTS,
) -> tuple[list[EvidenceChunk], ExternalResearchReport]:
    """Collect bounded search-result evidence without persistent storage."""
    selected_queries = [" ".join(query.split()) for query in queries if query.strip()][:max(1, min(max_queries, MAX_EXTERNAL_QUERIES))]
    report = ExternalResearchReport(queries=selected_queries, enabled=external_research_enabled())
    if not report.enabled:
        return [], report
    chunks: list[EvidenceChunk] = []
    seen_urls: set[str] = set()
    for query_index, query in enumerate(selected_queries, start=1):
        endpoint = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            request = Request(endpoint, headers={"User-Agent": "AI-Document-Studio/1.0"})
            with urlopen(request, timeout=6) as response:  # nosec B310 - fixed HTTPS search endpoint
                raw = response.read(600_000).decode("utf-8", errors="ignore")
            parser = _SearchParser()
            parser.feed(raw)
            for result_index, (title, href, snippet) in enumerate(parser.results[:max_results], start=1):
                target = _result_url(href)
                if not target or target in seen_urls:
                    continue
                seen_urls.add(target)
                host = (urlparse(target).hostname or "public-source").removeprefix("www.")
                text = f"{title}. {snippet}".strip()
                if len(text) < 30:
                    continue
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"external-{query_index:02d}-{result_index:02d}",
                        page=1,
                        section="Public search result",
                        source=f"Web Research: {host}",
                        text=text[:MAX_EXTERNAL_CHARS],
                    )
                )
        except Exception as error:  # public search is optional
            report.errors.append(f"Query {query_index}: {error}")
    report.results = len(chunks)
    return chunks, report


def queries_for_section(title: str, objective: str, questions: list[str]) -> list[str]:
    context = " ".join([title, objective, *questions]).strip()
    lower = context.lower()
    queries = [f"{title} Singapore market trends credible public sources"]
    if any(term in lower for term in ("competitive", "competitor", "positioning")):
        queries.append(f"Singapore {title} competitor landscape category leaders")
    if any(term in lower for term in ("market", "customer", "sizing", "international")):
        queries.append(f"Singapore {title} market size customer trends government industry")
    if any(term in lower for term in ("international", "expansion", "partnership")):
        queries.append(f"{title} international expansion readiness public evidence")
    return queries[:MAX_EXTERNAL_QUERIES]

"""Bounded, session-only website evidence for client document production."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from models import EvidenceChunk


MAX_WEBSITE_PAGES = 5
MAX_WEBSITE_CHARS_PER_PAGE = 8000
MAX_WEBSITE_BYTES = 1_000_000
MAX_DISCOVERED_LINKS = 24


@dataclass
class WebsiteFetchReport:
    requested_url: str
    resolved_url: str | None = None
    status_code: int | None = None
    pages_discovered: list[str] = field(default_factory=list)
    pages_fetched: list[str] = field(default_factory=list)
    pages_rejected: list[str] = field(default_factory=list)
    indexed_pages: list[str] = field(default_factory=list)
    character_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.pages_fetched:
            return "Website unavailable — generating from supplied requirements only."
        return (
            f"Website research: {len(self.pages_discovered)} pages discovered, "
            f"{len(self.pages_fetched)} fetched, {len(self.indexed_pages)} indexed."
        )


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1


def _host_key(host: str | None) -> str:
    return (host or "").lower().removeprefix("www.")


def _canonical_url(value: str, base: str | None = None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if base:
        raw = urljoin(base, raw)
    parsed = urlparse(raw)
    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunparse(("https", parsed.netloc.lower(), path, "", "", ""))


def _is_fetchable_link(url: str, site_host: str) -> bool:
    parsed = urlparse(url)
    if _host_key(parsed.hostname) != _host_key(site_host):
        return False
    return not parsed.path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".pdf", ".zip"))


def _link_priority(url: str) -> tuple[int, str]:
    path = urlparse(url).path.lower()
    preferred = (
        "about", "collection", "category", "shop", "product", "shipping",
        "delivery", "faq", "contact", "company", "policy", "terms",
    )
    rank = next((index for index, marker in enumerate(preferred) if marker in path), len(preferred))
    return rank, path


def _extract_text(raw: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", raw)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Strip storefront chrome that is navigation state rather than company
    # evidence, while retaining product, category, proposition, and fulfilment
    # text for retrieval.
    noise = re.compile(
        r"(?:accept\s+decline|your\s+cart\s+is\s+empty|continue\s+shopping|go\s+to\s+item\s+\d+|"
        r"refer\s+to\s+our\s+privacy\s+policy|close\s+menu|skip\s+to\s+content)",
        flags=re.I,
    )
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]
    cleaned = " ".join(sentence for sentence in sentences if not noise.search(sentence))
    return re.sub(r"\s+", " ", cleaned).strip()


def fetch_website_evidence(
    url: str,
    max_pages: int = MAX_WEBSITE_PAGES,
    *,
    return_report: bool = False,
):
    """Fetch a bounded same-domain crawl and optionally return diagnostics."""
    requested = (url or "").strip()
    report = WebsiteFetchReport(requested_url=requested)
    normalized = _canonical_url(requested)
    if not normalized:
        report.errors.append("Only an http/https website URL is supported.")
        result = ([], report.summary, report)
        return result if return_report else result[:2]
    site_host = urlparse(normalized).hostname or ""
    # ``example.com`` is a reserved documentation host, not a client source;
    # rejecting it keeps the demo from presenting placeholder copy as evidence
    # while allowing real HTTP sites to upgrade to HTTPS and follow redirects.
    if _host_key(site_host) == "example.com":
        report.errors.append("Reserved placeholder domain was not indexed.")
        result = ([], "Website unavailable — generating from supplied requirements only.", report)
        return result if return_report else result[:2]
    queue = [normalized]
    seen: set[str] = set()
    chunks: list[EvidenceChunk] = []
    while queue and len(report.pages_fetched) < max(1, min(max_pages, MAX_WEBSITE_PAGES)):
        page_url = queue.pop(0)
        if page_url in seen:
            continue
        seen.add(page_url)
        report.pages_discovered.append(page_url)
        try:
            request = Request(page_url, headers={"User-Agent": "AI-Document-Studio/1.0"})
            with urlopen(request, timeout=8) as response:  # nosec B310 - canonical https URL
                status = int(getattr(response, "status", 200) or 200)
                final_url = _canonical_url(response.geturl()) or page_url
                raw = response.read(MAX_WEBSITE_BYTES).decode("utf-8", errors="ignore")
            if report.resolved_url is None:
                report.resolved_url = final_url
                report.status_code = status
            if _host_key(urlparse(final_url).hostname) != _host_key(site_host):
                report.pages_rejected.append(f"{page_url} (redirected off-domain)")
                continue
            report.pages_fetched.append(final_url)
            text = _extract_text(raw)
            report.character_counts[final_url] = len(text)
            parser = _LinkParser()
            try:
                parser.feed(raw)
            except Exception:
                parser.links = []
            discovered = []
            for href in parser.links:
                candidate = _canonical_url(href, final_url)
                if candidate and _is_fetchable_link(candidate, site_host) and candidate not in seen:
                    discovered.append(candidate)
            for candidate in sorted(set(discovered), key=_link_priority):
                if candidate not in queue and len(report.pages_discovered) + len(queue) < MAX_DISCOVERED_LINKS:
                    queue.append(candidate)
            if len(text) < 80:
                report.pages_rejected.append(f"{final_url} (insufficient extracted text)")
                continue
            page_number = len(chunks) + 1
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"website-{_host_key(site_host)}-{page_number:02d}",
                    page=page_number,
                    section=urlparse(final_url).path or "/",
                    source=f"Website: {_host_key(site_host)}",
                    text=text[:MAX_WEBSITE_CHARS_PER_PAGE],
                )
            )
            report.indexed_pages.append(final_url)
        except Exception as error:  # website context is optional
            report.pages_rejected.append(page_url)
            report.errors.append(f"{page_url}: {error}")
    if not report.resolved_url:
        report.resolved_url = normalized
    if not chunks:
        report.errors.append("No usable website pages were indexed.")
    notice = None if chunks else report.summary
    result = (chunks, notice, report)
    return result if return_report else result[:2]

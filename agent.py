"""The explicit decide -> retrieve -> assess -> refine loop."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from collections.abc import Callable

from llm import Reasoner
from models import (
    AgentState,
    AgentTrace,
    CitationValidation,
    IterationTrace,
    RetrievedPreview,
    SearchRecord,
)


class Agent:
    SEARCH_REFORMULATION_RETRIES = 2

    def __init__(self, retriever, reasoner: Reasoner, max_iterations: int = 4, k: int = 6):
        self.retriever = retriever
        self.reasoner = reasoner
        self.max_iterations = max_iterations
        self.k = k

    @staticmethod
    def deduplicate(chunks):
        unique = {}
        for chunk in chunks:
            current = unique.get(chunk.chunk_id)
            if current is None or chunk.score > current.score:
                unique[chunk.chunk_id] = chunk
        return list(unique.values())

    @staticmethod
    def validate_citations(answer: str, chunks) -> tuple[str, CitationValidation]:
        retrieved_pairs = {(chunk.source, chunk.page) for chunk in chunks}
        retrieved_pages = sorted({chunk.page for chunk in chunks})
        citation_pattern = re.compile(r"\[([^\[\]\n]+?) p\.(\d+)\]")
        cited = [(match.group(1), int(match.group(2))) for match in citation_pattern.finditer(answer)]

        def keep_valid(match):
            source = match.group(1)
            page = int(match.group(2))
            return match.group(0) if (source, page) in retrieved_pairs else ""

        cleaned = citation_pattern.sub(keep_valid, answer)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        valid_pairs = sorted({pair for pair in cited if pair in retrieved_pairs}, key=lambda pair: (pair[0], pair[1]))
        valid_cited_pages = sorted({page for _, page in valid_pairs})
        uncited_claims = []
        substantive_lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith((
                "**Knowledge Base:**",
                "**Audience:**",
                "**Client Brief:**",
                "**Reference Precedent:**",
                "**Client Sources:**",
                "**Company Website:**",
            )):
                continue
            if stripped.startswith((
                "Scope.",
                "Analysis focus.",
                "Current evidence status.",
                "Evidence status.",
                "Working hypothesis.",
                "Data required.",
                "External research limitation.",
                "External research status.",
                "Recommended next step.",
                "The supplied material does not establish",
                "A useful review should",
                "The missing inputs are likely to include",
                "The next analytical step is to test",
                "Public competitor, market, and internationalisation research was not available",
                "Public competitor, market, and internationalisation sources were not available",
                "Any recommendation should be conditional on that validation",
            )):
                continue
            is_heading = stripped.endswith(":") or len(stripped.split()) <= 4 and not stripped.endswith((".", "]"))
            if is_heading:
                continue
            if Agent._is_status_statement(stripped):
                continue
            substantive_lines.append(stripped)
            if re.search(r"[A-Za-z]{4,}", stripped) and not citation_pattern.search(stripped):
                uncited_claims.append(stripped)
        validation = CitationValidation(
            # Requirements-only sections are deliberately uncited: with no
            # retrieved source there is no factual claim to ground.  The
            # document QC still records the evidence gap separately.
            valid=True if not retrieved_pairs else not uncited_claims and (bool(valid_cited_pages) or not substantive_lines),
            cited_pages=valid_cited_pages,
            retrieved_pages=retrieved_pages,
            cited_references=[f"{source} p.{page}" for source, page in valid_pairs],
            retrieved_references=[f"{source} p.{page}" for source, page in sorted(retrieved_pairs, key=lambda pair: (pair[0], pair[1]))],
            uncited_claims=[] if not retrieved_pairs else uncited_claims,
        )
        return cleaned, validation

    @staticmethod
    def _is_status_statement(line: str) -> bool:
        """Return true only for explicit answer-status prose, not corpus claims."""
        normalized = " ".join(line.lower().split())
        return any(
            normalized.startswith(prefix)
            for prefix in (
                "the evidence remained incomplete",
                "the available evidence is incomplete",
                "the retrieved evidence is incomplete",
                "insufficient evidence was retrieved",
                "no evidence was retrieved",
                "an evidence-grounded answer cannot be generated",
            )
        )

    @staticmethod
    def normalize_query(query: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", query.lower()))

    @classmethod
    def queries_are_near_duplicates(cls, first: str, second: str) -> bool:
        first_normalized = cls.normalize_query(first)
        second_normalized = cls.normalize_query(second)
        if first_normalized == second_normalized:
            return True
        first_terms = set(first_normalized.split())
        second_terms = set(second_normalized.split())
        if not first_terms or not second_terms:
            return False
        jaccard = len(first_terms & second_terms) / len(first_terms | second_terms)
        sequence_similarity = SequenceMatcher(None, first_normalized, second_normalized).ratio()
        return jaccard >= 0.8 or sequence_similarity >= 0.9

    def _next_unique_search(self, state: AgentState):
        prior_queries = [search.query for search in state.searches] + state.rejected_search_queries
        for _ in range(self.SEARCH_REFORMULATION_RETRIES + 1):
            decision = self.reasoner.decide_search(state)
            if not any(self.queries_are_near_duplicates(decision.search_query, prior) for prior in prior_queries):
                state.search_strategy_feedback = None
                return decision
            state.rejected_search_queries.append(decision.search_query)
            prior_queries.append(decision.search_query)
            missing = state.assessments[-1].missing_information if state.assessments else []
            state.search_strategy_feedback = (
                "The proposed query duplicated a previous strategy. Reformulate it using the "
                "original question, previous search queries, retrieved evidence, and this still-"
                f"missing information: {missing or ['not yet established']}."
            )
        return None

    @staticmethod
    def _coverage_signature(assessment):
        return (
            tuple(sorted(item.lower().strip() for item in assessment.supported_information)),
            tuple(sorted(item.lower().strip() for item in assessment.missing_information)),
        )

    @classmethod
    def _no_progress(cls, previous: IterationTrace, current: IterationTrace) -> bool:
        previous_ids = {chunk.chunk_id for chunk in previous.retrieved}
        current_ids = {chunk.chunk_id for chunk in current.retrieved}
        if not previous_ids or not current_ids:
            return False
        overlap = len(previous_ids & current_ids) / len(previous_ids | current_ids)
        return overlap >= 0.8 and cls._coverage_signature(previous.assessment) == cls._coverage_signature(current.assessment)

    def _search_loop(self, question: str, on_iteration: Callable[[IterationTrace], None] | None = None):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        started = datetime.now(timezone.utc)
        state = AgentState(original_question=question)
        trace = AgentTrace(question=question, started_at=started.isoformat())
        for iteration in range(1, self.max_iterations + 1):
            state.iteration = iteration
            decision = self._next_unique_search(state)
            if decision is None:
                state.stop_reason = "no_new_search_strategy"
                break
            retrieved = self.retriever.search(decision.search_query, self.k)
            state.searches.append(SearchRecord(query=decision.search_query, reason=decision.reason))
            state.gathered_evidence = self.deduplicate(state.gathered_evidence + retrieved)
            assessment = self.reasoner.assess(state)
            state.assessments.append(assessment)
            item = IterationTrace(
                iteration=iteration,
                search_decision=decision,
                retrieved=[
                    RetrievedPreview(
                        chunk_id=chunk.chunk_id,
                        page=chunk.page,
                        score=round(chunk.score, 4),
                        section=chunk.section,
                        text_preview=chunk.text[:240].replace("\n", " "),
                    )
                    for chunk in retrieved
                ],
                assessment=assessment,
            )
            trace.iterations.append(item)
            if on_iteration:
                on_iteration(item)
            if assessment.sufficient:
                state.stop_reason = "sufficient_evidence"
                break
            if len(trace.iterations) > 1 and self._no_progress(trace.iterations[-2], item):
                state.stop_reason = "no_new_evidence"
                break
        else:
            state.stop_reason = "max_iterations" if state.gathered_evidence else "no_evidence"

        if state.stop_reason is None:
            state.stop_reason = "no_evidence" if not state.gathered_evidence else "max_iterations"
        return started, state, trace

    @staticmethod
    def _finish_trace(started: datetime, state: AgentState, trace: AgentTrace) -> AgentTrace:
        trace.stop_reason = state.stop_reason
        trace.final_answer = state.final_answer
        trace.total_iterations = len(trace.iterations)
        trace.total_unique_evidence_chunks = len(state.gathered_evidence)
        completed = datetime.now(timezone.utc)
        trace.completed_at = completed.isoformat()
        trace.duration_ms = int((completed - started).total_seconds() * 1000)
        trace.citations = sorted(
            set(re.findall(r"\[[^\[\]\n]+? p\.\d+\]", state.final_answer or "")),
            key=lambda citation: (re.sub(r" p\.\d+\]$", "", citation), int(re.search(r"p\.(\d+)", citation).group(1))),
        )
        return trace

    def research(self, question: str, on_iteration: Callable[[IterationTrace], None] | None = None):
        started, state, trace = self._search_loop(question, on_iteration)
        state.final_answer = ""
        trace.citation_validation = CitationValidation(
            valid=True,
            retrieved_pages=sorted({chunk.page for chunk in state.gathered_evidence}),
            retrieved_references=[
                f"{source} p.{page}"
                for source, page in sorted({(chunk.source, chunk.page) for chunk in state.gathered_evidence})
            ],
        )
        self._finish_trace(started, state, trace)
        return state, trace

    def run(self, question: str, on_iteration: Callable[[IterationTrace], None] | None = None):
        started, state, trace = self._search_loop(question, on_iteration)
        if state.gathered_evidence:
            state.final_answer = self.reasoner.answer(state)
            state.final_answer, trace.citation_validation = self.validate_citations(
                state.final_answer, state.gathered_evidence
            )
        else:
            state.final_answer = "No evidence was retrieved, so an evidence-grounded answer cannot be generated."
        self._finish_trace(started, state, trace)
        return state, trace

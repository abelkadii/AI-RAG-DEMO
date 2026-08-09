"""The explicit decide -> retrieve -> assess -> refine loop."""

from __future__ import annotations

import re
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
        retrieved_pages = sorted({chunk.page for chunk in chunks})
        retrieved_page_set = set(retrieved_pages)
        cited_pages = [int(match.group(1)) for match in re.finditer(r"\[AWS-WAF p\.(\d+)\]", answer)]

        def keep_valid(match):
            page = int(match.group(1))
            return match.group(0) if page in retrieved_page_set else ""

        cleaned = re.sub(r"\[AWS-WAF p\.(\d+)\]", keep_valid, answer)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        valid_cited_pages = sorted({page for page in cited_pages if page in retrieved_page_set})
        uncited_claims = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            is_heading = len(stripped.split()) <= 4 and not stripped.endswith((".", "]"))
            if is_heading:
                continue
            if re.search(r"[A-Za-z]{4,}", stripped) and not re.search(r"\[AWS-WAF p\.\d+\]", stripped):
                uncited_claims.append(stripped)
        validation = CitationValidation(
            valid=bool(valid_cited_pages) and not uncited_claims,
            cited_pages=valid_cited_pages,
            retrieved_pages=retrieved_pages,
            uncited_claims=uncited_claims,
        )
        return cleaned, validation

    def run(self, question: str, on_iteration: Callable[[IterationTrace], None] | None = None):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        started = datetime.now(timezone.utc)
        state = AgentState(original_question=question)
        trace = AgentTrace(question=question, started_at=started.isoformat())
        for iteration in range(1, self.max_iterations + 1):
            state.iteration = iteration
            decision = self.reasoner.decide_search(state)
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
        else:
            state.stop_reason = "max_iterations" if state.gathered_evidence else "no_evidence"

        if state.gathered_evidence:
            state.final_answer = self.reasoner.answer(state)
            state.final_answer, trace.citation_validation = self.validate_citations(
                state.final_answer, state.gathered_evidence
            )
        else:
            state.final_answer = "No evidence was retrieved, so an evidence-grounded answer cannot be generated."
        trace.stop_reason = state.stop_reason
        trace.final_answer = state.final_answer
        trace.total_iterations = len(trace.iterations)
        trace.total_unique_evidence_chunks = len(state.gathered_evidence)
        completed = datetime.now(timezone.utc)
        trace.completed_at = completed.isoformat()
        trace.duration_ms = int((completed - started).total_seconds() * 1000)
        trace.citations = sorted(
            set(re.findall(r"\[AWS-WAF p\.\d+\]", state.final_answer)),
            key=lambda citation: int(re.search(r"\d+", citation).group()),
        )
        return state, trace

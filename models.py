"""Shared, serializable models for retrieval and the agent trajectory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class EvidenceChunk(BaseModel):
    chunk_id: str
    page: int
    text: str
    score: float = 0.0
    section: str | None = None
    source: str = "AWS-WAF"


class SearchDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    search_query: str = Field(
        validation_alias=AliasChoices("search_query", "query"), serialization_alias="query"
    )
    reason: str


class EvidenceAssessment(BaseModel):
    sufficient: bool
    reason: str
    missing_information: list[str] = Field(default_factory=list)
    suggested_next_search: str | None = None
    supported_information: list[str] = Field(default_factory=list)
    partially_supported_information: list[str] = Field(default_factory=list)
    unsupported_information: list[str] = Field(default_factory=list)

    @field_validator(
        "missing_information",
        "supported_information",
        "partially_supported_information",
        "unsupported_information",
        mode="before",
    )
    @classmethod
    def normalize_information_items(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("label") or item.get("claim") or item.get("description")
                identifier = item.get("chunk_id") or item.get("id") or item.get("page")
                if text and identifier:
                    normalized.append(f"{text} ({identifier})")
                elif text:
                    normalized.append(str(text))
                elif identifier:
                    normalized.append(str(identifier))
                else:
                    normalized.append(str(item))
            else:
                normalized.append(str(item))
        return normalized


class CitationValidation(BaseModel):
    valid: bool = False
    cited_pages: list[int] = Field(default_factory=list)
    retrieved_pages: list[int] = Field(default_factory=list)
    cited_references: list[str] = Field(default_factory=list)
    retrieved_references: list[str] = Field(default_factory=list)
    uncited_claims: list[str] = Field(default_factory=list)


class RetrievedPreview(BaseModel):
    chunk_id: str
    page: int
    score: float
    section: str | None = None
    text_preview: str


class IterationTrace(BaseModel):
    iteration: int
    search_decision: SearchDecision
    retrieved: list[RetrievedPreview]
    assessment: EvidenceAssessment


class SearchRecord(BaseModel):
    query: str
    reason: str


class AgentState(BaseModel):
    original_question: str
    iteration: int = 0
    searches: list[SearchRecord] = Field(default_factory=list)
    rejected_search_queries: list[str] = Field(default_factory=list)
    search_strategy_feedback: str | None = None
    gathered_evidence: list[EvidenceChunk] = Field(default_factory=list)
    assessments: list[EvidenceAssessment] = Field(default_factory=list)
    stop_reason: Literal[
        "sufficient_evidence",
        "max_iterations",
        "no_evidence",
        "no_new_search_strategy",
        "no_new_evidence",
    ] | None = None
    final_answer: str | None = None


class AgentTrace(BaseModel):
    question: str
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    duration_ms: int | None = None
    iterations: list[IterationTrace] = Field(default_factory=list)
    total_iterations: int = 0
    total_unique_evidence_chunks: int = 0
    stop_reason: str | None = None
    final_answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    citation_validation: CitationValidation = Field(default_factory=CitationValidation)

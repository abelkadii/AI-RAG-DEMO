"""Serializable models for the document production workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from models import AgentTrace, CitationValidation, EvidenceChunk


class DocumentSpec(BaseModel):
    title: str = "AWS Well-Architected Architecture Assessment & Remediation Plan"
    client_brief: str
    audience: str = "Executive and technical stakeholders"
    # ``Demo`` is retained as a compatibility value for older traces and API
    # callers.  The public UI exposes the four production depth profiles.
    target_depth: Literal["Brief", "Standard", "Detailed", "Comprehensive", "Demo"] = "Standard"
    target_word_count: int | None = Field(default=None, ge=100, le=10000)
    knowledge_base: str = "AWS Well-Architected Framework"
    source_kind: Literal["aws_sample", "uploaded"] = "aws_sample"
    deliverable_type: Literal[
        "Auto",
        "Summary / Brief",
        "Consulting Assessment",
        "Research Report",
        "Curriculum / Teaching Material",
        "Custom",
    ] = "Auto"


class DocumentSectionPlan(BaseModel):
    section_id: str
    title: str
    objective: str
    research_question: str = ""
    research_questions: list[str] = Field(default_factory=list)
    approximate_word_budget: int | None = Field(default=None, ge=25)
    requirements: list[str] = Field(default_factory=list)

    @property
    def questions(self) -> list[str]:
        """Return the normalized research-question list for old/new traces."""
        return self.research_questions or ([self.research_question] if self.research_question else [])


class DocumentPlan(BaseModel):
    title: str
    sections: list[DocumentSectionPlan]
    deliverable_type: str = "Consulting Assessment"
    target_depth: str = "Standard"
    target_word_count: int | None = None
    source_survey: list[EvidenceChunk] = Field(default_factory=list)
    source_topics: list[str] = Field(default_factory=list)


class SectionQC(BaseModel):
    passed: bool
    requirements_covered: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_valid: bool = False
    issues: list[str] = Field(default_factory=list)
    revision_instructions: str | None = None


class GeneratedSection(BaseModel):
    section_id: str
    title: str
    objective: str
    content_markdown: str
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    research_trace: AgentTrace
    qc: SectionQC
    revised: bool = False
    revision_count: int = 0


class DocumentQC(BaseModel):
    passed: bool
    sections_present: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    citation_valid: bool = False
    major_issues: list[str] = Field(default_factory=list)
    recommendations_align_with_findings: bool = True
    summary: str = ""
    target_word_count: int | None = None
    final_word_count: int = 0
    unique_pages_researched: int = 0
    unique_pages_cited: int = 0
    cross_section_duplication: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    unsupported_recommendations: list[str] = Field(default_factory=list)


class DocumentTrace(BaseModel):
    spec: DocumentSpec
    plan: DocumentPlan
    sections: list[GeneratedSection]
    final_qc: DocumentQC
    final_markdown: str
    citation_validation: CitationValidation
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    duration_ms: int | None = None
    total_research_iterations: int = 0
    total_unique_evidence_pages: int = 0
    total_retrieved_evidence_chunks: int = 0
    total_unique_cited_pages: int = 0
    final_word_count: int = 0
    target_word_count: int | None = None

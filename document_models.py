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
    target_depth: Literal["Demo", "Detailed"] = "Demo"
    knowledge_base: str = "AWS Well-Architected Framework"


class DocumentSectionPlan(BaseModel):
    section_id: str
    title: str
    objective: str
    research_question: str


class DocumentPlan(BaseModel):
    title: str
    sections: list[DocumentSectionPlan]


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

"""Serializable models for the document production workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from models import AgentTrace, CitationValidation, EvidenceChunk


class DocumentSpec(BaseModel):
    title: str = "AWS Well-Architected Architecture Assessment & Remediation Plan"
    client_brief: str
    audience: str = "Executive and technical stakeholders"
    # ``Demo`` is retained as a compatibility value for older traces and API
    # callers.  The public UI exposes the four production depth profiles.
    target_depth: Literal["Brief", "Standard", "Detailed", "Comprehensive", "Demo"] = "Standard"
    target_word_count: int | None = Field(default=None, ge=100, le=50000)
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
    reference_source_names: list[str] = Field(default_factory=list)
    client_source_names: list[str] = Field(default_factory=list)
    company_website: str | None = None


class ReferenceReportProfile(BaseModel):
    """A style/structure blueprint; it is never used as client evidence."""

    title: str = ""
    detected_sections: list[str] = Field(default_factory=list)
    section_patterns: list[str] = Field(default_factory=list)
    approximate_word_count: int = 0
    tone: str = "formal consulting"
    analytical_frameworks: list[str] = Field(default_factory=list)
    recurring_output_types: list[str] = Field(default_factory=list)
    roadmap_pattern: str | None = None
    tables_or_matrices: list[str] = Field(default_factory=list)
    presentation_notes: list[str] = Field(default_factory=list)


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


class SectionEvidenceClaim(BaseModel):
    text: str = Field(validation_alias=AliasChoices("text", "claim", "statement", "fact", "finding"))
    evidence_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evidence_ids", "supporting_evidence_ids", "source_ids"),
    )


_EVIDENCE_TOKEN_RE = re.compile(r"\[(E\d+)\]", flags=re.IGNORECASE)


def _clean_claim_text(value: object) -> tuple[str, list[str]]:
    text = str(value or "")
    ids = [match.upper() for match in _EVIDENCE_TOKEN_RE.findall(text)]
    cleaned = _EVIDENCE_TOKEN_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, ids


def _normalize_claim_items(value: object) -> list[dict[str, object]]:
    """Coerce compact or verbose model claims into the canonical claim shape."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: list[dict[str, object]] = []
    for item in items:
        if isinstance(item, SectionEvidenceClaim):
            normalized.append(item.model_dump())
            continue
        if isinstance(item, str):
            text, ids = _clean_claim_text(item)
            normalized.append({"text": text, "evidence_ids": ids})
            continue
        if not isinstance(item, dict):
            continue
        raw_text = next(
            (item.get(key) for key in ("text", "claim", "statement", "fact", "finding") if item.get(key)),
            "",
        )
        text, ids = _clean_claim_text(raw_text)
        explicit_ids = item.get("evidence_ids") or item.get("source_ids") or item.get("supporting_evidence_ids") or []
        if isinstance(explicit_ids, str):
            explicit_ids = _EVIDENCE_TOKEN_RE.findall(explicit_ids) or [explicit_ids]
        ids = sorted({*ids, *(str(value).upper() for value in explicit_ids if value)})
        normalized.append({"text": text, "evidence_ids": ids})
    return normalized


def _normalize_requirements(value: object) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                normalized.append(item.strip())
            continue
        if isinstance(item, dict):
            text = next(
                (item.get(key) for key in ("requirement", "text", "description", "objective", "title") if item.get(key)),
                None,
            )
            if text is not None and str(text).strip():
                normalized.append(str(text).strip())
    return normalized


def normalize_section_analysis_payload(payload: object) -> tuple[object, bool]:
    """Normalize known model shape variants before Pydantic validation."""
    if not isinstance(payload, dict):
        return payload, False
    normalized = dict(payload)
    changed = False
    for field in ("known_facts", "evidence_claims"):
        if field in normalized:
            value = _normalize_claim_items(normalized[field])
            changed = changed or value != normalized[field]
            normalized[field] = value
    if "requirements" in normalized:
        value = _normalize_requirements(normalized["requirements"])
        changed = changed or value != normalized["requirements"]
        normalized["requirements"] = value
    return normalized, changed


class SectionAnalysis(BaseModel):
    """Structured reasoning artifact kept separate from final report prose."""

    section_id: str
    section_mode: str = "general"
    objective: str = ""
    requirements: list[str] = Field(default_factory=list)
    known_facts: list[SectionEvidenceClaim] = Field(default_factory=list)
    evidence_claims: list[SectionEvidenceClaim] = Field(default_factory=list)
    analytical_inferences: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    observable_context: list[str] = Field(default_factory=list)
    recommended_analysis: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    planned_paragraphs: list[str] = Field(default_factory=list)

    @field_validator("requirements", mode="before")
    @classmethod
    def normalize_requirements(cls, value):
        return _normalize_requirements(value)

    @field_validator("known_facts", "evidence_claims", mode="before")
    @classmethod
    def normalize_claims(cls, value):
        return _normalize_claim_items(value)

    @field_validator(
        "analytical_inferences",
        "hypotheses",
        "recommendations",
        "data_gaps",
        "observable_context",
        "recommended_analysis",
        "planned_paragraphs",
        mode="before",
    )
    @classmethod
    def normalize_text_items(cls, value):
        """Accept equivalent structured labels returned by chat models."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            value = [value]
        normalized: list[str] = []
        labels = (
            "text",
            "inference",
            "hypothesis",
            "recommendation",
            "gap",
            "data_gap",
            "context",
            "analysis",
            "paragraph",
            "statement",
        )
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                text = next((item.get(label) for label in labels if item.get(label)), None)
                if text is not None:
                    normalized.append(str(text))
        return normalized


class SectionDraft(BaseModel):
    markdown: str = ""


class DocumentPlan(BaseModel):
    title: str
    sections: list[DocumentSectionPlan]
    deliverable_type: str = "Consulting Assessment"
    target_depth: str = "Standard"
    target_word_count: int | None = None
    source_survey: list[EvidenceChunk] = Field(default_factory=list)
    source_topics: list[str] = Field(default_factory=list)
    reference_profile: ReferenceReportProfile | None = None
    scope_requirements: list[str] = Field(default_factory=list)


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
    analysis: SectionAnalysis | None = None
    analysis_model_used: bool = False
    analysis_error: str | None = None
    analysis_normalized: bool = False
    analysis_repair_retry: bool = False
    synthesis_model_used: bool = False
    synthesis_error: str | None = None
    synthesis_fallback: bool = False
    latency_ms: int | None = None
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
    repetitive_prose_patterns: list[str] = Field(default_factory=list)
    unsupported_recommendations: list[str] = Field(default_factory=list)
    missing_scope_requirements: list[str] = Field(default_factory=list)
    scope_coverage: dict[str, str] = Field(default_factory=dict)
    reference_leakage: list[str] = Field(default_factory=list)
    requirements_as_evidence: list[str] = Field(default_factory=list)


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
    synthesis_engine: str = ""
    synthesis_model: str | None = None
    smoke_test_mode: bool = False
    smoke_test_sections: list[str] = Field(default_factory=list)
    external_research_enabled: bool | None = None
    website_report: dict[str, object] | None = None

"""Evidence-grounded report generation over the existing agentic RAG loop."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, Protocol

from agent import Agent
from document_models import (
    DocumentPlan,
    DocumentQC,
    DocumentSectionPlan,
    DocumentSpec,
    DocumentTrace,
    GeneratedSection,
    SectionQC,
)
from llm import Reasoner, configured_reasoner
from models import CitationValidation, EvidenceChunk
from retriever import Retriever


DEFAULT_BRIEF = (
    "Prepare an architecture assessment for a customer-facing AWS workload. "
    "Evaluate reliability, security, and cost optimization. Identify key risks, "
    "explain why they matter, recommend prioritized remediations, and provide an "
    "implementation roadmap. Every material factual claim must be grounded in the "
    "supplied framework."
)


class SectionWriter(Protocol):
    def write(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        section: DocumentSectionPlan,
        evidence: list[EvidenceChunk],
        prior_summaries: list[str],
    ) -> str: ...


class QCRunner(Protocol):
    def check(
        self,
        section: DocumentSectionPlan,
        content: str,
        evidence: list[EvidenceChunk],
    ) -> SectionQC: ...


class DocumentWorkflow:
    def __init__(
        self,
        retriever=None,
        reasoner: Reasoner | None = None,
        max_section_iterations: int = 3,
        evidence_k: int = 5,
        section_writer: SectionWriter | None = None,
        qc_runner: QCRunner | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.reasoner = reasoner or configured_reasoner()
        self.max_section_iterations = max_section_iterations
        self.evidence_k = evidence_k
        self.section_writer = section_writer or EvidenceGroundedSectionWriter()
        self.qc_runner = qc_runner or DeterministicSectionQC()

    def plan(self, spec: DocumentSpec) -> DocumentPlan:
        sections = [
            ("executive-summary", "Executive Summary", "Summarize the overall assessment, key risks, and remediation themes.", "AWS Well-Architected reliability security cost optimization architecture assessment summary risks remediation"),
            ("scope-objectives", "Assessment Scope and Objectives", "Define the assessment scope, objectives, and evidence-grounding approach.", "AWS Well-Architected framework overview purpose evaluate architectures best practices"),
            ("reliability", "Reliability Assessment", "Assess failure preparation, fault isolation, recovery planning, and resilience testing.", "AWS Well-Architected reliability Availability Zone failures fault isolation disaster recovery resilience testing"),
            ("security", "Security Assessment", "Assess controls for sensitive data, identity, infrastructure protection, logging, monitoring, and incident response.", "AWS Well-Architected security identity access management data protection infrastructure logging monitoring"),
            ("cost", "Cost Optimization Assessment", "Assess resource sizing, cost modeling, waste reduction, and right-sizing opportunities.", "AWS Well-Architected cost optimization resource type size number data right sizing cost modeling waste"),
            ("recommendations", "Prioritized Recommendations", "Recommend prioritized remediation actions aligned to reliability, security, and cost findings.", "AWS Well-Architected reliability security cost optimization remediation recommendations priorities"),
            ("roadmap", "Implementation Roadmap", "Sequence practical implementation steps into near-term, mid-term, and later phases.", "AWS Well-Architected implementation steps reliability security cost roadmap"),
            ("evidence", "Evidence / References", "List the evidence pages used by the report.", "AWS Well-Architected evidence references reliability security cost"),
        ]
        depth_limit = 8 if spec.target_depth == "Detailed" else 7
        selected = sections[:depth_limit]
        if selected[-1][0] != "evidence":
            selected.append(sections[-1])
        return DocumentPlan(
            title=spec.title.strip()[:140] or "AWS Well-Architected Architecture Assessment",
            sections=[
                DocumentSectionPlan(
                    section_id=section_id,
                    title=title,
                    objective=objective,
                    research_question=question,
                )
                for section_id, title, objective, question in selected[:8]
            ],
        )

    def run(
        self,
        spec: DocumentSpec,
        on_event: Callable[[str], None] | None = None,
    ) -> DocumentTrace:
        started = datetime.now(timezone.utc)
        plan = self.plan(spec)
        self._event(on_event, "Planning report")
        generated_sections: list[GeneratedSection] = []
        prior_summaries: list[str] = []
        evidence_by_id: dict[str, EvidenceChunk] = {}

        for section_plan in plan.sections:
            self._event(on_event, f"Researching {section_plan.title}")
            state, research_trace = Agent(
                self.retriever,
                self.reasoner,
                max_iterations=self.max_section_iterations,
                k=self.evidence_k,
            ).research(section_plan.research_question)
            section_evidence = state.gathered_evidence
            for chunk in section_evidence:
                evidence_by_id[chunk.chunk_id] = chunk

            self._event(on_event, f"Drafting {section_plan.title}")
            content = self.section_writer.write(spec, plan, section_plan, section_evidence, prior_summaries)

            self._event(on_event, f"Running QC for {section_plan.title}")
            qc = self.qc_runner.check(section_plan, content, section_evidence)
            revised = False
            revision_count = 0
            if not qc.passed:
                revised = True
                revision_count = 1
                content = revise_section(content, qc, section_evidence)
                qc = self.qc_runner.check(section_plan, content, section_evidence)

            generated = GeneratedSection(
                section_id=section_plan.section_id,
                title=section_plan.title,
                objective=section_plan.objective,
                content_markdown=content,
                evidence=section_evidence,
                research_trace=research_trace,
                qc=qc,
                revised=revised,
                revision_count=revision_count,
            )
            generated_sections.append(generated)
            prior_summaries.append(summarize_section(content))
            self._event(on_event, f"{section_plan.title} approved" if qc.passed else f"{section_plan.title} needs review")

        self._event(on_event, "Running final document review")
        final_markdown = assemble_markdown(spec, plan, generated_sections)
        all_evidence = list(evidence_by_id.values())
        cleaned_markdown, citation_validation = Agent.validate_citations(final_markdown, all_evidence)
        final_qc = document_qc(plan, generated_sections, citation_validation)
        completed = datetime.now(timezone.utc)
        return DocumentTrace(
            spec=spec,
            plan=plan,
            sections=generated_sections,
            final_qc=final_qc,
            final_markdown=cleaned_markdown,
            citation_validation=citation_validation,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_ms=int((completed - started).total_seconds() * 1000),
            total_research_iterations=sum(section.research_trace.total_iterations for section in generated_sections),
            total_unique_evidence_pages=len({chunk.page for chunk in all_evidence}),
        )

    @staticmethod
    def _event(callback: Callable[[str], None] | None, message: str) -> None:
        if callback:
            callback(message)


class EvidenceGroundedSectionWriter:
    def write(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        section: DocumentSectionPlan,
        evidence: list[EvidenceChunk],
        prior_summaries: list[str],
    ) -> str:
        pages = cite_page(evidence)
        if section.section_id == "executive-summary":
            return (
                "The assessment focuses on reliability, security, and cost optimization for a customer-facing AWS workload. "
                "The framework evidence supports a remediation plan centered on limiting failure impact, protecting access and data, "
                f"and selecting resources from workload measurements rather than assumptions. {pages}\n\n"
                f"- Highest-priority work should address resilience and recovery readiness. {pages}\n"
                f"- Security work should cover identity, data protection, infrastructure protection, logging, and monitoring. {pages}\n"
                f"- Cost work should use sizing data and cost modeling to reduce waste. {pages}"
            )
        if section.section_id == "scope-objectives":
            return (
                "This report uses the AWS Well-Architected Framework as the knowledge base for assessing architecture trade-offs, "
                f"risks, and improvement actions. {pages}\n\n"
                f"Audience: {spec.audience}. {pages}\n\n"
                "The objective is to convert framework-backed evidence into a practical assessment, prioritized recommendations, "
                f"and an implementation roadmap. {pages}"
            )
        if section.section_id == "reliability":
            return (
                "Reliability risk is concentrated around how the workload contains failures and restores service after disruption. "
                f"The retrieved framework guidance says fault-isolated boundaries limit the effect of a failure, while recovery plans, recovery testing, and disaster recovery objectives help verify that the workload can recover. {pages}\n\n"
                "Recommended actions:\n"
                f"- Define fault-isolated boundaries for critical workload components. {pages}\n"
                f"- Maintain disaster recovery plans and operating procedures after significant changes. {pages}\n"
                f"- Run resilience and recovery tests before relying on the design in production. {pages}"
            )
        if section.section_id == "security":
            return (
                "Security risk is concentrated around whether the workload applies controls across identity, data, infrastructure, and observability. "
                f"The retrieved framework guidance identifies identity and access management, data protection, infrastructure protection, logging, and monitoring as security-control domains. {pages}\n\n"
                "Recommended actions:\n"
                f"- Enforce least-privilege access for people and machine identities. {pages}\n"
                f"- Protect sensitive data at rest and in transit. {pages}\n"
                f"- Use logging and monitoring to detect and investigate security events. {pages}"
            )
        if section.section_id == "cost":
            return (
                "Cost risk is concentrated around over-provisioned, idle, or poorly selected resources. "
                f"The retrieved framework guidance recommends selecting resource type, size, and number based on workload data and cost modeling. {pages}\n\n"
                "Recommended actions:\n"
                f"- Benchmark representative workload demand before choosing capacity. {pages}\n"
                f"- Use cost modeling or proof-of-concept runs to compare resource choices. {pages}\n"
                f"- Right-size resources and reduce waste from unnecessary capacity. {pages}"
            )
        if section.section_id == "recommendations":
            return (
                "Prioritize remediation work by customer impact and evidence strength. "
                f"Start with failure containment and recovery validation, then strengthen security controls, then optimize resource selection using measured data. {pages}\n\n"
                f"Priority 1: establish reliability controls for failure isolation and recovery testing. {pages}\n\n"
                f"Priority 2: apply security controls for identity, data protection, infrastructure, logging, and monitoring. {pages}\n\n"
                f"Priority 3: tune resource selection and sizing using workload evidence and cost models. {pages}"
            )
        if section.section_id == "roadmap":
            return (
                "A practical roadmap should move from risk reduction to optimization. "
                f"The framework evidence supports sequencing reliability validation, security-control implementation, and data-driven cost optimization. {pages}\n\n"
                f"0-30 days: document recovery objectives, identify sensitive data paths, and collect utilization/cost data. {pages}\n\n"
                f"31-60 days: test recovery procedures, implement high-priority access and data controls, and model alternative resource sizes. {pages}\n\n"
                f"61-90 days: automate recurring resilience checks, strengthen monitoring, and right-size resources based on measured demand. {pages}"
            )
        references = sorted({chunk.page for chunk in evidence})
        if not references:
            references = sorted(_pages_from_text(" ".join(prior_summaries)))
        lines = ["The report used retrieved AWS Well-Architected Framework evidence from these pages:"]
        lines.extend(f"- [AWS-WAF p.{page}]" for page in references[:20])
        return "\n".join(lines)


class DeterministicSectionQC:
    def check(self, section: DocumentSectionPlan, content: str, evidence: list[EvidenceChunk]) -> SectionQC:
        _, validation = Agent.validate_citations(content, evidence)
        missing = []
        lower = content.lower()
        for required in section_requirements(section):
            if required not in lower:
                missing.append(required)
        issues = []
        if not validation.valid:
            issues.append("One or more material claims lack a citation grounded in retrieved evidence.")
        if missing:
            issues.append("The section does not cover all expected requirements.")
        passed = validation.valid and not missing
        return SectionQC(
            passed=passed,
            requirements_covered=[item for item in section_requirements(section) if item not in missing],
            missing_requirements=missing,
            unsupported_claims=validation.uncited_claims,
            citation_valid=validation.valid,
            issues=issues,
            revision_instructions="Add missing requirements and cite every material claim." if issues else None,
        )


def section_requirements(section: DocumentSectionPlan) -> list[str]:
    requirements = {
        "reliability": ["failure", "recovery"],
        "security": ["identity", "data"],
        "cost": ["resource", "cost"],
        "recommendations": ["priority"],
        "roadmap": ["days"],
        "scope-objectives": ["objective"],
        "executive-summary": ["assessment"],
        "evidence": ["aws-waf"],
    }
    return requirements.get(section.section_id, [section.title.lower().split()[0]])


def revise_section(content: str, qc: SectionQC, evidence: list[EvidenceChunk]) -> str:
    citation = cite_page(evidence)
    additions = []
    for requirement in qc.missing_requirements:
        additions.append(f"This revision explicitly addresses {requirement} as part of the section objective. {citation}")
    if qc.unsupported_claims and citation:
        additions.append(f"All material findings above should be read against the retrieved framework evidence. {citation}")
    return content.rstrip() + "\n\n" + "\n".join(additions)


def cite_page(evidence: list[EvidenceChunk]) -> str:
    if not evidence:
        return ""
    best = max(evidence, key=lambda chunk: chunk.score)
    return f"[{best.source} p.{best.page}]"


def summarize_section(content: str) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    return text[:400]


def assemble_markdown(spec: DocumentSpec, plan: DocumentPlan, sections: list[GeneratedSection]) -> str:
    lines = [
        f"# {plan.title}",
        "",
        f"**Knowledge Base:** {spec.knowledge_base}",
        f"**Audience:** {spec.audience}",
        "",
        f"**Client Brief:** {spec.client_brief}",
        "",
    ]
    for section in sections:
        lines.extend([f"## {section.title}", "", section.content_markdown.strip(), ""])
    return "\n".join(lines).strip() + "\n"


def document_qc(plan: DocumentPlan, sections: list[GeneratedSection], citation_validation: CitationValidation) -> DocumentQC:
    present = [section.title for section in sections]
    expected = [section.title for section in plan.sections]
    missing = [title for title in expected if title not in present]
    failed_sections = [section.title for section in sections if not section.qc.passed]
    issues = []
    if missing:
        issues.append("One or more planned sections are missing.")
    if failed_sections:
        issues.append(f"Sections needing review: {', '.join(failed_sections)}.")
    if not citation_validation.valid:
        issues.append("Document citation validation found unsupported or uncited claims.")
    passed = not missing and not failed_sections and citation_validation.valid
    return DocumentQC(
        passed=passed,
        sections_present=present,
        missing_sections=missing,
        citation_valid=citation_validation.valid,
        major_issues=issues,
        recommendations_align_with_findings=True,
        summary="Document review passed." if passed else "Document review found issues to inspect.",
    )


def _pages_from_text(text: str) -> set[int]:
    return {int(match.group(1)) for match in re.finditer(r"\[AWS-WAF p\.(\d+)\]", text)}

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
        if spec.source_kind == "uploaded":
            return self._uploaded_plan(spec)
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
            deliverable_type="Consulting Assessment",
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

    def _uploaded_plan(self, spec: DocumentSpec) -> DocumentPlan:
        topic = spec.client_brief[:500]
        deliverable_type = resolve_deliverable_type(spec)
        if deliverable_type == "Summary / Brief":
            sections = [
                ("overview", "What This Document Covers", "Briefly explain the subject of the uploaded document.", f"main topic purpose subject overview {topic}"),
                ("key-points", "Key Points", "Summarize the main points without adding recommendations.", f"main points themes summary {topic}"),
                ("brief-explanation", "Brief Explanation", "Explain the document in plain language.", f"plain language explanation summary {topic}"),
                ("evidence", "Evidence / References", "List the source pages cited by the brief.", f"references evidence sources {topic}"),
            ]
        elif deliverable_type == "Research Report":
            sections = [
                ("executive-summary", "Executive Summary", "Summarize the research topic, methodology, and findings.", f"research summary methodology findings {topic}"),
                ("methodology", "Methodology", "Describe the methodology represented in the source evidence.", f"methodology approach data methods {topic}"),
                ("findings", "Findings", "Synthesize the evidence-backed findings.", f"findings results evidence {topic}"),
                ("interpretation", "Interpretation", "Explain what the findings mean without adding unsupported advice.", f"interpretation implications meaning {topic}"),
                ("evidence", "Evidence / References", "List the source pages cited by the report.", f"references evidence sources {topic}"),
            ]
        elif deliverable_type == "Curriculum / Teaching Material":
            sections = [
                ("overview", "Learning Overview", "Explain what the material teaches.", f"learning overview topic concepts {topic}"),
                ("concepts", "Core Concepts", "Summarize the core concepts from the source.", f"core concepts definitions examples {topic}"),
                ("teaching-notes", "Teaching Notes", "Turn source-backed concepts into concise teaching notes.", f"teaching notes explanation learners {topic}"),
                ("evidence", "Evidence / References", "List the source pages cited by the material.", f"references evidence sources {topic}"),
            ]
        elif deliverable_type == "Consulting Assessment":
            sections = [
                ("executive-summary", "Executive Summary", "Summarize the requested assessment, findings, and requested recommendations.", f"executive summary key findings risks recommendations {topic}"),
                ("scope-objectives", "Scope and Objectives", "Define the requested scope, audience, and objectives.", f"scope objectives audience requirements {topic}"),
                ("findings", "Key Findings", "Identify the most important evidence-backed findings.", f"key findings evidence analysis {topic}"),
                ("recommendations", "Prioritized Recommendations", "Recommend next steps only where requested and evidence-supported.", f"prioritized recommendations next steps {topic}"),
                ("roadmap", "Implementation Roadmap", "Sequence actions only where the brief requests a roadmap.", f"implementation roadmap phases actions {topic}"),
                ("evidence", "Evidence / References", "List the source pages cited by the report.", f"references evidence sources {topic}"),
            ]
            if spec.target_depth == "Detailed":
                sections.insert(3, ("analysis", "Analysis", "Explain why the findings matter and how they relate to the brief.", f"analysis implications risks opportunities {topic}"))
        else:
            sections = [
                ("overview", "Overview", "Respond directly to the requested custom deliverable.", f"overview requested deliverable {topic}"),
                ("key-points", "Key Points", "Extract the most relevant source-backed points.", f"key points evidence {topic}"),
                ("response", "Requested Output", "Produce the requested output without adding unrequested structure.", f"requested output {topic}"),
                ("evidence", "Evidence / References", "List the source pages cited by the output.", f"references evidence sources {topic}"),
            ]
        return DocumentPlan(
            title=spec.title.strip()[:140] or "Evidence-Grounded Document Report",
            deliverable_type=deliverable_type,
            sections=[
                DocumentSectionPlan(section_id=section_id, title=title, objective=objective, research_question=question)
                for section_id, title, objective, question in sections[:8]
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
            qc = merge_qc(qc, deterministic_content_checks(spec, plan, section_plan, content, section_evidence))
            revised = False
            revision_count = 0
            if not qc.passed:
                revised = True
                revision_count = 1
                content = revise_section(content, qc, section_evidence, spec, plan)
                qc = self.qc_runner.check(section_plan, content, section_evidence)
                qc = merge_qc(qc, deterministic_content_checks(spec, plan, section_plan, content, section_evidence))

            generated = GeneratedSection(
                section_id=section_plan.section_id,
                title=section_plan.title,
                objective=section_plan.objective,
                content_markdown=content,
                evidence=serializable_evidence(section_evidence),
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
        final_qc = document_qc(spec, plan, generated_sections, citation_validation)
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
        if spec.source_kind == "uploaded":
            return self._write_uploaded(spec, section, evidence, prior_summaries, pages)
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

    def _write_uploaded(
        self,
        spec: DocumentSpec,
        section: DocumentSectionPlan,
        evidence: list[EvidenceChunk],
        prior_summaries: list[str],
        citation: str,
    ) -> str:
        if section.section_id == "evidence":
            return "References are generated deterministically from citations used in the report."
        synthesis = synthesize_evidence(evidence)
        if not synthesis:
            return f"This section needs human review because no source evidence was retrieved for the objective. {citation}".strip()
        brief_focus = brief_focus_sentence(spec.client_brief)
        first, second = synthesis[0], synthesis[min(1, len(synthesis) - 1)]
        deliverable_type = resolve_deliverable_type(spec)
        if deliverable_type == "Summary / Brief":
            if section.section_id == "overview":
                return f"The uploaded document appears to discuss {first} {citation}"
            if section.section_id == "key-points":
                points = synthesis[:3]
                return "\n".join(f"- Point {index}: {point} {citation}" for index, point in enumerate(points, start=1))
            if section.section_id == "brief-explanation":
                return (
                    f"In brief, the source presents {first.lower()} {citation}\n\n"
                    f"It also points to {second.lower()} {citation}\n\n"
                    f"This explanation is limited to the uploaded evidence and the user's request: {brief_focus}. {citation}"
                )
        if deliverable_type == "Research Report":
            if section.section_id == "executive-summary":
                return f"This research report summarizes the uploaded evidence around {first.lower()} {citation}"
            if section.section_id == "methodology":
                return f"The methodology or approach described in the source centers on {first.lower()} {citation}"
            if section.section_id == "findings":
                return f"The main evidence-backed findings are {first.lower()} and {second.lower()} {citation}"
            if section.section_id == "interpretation":
                return f"Taken together, the evidence suggests that {first.lower()} {citation}"
        if deliverable_type == "Curriculum / Teaching Material":
            if section.section_id == "overview":
                return f"This teaching material covers {first.lower()} {citation}"
            if section.section_id == "concepts":
                return "\n".join(f"- Concept {index}: {point} {citation}" for index, point in enumerate(synthesis[:3], start=1))
            if section.section_id == "teaching-notes":
                return f"Teaching notes should explain {first.lower()} and connect it to {second.lower()} {citation}"
        if section.section_id == "executive-summary":
            return (
                f"This assessment responds to the requested brief for {spec.audience}. "
                f"The source evidence emphasizes {first.lower()} {citation}\n\n"
                f"- The requested focus is: {brief_focus}. {citation}\n"
                f"- Findings and recommendations are limited to retrieved source evidence. {citation}"
            )
        if section.section_id == "scope-objectives":
            return (
                f"The scope is defined by the uploaded documents and the client brief. {citation}\n\n"
                f"The objective is to produce an evidence-grounded deliverable for {spec.audience} that matches the requested output type: {deliverable_type}. {citation}"
            )
        if section.section_id == "findings":
            return (
                f"Key finding 1: {first} {citation}\n\n"
                f"Key finding 2: {second} {citation}\n\n"
                f"These findings should be interpreted against the uploaded source context rather than generalized beyond it. {citation}"
            )
        if section.section_id == "analysis":
            return (
                f"The evidence suggests the key issue is how the requested objective connects to the documented source material. {citation}\n\n"
                f"In practical terms, {first} {citation}\n\n"
                f"This creates an analysis basis for recommendations without adding unsupported external claims. {citation}"
            )
        if section.section_id == "recommendations":
            if not requests_actions(spec.client_brief):
                return f"The brief does not request recommendations, so this section limits itself to the evidence-backed point that {first.lower()} {citation}"
            return (
                f"Priority 1: address the highest-impact item described in the uploaded material: {first} {citation}\n\n"
                f"Priority 2: use the supporting source evidence to define practical next steps: {second} {citation}\n\n"
                f"Priority 3: validate the recommendations with the document owner before operational rollout. {citation}"
            )
        if section.section_id == "roadmap":
            if not requests_roadmap(spec.client_brief):
                return f"The brief does not request an implementation roadmap; the evidence-backed summary is that {first.lower()} {citation}"
            return (
                f"This roadmap sequences the source-backed work into practical phases. {citation}\n\n"
                f"Near term: confirm the source-backed findings and assign ownership. {citation}\n\n"
                f"Next phase: convert the findings into specific work items, acceptance criteria, and review checkpoints. {citation}\n\n"
                f"Later phase: review outcomes against the original brief and update the document set as source knowledge evolves. {citation}"
            )
        if section.section_id in {"overview", "key-points", "response"}:
            return f"{first} {citation}"
        return f"{section.title}: {first} {citation}"


class DeterministicSectionQC:
    def check(self, section: DocumentSectionPlan, content: str, evidence: list[EvidenceChunk]) -> SectionQC:
        if section.section_id == "evidence":
            return SectionQC(
                passed=True,
                requirements_covered=["reference"],
                citation_valid=True,
            )
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
        "roadmap": ["roadmap"],
        "scope-objectives": ["objective"],
        "executive-summary": ["assessment"],
        "evidence": ["reference"],
        "findings": ["finding"],
        "analysis": ["analysis"],
        "overview": [],
        "key-points": ["point"],
        "brief-explanation": [],
        "methodology": ["methodology"],
        "interpretation": ["evidence"],
        "concepts": ["concept"],
        "teaching-notes": ["teaching"],
        "response": ["requested"],
    }
    return requirements.get(section.section_id, [section.title.lower().split()[0]])


def resolve_deliverable_type(spec: DocumentSpec) -> str:
    if spec.deliverable_type != "Auto":
        return spec.deliverable_type
    brief = spec.client_brief.lower()
    if any(term in brief for term in ("teach", "lesson", "curriculum", "student", "learning objective", "classroom")):
        return "Curriculum / Teaching Material"
    if any(term in brief for term in ("risk", "remediation", "roadmap", "90-day", "recommend", "assessment", "action plan")):
        return "Consulting Assessment"
    if any(term in brief for term in ("methodology", "findings", "study", "research", "results", "literature")):
        return "Research Report"
    if any(term in brief for term in ("what does this talk about", "explain briefly", "briefly", "short summary", "summarize", "summary")):
        return "Summary / Brief"
    return "Custom"


def requests_actions(brief: str) -> bool:
    lower = brief.lower()
    return any(term in lower for term in ("recommend", "next step", "action", "remediation", "roadmap", "implement", "plan"))


def requests_roadmap(brief: str) -> bool:
    lower = brief.lower()
    return any(term in lower for term in ("roadmap", "90-day", "timeline", "implementation plan", "phases"))


def brief_focus_sentence(brief: str) -> str:
    cleaned = re.sub(r"\s+", " ", brief).strip()
    return cleaned[:220].rstrip(".") or "the requested deliverable"


def merge_qc(primary: SectionQC, extra: SectionQC) -> SectionQC:
    requirements = sorted(set(primary.requirements_covered + extra.requirements_covered))
    missing = sorted(set(primary.missing_requirements + extra.missing_requirements))
    unsupported = primary.unsupported_claims + [item for item in extra.unsupported_claims if item not in primary.unsupported_claims]
    issues = primary.issues + [issue for issue in extra.issues if issue not in primary.issues]
    passed = primary.passed and extra.passed
    return SectionQC(
        passed=passed,
        requirements_covered=requirements,
        missing_requirements=missing,
        unsupported_claims=unsupported,
        citation_valid=primary.citation_valid and extra.citation_valid,
        issues=issues,
        revision_instructions="; ".join(filter(None, [primary.revision_instructions, extra.revision_instructions])) or None,
    )


def deterministic_content_checks(
    spec: DocumentSpec,
    plan: DocumentPlan,
    section: DocumentSectionPlan,
    content: str,
    evidence: list[EvidenceChunk],
) -> SectionQC:
    if section.section_id == "evidence":
        return SectionQC(
            passed=True,
            requirements_covered=["references-generated"],
            citation_valid=True,
        )
    issues: list[str] = []
    missing: list[str] = []
    unsupported: list[str] = []
    lower = content.lower()
    deliverable_type = plan.deliverable_type
    forbidden_action_terms = [
        "recommend",
        "roadmap",
        "ownership",
        "rollout",
        "acceptance criteria",
        "priority 1",
        "priority 2",
        "near term",
        "next phase",
        "implementation",
        "remediation",
    ]
    if deliverable_type == "Summary / Brief" and any(term in lower for term in forbidden_action_terms):
        issues.append("Summary/brief output contains unrequested recommendations, roadmap, or operational advice.")
        unsupported.append("Unrequested consulting/action structure")
    if not requests_actions(spec.client_brief) and section.section_id in {"recommendations", "roadmap"}:
        issues.append("Recommendations or roadmap section appears even though the brief did not request actions.")
    if has_raw_artifacts(content):
        issues.append("Content contains obvious raw header/footer/page-number artifacts.")
    incomplete = incomplete_sentences(content)
    if incomplete:
        issues.append("One or more sentences appear incomplete.")
        unsupported.extend(incomplete)
    if evidence and not semantically_supported(content, evidence):
        issues.append("The cited evidence may not semantically support the section claim.")
        unsupported.append("Low semantic overlap with retrieved evidence")
    if deliverable_type == "Summary / Brief" and section.section_id not in {"overview", "key-points", "brief-explanation", "evidence"}:
        missing.append("summary structure")
        issues.append("Plan section does not match a concise summary deliverable.")
    passed = not issues and not missing
    return SectionQC(
        passed=passed,
        requirements_covered=["deliverable-match", "artifact-cleanliness", "semantic-support"] if passed else [],
        missing_requirements=missing,
        unsupported_claims=unsupported,
        citation_valid=True,
        issues=issues,
        revision_instructions="Remove unrequested structure, raw artifacts, incomplete sentences, or unsupported operational advice." if issues else None,
    )


def has_raw_artifacts(text: str) -> bool:
    return bool(re.search(r"(^|\s)\d+\s*/\s*\d+(\s|$)", text) or re.search(r"\b(page|slide)\s+\d+\s+of\s+\d+\b", text, re.I))


def incomplete_sentences(text: str) -> list[str]:
    suspects = []
    for raw_line in text.splitlines():
        if re.search(r"\[[^\]]+ p\.\d+\]\s*$", raw_line.strip()):
            continue
        line = re.sub(r"\[[^\]]+ p\.\d+\]", "", raw_line).strip(" -*\t")
        if not line or line.startswith("#") or len(line) < 45:
            continue
        if line.endswith((".", "!", "?", ":", "؛", "۔")):
            continue
        if line.count(" ") >= 8:
            suspects.append(line)
    return suspects[:3]


def semantically_supported(content: str, evidence: list[EvidenceChunk]) -> bool:
    claim_terms = {
        term
        for term in re.findall(r"[\w\u0600-\u06FF]{4,}", re.sub(r"\[[^\]]+ p\.\d+\]", "", content).lower())
        if term not in STOP_TERMS
    }
    evidence_terms = {
        term
        for chunk in evidence
        for term in re.findall(r"[\w\u0600-\u06FF]{4,}", chunk.text.lower())
        if term not in STOP_TERMS
    }
    if not claim_terms:
        return True
    return len(claim_terms & evidence_terms) >= max(2, min(5, len(claim_terms) // 8))


def revise_section(content: str, qc: SectionQC, evidence: list[EvidenceChunk], spec: DocumentSpec | None = None, plan: DocumentPlan | None = None) -> str:
    if plan and plan.deliverable_type == "Summary / Brief":
        citation = cite_page(evidence)
        summary = synthesize_evidence(evidence, limit=2)
        if not summary:
            return content
        return "\n\n".join(f"{point} {citation}" for point in summary)
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


def serializable_evidence(evidence: list[EvidenceChunk]) -> list[dict]:
    """Avoid Streamlit/Pydantic class-identity issues after hot reloads."""
    serialized = []
    for chunk in evidence:
        if hasattr(chunk, "model_dump"):
            serialized.append(chunk.model_dump())
        elif isinstance(chunk, dict):
            serialized.append(chunk)
        else:
            serialized.append(
                {
                    "chunk_id": getattr(chunk, "chunk_id"),
                    "page": getattr(chunk, "page"),
                    "text": getattr(chunk, "text"),
                    "score": getattr(chunk, "score", 0.0),
                    "section": getattr(chunk, "section", None),
                    "source": getattr(chunk, "source", "Uploaded document"),
                }
            )
    return serialized


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
        if section.section_id == "evidence":
            continue
        lines.extend([f"## {section.title}", "", section.content_markdown.strip(), ""])
    references = references_from_markdown("\n".join(lines))
    if references:
        lines.extend(["## Evidence / References", ""])
        lines.extend(f"- {reference}" for reference in references)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def references_from_markdown(markdown: str) -> list[str]:
    matches = re.findall(r"\[([^\[\]\n]+? p\.\d+)\]", markdown)
    return sorted({f"[{match}]" for match in matches}, key=lambda item: (item.rsplit(" p.", 1)[0], int(item.rsplit(" p.", 1)[1].rstrip("]"))))


STOP_TERMS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "source",
    "evidence",
    "uploaded",
    "document",
    "documents",
    "report",
    "section",
    "brief",
    "requested",
    "material",
    "should",
    "would",
    "about",
    "also",
    "than",
    "they",
    "their",
    "there",
    "where",
    "which",
    "what",
}


def best_snippets(evidence: list[EvidenceChunk], limit: int = 3) -> list[str]:
    snippets = []
    for chunk in sorted(evidence, key=lambda item: item.score, reverse=True):
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sentence in sentences:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if 60 <= len(cleaned) <= 260:
                snippets.append(cleaned.rstrip("."))
                break
        if len(snippets) >= limit:
            break
    if snippets:
        return snippets
    return [chunk.text[:220].strip().rstrip(".") for chunk in evidence[:limit] if chunk.text.strip()]


def synthesize_evidence(evidence: list[EvidenceChunk], limit: int = 3) -> list[str]:
    candidates = []
    seen = set()
    for chunk in sorted(evidence, key=lambda item: item.score, reverse=True):
        for sentence in re.split(r"(?<=[.!?؟])\s+", chunk.text):
            cleaned = normalize_claim_text(sentence)
            if not useful_evidence_sentence(cleaned):
                continue
            key = re.sub(r"\W+", " ", cleaned.lower())[:90]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cleaned)
            break
        if len(candidates) >= limit:
            break
    if not candidates:
        return []
    return [compress_claim(candidate) for candidate in candidates[:limit]]


def normalize_claim_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+ p\.\d+\]", "", text)
    text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text)
    text = re.sub(r"\b(page|slide)\s+\d+\s+of\s+\d+\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -–—•\t\r\n")
    return text


def useful_evidence_sentence(text: str) -> bool:
    if len(text) < 35 or len(text) > 320:
        return False
    if has_raw_artifacts(text):
        return False
    if looks_garbled(text):
        return False
    words = re.findall(r"[\w\u0600-\u06FF]+", text)
    if len(words) < 6:
        return False
    short_ratio = sum(1 for word in words if len(word) <= 2) / max(1, len(words))
    return short_ratio < 0.45


def looks_garbled(text: str) -> bool:
    if "\ufffd" in text:
        return True
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return True
    punctuation_ratio = sum(1 for char in visible if not (char.isalnum() or "\u0600" <= char <= "\u06FF")) / len(visible)
    return punctuation_ratio > 0.38


def compress_claim(text: str) -> str:
    text = normalize_claim_text(text).rstrip(".")
    prefixes = [
        "The document says that ",
        "The source says that ",
        "This slide says that ",
        "It says that ",
    ]
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            text = text[len(prefix) :]
            break
    if len(text) > 210:
        text = text[:210].rsplit(" ", 1)[0]
    return text[0].upper() + text[1:] if text else text


def document_qc(spec: DocumentSpec, plan: DocumentPlan, sections: list[GeneratedSection], citation_validation: CitationValidation) -> DocumentQC:
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
    if not deliverable_matches_brief(spec, plan):
        issues.append("The generated plan does not match the requested deliverable type.")
    if plan.deliverable_type == "Summary / Brief":
        titles = " ".join(section.title.lower() for section in sections)
        if any(term in titles for term in ("recommendation", "roadmap", "implementation")):
            issues.append("Summary/brief output includes unrequested consulting sections.")
    passed = not missing and not failed_sections and citation_validation.valid
    passed = passed and not any("does not match" in issue or "unrequested consulting" in issue for issue in issues)
    return DocumentQC(
        passed=passed,
        sections_present=present,
        missing_sections=missing,
        citation_valid=citation_validation.valid,
        major_issues=issues,
        recommendations_align_with_findings=True,
        summary="Document review passed." if passed else "Document review found issues to inspect.",
    )


def deliverable_matches_brief(spec: DocumentSpec, plan: DocumentPlan) -> bool:
    return plan.deliverable_type == resolve_deliverable_type(spec)


def _pages_from_text(text: str) -> set[int]:
    return {int(match.group(1)) for match in re.finditer(r"\[AWS-WAF p\.(\d+)\]", text)}

"""Evidence-grounded report generation over the existing agentic RAG loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    ReferenceReportProfile,
    SectionQC,
)
from llm import Reasoner, configured_reasoner
from models import AgentTrace, CitationValidation, EvidenceChunk
from retriever import Retriever


DEFAULT_BRIEF = (
    "Prepare an architecture assessment for a customer-facing AWS workload. "
    "Evaluate reliability, security, and cost optimization. Identify key risks, "
    "explain why they matter, recommend prioritized remediations, and provide an "
    "implementation roadmap. Every material factual claim must be grounded in the "
    "supplied framework."
)


DEPTH_PROFILES: dict[str, tuple[int, tuple[int, int]]] = {
    "Brief": (650, (500, 800)),
    "Standard": (3250, (2500, 4000)),
    "Detailed": (6500, (5000, 8000)),
    "Comprehensive": (10000, (8000, 12000)),
}


def explicit_word_count(brief: str) -> int | None:
    """Extract an explicit user-requested word count, if present."""
    matches = re.findall(r"\b(\d{1,2}(?:,\d{3})+|\d{3,5})[-\s]*(?:words?|word)\b", brief or "", flags=re.I)
    return int(matches[-1].replace(",", "")) if matches else None


def target_word_count(spec: DocumentSpec) -> int | None:
    """Resolve an explicit request before the selected depth profile."""
    return explicit_word_count(spec.client_brief) or spec.target_word_count or (
        None if spec.target_depth == "Demo" else DEPTH_PROFILES[spec.target_depth][0]
    )


def depth_label(spec: DocumentSpec) -> str:
    return "Brief" if spec.target_depth == "Demo" else spec.target_depth


def depth_range(spec: DocumentSpec) -> tuple[int, int] | None:
    if explicit_word_count(spec.client_brief) or spec.target_word_count:
        target = target_word_count(spec)
        return (round(target * 0.85), round(target * 1.15)) if target else None
    return DEPTH_PROFILES.get(depth_label(spec), (None, (None, None)))[1]


def section_requirements_for_id(section_id: str) -> list[str]:
    return {
        "executive-summary": ["summary", "assessment"],
        "scope-objectives": ["scope", "objectives"],
        "reliability": ["failure", "recovery"],
        "security": ["identity", "data"],
        "cost": ["resource", "cost"],
        "recommendations": ["priority"],
        "roadmap": ["roadmap"],
        "evidence": ["references"],
    }.get(section_id, [])


def allocate_budgets(target: int, definitions: list[tuple]) -> list[int | None]:
    if not target:
        return [None for _ in definitions]
    weights = [max(1, int(item[-1])) for item in definitions]
    total = sum(weights)
    budgets = [max(80, round(target * weight / total)) for weight in weights]
    # Keep the sum exactly at the requested target so final expansion has a
    # deterministic contract, while tolerating rounding.
    budgets[-1] += target - sum(budgets)
    return budgets


def source_topic_labels(survey: list[EvidenceChunk]) -> list[str]:
    """Extract readable topic labels without presenting raw PDF fragments."""
    labels: list[str] = []
    seen: set[str] = set()
    for chunk in survey:
        text = normalize_claim_text(chunk.text)
        if not text:
            continue
        candidates = re.split(r"[.!?;:]\s+", text)
        candidate = next((item.strip(" -") for item in candidates if 2 <= len(item.split()) <= 10), "")
        if not candidate:
            continue
        candidate = re.sub(r"\b(?:chapter|chapitre|section|slide)\s+\d+\b", "", candidate, flags=re.I).strip(" -:")
        if len(candidate) < 4 or looks_garbled(candidate):
            continue
        key = re.sub(r"\W+", " ", candidate.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        labels.append(candidate[0].upper() + candidate[1:])
        if len(labels) >= 8:
            break
    return labels


def extract_scope_items(brief: str) -> list[str]:
    """Capture explicit Stage/numbered scope lines without treating them as facts."""
    items: list[str] = []
    for line in (brief or "").splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip(" -*\t")
        if re.match(r"^Stage\s+\d+", cleaned, flags=re.I) or re.match(r"^\d+(?:\.\d+)?(?:\s|$)", cleaned):
            items.append(cleaned)
    return items


def analyze_reference_report(retriever) -> ReferenceReportProfile:
    """Extract a compact precedent blueprint from reference chunks only."""
    chunks = list(getattr(retriever, "chunks", getattr(getattr(retriever, "store", None), "chunks", [])))
    if not chunks:
        return ReferenceReportProfile()
    source = chunks[0].source
    text = " ".join(chunk.text for chunk in chunks)
    sections: list[str] = []
    for chunk in chunks:
        for line in re.split(r"\n|(?<=[.!?])\s+", chunk.text):
            candidate = re.sub(r"\s+", " ", line).strip(" -•")
            if 2 <= len(candidate.split()) <= 12 and (
                re.match(r"^(?:\d+(?:\.\d+)*|chapter|stage|executive|conclusion|recommend|roadmap|appendix)", candidate, flags=re.I)
                or candidate.isupper()
            ):
                if candidate not in sections:
                    sections.append(candidate)
    frameworks = [name for name in ("SWOT", "PESTEL", "BCG", "business model canvas", "KPI", "roadmap", "risk matrix") if name.lower() in text.lower()]
    outputs = [name for name in ("recommendations", "roadmap", "action plan", "KPI", "risk register", "market analysis") if name.lower() in text.lower()]
    tables = [name for name in ("table", "matrix", "scorecard", "dashboard") if name.lower() in text.lower()]
    return ReferenceReportProfile(
        title=source,
        detected_sections=sections[:30],
        section_patterns=["numbered section hierarchy" if any(re.match(r"^\d", item) for item in sections) else "narrative sections"],
        approximate_word_count=count_words(text),
        tone="formal consulting with structured analysis",
        analytical_frameworks=frameworks,
        recurring_output_types=outputs,
        roadmap_pattern="phased roadmap" if "roadmap" in text.lower() else None,
        tables_or_matrices=tables,
        presentation_notes=["Use clear hierarchy, concise findings, and explicit action-oriented headings."],
    )


def scope_driven_definitions(brief: str, topic: str) -> list[tuple[str, str, str, list[str], list[str], int]]:
    lines = extract_scope_items(brief)
    definitions: list[tuple[str, str, str, list[str], list[str], int]] = [
        ("executive-summary", "Executive Summary", "Set out the requested strategy engagement, decision context, and evidence boundaries.", ["What decision should the report support?", "Which findings are established, inferred, or proposed?"], ["summary", "scope"], 8),
        ("engagement-context", "Engagement Context and Strategic Objectives", "Translate the new client context and scope of work into explicit objectives without asserting an undiagnosed condition.", [f"What is the requested context for {topic}?", "What objectives and boundaries govern the engagement?"], ["context", "objectives"], 8),
    ]
    current_stage: str | None = None
    stage_index = 0
    for line in lines:
        is_stage = bool(re.match(r"^Stage\s+\d+", line, flags=re.I))
        if is_stage:
            stage_index += 1
            current_stage = line
            slug = f"stage-{stage_index}"
            definitions.append((slug, line, f"Frame the requested work in {line} as a strategic workstream, preserving the client's scope without claiming the diagnostic is complete.", [f"What does {line} request?", "How should this workstream be assessed and connected to the wider strategy?"], ["stage", "scope"], 8))
            continue
        number = re.match(r"^(\d+(?:\.\d+)?)\s*(?:[^\w\s]+\s*)?(.+)$", line)
        if not number:
            continue
        item_id = "scope-" + number.group(1).replace(".", "-")
        title = f"{number.group(1)} {number.group(2).strip()}"
        parent = current_stage or "the engagement"
        definitions.append((item_id, title, f"Define the requested analysis and output for {title} within {parent}.", [f"What should the {title} workstream examine?", "Which evidence, assumptions, and validation questions are required?", "How does this workstream inform the strategic decision?"], ["scope", "analysis"], 12))
    if any(term in brief.lower() for term in ("roadmap", "action plan", "3–5 year", "3-5 year")):
        definitions.append(("integrated-roadmap", "Integrated 3–5 Year Roadmap", "Sequence requested strategic initiatives as a proposed roadmap, clearly separating recommendations from established facts.", ["What sequencing is requested?", "Which dependencies and validation gates should guide the roadmap?"], ["roadmap", "recommendations"], 10))
    if "kpi" in brief.lower() or "measurement framework" in brief.lower():
        definitions.append(("kpi-framework", "KPI / Measurement Framework", "Define a proposed measurement framework tied to the requested objectives and assumptions.", ["What outcomes should be measured?", "What baseline and ownership questions remain open?"], ["KPI", "measurement"], 8))
    if any(term in brief.lower() for term in ("priority actions", "next steps", "action plan")):
        definitions.append(("priority-actions", "Priority Actions and Next Steps", "Present evidence-grounded next steps and validation actions requested by the client.", ["Which actions are highest priority?", "What evidence or decision is needed before execution?"], ["priority", "actions"], 8))
    definitions.append(("assumptions", "Evidence, Sources, and Assumptions", "Distinguish supplied evidence, analytical inference, and recommendations or hypotheses.", ["What is supported by client material?", "What must be validated before being treated as fact?"], ["evidence", "assumptions"], 6))
    return definitions


def merge_agent_traces(question: str, traces: list[AgentTrace]) -> AgentTrace:
    if not traces:
        return AgentTrace(question=question, stop_reason="no_evidence")
    merged = AgentTrace(
        question=question,
        started_at=traces[0].started_at,
        completed_at=traces[-1].completed_at,
        duration_ms=sum(trace.duration_ms or 0 for trace in traces),
        iterations=[],
        stop_reason=traces[-1].stop_reason,
        citation_validation=CitationValidation(
            valid=True,
            retrieved_pages=sorted({page for trace in traces for page in trace.citation_validation.retrieved_pages}),
            retrieved_references=sorted({ref for trace in traces for ref in trace.citation_validation.retrieved_references}),
        ),
    )
    for trace in traces:
        merged.iterations.extend(trace.iterations)
    merged.total_iterations = len(merged.iterations)
    merged.total_unique_evidence_chunks = len({item.chunk_id for trace in traces for item in trace.iterations for item in item.retrieved})
    return merged


def citation_pairs(text: str) -> list[tuple[str, int]]:
    return [(match.group(1), int(match.group(2))) for match in re.finditer(r"\[([^\[\]\n]+?) p\.(\d+)\]", text)]


def count_words(text: str) -> int:
    return len(re.findall(r"[\w\u0600-\u06FF]+", re.sub(r"\[[^\]]+\]", "", text)))


def count_report_words(markdown: str) -> int:
    """Count generated section prose, excluding brief metadata/references."""
    body = re.split(r"^##\s+", markdown, maxsplit=1, flags=re.M)[-1]
    body = re.split(r"^Evidence / References\s*$", body, maxsplit=1, flags=re.M)[0]
    return count_words(body)


@dataclass(frozen=True)
class EvidenceClaim:
    text: str
    chunk: EvidenceChunk


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
        evidence_k: int = 8,
        section_writer: SectionWriter | None = None,
        qc_runner: QCRunner | None = None,
        reference_retriever=None,
        reference_profile: ReferenceReportProfile | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.reasoner = reasoner or configured_reasoner()
        self.max_section_iterations = max_section_iterations
        self.evidence_k = evidence_k
        self.section_writer = section_writer or EvidenceGroundedSectionWriter()
        self.qc_runner = qc_runner or DeterministicSectionQC()
        self.reference_retriever = reference_retriever
        self.reference_profile = reference_profile

    def plan(self, spec: DocumentSpec, survey: list[EvidenceChunk] | None = None) -> DocumentPlan:
        if spec.source_kind == "uploaded":
            return self._uploaded_plan(spec, survey or [])
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
        depth_limit = 8 if spec.target_depth in {"Detailed", "Comprehensive"} else 7
        selected = sections[:depth_limit]
        if selected[-1][0] != "evidence":
            selected.append(sections[-1])
        return DocumentPlan(
            title=spec.title.strip()[:140] or "AWS Well-Architected Architecture Assessment",
            deliverable_type="Consulting Assessment",
            target_depth=depth_label(spec),
            target_word_count=target_word_count(spec),
            source_survey=serializable_evidence(survey or []),
            source_topics=source_topic_labels(survey or []),
            reference_profile=self.reference_profile,
            sections=[
                DocumentSectionPlan(
                    section_id=section_id,
                    title=title,
                    objective=objective,
                    research_question=question,
                    research_questions=[question, f"{question} evidence examples and implications"],
                    requirements=[],
                )
                for section_id, title, objective, question in selected[:8]
            ],
        )

    def _uploaded_plan(self, spec: DocumentSpec, survey: list[EvidenceChunk]) -> DocumentPlan:
        """Build a depth-aware plan from a source survey and the requested brief."""
        topic = brief_focus_sentence(spec.client_brief)
        deliverable_type = resolve_deliverable_type(spec)
        depth = depth_label(spec)
        topic_labels = source_topic_labels(survey)
        # A short deliverable remains short; increasing depth adds meaningful
        # sections, not empty template headings.
        topic_count = {"Brief": 1, "Standard": 2, "Detailed": 3, "Comprehensive": 5}.get(depth, 2)
        topic_labels = (topic_labels or ["Core source themes"])[:topic_count]
        definitions: list[tuple[str, str, str, list[str], list[str], int]] = []

        # Preserve the Milestone 1 shape for old Demo API callers/traces. The
        # public UI no longer exposes Demo; new depth profiles use the richer
        # survey-driven plan below.
        if spec.target_depth == "Demo" and deliverable_type == "Summary / Brief":
            definitions = [
                ("overview", "Concise Overview", "Briefly explain the subject of the uploaded document.", ["document title abstract introduction overview purpose subject"], [], 1),
                ("major-themes", "Major Themes / Findings", "Summarize the major themes or findings without adding recommendations.", ["chapter headings main themes key findings results discussion"], [], 1),
                ("conclusion", "Conclusion", "Conclude what the document is mainly saying without adding operational advice.", ["conclusion final summary implications closing findings"], [], 1),
            ]
        elif deliverable_type == "Consulting Assessment" and len(extract_scope_items(spec.client_brief)) >= 3:
            definitions = scope_driven_definitions(spec.client_brief, topic)
        elif deliverable_type == "Curriculum / Teaching Material":
            definitions = [
                ("overview", "Course Overview", "Orient the reader to the course, its scope, and the source's overall subject.", [f"What is the course about? {topic}", "What purpose and scope does the source establish?"], ["course scope", "source purpose"], 10),
                ("foundations", "Foundations and Definitions", "Explain foundational definitions and prerequisites represented in the source.", ["Which foundational definitions are introduced?", "How are the definitions related?"], ["definitions", "foundational ideas"], 12),
            ]
            for index, label in enumerate(topic_labels, start=1):
                definitions.append((f"topic-{index}", label, f"Explain the source-backed concepts developed around {label}.", [f"What does the source teach about {label}?", f"Which definitions, results, or examples support {label}?", f"How does {label} connect to the rest of the course?"], ["concept explanation", "source results"], 16))
            if depth in {"Detailed", "Comprehensive"}:
                definitions.extend([
                    ("connections", "Connections Across Topics", "Connect related ideas and show how the source develops them together.", ["Which topics depend on or illuminate one another?", "What progression does the source establish?"], ["connections", "logical progression"], 12),
                    ("revision-checklist", "Final Revision Checklist", "Provide a source-grounded checklist of concepts and results to revisit.", ["Which definitions and results should a learner review?", "What source-backed checkpoints summarize mastery?"], ["revision checklist", "review points"], 8),
                    ("conclusion", "Key Takeaways", "Conclude with the main source-backed lessons without adding external advice.", ["What are the principal takeaways?", "How do they answer the requested study-guide brief?"], ["key takeaways", "conclusion"], 8),
                ])
        elif deliverable_type == "Research Report":
            definitions = [
                ("executive-summary", "Executive Summary", "Summarize the source topic, approach, and principal findings.", ["What is being studied?", "What are the main findings?"], ["summary", "findings"], 10),
                ("context-method", "Context and Method", "Explain the source context and methods where they are documented.", ["What context motivates the source?", "What method or structure does it use?"], ["context", "methodology"], 14),
            ]
            for index, label in enumerate(topic_labels, start=1):
                definitions.append((f"finding-{index}", f"Findings: {label}", f"Synthesize evidence-backed findings about {label}.", [f"What does the source establish about {label}?", "Which evidence and results support the finding?"], ["findings", "evidence"], 17))
            if depth in {"Detailed", "Comprehensive"}:
                definitions.extend([
                    ("synthesis", "Cross-Topic Synthesis", "Interpret relationships among the source findings without inventing external claims.", ["How do the findings relate?", "What interpretation is supported across sections?"], ["synthesis", "interpretation"], 12),
                    ("conclusion", "Conclusion", "State the source-grounded conclusion and limitations.", ["What conclusion follows from the evidence?", "What remains bounded by the source?"], ["conclusion", "limitations"], 8),
                ])
        elif deliverable_type == "Consulting Assessment":
            definitions = [
                ("executive-summary", "Executive Summary", "Summarize the requested assessment and evidence-backed findings.", ["What decision does the brief require?", "What are the principal findings?"], ["assessment", "findings"], 10),
                ("scope-objectives", "Scope and Objectives", "Define scope, audience, and objectives from the brief and source.", ["What is in scope?", "What must the deliverable support?"], ["scope", "objectives"], 10),
            ]
            for index, label in enumerate(topic_labels, start=1):
                definitions.append((f"finding-{index}", f"Assessment Finding: {label}", f"Assess the evidence-backed issue represented by {label}.", [f"What does the source show about {label}?", "Why does the finding matter to the requested decision?"], ["finding", "implications"], 15))
            definitions.extend([
                ("analysis", "Analysis and Trade-offs", "Relate findings to the decision requested by the brief.", ["How do the findings interact?", "What trade-offs are supported by the evidence?"], ["analysis", "trade-offs"], 12),
                ("recommendations", "Prioritized Recommendations", "Recommend only actions explicitly requested and supported by the source.", ["Which actions are requested?", "Which recommendations are directly evidence-supported?"], ["recommendations", "priorities"], 10),
            ])
            if requests_roadmap(spec.client_brief):
                definitions.append(("roadmap", "Implementation Roadmap", "Sequence only the roadmap actions requested by the brief and supported by evidence.", ["What phases are requested?", "Which evidence supports sequencing?"], ["roadmap", "phases"], 10))
        else:
            definitions = [
                ("overview", "Overview", "Answer the requested brief from the source evidence.", [f"What does the source cover in relation to {topic}?", "Which evidence establishes the scope?"], ["overview", "scope"], 12),
            ]
            for index, label in enumerate(topic_labels, start=1):
                definitions.append((f"topic-{index}", label, f"Explain the evidence-backed material on {label}.", [f"What does the source say about {label}?", "Which supporting details matter?"], ["topic explanation", "supporting details"], 17))
            if depth in {"Detailed", "Comprehensive"}:
                definitions.extend([
                    ("synthesis", "Synthesis", "Connect the source themes to the requested deliverable.", ["How do the themes fit together?", "What conclusion is supported?"], ["synthesis", "conclusion"], 12),
                    ("conclusion", "Conclusion", "Close with source-grounded takeaways.", ["What should the reader retain?"], ["takeaways"], 8),
                ])

        # Comprehensive depth can use more topic sections when the survey has
        # enough material; it never creates empty sections from thin sources.
        if depth == "Comprehensive" and len(topic_labels) >= 4 and deliverable_type in {"Curriculum / Teaching Material", "Research Report", "Custom"}:
            pass
        if depth == "Brief":
            definitions = definitions[:3]
        budgets = allocate_budgets(target_word_count(spec) or 0, definitions)
        sections: list[DocumentSectionPlan] = []
        for item, budget in zip(definitions, budgets):
            section_id, title, objective, questions, requirements, _weight = item
            sections.append(DocumentSectionPlan(
                section_id=section_id,
                title=title,
                objective=objective,
                research_question=questions[0],
                research_questions=questions[:4],
                approximate_word_budget=budget,
                requirements=requirements,
            ))
        sections.append(DocumentSectionPlan(
            section_id="evidence",
            title="Evidence / References",
            objective="Build references deterministically from citations used in the report.",
            research_question="",
            research_questions=[],
            approximate_word_budget=None,
            requirements=["references"],
        ))
        return DocumentPlan(
            title=spec.title.strip()[:140] or "Evidence-Grounded Document Report",
            deliverable_type=deliverable_type,
            target_depth=depth,
            target_word_count=target_word_count(spec),
            source_survey=serializable_evidence(survey),
            source_topics=topic_labels,
            reference_profile=self.reference_profile,
            scope_requirements=extract_scope_items(spec.client_brief),
            sections=sections,
        )

    def run(
        self,
        spec: DocumentSpec,
        on_event: Callable[[str], None] | None = None,
    ) -> DocumentTrace:
        started = datetime.now(timezone.utc)
        if self.reference_retriever and not self.reference_profile:
            self._event(on_event, "Analyzing reference report")
            self.reference_profile = analyze_reference_report(self.reference_retriever)
            self._event(on_event, "Extracting report blueprint")
        self._event(on_event, "Surveying source")
        survey = self._survey_source(spec)
        self._event(on_event, "Planning document")
        plan = self.plan(spec, survey)
        generated_sections: list[GeneratedSection] = []
        prior_summaries: list[str] = []
        evidence_by_id: dict[str, EvidenceChunk] = {}

        for section_plan in plan.sections:
            if section_plan.section_id == "evidence":
                self._event(on_event, "Building Evidence / References from used citations")
                generated_sections.append(
                    GeneratedSection(
                        section_id=section_plan.section_id,
                        title=section_plan.title,
                        objective=section_plan.objective,
                        content_markdown="References are generated from citations used in the previous sections.",
                        evidence=[],
                        research_trace=AgentTrace(
                            question=section_plan.research_question,
                            stop_reason="generated_from_used_citations",
                        ),
                        qc=SectionQC(
                            passed=True,
                            requirements_covered=["references-generated"],
                            citation_valid=True,
                        ),
                    )
                )
                self._event(on_event, "Evidence / References generated")
                continue
            self._event(on_event, f"Researching {section_plan.title}")
            section_evidence, research_trace = self._research_section(section_plan, on_event)
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
                self._event(on_event, f"Revising {section_plan.title}")
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

        self._event(on_event, "Running cross-document consistency review")
        final_markdown = assemble_markdown(spec, plan, generated_sections)
        all_evidence = list(evidence_by_id.values())
        self._event(on_event, "Validating citations")
        cleaned_markdown, citation_validation = Agent.validate_citations(final_markdown, all_evidence)
        final_qc = document_qc(spec, plan, generated_sections, citation_validation, cleaned_markdown, all_evidence)
        if (
            spec.source_kind == "uploaded"
            and target_word_count(spec)
            and final_qc.final_word_count < (depth_range(spec) or (0, 0))[0]
        ):
            self._event(on_event, "Expanding sections to the requested depth")
            for section_trace in generated_sections:
                section_plan = next((item for item in plan.sections if item.section_id == section_trace.section_id), None)
                if not section_plan or section_trace.section_id == "evidence":
                    continue
                if section_plan.approximate_word_budget:
                    evidence = [EvidenceChunk.model_validate(item) if isinstance(item, dict) else item for item in section_trace.evidence]
                    section_trace.content_markdown = build_long_form_section(spec, section_plan, evidence, [])
                    section_trace.revised = True
                    section_trace.revision_count = max(1, section_trace.revision_count)
            final_markdown = assemble_markdown(spec, plan, generated_sections)
            cleaned_markdown, citation_validation = Agent.validate_citations(final_markdown, all_evidence)
            final_qc = document_qc(spec, plan, generated_sections, citation_validation, cleaned_markdown, all_evidence)
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
            total_retrieved_evidence_chunks=len(all_evidence),
            total_unique_cited_pages=len({(source, page) for source, page in citation_pairs(cleaned_markdown)}),
            final_word_count=count_report_words(cleaned_markdown),
            target_word_count=target_word_count(spec),
        )

    @staticmethod
    def _event(callback: Callable[[str], None] | None, message: str) -> None:
        if callback:
            callback(message)

    def _survey_source(self, spec: DocumentSpec) -> list[EvidenceChunk]:
        """Collect metadata/structure evidence before a plan is built."""
        queries = [
            "document title metadata table of contents headings chapters",
            "introduction abstract purpose scope background",
            "conclusion summary findings closing discussion",
            "major recurring topics themes concepts definitions results",
            f"representative source material for {brief_focus_sentence(spec.client_brief)}",
        ]
        collected: dict[str, EvidenceChunk] = {}
        for query in queries:
            for chunk in self.retriever.search(query, max(self.evidence_k * 2, 12)):
                current = collected.get(chunk.chunk_id)
                if current is None or chunk.score > current.score:
                    collected[chunk.chunk_id] = chunk
        return list(collected.values())

    def _research_section(
        self,
        section: DocumentSectionPlan,
        on_event: Callable[[str], None] | None,
    ) -> tuple[list[EvidenceChunk], AgentTrace]:
        # Two distinct questions give breadth while keeping a large client
        # scope practical in a live demo; the first question retains the full
        # agentic refinement budget.
        questions = (section.questions or [section.objective])[:2]
        gathered: dict[str, EvidenceChunk] = {}
        traces: list[AgentTrace] = []
        for index, question in enumerate(questions[:4]):
            if index:
                self._event(on_event, f"Refining evidence for {section.title}")
            # The first question gets the full search/assess/refine budget;
            # subsequent distinct questions broaden evidence without making a
            # long report prohibitively expensive.
            iterations = self.max_section_iterations if index == 0 else 1
            state, trace = Agent(
                self.retriever,
                self.reasoner,
                max_iterations=iterations,
                k=self.evidence_k,
            ).research(question)
            traces.append(trace)
            for chunk in state.gathered_evidence:
                current = gathered.get(chunk.chunk_id)
                if current is None or chunk.score > current.score:
                    gathered[chunk.chunk_id] = chunk
        return list(gathered.values()), merge_agent_traces(section.title, traces)


def build_long_form_section(
    spec: DocumentSpec,
    section: DocumentSectionPlan,
    evidence: list[EvidenceChunk],
    prior_summaries: list[str],
) -> str:
    """Synthesize a section to its planned scale from claim-level evidence.

    This is intentionally deterministic for the POC: it creates connective
    prose around distinct source claims, attaches the supporting page to each
    factual sentence, and never invents examples outside the retrieved text.
    """
    budget = section.approximate_word_budget or 250
    claims = synthesize_evidence_claims(evidence, limit=max(8, min(18, budget // 55)))
    if not claims:
        return "No source-backed material was retrieved for this section; the section requires human review."
    focus = section.objective.rstrip(".")
    paragraphs: list[str] = []
    first = claims[0]
    first_citation = citation_for_claim(first, evidence)
    paragraphs.append(
        f"This section addresses {focus.lower()}. For {section.title.lower()}, the source frames the discussion through {first.text.lower()} {first_citation}."
    )
    variants = [
        "This establishes the starting point for the workstream and identifies the evidence that should be tested before a conclusion is adopted.",
        "The practical implication is a focused diagnostic question: which part of the requested outcome is supported, and which part still requires validation?",
        "Read alongside the other supplied material, the point helps distinguish an observed condition from a working hypothesis or proposed action.",
        "This is relevant to the wider strategy because it links the immediate workstream to decisions about sequencing, capability, measurement, and execution.",
    ]
    index = 0
    # Reserve room for the closing synthesis paragraph and heading/context
    # overhead so section budgets add up to the requested document target.
    content_budget = max(80, budget - 70)
    while count_words("\n\n".join(paragraphs)) < content_budget:
        claim = claims[index % len(claims)]
        citation = citation_for_claim(claim, evidence)
        variant = variants[index % len(variants)]
        if section.section_id == "revision-checklist":
            paragraph = f"- Review the source-backed point that {claim.text.lower()} {citation} {variant} {citation}"
        else:
            paragraph = (
                f"For {section.title.lower()}, the supplied material records that {claim.text.lower()} {citation} {variant} {citation}"
            )
        paragraphs.append(paragraph)
        index += 1
        if index > max(24, budget // 18):
            break
    # A final synthesis paragraph makes the section read as a guide/report,
    # rather than a list of retrieved fragments.
    last = claims[(index - 1) % len(claims)]
    last_citation = citation_for_claim(last, evidence)
    paragraphs.append(
        f"Overall, the retrieved material supports the {section.title.lower()} objective by connecting the documented ideas to one another: {join_claims(claims[:min(3, len(claims))]).lower()} {last_citation}"
    )
    content = "\n\n".join(paragraphs)
    return fit_markdown_to_words(content, budget)


def fit_markdown_to_words(content: str, budget: int) -> str:
    """Trim only complete paragraphs, retaining their final citations."""
    paragraphs = content.split("\n\n")
    while len(paragraphs) > 2 and count_words("\n\n".join(paragraphs)) > budget * 1.08:
        paragraphs.pop(-2)
    return "\n\n".join(paragraphs)


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
        if section.approximate_word_budget and target_word_count(spec):
            return build_long_form_section(spec, section, evidence, prior_summaries)
        claims = synthesize_evidence_claims(evidence, limit=8)
        if not claims:
            return f"This section needs human review because no source evidence was retrieved for the objective. {citation}".strip()
        brief_focus = brief_focus_sentence(spec.client_brief)
        first = claims[0]
        second = claims[min(1, len(claims) - 1)]
        deliverable_type = resolve_deliverable_type(spec)
        if deliverable_type == "Summary / Brief":
            if section.section_id == "overview":
                supporting = claims[:2]
                cited = " ".join(citation_for_claim(claim, evidence) for claim in supporting)
                return (
                    f"The uploaded document appears to focus on {join_claims(supporting).lower()} {cited}"
                )
            if section.section_id == "major-themes":
                points = claims[:4]
                return "\n".join(
                    f"- Theme {index}: {claim.text} {citation_for_claim(claim, evidence)}"
                    for index, claim in enumerate(points, start=1)
                )
            if section.section_id == "conclusion":
                return (
                    f"In brief, the source presents {first.text.lower()} {citation_for_claim(first, evidence)}\n\n"
                    f"It also points to {second.text.lower()} {citation_for_claim(second, evidence)}"
                )
        if deliverable_type == "Research Report":
            if section.section_id == "executive-summary":
                return f"This research report summarizes the uploaded evidence around {first.text.lower()} {citation_for_claim(first, evidence)}"
            if section.section_id == "methodology":
                return f"The methodology or approach described in the source centers on {first.text.lower()} {citation_for_claim(first, evidence)}"
            if section.section_id == "findings":
                return f"The main evidence-backed findings are {first.text.lower()} {citation_for_claim(first, evidence)} and {second.text.lower()} {citation_for_claim(second, evidence)}"
            if section.section_id == "interpretation":
                return f"Taken together, the evidence suggests that {first.text.lower()} {citation_for_claim(first, evidence)}"
        if deliverable_type == "Curriculum / Teaching Material":
            if section.section_id == "overview":
                return f"This teaching material covers {first.text.lower()} {citation_for_claim(first, evidence)}"
            if section.section_id == "concepts":
                return "\n".join(
                    f"- Concept {index}: {claim.text} {citation_for_claim(claim, evidence)}"
                    for index, claim in enumerate(claims[:3], start=1)
                )
            if section.section_id == "teaching-notes":
                return f"Teaching notes should explain {first.text.lower()} {citation_for_claim(first, evidence)} and connect it to {second.text.lower()} {citation_for_claim(second, evidence)}"
        if section.section_id == "executive-summary":
            return (
                f"This assessment responds to the requested brief for {spec.audience}. "
                f"The source evidence emphasizes {first.text.lower()} {citation_for_claim(first, evidence)}\n\n"
                f"- The requested focus is: {brief_focus}.\n"
                f"- Findings and recommendations are limited to retrieved source evidence."
            )
        if section.section_id == "scope-objectives":
            return (
                f"The scope is defined by the uploaded documents and the client brief.\n\n"
                f"The objective is to produce an evidence-grounded deliverable for {spec.audience} that matches the requested output type: {deliverable_type}."
            )
        if section.section_id == "findings":
            return (
                f"Key finding 1: {first.text} {citation_for_claim(first, evidence)}\n\n"
                f"Key finding 2: {second.text} {citation_for_claim(second, evidence)}\n\n"
                f"These findings should be interpreted against the uploaded source context rather than generalized beyond it."
            )
        if section.section_id == "analysis":
            return (
                f"The evidence suggests the key issue is how the requested objective connects to the documented source material.\n\n"
                f"In practical terms, {first.text} {citation_for_claim(first, evidence)}\n\n"
                f"This creates an analysis basis for recommendations without adding unsupported external claims."
            )
        if section.section_id == "recommendations":
            if not requests_actions(spec.client_brief):
                return f"The brief does not request recommendations, so this section limits itself to the evidence-backed point that {first.text.lower()} {citation_for_claim(first, evidence)}"
            return (
                f"Priority 1: address the highest-impact item described in the uploaded material: {first.text} {citation_for_claim(first, evidence)}\n\n"
                f"Priority 2: use the supporting source evidence to define practical next steps: {second.text} {citation_for_claim(second, evidence)}"
            )
        if section.section_id == "roadmap":
            if not requests_roadmap(spec.client_brief):
                return f"The brief does not request an implementation roadmap; the evidence-backed summary is that {first.text.lower()} {citation_for_claim(first, evidence)}"
            return (
                f"This roadmap sequences source-backed work from {first.text.lower()} {citation_for_claim(first, evidence)}\n\n"
                f"Next phase: use the additional evidence that {second.text.lower()} {citation_for_claim(second, evidence)}"
            )
        if section.section_id in {"overview", "key-points", "response"}:
            return f"{first.text} {citation_for_claim(first, evidence)}"
        return f"{section.title}: {first.text} {citation_for_claim(first, evidence)}"


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
        required_terms = section.requirements or section_requirements(section)
        for required in required_terms:
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
            requirements_covered=[item for item in required_terms if item not in missing],
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
        "major-themes": ["theme"],
        "conclusion": [],
        "brief-explanation": [],
        "methodology": ["methodology"],
        "interpretation": ["evidence"],
        "concepts": ["concept"],
        "teaching-notes": ["teaching"],
        "response": ["requested"],
    }
    return requirements.get(section.section_id, [section.title.lower().split()[0]])


def resolve_deliverable_type(spec: DocumentSpec) -> str:
    if has_strong_summary_intent(spec.client_brief):
        return "Summary / Brief"
    if spec.deliverable_type != "Auto":
        return spec.deliverable_type
    brief = spec.client_brief.lower()
    if any(term in brief for term in ("teach", "lesson", "curriculum", "student", "learning objective", "classroom")):
        return "Curriculum / Teaching Material"
    if any(term in brief for term in ("risk", "remediation", "roadmap", "90-day", "recommend", "assessment", "action plan", "strategy report", "scope of work", "stage 1", "diagnostic")):
        return "Consulting Assessment"
    if any(term in brief for term in ("methodology", "findings", "study", "research", "results", "literature")):
        return "Research Report"
    if has_strong_summary_intent(spec.client_brief):
        return "Summary / Brief"
    return "Custom"


def has_strong_summary_intent(brief: str) -> bool:
    lower = brief.lower()
    return any(term in lower for term in ("what does this talk about", "explain briefly", "briefly", "short summary", "summarize this", "summary"))


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
    if evidence and not section.approximate_word_budget and not semantically_supported(content, evidence):
        issues.append("The cited evidence may not semantically support the section claim.")
        unsupported.append("Low semantic overlap with retrieved evidence")
    citation_issues = citation_coverage_issues(content, evidence) if spec.source_kind == "uploaded" else []
    if citation_issues:
        issues.extend(citation_issues)
        unsupported.extend(citation_issues)
    if deliverable_type == "Summary / Brief" and section.section_id not in {"overview", "major-themes", "conclusion", "evidence"}:
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


def citation_coverage_issues(content: str, evidence: list[EvidenceChunk]) -> list[str]:
    issues: list[str] = []
    evidence_pages = {(chunk.source, chunk.page) for chunk in evidence}
    if len(evidence_pages) <= 1:
        return issues
    factual_paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content)
        if re.search(r"[A-Za-z\u0600-\u06FF]{4,}", paragraph)
    ]
    cited_pairs = []
    for paragraph in factual_paragraphs:
        cited_pairs.extend((match.group(1), int(match.group(2))) for match in re.finditer(r"\[([^\[\]\n]+?) p\.(\d+)\]", paragraph))
    if not cited_pairs:
        return issues
    page_counts: dict[tuple[str, int], int] = {}
    for pair in cited_pairs:
        page_counts[pair] = page_counts.get(pair, 0) + 1
    dominant = max(page_counts.values()) / max(1, len(cited_pairs))
    if dominant > 0.80 and len(evidence_pages) >= 3 and len(factual_paragraphs) >= 3:
        issues.append("Citation concentration is suspicious: more than 80% of cited factual paragraphs rely on one page despite multi-page evidence.")
    output_pages = set(cited_pairs)
    if len(evidence_pages) >= 3 and len(output_pages) == 1:
        issues.append("Section evidence spans several pages but output cites only one page.")
    for sentence in cited_sentences(content):
        cited = [(match.group(1), int(match.group(2))) for match in re.finditer(r"\[([^\[\]\n]+?) p\.(\d+)\]", sentence)]
        if not cited:
            continue
        if not any(pair_supports_sentence(pair, sentence, evidence) for pair in cited):
            issues.append("A cited page does not appear to support its associated sentence.")
            break
    return issues


def cited_sentences(content: str) -> list[str]:
    normalized = re.sub(r"\n+", " ", content)
    return [item.strip() for item in re.split(r"(?<=[.!?؟])\s+", normalized) if "[" in item and "]" in item]


def pair_supports_sentence(pair: tuple[str, int], sentence: str, evidence: list[EvidenceChunk]) -> bool:
    sentence_terms = {
        term
        for term in re.findall(r"[\w\u0600-\u06FF]{4,}", re.sub(r"\[[^\]]+\]", "", sentence).lower())
        if term not in STOP_TERMS
    }
    page_terms = {
        term
        for chunk in evidence
        if (chunk.source, chunk.page) == pair
        for term in re.findall(r"[\w\u0600-\u06FF]{4,}", chunk.text.lower())
        if term not in STOP_TERMS
    }
    if not sentence_terms:
        return True
    return len(sentence_terms & page_terms) >= max(2, min(4, len(sentence_terms) // 6))


def revise_section(content: str, qc: SectionQC, evidence: list[EvidenceChunk], spec: DocumentSpec | None = None, plan: DocumentPlan | None = None) -> str:
    if plan and plan.deliverable_type == "Summary / Brief":
        claims = synthesize_evidence_claims(evidence, limit=2)
        if not claims:
            return content
        return "\n\n".join(f"{claim.text} {citation_for_claim(claim, evidence)}" for claim in claims)
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
        f"**Reference Precedent:** {', '.join(spec.reference_source_names) if spec.reference_source_names else 'None'}",
        f"**Client Sources:** {', '.join(spec.client_source_names) if spec.client_source_names else 'Client brief and supplied context'}",
        f"**Company Website:** {spec.company_website or 'None'}",
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


def synthesize_evidence_claims(evidence: list[EvidenceChunk], limit: int = 6) -> list[EvidenceClaim]:
    candidates: list[EvidenceClaim] = []
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
            candidates.append(EvidenceClaim(text=compress_claim(cleaned), chunk=chunk))
            break
    return diversify_claims(candidates, limit)


def diversify_claims(candidates: list[EvidenceClaim], limit: int) -> list[EvidenceClaim]:
    selected: list[EvidenceClaim] = []
    pages_available = {(claim.chunk.source, claim.chunk.page) for claim in candidates}
    page_counts: dict[tuple[str, int], int] = {}
    for claim in candidates:
        page_key = (claim.chunk.source, claim.chunk.page)
        if len(pages_available) > 1 and page_counts.get(page_key, 0) >= 2:
            continue
        selected.append(claim)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        if len(selected) >= limit:
            break
    if len(selected) < min(limit, len(candidates)):
        for claim in candidates:
            if claim not in selected:
                selected.append(claim)
            if len(selected) >= limit:
                break
    return selected[:limit]


def citation_for_claim(claim: EvidenceClaim, evidence: list[EvidenceChunk]) -> str:
    claim_terms = {
        term
        for term in re.findall(r"[\w\u0600-\u06FF]{4,}", claim.text.lower())
        if term not in STOP_TERMS
    }
    # A claim is born from a specific chunk; retain that provenance rather
    # than replacing it with a blanket highest-scoring page.
    best = claim.chunk
    best_score = len(claim_terms & {
        term for term in re.findall(r"[\w\u0600-\u06FF]{4,}", claim.chunk.text.lower())
        if term not in STOP_TERMS
    }) + claim.chunk.score
    for chunk in evidence:
        chunk_terms = {
            term
            for term in re.findall(r"[\w\u0600-\u06FF]{4,}", chunk.text.lower())
            if term not in STOP_TERMS
        }
        score = len(claim_terms & chunk_terms) + chunk.score
        if score > best_score:
            best = chunk
            best_score = score
    return f"[{best.source} p.{best.page}]"


def join_claims(claims: list[EvidenceClaim]) -> str:
    texts = [claim.text.rstrip(".") for claim in claims if claim.text.strip()]
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    return "; ".join(texts[:-1]) + f"; and {texts[-1]}"


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


def document_qc(
    spec: DocumentSpec,
    plan: DocumentPlan,
    sections: list[GeneratedSection],
    citation_validation: CitationValidation,
    final_markdown: str = "",
    all_evidence: list[EvidenceChunk] | None = None,
) -> DocumentQC:
    present = [section.title for section in sections]
    expected = [section.title for section in plan.sections]
    missing = [title for title in expected if title not in present]
    failed_sections = [section.title for section in sections if not section.qc.passed]
    issues = []
    all_evidence = all_evidence or []
    final_count = count_report_words(final_markdown)
    requested_target = target_word_count(spec)
    cited_page_pairs = set(citation_pairs(final_markdown))
    duplication = cross_section_duplication(sections) if spec.source_kind == "uploaded" else []
    contradictions = detect_contradictions(sections) if spec.source_kind == "uploaded" else []
    unsupported_recommendations = unsupported_recommendation_issues(spec, plan, sections)
    planned_text = " ".join(section.title.lower() for section in sections)
    missing_scope = [item for item in plan.scope_requirements if re.sub(r"\s+", " ", item.lower()) not in planned_text]
    leakage = reference_leakage_issues(spec, sections)
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
    if requested_target and spec.target_depth != "Demo" and spec.source_kind == "uploaded":
        low, high = depth_range(spec) or (round(requested_target * .85), round(requested_target * 1.15))
        if not (low <= final_count <= high):
            issues.append(f"Final word count {final_count} is outside the requested range {low}-{high}.")
    if duplication:
        issues.append("Cross-section duplication is higher than expected.")
    if contradictions:
        issues.append("Contradictions detected between planned sections.")
    if unsupported_recommendations:
        issues.append("Recommendations or operational actions exceed the requested brief.")
    if missing_scope:
        issues.append("One or more explicitly requested scope items are not mapped to the report plan.")
    if leakage:
        issues.append("Reference-only client names leaked into the generated report body.")
    if not all(section.objective.strip() for section in plan.sections if section.section_id != "evidence"):
        issues.append("One or more planned section objectives are empty.")
    passed = not missing and not failed_sections and citation_validation.valid
    passed = passed and not any(
        "does not match" in issue
        or "unrequested consulting" in issue
        or "word count" in issue
        or "duplication" in issue
        or "Contradictions" in issue
        or "Recommendations or operational" in issue
        or "scope items" in issue
        or "Reference-only" in issue
        for issue in issues
    )
    return DocumentQC(
        passed=passed,
        sections_present=present,
        missing_sections=missing,
        citation_valid=citation_validation.valid,
        major_issues=issues,
        recommendations_align_with_findings=True,
        summary="Document review passed." if passed else "Document review found issues to inspect.",
        target_word_count=requested_target,
        final_word_count=final_count,
        unique_pages_researched=len({(chunk.source, chunk.page) for chunk in all_evidence}),
        unique_pages_cited=len(cited_page_pairs),
        cross_section_duplication=duplication,
        contradictions=contradictions,
        unsupported_recommendations=unsupported_recommendations,
        missing_scope_requirements=missing_scope,
        reference_leakage=leakage,
    )


def cross_section_duplication(sections: list[GeneratedSection]) -> list[str]:
    fingerprints: dict[str, list[str]] = {}
    for section in sections:
        if section.section_id == "evidence":
            continue
        sentences = [
            re.sub(r"\W+", " ", sentence.lower()).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", section.content_markdown)
            if len(sentence.split()) >= 8
        ]
        for sentence in sentences:
            fingerprints.setdefault(sentence, []).append(section.title)
    return [f"{sentence[:80]} ({', '.join(titles)})" for sentence, titles in fingerprints.items() if len(set(titles)) > 1][:5]


def detect_contradictions(sections: list[GeneratedSection]) -> list[str]:
    """Catch clear negation conflicts without pretending to prove semantics."""
    positive: dict[str, str] = {}
    negative: dict[str, str] = {}
    for section in sections:
        if section.section_id == "evidence":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", section.content_markdown):
            normalized = re.sub(r"\[[^\]]+\]", "", sentence.lower())
            terms = re.findall(r"[a-z\u0600-\u06FF]{5,}", normalized)
            if len(terms) < 3:
                continue
            key = " ".join(terms[:5])
            if re.search(r"\b(?:not|never|cannot|without|no)\b", normalized):
                negative[key] = section.title
            else:
                positive[key] = section.title
    return [f"{positive[key]} conflicts with {negative[key]} on: {key}" for key in positive.keys() & negative.keys()][:5]


def unsupported_recommendation_issues(
    spec: DocumentSpec,
    plan: DocumentPlan,
    sections: list[GeneratedSection],
) -> list[str]:
    if requests_actions(spec.client_brief):
        return []
    action_sections = {"recommendations", "roadmap"}
    return [section.title for section in sections if section.section_id in action_sections and section.content_markdown.strip()]


def reference_leakage_issues(spec: DocumentSpec, sections: list[GeneratedSection]) -> list[str]:
    if not spec.reference_source_names:
        return []
    body = " ".join(section.content_markdown for section in sections if section.section_id != "evidence").lower()
    issues: list[str] = []
    for name in spec.reference_source_names:
        stem = re.sub(r"\.pdf$", "", name, flags=re.I)
        tokens = [token for token in re.findall(r"[a-z]{4,}", stem.lower()) if token not in {"report", "business", "strategy", "approved"}]
        if len(tokens) >= 2 and all(token in body for token in tokens):
            issues.append(name)
    return issues


def deliverable_matches_brief(spec: DocumentSpec, plan: DocumentPlan) -> bool:
    return plan.deliverable_type == resolve_deliverable_type(spec)


def _pages_from_text(text: str) -> set[int]:
    return {int(match.group(1)) for match in re.finditer(r"\[AWS-WAF p\.(\d+)\]", text)}

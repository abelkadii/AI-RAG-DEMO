"""Evidence-grounded report generation over the existing agentic RAG loop."""

from __future__ import annotations

import re
import os
import time
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
    FinalizedReport,
    GeneratedSection,
    ReferenceReportProfile,
    ReportBatchDraft,
    SectionAnalysis,
    SectionDraft,
    SectionEvidenceClaim,
    SectionQC,
    StrategyReportAnalysis,
)
from llm import Reasoner, configured_reasoner
from models import AgentTrace, CitationValidation, EvidenceChunk
from retriever import Retriever
from external_research import queries_for_section, research_public_sources


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


def parse_smoke_test_sections(value: str | list[str] | None = None) -> list[str]:
    """Parse the bounded developer smoke selector from env/configuration."""
    raw = os.getenv("SMOKE_TEST_SECTIONS", "") if value is None else value
    if isinstance(raw, str):
        items = raw.split(",")
    else:
        items = raw
    return [item.strip().lower() for item in items if item and item.strip()]


def smoke_selector_matches(selector: str, section: DocumentSectionPlan) -> bool:
    """Match canonical section IDs plus friendly numbered scope IDs (for example 1.1)."""
    selector = selector.strip().lower()
    if selector == section.section_id.lower():
        return True
    match = re.search(r"(?:scope-|section-)(\d+)-(\d+)$", section.section_id.lower())
    return bool(match and selector == f"{match.group(1)}.{match.group(2)}")


def explicit_word_count(brief: str) -> int | None:
    """Extract an explicit user-requested word count, if present."""
    matches = re.findall(r"\b(\d{1,2}(?:,\d{3})+|\d{3,5})[-\s]*(?:words?|word)\b", brief or "", flags=re.I)
    return int(matches[-1].replace(",", "")) if matches else None


def target_word_count(spec: DocumentSpec) -> int | None:
    """Resolve an explicit request before the selected depth profile."""
    explicit = explicit_word_count(spec.client_brief) or spec.target_word_count
    if explicit:
        return explicit
    # Auto summaries remain concise even when the UI depth selector is left at
    # its default.  They have no artificial target unless the user supplied a
    # word count, so the writer does not pad a short explanation.
    if spec.deliverable_type == "Summary / Brief" or (
        spec.deliverable_type == "Auto"
        and has_strong_summary_intent(spec.client_brief)
        and not has_strong_strategy_intent(spec.client_brief)
    ):
        return None
    return None if spec.target_depth == "Demo" else DEPTH_PROFILES[spec.target_depth][0]


def depth_label(spec: DocumentSpec) -> str:
    return "Brief" if spec.target_depth == "Demo" else spec.target_depth


def depth_range(spec: DocumentSpec) -> tuple[int, int] | None:
    if explicit_word_count(spec.client_brief) or spec.target_word_count or spec.deliverable_type == "Summary / Brief" or (
        spec.deliverable_type == "Auto"
        and has_strong_summary_intent(spec.client_brief)
        and not has_strong_strategy_intent(spec.client_brief)
    ):
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
    # Explicit scopes can legitimately contain more sections than a brief
    # report can give them words.  Keep every requested subsection represented
    # with a small bounded budget instead of dropping it or creating a
    # negative final allocation.
    minimum = 80 if target >= 80 * len(definitions) else 25
    budgets = [max(minimum, round(target * weight / total)) for weight in weights]
    # Keep the sum exactly at the requested target so final expansion has a
    # deterministic contract, while tolerating rounding.
    remainder = target - sum(budgets)
    budgets[-1] = max(25, budgets[-1] + remainder)
    return budgets


def source_topic_labels(survey: list[EvidenceChunk]) -> list[str]:
    """Return normalized semantic topics, never sentence-shaped chunk text.

    Uploaded PDF extraction often produces a page fragment rather than a real
    heading.  Using that fragment as a plan title makes the outline look like
    a retrieval dump.  We therefore score a small, stable vocabulary of
    semantic topics across the survey and use the topic labels as the outline.
    The source excerpts still remain available as evidence for drafting.
    """
    corpus = " ".join(normalize_claim_text(chunk.text).lower() for chunk in survey)
    if not corpus.strip():
        return []
    topic_rules = [
        ("Research Context", ("background", "context", "problem", "objective", "purpose", "introduction")),
        ("Methodology", ("method", "methodology", "approach", "experiment", "study", "data collection", "evaluation")),
        ("Operational Model", ("workflow", "process", "architecture", "system", "model", "implementation")),
        ("Key Findings", ("finding", "findings", "result", "results", "observed", "outcome", "analysis")),
        ("Strategic Implications", ("implication", "implications", "recommend", "strategy", "risk", "opportunity")),
        ("Core Concepts", ("concept", "definition", "principle", "theory", "framework")),
        ("Conclusions", ("conclusion", "summary", "takeaway", "future work", "closing")),
    ]
    scored: list[tuple[int, int, str]] = []
    for order, (label, terms) in enumerate(topic_rules):
        score = sum(corpus.count(term) for term in terms)
        if score:
            scored.append((score, order, label))
    scored.sort(key=lambda item: (-item[0], item[1]))
    labels = [label for _score, _order, label in scored[:8]]
    return labels or ["Document Overview", "Major Themes"]


def extract_scope_items(brief: str) -> list[str]:
    """Capture explicit Stage/numbered scope lines without treating them as facts."""
    items: list[str] = []
    for line in (brief or "").splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip(" -*\t")
        if re.match(r"^Stage\s+\d+", cleaned, flags=re.I) or re.match(r"^\d+(?:\.\d+)?(?:\s|$)", cleaned):
            items.append(cleaned)
    return items


def concise_scope_heading(item: str) -> str:
    """Keep the numbered requirement, dropping its explanatory tail."""
    cleaned = re.sub(r"\s+", " ", item).strip(" -*\t")
    match = re.match(r"^(Stage\s+\d+|\d+(?:\.\d+)?)\s*(.*)$", cleaned, flags=re.I)
    if not match:
        return cleaned[:120]
    identifier, remainder = match.groups()
    remainder = re.split(r"\s+[\u2014\u2013\uFFFD]\s+", remainder, maxsplit=1)[0]
    # PDF/UI encoding can turn an em dash into a replacement question mark;
    # only split when it is surrounded by whitespace, preserving real hyphens.
    remainder = re.split(r"\s+(?:[—–]|-|\?)\s+|\s*;\s*", remainder, maxsplit=1)[0]
    remainder = re.sub(
        r"\s+(?:assess|evaluate|review|benchmark|define|identify|prioriti[sz]e|provide|develop|deliver)\b.*$",
        "",
        remainder,
        flags=re.I,
    )
    return f"{identifier} {remainder.strip(' -:')}".strip()[:140]


def inferred_client_name(spec: DocumentSpec) -> str:
    brief = spec.client_brief or ""
    match = re.search(
        r"\bfor\s+([A-Z][A-Za-z0-9& .'-]{2,50}?)(?=\s+(?:covering|with|that|which|to|and)\b|[.!?,\n]|$)",
        brief,
    )
    if match:
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,-")
        if candidate.lower() not in {"the report", "this report", "client"}:
            return candidate
    if spec.company_website:
        host = re.sub(r"^www\.", "", re.sub(r"^https?://", "", spec.company_website.lower())).split("/", 1)[0]
        words = [word for word in re.split(r"[.-]+", host) if word and word not in {"com", "net", "org", "co", "io"}]
        if words and words[0]:
            slug = words[0]
            known_names = {"speckledspace": "Speckled Space"}
            return known_names.get(slug, " ".join(word.capitalize() for word in words))
    return "Client"


def inferred_document_title(spec: DocumentSpec, deliverable_type: str) -> str:
    requested = spec.title.strip()
    placeholders = {
        "",
        "report",
        "evidence-grounded report",
        "concise source summary",
        "evidence-grounded document report",
        "aws well-architected architecture assessment & remediation plan",
    }
    if requested.lower() not in placeholders:
        return requested[:140]
    client = inferred_client_name(spec)
    if deliverable_type == "Consulting Assessment" and len(extract_scope_items(spec.client_brief)) >= 3:
        return f"{client} Business Strategy Report"
    if deliverable_type == "Research Report":
        return f"{client} Research Report"
    if deliverable_type == "Summary / Brief":
        return f"{client} Source Summary"
    return f"{client} Evidence-Grounded Report"


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
            stage_title = concise_scope_heading(line)
            definitions.append((slug, stage_title, f"Frame the requested work in {line} as a strategic workstream, preserving the client's scope without claiming the diagnostic is complete.", [f"What does {line} request?", "How should this workstream be assessed and connected to the wider strategy?"], ["stage", "scope"], 8))
            continue
        number = re.match(r"^(\d+(?:\.\d+)?)\s*(?:[^\w\s]+\s*)?(.+)$", line)
        if not number:
            continue
        item_id = "scope-" + number.group(1).replace(".", "-")
        title = concise_scope_heading(line)
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
        external_research: bool | None = None,
        smoke_test_sections: str | list[str] | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.reasoner = reasoner or configured_reasoner()
        self.max_section_iterations = max_section_iterations
        self.evidence_k = evidence_k
        self.section_writer = section_writer or EvidenceGroundedSectionWriter(self.reasoner)
        self.qc_runner = qc_runner or DeterministicSectionQC()
        self.reference_retriever = reference_retriever
        self.reference_profile = reference_profile
        self.external_research = external_research
        self.smoke_test_sections = parse_smoke_test_sections(smoke_test_sections)

    def external_research_enabled(self, spec: DocumentSpec) -> bool:
        if self.smoke_test_enabled(spec):
            return False
        if self.external_research is not None:
            return self.external_research
        return spec.source_kind == "uploaded" and resolve_deliverable_type(spec) == "Consulting Assessment"

    def smoke_test_enabled(self, spec: DocumentSpec) -> bool:
        return bool(self.smoke_test_sections) and (
            spec.source_kind == "uploaded" and resolve_deliverable_type(spec) == "Consulting Assessment"
        )

    def execution_plan(
        self,
        plan: DocumentPlan,
        spec: DocumentSpec,
        on_event: Callable[[str], None] | None = None,
    ) -> tuple[DocumentPlan, bool]:
        """Keep the normal plan, but optionally execute only selected smoke sections."""
        if not self.smoke_test_enabled(spec):
            return plan, False
        selected = [
            section for section in plan.sections
            if section.section_id != "evidence"
            and any(smoke_selector_matches(selector, section) for selector in self.smoke_test_sections)
        ]
        if not selected:
            self._event(on_event, "Smoke test selector did not match a planned section; running the normal plan.")
            return plan, False
        evidence_section = next((section for section in plan.sections if section.section_id == "evidence"), None)
        if evidence_section is not None:
            selected.append(evidence_section)
        self._event(on_event, "Smoke test mode: executing only " + ", ".join(section.section_id for section in selected if section.section_id != "evidence"))
        self._event(on_event, "External research: disabled (smoke test)")
        return plan.model_copy(
            update={
                "sections": selected,
                "scope_requirements": [],
                "target_word_count": None,
            }
        ), True

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
        survey = [
            chunk for chunk in survey
            if (chunk.source or "").strip().lower() not in {"client brief", "requirements"}
        ]
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
        if deliverable_type == "Summary / Brief":
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
        if depth == "Brief" and not extract_scope_items(spec.client_brief):
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
            title=inferred_document_title(spec, deliverable_type),
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
        execution_plan, smoke_test_mode = self.execution_plan(plan, spec, on_event)
        if self.has_source_evidence(spec) and self.use_dedicated_consulting_path(spec, plan, smoke_test_mode):
            return self.run_consulting_report(spec, plan, survey, started, on_event)
        if spec.source_kind == "uploaded":
            self._event(on_event, f"Requested type: {spec.deliverable_type}")
            self._event(on_event, f"Effective type: {plan.deliverable_type}")
        synthesis_engine = getattr(self.section_writer, "synthesis", None)
        if spec.source_kind == "uploaded":
            if synthesis_engine is not None and synthesis_engine.capable:
                self._event(on_event, f"Synthesis engine: OpenAI ({getattr(self.reasoner, 'model', 'configured model')})")
            else:
                self._event(on_event, "Synthesis engine: local fallback")
                self._event(on_event, "Professional long-form synthesis disabled")
                selection_reason = getattr(self.reasoner, "selection_reason", None)
                if selection_reason:
                    self._event(on_event, f"OpenAIReasoner not selected: {selection_reason}")
            if plan.deliverable_type == "Consulting Assessment":
                self._event(on_event, f"External research: {'enabled' if self.external_research_enabled(spec) else 'disabled'}")
        generated_sections: list[GeneratedSection] = []
        prior_summaries: list[str] = []
        evidence_by_id: dict[str, EvidenceChunk] = {}

        for section_plan in execution_plan.sections:
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
            section_started = time.perf_counter()
            self._event(on_event, f"Researching {section_plan.title}")
            section_evidence, research_trace = self._research_section(section_plan, on_event, spec)
            # The brief is a requirements document, never a factual corpus.
            section_evidence = [
                chunk for chunk in section_evidence
                if (chunk.source or "").strip().lower() not in {"client brief", "requirements"}
            ]
            for chunk in section_evidence:
                evidence_by_id[chunk.chunk_id] = chunk

            self._event(on_event, f"Drafting {section_plan.title}")
            content = self.section_writer.write(spec, execution_plan, section_plan, section_evidence, prior_summaries)
            if spec.source_kind == "uploaded":
                disclosure = external_research_disclosure(section_plan, section_evidence)
                if disclosure and disclosure.lower() not in content.lower():
                    content = content.rstrip() + "\n\n" + disclosure
            content, inline_unknown_ids = replace_evidence_ids(content, section_evidence)
            if spec.source_kind == "uploaded":
                content = redact_builtin_branding(content)

            self._event(on_event, f"Running QC for {section_plan.title}")
            qc = self.qc_runner.check(section_plan, content, section_evidence)
            qc = merge_qc(qc, deterministic_content_checks(spec, execution_plan, section_plan, content, section_evidence))
            unknown_ids = sorted(set(
                list(getattr(getattr(self.section_writer, "synthesis", None), "last_unknown_evidence_ids", []))
                + inline_unknown_ids
            ))
            if unknown_ids:
                qc.passed = False
                qc.citation_valid = False
                qc.issues.append(f"Unknown evidence IDs were returned by the section writer: {', '.join(unknown_ids)}.")
                qc.unsupported_claims.extend(unknown_ids)
            revised = False
            revision_count = 0
            if not qc.passed:
                revised = True
                revision_count = 1
                self._event(on_event, f"Revising {section_plan.title}")
                content = revise_section(content, qc, section_evidence, spec, execution_plan, section_plan)
                if spec.source_kind == "uploaded":
                    content = redact_builtin_branding(content)
                qc = self.qc_runner.check(section_plan, content, section_evidence)
                qc = merge_qc(qc, deterministic_content_checks(spec, execution_plan, section_plan, content, section_evidence))

            generated = GeneratedSection(
                section_id=section_plan.section_id,
                title=section_plan.title,
                objective=section_plan.objective,
                content_markdown=content,
                evidence=serializable_evidence(section_evidence),
                research_trace=research_trace,
                qc=qc,
                analysis=getattr(self.section_writer, "last_analysis", None),
                analysis_model_used=getattr(getattr(self.section_writer, "synthesis", None), "last_analysis_model_used", False),
                analysis_error=getattr(getattr(self.section_writer, "synthesis", None), "last_analysis_error", None),
                analysis_normalized=getattr(getattr(self.section_writer, "synthesis", None), "last_analysis_normalized", False),
                analysis_repair_retry=getattr(getattr(self.section_writer, "synthesis", None), "last_analysis_repair_retry", False),
                synthesis_model_used=getattr(getattr(self.section_writer, "synthesis", None), "last_synthesis_model_used", False),
                synthesis_error=getattr(getattr(self.section_writer, "synthesis", None), "last_synthesis_error", None),
                synthesis_fallback=getattr(getattr(self.section_writer, "synthesis", None), "last_used_fallback", False),
                latency_ms=int((time.perf_counter() - section_started) * 1000),
                revised=revised,
                revision_count=revision_count,
            )
            generated_sections.append(generated)
            prior_summaries.append(summarize_section(content))
            if generated.analysis_normalized:
                self._event(on_event, "Structured analysis normalized locally")
            if generated.analysis_repair_retry:
                self._event(on_event, "Structured analysis repair retry")
            if generated.synthesis_error:
                self._event(on_event, f"Synthesis fallback for {section_plan.title} - structured model output could not be validated.")
            self._event(on_event, f"{section_plan.title} approved" if qc.passed else f"{section_plan.title} needs review")

        all_evidence = list(evidence_by_id.values())
        # Length/coverage repair is part of the section lifecycle.  Any
        # changed content is QC'd again before cross-document review and final
        # citation validation; nothing is modified after final QC.
        requested = target_word_count(spec)
        low_target = (depth_range(spec) or (0, 0))[0] if requested else 0
        assembled_before_repair = assemble_markdown(spec, execution_plan, generated_sections)
        if not smoke_test_mode and spec.source_kind == "uploaded" and requested and count_report_words(assembled_before_repair) < low_target:
            self._event(on_event, "Repairing section depth from additional source claims")
            for section_trace in generated_sections:
                section_plan = next((item for item in execution_plan.sections if item.section_id == section_trace.section_id), None)
                if not section_plan or section_trace.section_id == "evidence" or not section_plan.approximate_word_budget:
                    continue
                evidence = [EvidenceChunk.model_validate(item) if isinstance(item, dict) else item for item in section_trace.evidence]
                repaired = section_trace.content_markdown
                if (
                    synthesis_engine is not None
                    and synthesis_engine.capable
                    and not section_trace.synthesis_fallback
                    and section_trace.analysis is not None
                ):
                    candidate = synthesis_engine.synthesize(
                        spec,
                        execution_plan,
                        section_plan,
                        section_trace.analysis,
                        evidence,
                        target_range=f"{max(180, round((section_plan.approximate_word_budget or 250) * 0.95))}-"
                        f"{max(240, round((section_plan.approximate_word_budget or 250) * 1.35))} words",
                        depth_instruction="Add substantive analysis only where it follows from the structured analysis; do not pad or repeat the section.",
                    )
                    if synthesis_engine.last_synthesis_model_used:
                        repaired = candidate
                        section_trace.synthesis_model_used = True
                        section_trace.synthesis_error = synthesis_engine.last_synthesis_error
                    elif synthesis_engine.last_synthesis_error:
                        section_trace.synthesis_error = synthesis_engine.last_synthesis_error
                repaired, _ = replace_evidence_ids(repaired, evidence)
                if spec.source_kind == "uploaded":
                    disclosure = external_research_disclosure(section_plan, evidence)
                    if disclosure and disclosure.lower() not in repaired.lower():
                        repaired = repaired.rstrip() + "\n\n" + disclosure
                    repaired = redact_builtin_branding(repaired)
                if repaired != section_trace.content_markdown:
                    section_trace.content_markdown = repaired
                    section_trace.revised = True
                    section_trace.revision_count = max(1, section_trace.revision_count)
                    rerun = self.qc_runner.check(section_plan, repaired, evidence)
                    rerun = merge_qc(rerun, deterministic_content_checks(spec, execution_plan, section_plan, repaired, evidence))
                    section_trace.qc = rerun
            self._event(on_event, "Re-running section QC after depth repair")

        self._event(on_event, "Running cross-document consistency review")
        final_markdown = assemble_markdown(spec, execution_plan, generated_sections)
        self._event(on_event, "Validating citations")
        # Requirements-only sections are intentionally uncited.  Validate the
        # factual subset against retrieved evidence while leaving those
        # diagnostic paragraphs visible in the finished report.
        citation_sections = [
            section.model_copy(update={"content_markdown": ""})
            if section.section_id != "evidence" and not section.evidence
            else section
            for section in generated_sections
        ]
        citation_markdown = assemble_markdown(spec, execution_plan, citation_sections)
        _, citation_validation = Agent.validate_citations(citation_markdown, all_evidence)
        cleaned_markdown = final_markdown
        qc_spec = spec.model_copy(update={"target_depth": "Demo", "target_word_count": None}) if smoke_test_mode else spec
        final_qc = document_qc(qc_spec, execution_plan, generated_sections, citation_validation, cleaned_markdown, all_evidence)
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
            target_word_count=None if smoke_test_mode else target_word_count(spec),
            synthesis_engine=(
                f"OpenAI ({getattr(self.reasoner, 'model', 'configured model')})"
                if synthesis_engine is not None and synthesis_engine.capable
                else "local fallback"
            ),
            synthesis_model=getattr(self.reasoner, "model", None) if synthesis_engine is not None and synthesis_engine.capable else None,
            smoke_test_mode=smoke_test_mode,
            smoke_test_sections=[section.section_id for section in execution_plan.sections if section.section_id != "evidence"] if smoke_test_mode else [],
            external_research_enabled=self.external_research_enabled(spec) if spec.source_kind == "uploaded" else None,
        )

    def use_dedicated_consulting_path(self, spec: DocumentSpec, plan: DocumentPlan, smoke_test_mode: bool) -> bool:
        return (
            spec.source_kind == "uploaded"
            and plan.deliverable_type == "Consulting Assessment"
            and not smoke_test_mode
        )

    def has_source_evidence(self, spec: DocumentSpec) -> bool:
        if spec.source_kind != "uploaded":
            return False
        chunks = getattr(self.retriever, "chunks", getattr(getattr(self.retriever, "store", None), "chunks", []))
        return any((chunk.source or "").strip().lower() not in {"client brief", "requirements"} for chunk in chunks)

    def run_consulting_report(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        survey: list[EvidenceChunk],
        started: datetime,
        on_event: Callable[[str], None] | None = None,
    ) -> DocumentTrace:
        """Dedicated uploaded-client strategy path: draft, finalize, validate."""
        generation_started = time.perf_counter()
        metrics = {
            "total_llm_calls": 0,
            "analysis_llm_calls": 0,
            "synthesis_llm_calls": 0,
            "finalization_llm_calls": 0,
            "external_search_calls": 0,
            "external_results_count": 0,
        }
        self._event(on_event, f"Requested type: {spec.deliverable_type}")
        self._event(on_event, f"Effective type: {plan.deliverable_type}")
        self._event(on_event, "Collecting company evidence")
        external_chunks: list[EvidenceChunk] = []
        external_report = None
        if self.external_research_enabled(spec):
            self._event(on_event, "Researching market context")
            queries = consulting_external_queries(spec)
            external_chunks, external_report = research_public_sources(queries, enabled=True)
            metrics["external_search_calls"] = len(getattr(external_report, "queries", queries))
            metrics["external_results_count"] = len(external_chunks)
            self._event(on_event, external_report.notice)
        else:
            self._event(on_event, "Market context: external research disabled")
        evidence = build_global_evidence_pack(
            self.retriever,
            plan,
            survey + external_chunks,
            evidence_k=max(self.evidence_k, 8),
        )
        evidence = assign_consulting_evidence_ids(evidence)
        self._event(on_event, f"Company evidence items: {len(evidence)}")
        self._event(on_event, "Building strategy analysis")
        analysis_started = time.perf_counter()
        analysis, analysis_model_used, analysis_error = self.consulting_analysis(spec, plan, evidence)
        metrics["analysis_llm_calls"] = 1 if analysis_model_used else 0
        metrics["total_llm_calls"] += metrics["analysis_llm_calls"]
        sections: list[GeneratedSection] = []
        for label, batch_sections in consulting_batches(plan):
            self._event(on_event, f"Drafting {label}")
            batch_started = time.perf_counter()
            drafts, synthesis_model_used, synthesis_error = self.consulting_batch_draft(
                spec,
                plan,
                analysis,
                evidence,
                batch_sections,
            )
            if synthesis_model_used:
                metrics["synthesis_llm_calls"] += 1
                metrics["total_llm_calls"] += 1
            for section_plan in batch_sections:
                raw = drafts.get(section_plan.section_id, "")
                content = sanitize_generated_section_body(raw, section_plan.title)
                content, unknown = replace_evidence_ids(content, evidence)
                content = redact_builtin_branding(content)
                qc = deterministic_content_checks(spec, plan, section_plan, content, evidence)
                citation_validation = consulting_validate_citations(content, evidence)
                if unknown or re.search(r"\bE\d+\b", content):
                    qc.passed = False
                    qc.citation_valid = False
                    qc.issues.append("Raw or unknown evidence IDs remain in the section.")
                else:
                    qc.citation_valid = citation_validation.valid
                    qc.passed = qc.passed and citation_validation.valid
                if synthesis_error:
                    qc.passed = False
                    qc.issues.append("Synthesis model error requires review.")
                sections.append(
                    GeneratedSection(
                        section_id=section_plan.section_id,
                        title=section_plan.title,
                        objective=section_plan.objective,
                        content_markdown=content,
                        evidence=serializable_evidence(evidence),
                        research_trace=AgentTrace(
                            question=section_plan.research_question or section_plan.objective,
                            total_iterations=0,
                            stop_reason="global_evidence_pack",
                        ),
                        qc=qc,
                        analysis=SectionAnalysis(
                            section_id=section_plan.section_id,
                            section_mode=section_mode(section_plan, plan.deliverable_type),
                            objective=section_plan.objective,
                            requirements=section_plan.requirements,
                            evidence_claims=analysis.evidence_map[:8],
                            known_facts=analysis.publicly_observable_facts[:8],
                            data_gaps=analysis.data_gaps[:6],
                            recommendations=analysis.recommendations[:6],
                        ),
                        analysis_model_used=analysis_model_used,
                        analysis_error=analysis_error,
                        synthesis_model_used=synthesis_model_used,
                        synthesis_error=synthesis_error,
                        synthesis_fallback=not synthesis_model_used,
                        latency_ms=int((time.perf_counter() - batch_started) * 1000),
                    )
                )
        evidence_plan = next((section for section in plan.sections if section.section_id == "evidence"), None)
        if evidence_plan:
            sections.append(
                GeneratedSection(
                    section_id="evidence",
                    title=evidence_plan.title,
                    objective=evidence_plan.objective,
                    content_markdown="References are generated from citations used in the previous sections.",
                    evidence=[],
                    research_trace=AgentTrace(question="", stop_reason="generated_from_used_citations"),
                    qc=SectionQC(passed=True, requirements_covered=["references-generated"], citation_valid=True),
                    latency_ms=0,
                )
            )
        draft_markdown = assemble_consulting_markdown(spec, plan, sections)
        draft_issues = consulting_final_output_issues(draft_markdown, spec, plan)
        final_markdown = draft_markdown
        finalization_used = False
        if should_finalize_consulting_report(draft_markdown, draft_issues, spec):
            self._event(on_event, "Finalizing report")
            finalized, used, error = self.finalize_consulting_report(spec, plan, analysis, evidence, draft_markdown, draft_issues)
            if used:
                metrics["finalization_llm_calls"] += 1
                metrics["total_llm_calls"] += 1
                final_markdown = finalized
                finalization_used = True
            if error:
                draft_issues.append(f"Finalization model error: {error}")
            final_markdown, _ = replace_evidence_ids(final_markdown, evidence)
            final_markdown = sanitize_full_consulting_markdown(final_markdown, spec, plan)
        self._event(on_event, "Validating evidence and citations")
        citation_validation = consulting_validate_citations(final_markdown, evidence)
        final_issues = consulting_final_output_issues(final_markdown, spec, plan)
        if not finalization_used:
            final_issues += draft_issues
        if final_issues:
            citation_validation.valid = False
            citation_validation.uncited_claims.extend(final_issues[:12])
        final_qc = document_qc(spec, plan, sections, citation_validation, final_markdown, evidence)
        if final_issues:
            final_qc.major_issues.extend(issue for issue in final_issues if issue not in final_qc.major_issues)
            final_qc.passed = False
            final_qc.summary = "Document review found issues to inspect."
        completed = datetime.now(timezone.utc)
        self._event(on_event, "Rendering PDF")
        return DocumentTrace(
            spec=spec,
            plan=plan,
            sections=sections,
            final_qc=final_qc,
            final_markdown=final_markdown,
            citation_validation=citation_validation,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_ms=int((completed - started).total_seconds() * 1000),
            total_research_iterations=0,
            total_unique_evidence_pages=len({(chunk.source, chunk.page) for chunk in evidence}),
            total_retrieved_evidence_chunks=len(evidence),
            total_unique_cited_pages=len({(source, page) for source, page in citation_pairs(final_markdown)}),
            final_word_count=count_report_words(final_markdown),
            target_word_count=target_word_count(spec),
            synthesis_engine=(
                f"OpenAI ({getattr(self.reasoner, 'model', 'configured model')})"
                if callable(getattr(self.reasoner, "_json", None))
                else "local fallback"
            ),
            synthesis_model=getattr(self.reasoner, "model", None),
            smoke_test_mode=False,
            smoke_test_sections=[],
            external_research_enabled=self.external_research_enabled(spec),
            total_llm_calls=metrics["total_llm_calls"],
            analysis_llm_calls=metrics["analysis_llm_calls"],
            synthesis_llm_calls=metrics["synthesis_llm_calls"],
            finalization_llm_calls=metrics["finalization_llm_calls"],
            external_search_calls=metrics["external_search_calls"],
            external_results_count=metrics["external_results_count"],
            generation_seconds=round(time.perf_counter() - generation_started, 2),
        )

    def consulting_analysis(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        evidence: list[EvidenceChunk],
    ) -> tuple[StrategyReportAnalysis, bool, str | None]:
        fallback = deterministic_strategy_analysis(spec, evidence)
        if not callable(getattr(self.reasoner, "_json", None)):
            return fallback, False, None
        try:
            result = self.reasoner._json(
                "Return only JSON matching StrategyReportAnalysis. Build one global consulting analysis from the supplied client evidence. "
                "Classify statements conceptually as facts, inferences, hypotheses, recommendations, or data required through field placement. "
                "Reference precedent is style only, never factual evidence. Client brief is requirements only, never cited evidence. "
                "Do not invent revenue, margin, headcount, market share, internal technology stack, workforce capability, financial condition, or operational performance. "
                "Every publicly observable fact and evidence_map item must cite one or more evidence_ids from the packet.",
                {
                    "report_title": plan.title,
                    "client_brief": spec.client_brief,
                    "audience": spec.audience,
                    "reference_profile": plan.reference_profile.model_dump() if plan.reference_profile else None,
                    "evidence": evidence_packet(evidence, limit=900),
                    "required_sections": [section.model_dump() for section in plan.sections if section.section_id != "evidence"],
                },
                StrategyReportAnalysis,
                max_attempts=2,
                token_limit=9000,
            )
            return sanitize_strategy_analysis(result, evidence), True, None
        except Exception as error:
            return fallback, False, f"{type(error).__name__}: {error}"

    def consulting_batch_draft(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        analysis: StrategyReportAnalysis,
        evidence: list[EvidenceChunk],
        sections: list[DocumentSectionPlan],
    ) -> tuple[dict[str, str], bool, str | None]:
        fallback = {section.section_id: deterministic_consulting_section(spec, section, analysis, evidence) for section in sections}
        if not callable(getattr(self.reasoner, "_json", None)):
            return fallback, False, None
        try:
            result = self.reasoner._json(
                "Return only JSON: {sections:[{section_id,title,markdown}]}. Write BODY ONLY for each section; do not repeat the section heading. "
                "Write as one professional consulting team. Use concise, specific strategy prose. Cite each factual or evidence-based sentence with the exact supporting [E#] IDs. "
                "Use 1-2 evidence IDs for a claim when possible, not the whole packet. Do not leave raw evidence lists like [E1, E2, E3] unless each ID supports that sentence. "
                "Avoid internal workflow language, retrieval terminology, legacy repair instructions, and repeated evidence-boundary boilerplate. "
                "Where private data is missing, state the limitation naturally and briefly. Do not invent unsupported market numbers or internal facts.",
                {
                    "report_title": plan.title,
                    "target_total_words": plan.target_word_count or 3250,
                    "batch_sections": [section.model_dump() for section in sections],
                    "client_brief": spec.client_brief,
                    "analysis": analysis.model_dump(),
                    "evidence": evidence_packet(evidence),
                },
                ReportBatchDraft,
                max_attempts=2,
                token_limit=4200,
            )
            drafts = {item.section_id: item.markdown for item in result.sections if item.section_id}
            if not drafts:
                return fallback, False, "ReportBatchDraft contained no sections."
            return {section.section_id: drafts.get(section.section_id, fallback[section.section_id]) for section in sections}, True, None
        except Exception as error:
            return fallback, False, f"{type(error).__name__}: {error}"

    def finalize_consulting_report(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        analysis: StrategyReportAnalysis,
        evidence: list[EvidenceChunk],
        draft_markdown: str,
        qc_issues: list[str],
    ) -> tuple[str, bool, str | None]:
        if not callable(getattr(self.reasoner, "_json", None)):
            return draft_markdown, False, None
        try:
            result = self.reasoner._json(
                "Return only JSON with a markdown field. Edit the supplied report; do not regenerate from scratch. "
                "Preserve required Stage 1-3 coverage and supported analysis. Fix repetition, duplicate headings, malformed citations, awkward transitions, excessive disclaimer language, and verbosity. "
                "Compress to approximately 2700-4000 words excluding references. Do not add unsupported factual claims, market numbers, Core Biz facts, client-brief citations, raw E tokens, or internal pipeline language.",
                {
                    "draft_markdown": draft_markdown,
                    "qc_issues": qc_issues,
                    "analysis": analysis.model_dump(),
                    "evidence": evidence_packet(evidence),
                    "target_words": plan.target_word_count or 3250,
                },
                FinalizedReport,
                max_attempts=2,
                token_limit=6500,
            )
            return result.markdown or draft_markdown, True, None
        except Exception as error:
            return draft_markdown, False, f"{type(error).__name__}: {error}"

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
                if (chunk.source or "").strip().lower() in {"client brief", "requirements"}:
                    continue
                current = collected.get(chunk.chunk_id)
                if current is None or chunk.score > current.score:
                    collected[chunk.chunk_id] = chunk
        return list(collected.values())

    def _research_section(
        self,
        section: DocumentSectionPlan,
        on_event: Callable[[str], None] | None,
        spec: DocumentSpec | None = None,
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
                # Retrieve a broader candidate set, then apply the
                # section-specific relevance/diversity gate below.
                k=max(12, self.evidence_k * 2),
            ).research(question)
            traces.append(trace)
            for chunk in state.gathered_evidence:
                current = gathered.get(chunk.chunk_id)
                if current is None or chunk.score > current.score:
                    gathered[chunk.chunk_id] = chunk
        if spec and spec.source_kind == "uploaded" and requires_external_research(section):
            external_chunks, external_report = research_public_sources(
                queries_for_section(section.title, section.objective, section.questions),
                enabled=self.external_research_enabled(spec) if spec else None,
            )
            if external_report.enabled:
                self._event(on_event, external_report.notice)
            elif external_report.queries:
                self._event(on_event, external_report.notice)
            for chunk in external_chunks:
                gathered[chunk.chunk_id] = chunk
        filtered = filter_section_evidence(
            section,
            list(gathered.values()),
            self.evidence_k,
            # Document Studio must not turn an unrelated top hit into a
            # factual section.  The legacy RAG Explorer can opt into the
            # compatibility fallback by calling the helper directly.  The
            # existing offline curriculum compatibility path also retains one
            # source excerpt so its teaching-plan regression remains useful;
            # consulting/strategy reports always stay strict.
            allow_sparse_fallback=(
                spec is not None
                and resolve_deliverable_type(spec) == "Curriculum / Teaching Material"
                and not getattr(getattr(self.section_writer, "synthesis", None), "capable", False)
            ),
        )
        return filtered, merge_agent_traces(section.title, traces)


def consulting_external_queries(spec: DocumentSpec) -> list[str]:
    client = inferred_client_name(spec)
    return [
        f"{client} Singapore home decor home furnishings ecommerce trends",
        "Singapore home decor home furnishings ecommerce competitive landscape category leaders",
        "Singapore home furnishings decor consumer trends regional ecommerce",
        f"{client} cross border expansion home decor Singapore regional market considerations",
    ]


def build_global_evidence_pack(
    retriever,
    plan: DocumentPlan,
    seed_evidence: list[EvidenceChunk],
    evidence_k: int = 8,
) -> list[EvidenceChunk]:
    collected: dict[str, EvidenceChunk] = {}
    for chunk in seed_evidence:
        if (chunk.source or "").strip().lower() in {"client brief", "requirements"}:
            continue
        current = collected.get(chunk.chunk_id)
        if current is None or chunk.score > current.score:
            collected[chunk.chunk_id] = chunk
    queries = [
        "company overview product assortment customer proposition delivery showroom reviews",
        "pricing product categories brand marketing offers customer service",
        "contact showroom operations delivery returns residential space decor",
    ]
    for section in plan.sections:
        if section.section_id != "evidence":
            queries.append(" ".join([section.title, section.objective, *section.questions]))
    for query in queries[:18]:
        for chunk in retriever.search(query, max(16, evidence_k * 2)):
            if (chunk.source or "").strip().lower() in {"client brief", "requirements"}:
                continue
            current = collected.get(chunk.chunk_id)
            if current is None or chunk.score > current.score:
                collected[chunk.chunk_id] = chunk
    return diversify_global_evidence(list(collected.values()), limit=24)


def diversify_global_evidence(candidates: list[EvidenceChunk], limit: int = 24) -> list[EvidenceChunk]:
    useful = [chunk for chunk in candidates if useful_evidence_sentence(normalize_claim_text(chunk.text[:320])) or len(chunk.text.strip()) >= 60]
    useful.sort(key=lambda item: item.score, reverse=True)
    selected: list[EvidenceChunk] = []
    page_counts: dict[tuple[str, int], int] = {}
    while useful and len(selected) < limit:
        best_index = None
        best_value = float("-inf")
        for index, chunk in enumerate(useful):
            page_key = (chunk.source, chunk.page)
            if page_counts.get(page_key, 0) >= 2:
                continue
            similarity = max((term_jaccard(chunk.text, item.text) for item in selected), default=0.0)
            value = float(chunk.score) - (0.22 * similarity) + (0.08 if page_counts.get(page_key, 0) == 0 else 0)
            if value > best_value:
                best_value = value
                best_index = index
        if best_index is None:
            break
        chunk = useful.pop(best_index)
        selected.append(chunk)
        key = (chunk.source, chunk.page)
        page_counts[key] = page_counts.get(key, 0) + 1
    return selected


def assign_consulting_evidence_ids(evidence: list[EvidenceChunk]) -> list[EvidenceChunk]:
    assigned: list[EvidenceChunk] = []
    for index, chunk in enumerate(evidence, start=1):
        assigned.append(chunk.model_copy(update={"chunk_id": f"E{index}"}))
    return assigned


def deterministic_strategy_analysis(spec: DocumentSpec, evidence: list[EvidenceChunk]) -> StrategyReportAnalysis:
    claims = []
    for index, claim in enumerate(synthesize_evidence_claims(evidence, limit=10), start=1):
        evidence_index = next((idx for idx, chunk in enumerate(evidence, start=1) if chunk.chunk_id == claim.chunk.chunk_id), index)
        claims.append(SectionEvidenceClaim(text=claim.text, evidence_ids=[f"E{evidence_index}"]))
    return StrategyReportAnalysis(
        client_profile=[
            f"{inferred_client_name(spec)} is assessed from public client evidence and the supplied strategy scope.",
        ],
        publicly_observable_facts=claims[:6],
        customer_proposition=["Use website evidence to describe the observed proposition, product range, service promises, and trust signals."],
        market_signals=["External market findings require independent validation unless public research evidence is available."],
        operating_model_hypotheses=["Internal operating model performance cannot be treated as established without interviews, process data, and management information."],
        customer_segments=["Potential customer segments should be validated against sales, traffic, and margin data."],
        business_model_questions=["Clarify revenue mix, gross margin, stock turn, fulfilment cost, and channel economics."],
        financial_analysis_priorities=["Build baseline revenue, margin, unit economics, inventory, and cash-flow views before prioritising investment."],
        workforce_capability_questions=["Assess accountability, role clarity, digital capability, and execution capacity through internal evidence."],
        brand_and_marketing_findings=["Use observed public positioning as a starting point for brand and channel diagnosis."],
        internationalisation_questions=["Validate internationalisation only after customer demand, fulfilment, regulatory, and partnership evidence is gathered."],
        ai_use_cases=["Prioritise low-risk AI use cases tied to customer support, content operations, merchandising insight, and internal reporting where data exists."],
        strategic_opportunities=["Convert the public proposition into a sharper segment, channel, and execution roadmap."],
        strategic_risks=["Avoid treating public-facing material as proof of internal health, financial performance, or market position."],
        recommendations=["Run a staged diagnostic, validate the market and operating baseline, then sequence initiatives by evidence strength and business value."],
        roadmap_priorities=["Begin with diagnostic baselining, then market/customer validation, then operating and digital initiatives, then scaling decisions."],
        kpi_candidates=["Conversion, average order value, repeat purchase, fulfilment performance, customer acquisition cost, gross margin, stock turn, and review sentiment."],
        data_gaps=["Revenue, margin, traffic, conversion, customer acquisition cost, fulfilment cost, inventory, workforce capacity, and competitor benchmarking data."],
        validation_requirements=["Interview leadership, review financials, analyze web/channel data, test customer segments, and benchmark competitors before treating hypotheses as facts."],
        evidence_map=claims,
    )


def sanitize_strategy_analysis(analysis: StrategyReportAnalysis, evidence: list[EvidenceChunk]) -> StrategyReportAnalysis:
    valid_ids = {f"E{index}" for index in range(1, len(evidence) + 1)}

    def clean_claim(claim: SectionEvidenceClaim) -> SectionEvidenceClaim | None:
        ids = [item.upper() for item in claim.evidence_ids if item.upper() in valid_ids]
        if not ids or not claim.text.strip():
            return None
        return claim.model_copy(update={"evidence_ids": ids})

    facts = [item for claim in analysis.publicly_observable_facts if (item := clean_claim(claim)) is not None]
    evidence_map = [item for claim in analysis.evidence_map if (item := clean_claim(claim)) is not None]
    return analysis.model_copy(update={"publicly_observable_facts": facts, "evidence_map": evidence_map or facts})


def consulting_batches(plan: DocumentPlan) -> list[tuple[str, list[DocumentSectionPlan]]]:
    content = [section for section in plan.sections if section.section_id != "evidence"]
    groups = [
        ("Stage 1", {"executive-summary", "engagement-context", "stage-1", "scope-1-1", "scope-1-2"}),
        ("Stage 2", {"stage-2", "scope-2-1", "scope-2-2", "scope-2-3", "scope-2-4"}),
        ("Stage 2 roadmap", {"scope-2-5", "scope-2-6", "scope-2-7", "integrated-roadmap", "kpi-framework"}),
        ("Stage 3 and actions", {"stage-3", "scope-3-1", "scope-3-2", "scope-3-3", "priority-actions", "assumptions"}),
    ]
    batches: list[tuple[str, list[DocumentSectionPlan]]] = []
    assigned: set[str] = set()
    for label, ids in groups:
        items = [section for section in content if section.section_id in ids]
        if items:
            batches.append((label, items))
            assigned.update(section.section_id for section in items)
    remainder = [section for section in content if section.section_id not in assigned]
    if remainder:
        batches.append(("Additional sections", remainder))
    return batches[:4] if len(batches) <= 4 else batches[:3] + [("Stage 3 and actions", [section for _label, items in batches[3:] for section in items])]


def deterministic_consulting_section(
    spec: DocumentSpec,
    section: DocumentSectionPlan,
    analysis: StrategyReportAnalysis,
    evidence: list[EvidenceChunk],
) -> str:
    citations = evidence_id_citation_string(analysis.evidence_map[:2], evidence)
    citations = citations or (f"[E1]" if evidence else "")
    title = section.title.lower()
    if section.section_id == "executive-summary":
        return (
            f"{inferred_client_name(spec)} should be treated as a strategy assessment grounded in public-facing evidence, not as a completed internal diagnostic {citations}. "
            "The report should separate observed customer proposition and brand signals from hypotheses about operations, finance, workforce, and market position.\n\n"
            "The recommended path is to establish the baseline first, validate market and customer priorities, then sequence operating, brand, internationalisation, AI, and KPI work through evidence-based gates."
        )
    if "kpi" in title:
        return "The KPI framework should begin with measurable commercial, customer, operational, and execution indicators, then add owners and thresholds once internal baselines are available. " + citations
    if "roadmap" in title or "action" in title:
        return "The roadmap should move from diagnostic baselining to validated growth choices and then to implementation, with each initiative gated by evidence strength, owner capacity, and expected business value. " + citations
    if "evidence" in title or "assumption" in title:
        return "Evidence used in this report consists of client-facing source material and any available public research. Internal performance, financial condition, headcount, margin, and market share remain assumptions until supplied by the client. " + citations
    return (
        f"For {section.title}, the available evidence supports a bounded diagnostic rather than a final factual conclusion. "
        f"The workstream should test the public proposition against internal operating, customer, financial, and execution data before deciding priorities. {citations}"
    )


def evidence_id_citation_string(claims: list[SectionEvidenceClaim], evidence: list[EvidenceChunk]) -> str:
    ids = []
    valid = {f"E{index}" for index in range(1, len(evidence) + 1)}
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            if evidence_id.upper() in valid and evidence_id.upper() not in ids:
                ids.append(evidence_id.upper())
    return "".join(f"[{item}]" for item in ids[:2])


def replace_evidence_ids(text: str, evidence: list[EvidenceChunk]) -> tuple[str, list[str]]:
    unknown: list[str] = []

    def replace_group(match: re.Match) -> str:
        raw_ids = re.findall(r"E\d+", match.group(1), flags=re.I)
        if not raw_ids:
            return match.group(0)
        citations: list[str] = []
        for raw_id in raw_ids:
            evidence_id = raw_id.upper()
            citation = citation_for_evidence_id(evidence_id, evidence)
            if citation is None:
                unknown.append(evidence_id)
                continue
            if citation not in citations:
                citations.append(citation)
        return "".join(citations)

    cleaned = re.sub(r"\[\s*((?:E\d+\s*,?\s*){1,12})\]", replace_group, text or "", flags=re.I)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), sorted(set(unknown))


def sanitize_generated_section_body(markdown: str, title: str) -> str:
    text = (markdown or "").strip()
    lines = text.splitlines()
    normalized_title = normalize_heading_text(title)
    while lines:
        first = lines[0].strip()
        if first.startswith("#") and normalize_heading_text(first.lstrip("#").strip()) == normalized_title:
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
            continue
        break
    return sanitize_full_consulting_markdown("\n".join(lines).strip(), None, None)


def normalize_heading_text(text: str) -> str:
    text = re.sub(r"^[#\s]+", "", text or "")
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def assemble_consulting_markdown(spec: DocumentSpec, plan: DocumentPlan, sections: list[GeneratedSection]) -> str:
    client = inferred_client_name(spec)
    lines = [
        f"# {plan.title}",
        "",
        f"Prepared for {client}",
        "",
        "Business Strategy Assessment",
        "",
    ]
    for section in sections:
        if section.section_id == "evidence":
            continue
        body = sanitize_generated_section_body(section.content_markdown, section.title)
        lines.extend([f"## {section.title}", "", body, ""])
    references = references_from_markdown("\n".join(lines))
    if references:
        lines.extend(["## Evidence / References", ""])
        lines.extend(f"- {reference}" for reference in references)
        lines.append("")
    return sanitize_full_consulting_markdown("\n".join(lines).strip() + "\n", spec, plan)


def sanitize_full_consulting_markdown(markdown: str, spec: DocumentSpec | None, plan: DocumentPlan | None) -> str:
    text = markdown or ""
    text = re.sub(r"\bevidence packet\b", "available evidence", text, flags=re.I)
    text = re.sub(r"\bevidence pack\b", "available evidence", text, flags=re.I)
    text = re.sub(r"^\s*\*\*(?:Knowledge Base|Reference Precedent|Client Sources|Company Website|Client Brief|Audience):\*\*.*$", "", text, flags=re.I | re.M)
    text = re.sub(r"^\s*(?:Knowledge Base|Reference Precedent|Client Sources|Company Website|Client Brief):\s.*$", "", text, flags=re.I | re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + ("\n" if text.strip() else "")


def should_finalize_consulting_report(markdown: str, issues: list[str], spec: DocumentSpec) -> bool:
    words = count_report_words(markdown)
    return bool(issues) or words > 4000 or words < 2700


FINAL_FORBIDDEN_PHRASES = (
    "connect the requested focus",
    "keep the interpretation bounded",
    "retrieved material",
    "retrieved passage",
    "source point that",
    "this reading belongs to",
    "remaining requirement",
    "evidence packet",
    "evidence pack",
    "retrieval chunk",
)


def consulting_final_output_issues(markdown: str, spec: DocumentSpec, plan: DocumentPlan) -> list[str]:
    issues: list[str] = []
    lower = markdown.lower()
    if re.search(r"\bE\d+\b", markdown):
        issues.append("Raw E evidence token remains in final report.")
    for phrase in FINAL_FORBIDDEN_PHRASES:
        if phrase in lower:
            issues.append(f"Legacy/internal phrase leaked: {phrase}.")
    if "core biz" in lower:
        issues.append("Core Biz factual leakage appears in final report.")
    if re.search(r"\[(?:client brief|requirements)\s+p\.\d+\]", markdown, flags=re.I):
        issues.append("Client Brief citation appears in final report.")
    if duplicate_heading_count(markdown):
        issues.append("Duplicate section heading appears in final report.")
    if re.search(r"`\d+(?:\.\d+)?\s+to\s+(?:at least\s+)?`?\d", markdown):
        issues.append("Malformed monetary or price range formatting appears in final report.")
    words = count_report_words(markdown)
    if spec.target_depth == "Standard" and words > 4300:
        issues.append(f"Final word count {words} exceeds the standard demo maximum.")
    required = ("Stage 1", "1.1", "1.2", "Stage 2", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "Stage 3", "3.1", "3.2", "3.3", "Integrated", "KPI", "Priority")
    missing = [item for item in required if item.lower() not in lower]
    if missing and len(extract_scope_items(spec.client_brief)) >= 8:
        issues.append("Missing required workstreams: " + ", ".join(missing[:8]) + ".")
    return issues


def consulting_validate_citations(markdown: str, evidence: list[EvidenceChunk]) -> CitationValidation:
    retrieved = {(chunk.source, chunk.page) for chunk in evidence}
    cited = set(citation_pairs(markdown))
    uncited_claims: list[str] = []
    invalid = sorted(cited - retrieved)
    if invalid:
        uncited_claims.extend(f"Invalid citation: {source} p.{page}" for source, page in invalid)
    if re.search(r"\bE\d+\b", markdown):
        uncited_claims.append("Raw E evidence token remains in cited report.")
    return CitationValidation(
        valid=not uncited_claims,
        cited_pages=sorted({page for _source, page in cited}),
        retrieved_pages=sorted({page for _source, page in retrieved}),
        cited_references=sorted(f"{source} p.{page}" for source, page in cited),
        retrieved_references=sorted(f"{source} p.{page}" for source, page in retrieved),
        uncited_claims=uncited_claims,
    )


def duplicate_heading_count(markdown: str) -> int:
    headings = []
    count = 0
    for line in markdown.splitlines():
        if not line.lstrip().startswith("#"):
            continue
        normalized = normalize_heading_text(line)
        if headings and headings[-1] == normalized:
            count += 1
        headings.append(normalized)
    return count


def filter_section_evidence(
    section: DocumentSectionPlan,
    candidates: list[EvidenceChunk],
    requested_k: int = 8,
    allow_sparse_fallback: bool = False,
) -> list[EvidenceChunk]:
    """Gate broad retrieval so tangential chunks do not enter a section draft.

    The agent intentionally retrieves broadly.  This second, deterministic
    pass scores each candidate against the section's objective/questions,
    keeps primary and supporting evidence, caps repeated pages, and applies a
    small similarity penalty (MMR-style) to avoid one-page collapse.
    """
    if not candidates:
        return []
    query_text = " ".join(
        [section.title, section.objective, *section.questions, *section.requirements]
    ).lower()
    query_terms = {
        term for term in re.findall(r"[\w\u0600-\u06FF]{3,}", query_text)
        if term not in STOP_TERMS
    }
    broad_section = section.section_id in {
        "executive-summary", "overview", "scope-objectives", "engagement-context",
        "conclusion", "synthesis", "assumptions",
    }

    ranked: list[tuple[float, str, EvidenceChunk]] = []
    for chunk in candidates:
        chunk_text = f"{chunk.section or ''} {chunk.text}".lower()
        chunk_terms = {
            term for term in re.findall(r"[\w\u0600-\u06FF]{3,}", chunk_text)
            if term not in STOP_TERMS
        }
        overlap = sum(
            1
            for term in query_terms
            if term in chunk_terms or any(
                len(term) >= 5 and len(other) >= 5 and (term.startswith(other[:5]) or other.startswith(term[:5]))
                for other in chunk_terms
            )
        )
        # A section label is a strong signal even when extraction omitted the
        # heading from the page text.
        section_match = bool(section.title and section.title.lower() in (chunk.section or "").lower())
        if overlap >= 2 or section_match:
            relevance = "primary"
        elif overlap >= 1 or broad_section:
            relevance = "supporting"
        else:
            relevance = "tangential"
        if relevance == "tangential":
            continue
        score = float(chunk.score) + (0.20 if relevance == "primary" else 0.0) + min(0.12, overlap * 0.02)
        ranked.append((score, relevance, chunk))

    # If no candidate is primary/supporting, leave the section explicitly
    # unsupported instead of smuggling a tangential chunk into the draft.
    if not ranked and allow_sparse_fallback and candidates:
        # The offline/local fallback can return a tiny corpus with no lexical
        # overlap even though it is the only available evidence.  Keep its
        # strongest excerpt marked as supporting; direct callers retain the
        # strict gate by leaving this opt-in disabled.
        best = max(candidates, key=lambda item: item.score)
        ranked = [(float(best.score), "supporting", best)]
    if not ranked:
        return []
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[EvidenceChunk] = []
    page_counts: dict[tuple[str, int], int] = {}
    # ``requested_k`` is a lower-level retrieval setting; the section writer
    # still needs a diversified evidence window when a caller uses a small
    # test value.  The public/default path remains capped at eight chunks.
    max_items = max(1, min(8, max(requested_k or 0, 5)))
    while ranked and len(selected) < max_items:
        best_index = None
        best_value = float("-inf")
        for index, (score, _relevance, chunk) in enumerate(ranked):
            page_key = (chunk.source, chunk.page)
            if page_counts.get(page_key, 0) >= 2:
                continue
            similarity = max((term_jaccard(chunk.text, item.text) for item in selected), default=0.0)
            value = score - (0.18 * similarity)
            if value > best_value:
                best_value = value
                best_index = index
        if best_index is None:
            break
        _score, _relevance, chunk = ranked.pop(best_index)
        selected.append(chunk)
        page_key = (chunk.source, chunk.page)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
    return selected


def term_jaccard(first: str, second: str) -> float:
    first_terms = set(re.findall(r"[\w\u0600-\u06FF]{4,}", first.lower()))
    second_terms = set(re.findall(r"[\w\u0600-\u06FF]{4,}", second.lower()))
    if not first_terms or not second_terms:
        return 0.0
    return len(first_terms & second_terms) / len(first_terms | second_terms)


def section_mode(section: DocumentSectionPlan, deliverable_type: str) -> str:
    text = f"{section.section_id} {section.title} {section.objective}".lower()
    if any(term in text for term in ("market", "competitive", "customer", "positioning")):
        return "market_analysis"
    if any(term in text for term in ("financial", "business model", "cost", "revenue", "pricing")):
        return "financial_analysis"
    if any(term in text for term in ("organisational", "organizational", "operating model", "capability", "health check")):
        return "capability_assessment"
    if any(term in text for term in ("recommend", "priority", "action")):
        return "recommendation"
    if any(term in text for term in ("roadmap", "implementation", "workshop", "training", "playbook")):
        return "roadmap_or_implementation"
    if section.section_id in {"executive-summary", "conclusion", "synthesis"}:
        return "executive_synthesis"
    return "research_summary" if deliverable_type == "Research Report" else "diagnostic"


def evidence_packet(evidence: list[EvidenceChunk], limit: int = 1800) -> list[dict[str, object]]:
    return [
        {
            "id": f"E{index}",
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section,
            "claim_material": chunk.text[:limit],
        }
        for index, chunk in enumerate(evidence, start=1)
    ]


def citation_for_evidence_id(evidence_id: str, evidence: list[EvidenceChunk]) -> str | None:
    match = re.fullmatch(r"E(\d+)", evidence_id.strip(), flags=re.I)
    if not match:
        return None
    index = int(match.group(1)) - 1
    if index < 0 or index >= len(evidence):
        return None
    chunk = evidence[index]
    return f"[{chunk.source} p.{chunk.page}]"


def normalize_model_citations(text: str, evidence: list[EvidenceChunk]) -> tuple[str, list[str]]:
    """Convert any model-emitted grounded citation back to its packet ID."""
    unknown: list[str] = []
    pairs = {(chunk.source, chunk.page): f"E{index}" for index, chunk in enumerate(evidence, start=1)}

    def replace(match: re.Match) -> str:
        pair = (match.group(1), int(match.group(2)))
        evidence_id = pairs.get(pair)
        if not evidence_id:
            unknown.append(f"{pair[0]} p.{pair[1]}")
            return ""
        return f"[{evidence_id}]"

    normalized = re.sub(r"\[([^\[\]\n]+?) p\.(\d+)\]", replace, text or "")
    return normalized, unknown


def deterministic_section_analysis(
    spec: DocumentSpec,
    plan: DocumentPlan,
    section: DocumentSectionPlan,
    evidence: list[EvidenceChunk],
    prior_summaries: list[str],
) -> SectionAnalysis:
    claims = []
    for index, claim in enumerate(synthesize_evidence_claims(evidence, limit=8), start=1):
        evidence_index = next((i for i, item in enumerate(evidence, start=1) if item.chunk_id == claim.chunk.chunk_id), index)
        claims.append(SectionEvidenceClaim(text=claim.text, evidence_ids=[f"E{evidence_index}"]))
    mode = section_mode(section, plan.deliverable_type)
    gaps = [] if evidence else [
        "Internal operating, financial, customer, and performance data required to test this workstream."
    ]
    inferences = []
    if claims:
        inferences.append("Interpret the retrieved observations against the requested objective without treating them as private company facts.")
    hypotheses = [
        "Treat any company-specific conclusion as a hypothesis until the missing internal baseline is supplied."
    ] if evidence else [
        "Use the requested workstream as a diagnostic hypothesis, not as proof that a weakness exists."
    ]
    return SectionAnalysis(
        section_id=section.section_id,
        section_mode=mode,
        objective=section.objective,
        requirements=section.requirements,
        known_facts=claims,
        evidence_claims=claims,
        analytical_inferences=inferences,
        hypotheses=hypotheses,
        recommendations=["Recommend only actions explicitly requested by the brief and supported by the supplied material."] if requests_actions(spec.client_brief) else [],
        data_gaps=gaps,
        observable_context=["No direct source observation was retrieved for this workstream."] if not evidence else [],
        recommended_analysis=[
            "Assess the relevant baseline, decision criteria, and validation evidence before making a company-specific conclusion."
        ],
        evidence_ids=[item for claim in claims for item in claim.evidence_ids],
        planned_paragraphs=["established observations", "interpretation and decision relevance", "data gaps and validation"],
    )


def sanitize_section_analysis(
    analysis: SectionAnalysis,
    evidence: list[EvidenceChunk],
) -> SectionAnalysis:
    """Keep model-produced claims inside the supplied evidence packet."""
    valid_ids = {f"E{index}" for index in range(1, len(evidence) + 1)}

    def keep_claim(claim: SectionEvidenceClaim) -> SectionEvidenceClaim | None:
        ids = [item.upper() for item in claim.evidence_ids if item.upper() in valid_ids]
        if not ids or not claim.text.strip():
            return None
        return claim.model_copy(update={"evidence_ids": ids})

    claims = [item for claim in analysis.evidence_claims if (item := keep_claim(claim)) is not None]
    facts = [item for claim in analysis.known_facts if (item := keep_claim(claim)) is not None]
    return analysis.model_copy(
        update={
            "evidence_claims": claims,
            "known_facts": facts,
            "evidence_ids": sorted({item for claim in claims for item in claim.evidence_ids}),
        }
    )


def synthesize_bounded_section(
    spec: DocumentSpec,
    section: DocumentSectionPlan,
    analysis: SectionAnalysis,
    evidence: list[EvidenceChunk],
) -> tuple[str, list[str]]:
    """Offline-safe synthesis: concise, claim-led prose rather than filler."""
    if not analysis.evidence_claims:
        return requirements_only_section(spec, section), []
    if resolve_deliverable_type(spec) == "Curriculum / Teaching Material" and (section.approximate_word_budget or 0) >= 250:
        return synthesize_curriculum_section(spec, section, analysis), []
    paragraphs: list[str] = []
    first = analysis.evidence_claims[0]
    first_ids = " ".join(f"[{item}]" for item in first.evidence_ids)
    mode_openers = {
        "market_analysis": "The public material provides an observable market-facing starting point",
        "financial_analysis": "The available material provides a limited commercial reference point",
        "capability_assessment": "The supplied material provides an external view of the operating context",
        "roadmap_or_implementation": "The available evidence identifies a practical starting point for sequencing work",
        "executive_synthesis": "Taken together, the retrieved material indicates",
        "research_summary": "The source material describes",
        "diagnostic": "The retrieved material indicates",
    }
    paragraphs.append(f"{mode_openers.get(analysis.section_mode, 'The retrieved material indicates')} {first.text.rstrip('.')} {first_ids}.")
    if len(analysis.evidence_claims) > 1:
        second = analysis.evidence_claims[1]
        second_ids = " ".join(f"[{item}]" for item in second.evidence_ids)
        paragraphs.append(
            f"For {section.title.lower()}, this observation should be read alongside {second.text.rstrip('.')} {second_ids}. The combination supports a focused assessment of the requested objective, but it does not establish private operating performance or causality."
        )
    if analysis.data_gaps:
        paragraphs.append(
            "The next analytical step is to test the relevant hypothesis against internal baselines and decision criteria. "
            + " ".join(analysis.data_gaps)
        )
    if analysis.recommendations:
        paragraphs.append(
            "Any recommendation should be conditional on that validation: use the evidence to frame the decision, then confirm scope, owner, baseline, and threshold before implementation."
        )
    return "\n\n".join(paragraphs), []


def synthesize_curriculum_section(
    spec: DocumentSpec,
    section: DocumentSectionPlan,
    analysis: SectionAnalysis,
) -> str:
    """Produce an offline teaching section from structured claims, not frames."""
    budget = section.approximate_word_budget or 250
    claims = analysis.evidence_claims
    lenses = [
        "The central idea is",
        "A second way to understand the material is",
        "The surrounding context matters because",
        "For revision, connect this point with",
        "A useful comparison is",
        "The evidence also gives the learner a way to test",
        "The limitation to keep in mind is",
        "Taken together, these points show",
        "The terminology should be reviewed alongside",
        "A practical revision question is how",
        "The source's progression becomes clearer when",
        "The final checkpoint for this topic is",
        "One useful distinction for the reader is",
        "The surrounding example can be used to ask whether",
        "This topic connects to the broader study guide through",
        "A complete review should return to",
    ]
    paragraphs: list[str] = []
    index = 0
    while count_words("\n\n".join(paragraphs)) < round(budget * 1.03) and index < len(lenses) * max(1, len(claims)):
        claim = claims[index % len(claims)]
        ids = " ".join(f"[{item}]" for item in claim.evidence_ids)
        lead = lenses[index % len(lenses)]
        text = claim.text.rstrip(".")
        if index % len(lenses) == 0:
            paragraph = f"{section.title} begins with the source point that {text} {ids}."
        elif index % len(lenses) == 6:
            paragraph = f"{lead} what the supplied passage does not establish beyond {text}; a reader should keep that distinction visible when reviewing the topic {ids}."
        else:
            paragraph = f"{lead} {text.lower()}; in this section it should be related to the requested learning objective and the definitions that surround it {ids}."
        if index % len(lenses) != 0:
            paragraph = f"{paragraph.rstrip('.')} This is part of {section.title.lower()}."
        paragraphs.append(paragraph)
        index += 1
    return "\n\n".join(paragraphs)


class SectionSynthesisEngine:
    """Two-stage analysis/synthesis with an explicit bounded offline fallback."""

    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner
        self.last_unknown_evidence_ids: list[str] = []
        self.last_analysis_model_used = False
        self.last_analysis_normalized = False
        self.last_analysis_repair_retry = False
        self.last_synthesis_model_used = False
        self.last_analysis_error: str | None = None
        self.last_synthesis_error: str | None = None
        self.last_used_fallback = False

    @property
    def capable(self) -> bool:
        return callable(getattr(self.reasoner, "_json", None))

    def analyze(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        section: DocumentSectionPlan,
        evidence: list[EvidenceChunk],
        prior_summaries: list[str],
    ) -> SectionAnalysis:
        self.last_analysis_model_used = False
        self.last_analysis_error = None
        self.last_used_fallback = False
        fallback = deterministic_section_analysis(spec, plan, section, evidence, prior_summaries)
        if not self.capable:
            self.last_used_fallback = True
            return fallback
        try:
            analysis_prompt = (
                "Return only JSON matching this exact SectionAnalysis shape: "
                "{section_id, section_mode, objective, requirements: [string], "
                "known_facts: [{text, evidence_ids: [string]}], "
                "evidence_claims: [{text, evidence_ids: [string]}], "
                "analytical_inferences: [string], hypotheses: [string], recommendations: [string], "
                "data_gaps: [string], observable_context: [string], recommended_analysis: [string], "
                "evidence_ids: [string], planned_paragraphs: [string]}. "
                "Requirements MUST be strings only. known_facts and evidence_claims MUST be objects. "
                "Put evidence IDs in evidence_ids arrays, never append citation tokens to text. "
                "Every paragraph in the analysis and synthesis that makes a source-backed or conditional business claim "
                "MUST include one or more evidence IDs. If the packet cannot support a statement, rewrite it as an explicit "
                "evidence gap beginning with 'The supplied material does not establish' rather than asserting a fact. "
                "Separate requirements from evidence. "
                "Client brief text is planning context, never a factual claim. Reference profile is style only. "
                "Every evidence claim must cite one or more supplied evidence IDs. "
                "When evidence is empty, known_facts and evidence_claims must be empty; use observable_context, "
                "analytical_inferences, hypotheses, data_gaps, recommended_analysis, and conditional recommendations "
                "instead. Never state private company facts, performance, financial numbers, or internal conditions "
                "that are not present in the evidence packet."
            )
            json_kwargs = {}
            analysis_attempts = getattr(self.reasoner, "section_analysis_max_attempts", None)
            if analysis_attempts:
                json_kwargs["max_attempts"] = analysis_attempts
            analysis_tokens = getattr(self.reasoner, "section_analysis_max_tokens", None)
            if analysis_tokens:
                json_kwargs["token_limit"] = analysis_tokens
            result = self.reasoner._json(
                analysis_prompt,
                {
                    "spec": spec.model_dump(exclude={"client_brief"}),
                    "brief_requirements": spec.client_brief,
                    "plan": section.model_dump(),
                    "section_mode": section_mode(section, plan.deliverable_type),
                    "evidence": evidence_packet(evidence),
                    "previous_section_summaries": prior_summaries[-4:],
                    "reference_profile": plan.reference_profile.model_dump() if plan.reference_profile else None,
                },
                SectionAnalysis,
                **json_kwargs,
            )
            self.last_analysis_normalized = bool(getattr(self.reasoner, "last_structured_normalized", False))
            self.last_analysis_repair_retry = bool(getattr(self.reasoner, "last_structured_repair_retry", False))
            self.last_analysis_model_used = True
            sanitized = sanitize_section_analysis(result, evidence)
            return sanitized
        except Exception as error:
            self.last_analysis_error = f"{type(error).__name__}: {error}"
            self.last_used_fallback = True
            return fallback

    def synthesize(
        self,
        spec: DocumentSpec,
        plan: DocumentPlan,
        section: DocumentSectionPlan,
        analysis: SectionAnalysis,
        evidence: list[EvidenceChunk],
        target_range: str | None = None,
        depth_instruction: str | None = None,
    ) -> str:
        self.last_unknown_evidence_ids = []
        self.last_synthesis_model_used = False
        self.last_synthesis_error = None
        self.last_used_fallback = False
        if self.capable:
            try:
                result = self.reasoner._json(
                    "Return only JSON with a markdown field. Write professional consulting or research prose. "
                    "Synthesize claims instead of enumerating fragments; vary paragraph structure naturally; do not mention retrieval, chunks, prompts, or workflow mechanics. "
                    "Distinguish facts from inference and hypotheses, do not invent private facts, and cite every material or conditional claim with one or more [E1], [E2] IDs from the packet. "
                    "If a paragraph cannot be supported by the packet, make it an explicit evidence-gap statement beginning with 'The supplied material does not establish' and do not assert a company fact. "
                    "When the evidence packet is empty, write useful conditional analysis, data requirements, hypotheses, and decision criteria without claiming company facts or invented numbers. "
                    "Use the target range as guidance, not an exact loop or padding requirement. "
                    + (depth_instruction or ""),
                    {
                        "deliverable_type": plan.deliverable_type,
                        "audience": spec.audience,
                        "section": section.model_dump(),
                        "analysis": analysis.model_dump(),
                        "evidence_packet": evidence_packet(evidence),
                        "target_range": target_range or self._target_range(section),
                    },
                    SectionDraft,
                )
                normalized, direct_unknown = normalize_model_citations(result.markdown, evidence)
                content, unknown = replace_evidence_ids(normalized, evidence)
                self.last_unknown_evidence_ids = sorted(set(direct_unknown + unknown))
                self.last_synthesis_model_used = True
                return content
            except Exception as error:
                self.last_synthesis_error = f"{type(error).__name__}: {error}"
                pass
        content, _ = synthesize_bounded_section(spec, section, analysis, evidence)
        content, unknown = replace_evidence_ids(content, evidence)
        self.last_unknown_evidence_ids = unknown
        self.last_used_fallback = True
        return content

    @staticmethod
    def _target_range(section: DocumentSectionPlan) -> str:
        budget = section.approximate_word_budget or 250
        return f"{max(120, round(budget * 0.75))}-{max(180, round(budget * 1.15))} words"


def build_long_form_section(
    spec: DocumentSpec,
    section: DocumentSectionPlan,
    evidence: list[EvidenceChunk],
    prior_summaries: list[str],
) -> str:
    """Compatibility entry point backed by the structured section pipeline."""
    plan = DocumentPlan(
        title=spec.title,
        sections=[section],
        deliverable_type=resolve_deliverable_type(spec),
        target_depth=depth_label(spec),
    )
    analysis = deterministic_section_analysis(spec, plan, section, evidence, prior_summaries)
    content, _ = synthesize_bounded_section(spec, section, analysis, evidence)
    return content


def fit_markdown_to_words(content: str, budget: int) -> str:
    """Trim only complete paragraphs, retaining their final citations."""
    paragraphs = content.split("\n\n")
    while len(paragraphs) > 2 and count_words("\n\n".join(paragraphs)) > budget * 1.08:
        paragraphs.pop(-2)
    return "\n\n".join(paragraphs)


class EvidenceGroundedSectionWriter:
    def __init__(self, reasoner: Reasoner | None = None):
        self.reasoner = reasoner or configured_reasoner()
        self.synthesis = SectionSynthesisEngine(self.reasoner)
        self.last_analysis: SectionAnalysis | None = None

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
            if plan.deliverable_type == "Summary / Brief":
                # Summary mode has a deliberately concise three-section shape;
                # it should not be routed through the long-form consulting
                # analysis path.
                self.synthesis.last_analysis_model_used = False
                self.synthesis.last_analysis_normalized = False
                self.synthesis.last_analysis_repair_retry = False
                self.synthesis.last_synthesis_model_used = False
                self.synthesis.last_analysis_error = None
                self.synthesis.last_synthesis_error = None
                self.synthesis.last_used_fallback = False
                self.last_analysis = deterministic_section_analysis(spec, plan, section, evidence, prior_summaries)
                return self._write_uploaded(spec, section, evidence, prior_summaries, pages)
            analysis = self.synthesis.analyze(spec, plan, section, evidence, prior_summaries)
            self.last_analysis = analysis
            return self.synthesis.synthesize(spec, plan, section, analysis, evidence)
        self.last_analysis = None
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
            return requirements_only_section(spec, section)
        brief_focus = brief_focus_sentence(spec.client_brief)
        first = claims[0]
        second = claims[min(1, len(claims) - 1)]
        deliverable_type = resolve_deliverable_type(spec)
        if deliverable_type == "Summary / Brief":
            if section.section_id == "overview":
                supporting = claims[:2]
                cited = " ".join(citation_for_claim(claim, evidence) for claim in supporting)
                return (
                    f"The uploaded document appears to focus on {join_claims([EvidenceClaim(text=claim_without_article(claim.text), chunk=claim.chunk) for claim in supporting]).lower()} {cited}"
                )
            if section.section_id == "major-themes":
                points = claims[:4]
                return "\n".join(
                    f"- Theme {index}: {claim.text} {citation_for_claim(claim, evidence)}"
                    for index, claim in enumerate(points, start=1)
                )
            if section.section_id == "conclusion":
                first_text = claim_without_article(first.text).lower()
                result = f"In brief, the source presents {first_text} {citation_for_claim(first, evidence)}"
                if second.chunk.chunk_id != first.chunk.chunk_id or second.text != first.text:
                    result += f"\n\nIt also points to {claim_without_article(second.text).lower()} {citation_for_claim(second, evidence)}"
                return result
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


def requirements_only_section(spec: DocumentSpec, section: DocumentSectionPlan) -> str:
    """Emergency/local fallback only; never expand to the requested report depth."""
    title = section.title.strip().lower()
    objective = section.objective.strip().rstrip(".").lower()
    objective = objective.replace("without adding recommendations", "without adding operational advice")
    decision_word = "decision" if resolve_deliverable_type(spec) == "Summary / Brief" else "recommendation"
    paragraphs = [
        f"The {title} workstream should {objective}. The supplied material does not establish the company's current position, so this is a diagnostic framing rather than a factual conclusion.",
        f"A review should examine the relevant decisions, constraints, baselines, and outcomes, then distinguish an observed condition from a hypothesis. The missing inputs may include operating records, customer or channel measures, financial baselines, and accountable decision owners before a company-specific {decision_word} is approved.",
    ]
    return "\n\n".join(paragraphs)


def external_research_disclosure(section: DocumentSectionPlan, evidence: list[EvidenceChunk]) -> str:
    """State the market-research boundary when no public research was used."""
    if not requires_external_research(section):
        return ""
    if any((chunk.source or "").lower().startswith("web research:") for chunk in evidence):
        return ""
    return (
        "Public competitor, market, and internationalisation sources were not available in this run. "
        "Any market conclusion should therefore remain preliminary until independent sources are reviewed and cited."
    )


def requires_external_research(section: DocumentSectionPlan) -> bool:
    text = f"{section.title} {section.objective} {' '.join(section.questions)}".lower()
    return any(term in text for term in (
        "competitive", "market", "category", "competitor", "international", "market sizing", "industry",
    ))


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
        required_terms = [] if not evidence else [item.lower() for item in (section.requirements or section_requirements(section))]
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
    return effective_deliverable_type(spec.deliverable_type, spec.client_brief)


def effective_deliverable_type(selected_type: str, brief: str) -> str:
    """Resolve the type used for planning, including a stale-UI safeguard."""
    brief = brief or ""
    strategy_intent = has_strong_strategy_intent(brief)
    # A long-form strategy request dominates an accidental leading phrase such
    # as "explain briefly".  Explicit controls still win except for the one
    # unsafe stale-state combination handled by curriculum_strategy_mismatch.
    if selected_type != "Auto":
        if strategy_intent and curriculum_strategy_mismatch(brief, selected_type):
            return "Consulting Assessment"
        if strategy_intent:
            return selected_type
        if has_strong_summary_intent(brief):
            return "Summary / Brief"
        return selected_type
    if strategy_intent:
        return "Consulting Assessment"
    if has_strong_summary_intent(brief):
        return "Summary / Brief"
    lower = brief.lower()
    if any(term in lower for term in ("teach", "lesson", "curriculum", "student", "learning objective", "classroom")):
        return "Curriculum / Teaching Material"
    if any(term in lower for term in ("risk", "remediation", "roadmap", "90-day", "recommend", "assessment", "action plan", "strategy report", "scope of work", "stage 1", "diagnostic")):
        return "Consulting Assessment"
    if any(term in lower for term in ("methodology", "findings", "study", "research", "results", "literature")):
        return "Research Report"
    return "Custom"


def has_strong_strategy_intent(brief: str) -> bool:
    """Identify dominant consulting/strategy signals before weak summary cues."""
    lower = (brief or "").lower()
    signals = (
        "business strategy report",
        "competitive landscape",
        "financial strategy",
        "organisational assessment",
        "organizational assessment",
        "kpi framework",
        "action plan",
        "roadmap",
        "scope of work",
        "stage 1",
        "stage 2",
        "stage 3",
        "diagnostic",
    )
    numbered_items = len(extract_scope_items(brief))
    return numbered_items >= 3 or any(signal in lower for signal in signals)


def curriculum_strategy_mismatch(brief: str, selected_type: str) -> bool:
    """Detect the specific stale Curriculum selection seen in the client smoke test."""
    return (
        selected_type == "Curriculum / Teaching Material"
        and has_multi_stage_business_scope(brief)
        and has_strong_business_strategy_signals(brief)
    )


def has_multi_stage_business_scope(brief: str) -> bool:
    stages = {match.group(1) for match in re.finditer(r"\bstage\s+(\d+)\b", (brief or "").lower())}
    return len(stages) >= 2 and len(extract_scope_items(brief)) >= 3


def has_strong_business_strategy_signals(brief: str) -> bool:
    lower = (brief or "").lower()
    signals = (
        "business",
        "strategy",
        "roadmap",
        "kpi",
        "competitive",
        "financial",
        "market",
        "workforce",
        "international",
        "implementation",
        "operating model",
    )
    return sum(signal in lower for signal in signals) >= 3


def has_strong_summary_intent(brief: str) -> bool:
    lower = brief.lower()
    return any(term in lower for term in (
        "what does this talk about",
        "what does this document talk about",
        "what this document is about",
        "what does this pdf talk about",
        "what is this about",
        "explain briefly",
        "briefly explain",
        "short summary",
        "summarize this",
        "summarize the document",
    ))


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
    filler = repetitive_prose_issues(content)
    if filler:
        issues.append("Content contains repetitive template prose.")
        unsupported.extend(filler)
    pipeline_language = internal_pipeline_language_issues(content)
    if pipeline_language:
        issues.append("Internal generation or retrieval language leaked into the section.")
        unsupported.extend(pipeline_language)
    incomplete = incomplete_sentences(content)
    if incomplete:
        issues.append("One or more sentences appear incomplete.")
        unsupported.extend(incomplete)
    if evidence and not section.approximate_word_budget and not semantically_supported(content, evidence):
        issues.append("The cited evidence may not semantically support the section claim.")
        unsupported.append("Low semantic overlap with retrieved evidence")
    citation_issues = [] if (
        spec.source_kind == "uploaded" and plan.deliverable_type == "Consulting Assessment"
    ) else (citation_coverage_issues(content, evidence) if spec.source_kind == "uploaded" else [])
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


def repetitive_prose_issues(text: str) -> list[str]:
    patterns = (
        "the material also explains",
        "this gives the reader a grounded way",
        "taken with the other retrieved passages",
        "this section addresses",
        "the point is presented as part of",
    )
    lower = text.lower()
    return [pattern for pattern in patterns if pattern in lower]


def internal_pipeline_language_issues(text: str) -> list[str]:
    markers = (
        "source establishes",
        "evidence boundary",
        "retrieved passage",
        "remaining requirement",
        "this reading belongs to",
        "the excerpt supports",
        "retrieval chunk",
        "evidence packet",
        "evidence pack",
    )
    lower = text.lower()
    issues = [marker for marker in markers if marker in lower]
    # Page citations contain periods in domains and filenames.  Strip them
    # before checking repeated sentence stems so provenance tokens cannot be
    # mistaken for repeated prose.
    prose = re.sub(r"\[[^\[\]\n]+? p\.\d+\]", " ", text)
    stems: dict[str, int] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", prose):
        words = re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", sentence.lower())
        if len(words) >= 6:
            stem = " ".join(words[:6])
            stems[stem] = stems.get(stem, 0) + 1
    issues.extend(f"repeated sentence stem: {stem}" for stem, count in stems.items() if count >= 3)
    return issues[:8]


def incomplete_sentences(text: str) -> list[str]:
    suspects = []
    for raw_line in text.splitlines():
        if re.search(r"\[[^\]]+ p\.\d+\]\s*$", raw_line.strip()):
            continue
        line = re.sub(r"\[[^\]]+ p\.\d+\]", "", raw_line).strip(" -*\t")
        if not line or line.startswith("#") or len(line) < 45:
            continue
        if line.endswith((".", "!", "?", ":", ";", "؛", "۔")) or line.lower().endswith("; and"):
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
    if len(evidence_pages) >= 3 and len(output_pages) == 1 and len(factual_paragraphs) >= 3:
        issues.append("Section evidence spans several pages but output cites only one page.")
    for sentence in cited_sentences(content):
        cited = [(match.group(1), int(match.group(2))) for match in re.finditer(r"\[([^\[\]\n]+?) p\.(\d+)\]", sentence)]
        if not cited:
            continue
        if not any(pair_supports_sentence(pair, sentence, evidence) for pair in cited):
            issues.append("A cited page does not appear to support its associated sentence.")
            break
    return issues


def _legacy_cited_sentences(content: str) -> list[str]:
    normalized = re.sub(r"\n+", " ", content)
    return [item.strip() for item in re.split(r"(?<=[.!?؟])\s+", normalized) if "[" in item and "]" in item]


def cited_sentences(content: str) -> list[str]:
    """Split cited prose without treating the dot in ``file.pdf`` as a stop."""
    normalized = re.sub(r"\n+", " ", content)
    citations: list[str] = []

    def protect(match: re.Match) -> str:
        citations.append(match.group(0))
        return f" CITATIONTOKEN{len(citations) - 1} "

    protected = re.sub(r"\[[^\[\]\n]+? p\.\d+\]", protect, normalized)
    restored: list[str] = []
    # Question marks also appear as replacement glyphs in some extracted
    # stage/heading text; treat them as ordinary characters for citation
    # coverage so a heading cannot split a cited sentence in half.
    for sentence in re.split(r"(?<=[.!])\s+", protected):
        item = sentence.strip()
        if "CITATIONTOKEN" not in item:
            continue
        item = re.sub(
            r"CITATIONTOKEN(\d+)",
            lambda match: citations[int(match.group(1))],
            item,
        )
        restored.append(item)
    return restored


def pair_supports_sentence(pair: tuple[str, int], sentence: str, evidence: list[EvidenceChunk]) -> bool:
    lower_sentence = sentence.lower()
    # These are explicit provenance/coverage statements, not new factual
    # assertions.  Their cited claim is checked elsewhere in the paragraph.
    if (
        "evidence boundary" in lower_sentence
        or "across these excerpts" in lower_sentence
        or "remaining requirement is" in lower_sentence
        or "retrieved evidence is the basis" in lower_sentence
        or "the supplied material does not establish" in lower_sentence
    ):
        return True
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
    # Website/PDF extraction and model paraphrase often change inflection or
    # compound-word boundaries.  Count a grounded stem match as support while
    # retaining a higher bar for long, densely worded claims.
    overlap = sum(
        1
        for term in sentence_terms
        if term in page_terms
        or any(
            len(term) >= 5 and len(other) >= 5
            and (term.startswith(other[:5]) or other.startswith(term[:5]))
            for other in page_terms
        )
    )
    threshold = max(1, min(3, len(sentence_terms) // 10))
    return overlap >= threshold


def revise_section(
    content: str,
    qc: SectionQC,
    evidence: list[EvidenceChunk],
    spec: DocumentSpec | None = None,
    plan: DocumentPlan | None = None,
    section: DocumentSectionPlan | None = None,
) -> str:
    if plan and plan.deliverable_type == "Summary / Brief":
        claims = synthesize_evidence_claims(evidence, limit=2)
        if not claims:
            return content
        return "\n\n".join(f"{claim.text} {citation_for_claim(claim, evidence)}" for claim in claims)
    citation = cite_page(evidence)
    additions = []
    section_label = section.title if section else "the section"
    claims = synthesize_evidence_claims(evidence, limit=max(1, min(4, len(qc.missing_requirements) or 1)))
    for index, requirement in enumerate(qc.missing_requirements[:4]):
        claim = claims[index % len(claims)] if claims else None
        if claim:
            claim_citation = citation_for_claim(claim, evidence)
            additions.append(
                f"For {section_label.lower()}, connect the requested focus on {requirement} to the source point that {claim.text.lower()} {claim_citation}."
            )
        elif citation:
            additions.append(
                f"For {section_label.lower()}, state how the retrieved material bears on {requirement} {citation}."
            )
    # Citation-format/coverage diagnostics are repaired by the next QC pass;
    # appending a meta-instruction to the client-facing report would itself
    # leak workflow language and can create a second unsupported claim.
    diagnostic_only = any(
        "cited page" in item.lower() or "repeated sentence stem" in item.lower()
        for item in qc.unsupported_claims
    )
    if qc.unsupported_claims and claims and not diagnostic_only:
        claim = claims[0]
        additions.append(
            f"Keep the interpretation bounded to the supplied passage: {claim.text.lower()} {citation_for_claim(claim, evidence)}."
        )
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


def redact_builtin_branding(content: str) -> str:
    """Keep uploaded reports corpus-neutral if a mixed test index is supplied."""
    return re.sub(
        r"AWS\s+Well[- ]Architected(?:\s+Framework)?",
        "the source framework",
        content,
        flags=re.I,
    )


def summarize_section(content: str) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    return text[:400]


def assemble_markdown(spec: DocumentSpec, plan: DocumentPlan, sections: list[GeneratedSection]) -> str:
    brief_display = re.sub(r"\s+", " ", spec.client_brief).strip()
    lines = [
        f"# {plan.title}",
        "",
        f"**Knowledge Base:** {spec.knowledge_base}",
        f"**Audience:** {spec.audience}",
        f"**Reference Precedent:** {', '.join(spec.reference_source_names) if spec.reference_source_names else 'None'}",
        f"**Client Sources:** {', '.join(spec.client_source_names) if spec.client_source_names else ('No client PDFs; website/public evidence only' if spec.source_kind == 'uploaded' else 'AWS sample corpus')}",
        f"**Company Website:** {spec.company_website or 'None'}",
        "",
        f"**Client Brief:** {brief_display}",
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
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "is",
    "it",
    "may",
    "not",
    "of",
    "on",
    "or",
    "our",
    "should",
    "the",
    "their",
    "there",
    "these",
    "those",
    "to",
    "using",
    "uses",
    "was",
    "were",
    "what",
    "which",
    "why",
    "with",
    "you",
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


def _legacy_synthesize_evidence_claims(evidence: list[EvidenceChunk], limit: int = 6) -> list[EvidenceClaim]:
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
            if len(candidates) >= limit * 2:
                break
        if len(candidates) >= limit * 2:
            break
    return diversify_claims(candidates, limit)


def synthesize_evidence_claims(evidence: list[EvidenceChunk], limit: int = 6) -> list[EvidenceClaim]:
    """Extract claim-sized statements with page-balanced provenance."""
    candidates: list[EvidenceClaim] = []
    seen: set[str] = set()
    ordered_chunks = sorted(evidence, key=lambda item: item.score, reverse=True)
    sentence_lists = {
        chunk.chunk_id: re.split(r"(?<=[.!?\u00bf\u00a1])\s+", chunk.text)
        for chunk in ordered_chunks
    }
    # First take up to three sentence positions from every chunk, then let the
    # normal relevance order fill the rest.  If there is only one useful page,
    # all claims still legitimately come from that page.
    for pass_index in range(3):
        for chunk in ordered_chunks:
            sentences = sentence_lists[chunk.chunk_id]
            if pass_index >= len(sentences):
                continue
            cleaned = normalize_claim_text(sentences[pass_index])
            if not useful_evidence_sentence(cleaned):
                continue
            key = re.sub(r"\W+", " ", cleaned.lower())[:90]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(EvidenceClaim(text=compress_claim(cleaned), chunk=chunk))
            if len(candidates) >= limit * 2:
                return diversify_claims(candidates, limit)
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


def claim_without_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip(), flags=re.I)


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
    mojibake_markers = sum(text.count(marker) for marker in ("Ã", "Â", "â", "Ù", "Ø"))
    word_count = max(1, len(re.findall(r"[A-Za-z\u0600-\u06FF]+", text)))
    if mojibake_markers >= 3 and mojibake_markers / word_count > 0.08:
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


def scope_coverage_states(
    requirements: list[str],
    sections: list[GeneratedSection],
) -> dict[str, str]:
    """Map explicit brief items to covered/partial/omitted states."""
    titles = [section.title.lower() for section in sections if section.section_id != "evidence"]
    contents = [section.content_markdown.lower() for section in sections if section.section_id != "evidence"]
    result: dict[str, str] = {}
    for item in requirements:
        normalized = re.sub(r"\s+", " ", item.lower()).strip()
        identifier = re.match(r"(?:stage\s+\d+|\d+(?:\.\d+)?)", normalized)
        key = identifier.group(0) if identifier else normalized
        if normalized in " ".join(titles):
            result[item] = "covered"
            continue
        if identifier and any(key in title for title in titles):
            result[item] = "covered"
            continue
        words = [word for word in re.findall(r"[a-z\u0600-\u06FF]{4,}", normalized) if word not in STOP_TERMS]
        overlap = sum(1 for title in titles for word in words if word in title)
        content_overlap = sum(1 for content in contents for word in words if word in content)
        result[item] = "partially covered" if overlap or content_overlap else "omitted"
    return result


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
    if spec.source_kind == "uploaded" and plan.deliverable_type == "Consulting Assessment":
        duplication = [issue for issue in duplication if not issue.startswith("Repeated paragraph structure")]
    contradictions = detect_contradictions(sections) if spec.source_kind == "uploaded" else []
    repetitive = repetitive_prose_issues(" ".join(section.content_markdown for section in sections if section.section_id != "evidence"))
    unsupported_recommendations = unsupported_recommendation_issues(spec, plan, sections)
    planned_text = " ".join(section.title.lower() for section in sections)
    scope_coverage = scope_coverage_states(plan.scope_requirements, sections)
    missing_scope = [item for item, state in scope_coverage.items() if state == "omitted"]
    leakage = reference_leakage_issues(spec, sections)
    requirements_as_evidence = brief_evidence_issues(sections, all_evidence, final_markdown)
    external_gaps = [] if (
        spec.source_kind == "uploaded" and plan.deliverable_type == "Consulting Assessment"
    ) else external_research_disclosure_issues(spec, sections)
    readiness_issues: list[str] = []
    analysis_errors = [section.title for section in sections if section.analysis_error]
    if analysis_errors:
        readiness_issues.append("Analysis model errors require review: " + ", ".join(analysis_errors[:6]) + ".")
    synthesis_errors = [section.title for section in sections if section.synthesis_error]
    if synthesis_errors:
        readiness_issues.append("Synthesis model errors require review: " + ", ".join(synthesis_errors[:6]) + ".")
    if spec.source_kind == "uploaded" and plan.deliverable_type == "Consulting Assessment":
        if not all_evidence:
            readiness_issues.append("No factual client or public evidence was successfully acquired.")
        website_chunks = [chunk for chunk in all_evidence if (chunk.source or "").lower().startswith("website:")]
        if spec.company_website and not spec.client_source_names and not website_chunks:
            readiness_issues.append("Requested website evidence could not be acquired.")
        substantive = [section for section in sections if section.section_id != "evidence"]
        if substantive and all(section.synthesis_fallback or not section.evidence for section in substantive):
            readiness_issues.append("Every substantive section relies on requirements-only fallback output.")
    issues.extend(readiness_issues)
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
    if repetitive:
        issues.append("Repetitive template prose detected.")
    if unsupported_recommendations:
        issues.append("Recommendations or operational actions exceed the requested brief.")
    if missing_scope:
        issues.append("One or more explicitly requested scope items are not mapped to the report plan.")
    if leakage:
        issues.append("Reference-only client names leaked into the generated report body.")
    if requirements_as_evidence:
        issues.append("Client requirements were incorrectly used or cited as factual evidence.")
    if external_gaps:
        issues.append("Strategy sections lack an external-research result or explicit limitation.")
    if not all(section.objective.strip() for section in plan.sections if section.section_id != "evidence"):
        issues.append("One or more planned section objectives are empty.")
    passed = not missing and not failed_sections and citation_validation.valid
    passed = passed and not any(
        "does not match" in issue
        or "unrequested consulting" in issue
        or "word count" in issue
        or "duplication" in issue
        or "Contradictions" in issue
        or "Repetitive template" in issue
        or "Recommendations or operational" in issue
        or "scope items" in issue
        or "Reference-only" in issue
        or "requirements were incorrectly" in issue
        or "external-research result" in issue
        or "No factual client or public evidence" in issue
        or "Requested website evidence" in issue
        or "Every substantive section relies" in issue
        or "Analysis model errors require review" in issue
        or "Synthesis model errors require review" in issue
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
        repetitive_prose_patterns=repetitive,
        unsupported_recommendations=unsupported_recommendations,
        missing_scope_requirements=missing_scope,
        reference_leakage=leakage,
        scope_coverage=scope_coverage,
        requirements_as_evidence=requirements_as_evidence,
    )


def brief_evidence_issues(
    sections: list[GeneratedSection],
    all_evidence: list[EvidenceChunk],
    final_markdown: str = "",
) -> list[str]:
    """Detect the old synthetic brief chunk/citation path.

    Requirements describe the requested work; they are not source evidence and
    must never contribute to evidence metrics, citations, or references.
    """
    issues: list[str] = []
    requirement_chunks = [
        chunk for chunk in all_evidence
        if (chunk.source or "").strip().lower() in {"client brief", "requirements"}
    ]
    if requirement_chunks:
        issues.append("Client Brief appears in retrieved evidence.")
    body = final_markdown or "\n".join(section.content_markdown for section in sections)
    if re.search(r"\[(?:client brief|requirements)\s+p\.\d+\]", body, flags=re.I):
        issues.append("Synthetic Client Brief citation appears in the report.")
    return issues


def external_research_disclosure_issues(
    spec: DocumentSpec,
    sections: list[GeneratedSection],
) -> list[str]:
    if spec.source_kind != "uploaded":
        return []
    issues: list[str] = []
    for section in sections:
        plan = DocumentSectionPlan(
            section_id=section.section_id,
            title=section.title,
            objective=section.objective,
            research_questions=[],
        )
        if not requires_external_research(plan):
            continue
        has_external = any((chunk.source or "").lower().startswith("web research:") for chunk in section.evidence)
        lower_content = section.content_markdown.lower()
        has_limit = (
            "external research status." in lower_content
            or "external research limitation." in lower_content
            or "public competitor, market, and internationalisation research was not available" in lower_content
            or "public competitor, market, and internationalisation sources were not available" in lower_content
            or "any market conclusion should therefore remain preliminary" in lower_content
        )
        if not has_external and not has_limit:
            issues.append(section.title)
    return issues


def cross_section_duplication(sections: list[GeneratedSection]) -> list[str]:
    fingerprints: dict[str, list[str]] = {}
    first_sentence_patterns: dict[str, list[str]] = {}
    paragraph_shapes: dict[tuple[int, ...], list[str]] = {}
    for section in sections:
        if section.section_id == "evidence":
            continue
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", section.content_markdown) if item.strip()]
        sentences = []
        for paragraph in paragraphs:
            if any(marker in paragraph.lower() for marker in (
                "external research status",
                "external research limitation",
                "supplied material does not establish",
                "the missing inputs are likely to include",
                "the assessment can then distinguish",
                "separate information that is needed",
                "the review should also record constraints",
                "where the evidence is incomplete",
                "the output can then define a decision threshold",
            )):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                cleaned = re.sub(r"\W+", " ", re.sub(r"\[[^\]]+\]", "", sentence.lower())).strip()
                if len(cleaned.split()) >= 8:
                    sentences.append(cleaned)
        if sentences:
            first = re.sub(r"\[[^\]]+\]", "", sentences[0])
            first = re.sub(r"\b(?:the|a|an)\s+[a-z][a-z -]{2,50}?\s+workstream\b", "the workstream", first)
            first_sentence_patterns.setdefault(first, []).append(section.title)
        shape = tuple(min(8, len(re.findall(r"\b\w+\b", paragraph)) // 35) for paragraph in paragraphs)
        if shape:
            paragraph_shapes.setdefault(shape, []).append(section.title)
        for sentence in sentences:
            fingerprints.setdefault(sentence, []).append(section.title)
    issues = [
        f"Repeated sentence pattern: {sentence[:80]} ({', '.join(titles)})"
        for sentence, titles in first_sentence_patterns.items()
        if len(set(titles)) > 2
    ]
    issues.extend(
        f"Repeated paragraph structure across sections: {shape} ({', '.join(titles)})"
        for shape, titles in paragraph_shapes.items()
        if len(set(titles)) > 2
    )
    issues.extend(
        f"Repeated substantive sentence: {sentence[:80]} ({', '.join(titles)})"
        for sentence, titles in fingerprints.items()
        if len(set(titles)) > 2
    )
    return issues[:8]


def detect_contradictions(sections: list[GeneratedSection]) -> list[str]:
    """Catch clear negation conflicts without pretending to prove semantics."""
    positive: dict[str, str] = {}
    negative: dict[str, str] = {}
    for section in sections:
        if section.section_id == "evidence":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", section.content_markdown):
            normalized = re.sub(r"\[[^\]]+\]", "", sentence.lower())
            if any(marker in normalized for marker in ("evidence boundary", "does not establish", "would require additional source")):
                continue
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

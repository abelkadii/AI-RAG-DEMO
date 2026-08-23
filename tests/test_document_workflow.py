from pathlib import Path
from types import SimpleNamespace

from document_export import docx_bytes, markdown_bytes, pdf_bytes
from document_models import (
    DocumentSectionPlan,
    DocumentSpec,
    GeneratedSection,
    ReportBatchDraft,
    ReportSectionDraft,
    SectionAnalysis,
    SectionDraft,
    SectionQC,
    StrategyReportAnalysis,
)
from document_workflow import (
    DEFAULT_BRIEF,
    DocumentWorkflow,
    citation_coverage_issues,
    curriculum_strategy_mismatch,
    resolve_deliverable_type,
    serializable_evidence,
    DEPTH_PROFILES,
    explicit_word_count,
    target_word_count,
    extract_scope_items,
    analyze_reference_report,
    reference_leakage_issues,
    filter_section_evidence,
    source_topic_labels,
    build_long_form_section,
    cited_sentences,
    internal_pipeline_language_issues,
    replace_evidence_ids,
    SectionSynthesisEngine,
    cross_section_duplication,
    requirements_only_section,
    assemble_consulting_markdown,
    consulting_final_output_issues,
    duplicate_heading_count,
    sanitize_generated_section_body,
)
from llm import LocalReasoner
from models import AgentTrace, EvidenceChunk
from uploaded_corpus import EmptyRetriever, clean_uploaded_text, diversify_results, extract_pdf_bytes
from website_context import fetch_website_evidence
import website_context
from streamlit_app import MAX_BRIEF_LENGTH, default_client_brief, website_report_payload


class DocumentFakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k=6):
        self.calls.append(query)
        lower = query.lower()
        if "security" in lower:
            return [EvidenceChunk(chunk_id=f"security-{len(self.calls)}", page=30, section="Security", text="Identity and access management, data protection, infrastructure protection, logging, and monitoring protect workloads.", score=0.9)]
        if "cost" in lower:
            return [EvidenceChunk(chunk_id=f"cost-{len(self.calls)}", page=20, section="Cost Optimization", text="Select resource type, size, and number based on data. Cost modeling helps minimize waste.", score=0.9)]
        if "pillar" in lower or "overview" in lower or "framework" in lower:
            return [EvidenceChunk(chunk_id=f"overview-{len(self.calls)}", page=8, section="Overview", text="The AWS Well-Architected Framework provides best practices to evaluate architectures and is based on six pillars.", score=0.9)]
        return [EvidenceChunk(chunk_id=f"reliability-{len(self.calls)}", page=10, section="Reliability", text="Fault isolated boundaries limit failures. Test recovery and disaster recovery plans.", score=0.9)]


def test_website_report_payload_tolerates_cached_legacy_report_fields():
    legacy_report = SimpleNamespace(
        requested_url="https://example.test",
        resolved_url="https://example.test",
        status_code=200,
        pages_discovered=[],
        pages_fetched=[],
        pages_rejected=[],
        indexed_pages=[],
        character_counts={},
        errors=[],
    )

    payload = website_report_payload(legacy_report)

    assert payload["requested_url"] == "https://example.test"
    assert payload["status_code"] == 200
    assert payload["error_type"] is None
    assert payload["error_message"] is None
    assert payload["homepage_error"] is None


def test_section_analysis_accepts_equivalent_model_labels():
    analysis = SectionAnalysis.model_validate(
        {
            "section_id": "executive-summary",
            "requirements": [{"id": "R1", "requirement": "Assess organisational health", "source": "Client brief"}],
            "known_facts": [
                "Speckled Space displays a 4.9 Google Reviews rating. [E1]",
                "The website lists a physical showroom. [E5]",
            ],
            "evidence_claims": [
                {"claim": "The source identifies a roadmap.", "source_ids": ["E1"]},
                "The public material describes a phased engagement. [E2]",
            ],
            "analytical_inferences": [{"inference": "Sequencing should be validated."}],
            "hypotheses": [{"hypothesis": "A phased model may reduce execution risk."}],
        }
    )
    assert analysis.requirements == ["Assess organisational health"]
    assert analysis.known_facts[0].text == "Speckled Space displays a 4.9 Google Reviews rating."
    assert analysis.known_facts[0].evidence_ids == ["E1"]
    assert analysis.known_facts[1].text == "The website lists a physical showroom."
    assert analysis.known_facts[1].evidence_ids == ["E5"]
    assert analysis.evidence_claims[0].text == "The source identifies a roadmap."
    assert analysis.evidence_claims[0].evidence_ids == ["E1"]
    assert analysis.evidence_claims[1].text == "The public material describes a phased engagement."
    assert analysis.evidence_claims[1].evidence_ids == ["E2"]
    assert analysis.analytical_inferences == ["Sequencing should be validated."]
    assert analysis.hypotheses == ["A phased model may reduce execution risk."]


def make_workflow(**kwargs):
    return DocumentWorkflow(
        retriever=DocumentFakeRetriever(),
        reasoner=LocalReasoner(),
        max_section_iterations=3,
        evidence_k=3,
        **kwargs,
    )


def make_spec():
    return DocumentSpec(client_brief=DEFAULT_BRIEF, target_depth="Demo")


def test_document_workflow_executes_end_to_end_with_fake_components():
    trace = make_workflow().run(make_spec())
    assert trace.plan.sections
    assert len(trace.sections) >= 5
    assert trace.final_markdown.startswith("# AWS Well-Architected")
    assert trace.total_research_iterations >= len([section for section in trace.sections if section.section_id != "evidence"])
    assert trace.final_qc.passed


def test_section_research_evidence_is_associated_with_correct_section():
    trace = make_workflow().run(make_spec())
    reliability = next(section for section in trace.sections if section.section_id == "reliability")
    assert reliability.evidence
    assert all(chunk.section == "Reliability" for chunk in reliability.evidence)
    assert reliability.research_trace.iterations


def test_document_citation_validation_is_preserved():
    trace = make_workflow().run(make_spec())
    assert trace.citation_validation.valid is True
    assert trace.citation_validation.cited_pages
    assert not trace.citation_validation.uncited_claims


def test_failed_section_qc_triggers_no_more_than_one_revision():
    qc = FailsOnceQC()
    trace = make_workflow(qc_runner=qc).run(make_spec())
    assert any(section.revised for section in trace.sections)
    assert max(section.revision_count for section in trace.sections) == 1


def test_passing_section_qc_does_not_trigger_revision():
    trace = make_workflow(qc_runner=AlwaysPassQC()).run(make_spec())
    assert all(section.revision_count == 0 for section in trace.sections)


def test_docx_and_markdown_exports_are_non_empty_and_contain_expected_content():
    trace = make_workflow().run(make_spec())
    docx = docx_bytes(trace)
    markdown = markdown_bytes(trace)
    assert len(docx) > 1000
    assert b"Executive Summary" in markdown
    assert b"Reliability Assessment" in markdown


def test_document_trace_serializes_correctly():
    trace = make_workflow().run(make_spec())
    payload = trace.model_dump_json()
    assert "final_qc" in payload
    assert "research_trace" in payload


def test_uploaded_pdf_chunks_use_uploaded_source_name():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This strategy report describes curriculum risks, recommendations, and implementation steps.")
    content = doc.tobytes()
    doc.close()

    chunks = extract_pdf_bytes(content, "Strategy_Report.pdf")
    assert chunks
    assert chunks[0].source == "Strategy_Report.pdf"
    assert chunks[0].page == 1


def test_uploaded_mode_uses_generic_sections_without_aws_specific_leakage():
    spec = DocumentSpec(
        title="Client Strategy Report",
        client_brief="Analyze the uploaded strategy document and recommend next steps.",
        knowledge_base="Uploaded documents: Strategy_Report.pdf",
        source_kind="uploaded",
        target_depth="Demo",
    )
    trace = make_workflow().run(spec)
    titles = [section.title for section in trace.plan.sections]
    assert "Reliability Assessment" not in titles
    assert "Security Assessment" not in titles
    assert "Cost Optimization Assessment" not in titles
    assert "AWS Well-Architected" not in trace.final_markdown


def test_auto_summary_brief_uses_concise_summary_structure_without_roadmap():
    spec = DocumentSpec(
        title="Presentation Summary",
        client_brief="what does this talk about, explain briefly",
        knowledge_base="Uploaded documents: Presentation.pdf",
        source_kind="uploaded",
        deliverable_type="Auto",
        target_depth="Demo",
    )
    trace = make_workflow().run(spec)
    assert trace.plan.deliverable_type == "Summary / Brief"
    assert [section.section_id for section in trace.plan.sections] == [
        "overview",
        "major-themes",
        "conclusion",
        "evidence",
    ]
    forbidden = ("Implementation Roadmap", "Prioritized Recommendations", "acceptance criteria", "rollout", "Priority 1")
    assert not any(term in trace.final_markdown for term in forbidden)
    assert trace.final_qc.passed


def test_auto_summary_intent_overrides_default_standard_depth():
    spec = DocumentSpec(
        client_brief="what does this document talk about, explain briefly",
        source_kind="uploaded",
        deliverable_type="Auto",
        target_depth="Standard",
    )
    assert resolve_deliverable_type(spec) == "Summary / Brief"
    assert target_word_count(spec) is None
    plan = make_workflow().plan(spec, [])
    assert [section.section_id for section in plan.sections] == ["overview", "major-themes", "conclusion", "evidence"]


def test_auto_methodology_summary_is_research_report_not_generic_brief():
    spec = DocumentSpec(client_brief="summarize the methodology and findings", source_kind="uploaded")
    assert resolve_deliverable_type(spec) == "Research Report"


def test_document_workflow_does_not_research_references_section():
    workflow = make_workflow()
    trace = workflow.run(make_spec())
    evidence_section = next(section for section in trace.sections if section.section_id == "evidence")
    assert evidence_section.research_trace.stop_reason == "generated_from_used_citations"
    assert not any("references evidence sources" in query.lower() for query in workflow.retriever.calls)


def test_auto_classifies_research_and_consulting_briefs_differently():
    research = DocumentSpec(client_brief="summarize the methodology and findings", source_kind="uploaded")
    consulting = DocumentSpec(client_brief="identify risks, recommend remediations, give a 90-day roadmap", source_kind="uploaded")
    assert resolve_deliverable_type(research) == "Research Report"
    assert resolve_deliverable_type(consulting) == "Consulting Assessment"


def test_strong_strategy_intent_overrides_accidental_briefly_prefix_in_auto_mode():
    spec = DocumentSpec(
        client_brief="what does this talk about, explain briefly. Create a new business strategy report with Stage 1, Stage 2 and Stage 3, including a roadmap and KPI framework.",
        source_kind="uploaded",
        deliverable_type="Auto",
    )
    assert resolve_deliverable_type(spec) == "Consulting Assessment"


def test_charley_strategy_scope_overrides_stale_curriculum_selection():
    brief = """Create a business strategy report for Speckled Space.

Stage 1 — Business Diagnostic & Competitive Landscape
1.1 Organisational Health Check & Operating Model Review
1.2 Competitive Landscape & Market Positioning Assessment

Stage 2 — Strategic Planning & Business Development
2.1 Business Model & Financial Strategy
2.2 Workforce Planning
2.3 Brand & Marketing Strategy
2.4 Internationalisation
2.5 AI Adoption Roadmap
2.6 Strategic Roadmap and KPI framework

Stage 3 — Implementation Planning and Knowledge Transfer"""
    spec = DocumentSpec(
        title="Evidence-Grounded Report",
        client_brief=brief,
        source_kind="uploaded",
        deliverable_type="Curriculum / Teaching Material",
        company_website="https://speckledspace.com/",
    )

    assert curriculum_strategy_mismatch(brief, spec.deliverable_type)
    assert resolve_deliverable_type(spec) == "Consulting Assessment"
    plan = make_workflow().plan(spec, [])
    assert plan.title == "Speckled Space Business Strategy Report"
    assert plan.deliverable_type == "Consulting Assessment"
    titles = " ".join(section.title for section in plan.sections)
    assert "Course Overview" not in titles
    assert "Foundations and Definitions" not in titles
    assert "Stage 1" in titles and "Stage 2" in titles and "Stage 3" in titles

    trace = DocumentWorkflow(
        retriever=DocumentFakeRetriever(),
        reasoner=LocalReasoner(),
        max_section_iterations=1,
        external_research=False,
    ).run(spec)
    assert trace.plan.deliverable_type == "Consulting Assessment"
    assert "A second way to understand" not in trace.final_markdown
    assert "This is part of course overview" not in trace.final_markdown.lower()


def test_summary_brief_overrides_mismatched_manual_curriculum_type():
    spec = DocumentSpec(
        client_brief="what does this talk about, explain briefly in 400 words",
        source_kind="uploaded",
        deliverable_type="Curriculum / Teaching Material",
    )
    assert resolve_deliverable_type(spec) == "Summary / Brief"


def test_uploaded_text_cleaning_removes_slide_counters_and_fragments_preserves_arabic():
    text = clean_uploaded_text(
        """
        Presentation Title
        17 / 17
        Confidential
        هذا العرض يشرح أهداف البرنامج ونتائجه المتوقعة.
        The presentation explains the program goals and expected outcomes.
        Slide 2 of 9
        """,
        "Presentation.pdf",
    )
    assert "17 / 17" not in text
    assert "Slide 2 of 9" not in text
    assert "Confidential" not in text
    assert "هذا العرض" in text


def test_uploaded_text_cleaning_repairs_extraction_hyphenation():
    text = clean_uploaded_text(
        "The presentation explains re- quired program goals and expected outcomes.",
        "Presentation.pdf",
    )
    assert "re- quired" not in text
    assert "required program goals" in text


def test_serializable_evidence_handles_hot_reload_model_identity():
    chunk = EvidenceChunk(chunk_id="doc-p001-c00", page=1, text="A complete source-backed sentence.", source="doc.pdf")
    serialized = serializable_evidence([chunk])
    assert serialized == [chunk.model_dump()]
    assert serialized[0]["source"] == "doc.pdf"


def test_document_plan_serializes_source_survey_evidence_for_streamlit_reruns():
    spec = DocumentSpec(
        client_brief="what does this talk about, explain briefly",
        source_kind="uploaded",
        target_depth="Standard",
    )
    survey = [EvidenceChunk(chunk_id="survey-1", page=2, text="The source introduces its subject.", source="doc.pdf")]
    plan = make_workflow().plan(spec, survey)
    assert plan.source_survey[0].chunk_id == "survey-1"
    assert plan.model_dump()["source_survey"][0]["source"] == "doc.pdf"


def test_uploaded_retrieval_diversification_limits_per_page_without_forcing_irrelevance():
    candidates = [
        EvidenceChunk(chunk_id=f"p1-{index}", page=1, source="doc.pdf", text=f"alpha topic detail {index}", score=1.0 - index * 0.01)
        for index in range(5)
    ] + [
        EvidenceChunk(chunk_id="p2-0", page=2, source="doc.pdf", text="beta topic detail", score=0.9),
        EvidenceChunk(chunk_id="p3-0", page=3, source="doc.pdf", text="gamma topic detail", score=0.88),
    ]
    selected = diversify_results(candidates, k=5, max_per_page=2)
    assert len(selected) == 4
    assert sum(1 for chunk in selected if chunk.page == 1) <= 2
    assert {chunk.page for chunk in selected} >= {1, 2, 3}


def test_citation_coverage_flags_single_page_concentration_when_evidence_is_broad():
    evidence = [
        EvidenceChunk(chunk_id="p1", page=1, source="doc.pdf", text="Alpha topic explains data collection and project scope.", score=0.9),
        EvidenceChunk(chunk_id="p2", page=2, source="doc.pdf", text="Beta topic explains methodology and evaluation.", score=0.8),
        EvidenceChunk(chunk_id="p3", page=3, source="doc.pdf", text="Gamma topic explains findings and conclusions.", score=0.7),
    ]
    content = "\n\n".join(
        [
            "Alpha topic explains data collection. [doc.pdf p.1]",
            "Alpha topic explains project scope. [doc.pdf p.1]",
            "Alpha topic explains background. [doc.pdf p.1]",
        ]
    )
    issues = citation_coverage_issues(content, evidence)
    assert any("Citation concentration" in issue for issue in issues)


def test_references_are_built_from_citations_used_in_report():
    trace = make_workflow().run(make_spec())
    references = trace.final_markdown.split("## Evidence / References", 1)[1]
    assert "[AWS-WAF p." in references or "[AWS Well-Architected Framework p." in references
    assert trace.citation_validation.cited_references


def test_depth_profiles_and_explicit_word_count_are_independent_from_deliverable_type():
    spec = DocumentSpec(
        client_brief="Create a detailed 3,000-word study guide.",
        source_kind="uploaded",
        deliverable_type="Curriculum / Teaching Material",
        target_depth="Brief",
    )
    assert DEPTH_PROFILES["Detailed"][1] == (5000, 8000)
    assert explicit_word_count(spec.client_brief) == 3000
    assert target_word_count(spec) == 3000


def test_long_brief_survives_document_spec_without_truncation():
    brief = "Speckled Space scope\n" + ("Validate the proposed strategic workstream and evidence boundary. " * 250)
    assert len(brief) > 10000
    spec = DocumentSpec(client_brief=brief, source_kind="uploaded")
    assert spec.client_brief == brief
    assert len(spec.client_brief) == len(brief)
    assert MAX_BRIEF_LENGTH >= 40000


def test_uploaded_workflow_brief_starts_empty_and_does_not_prefix_strategy_scope():
    strategy_scope = "Create a new business strategy report. Stage 1 - Diagnostic. Stage 2 - Planning. Stage 3 - Training."
    assert default_client_brief("Upload documents") == ""
    assert strategy_scope == strategy_scope.strip()
    assert not strategy_scope.startswith("what does this talk about")


def test_website_context_rejects_unbounded_or_insecure_urls_gracefully():
    chunks, notice = fetch_website_evidence("http://example.com")
    assert chunks == []
    assert notice


def test_shopify_like_website_ingestion_returns_chunks(monkeypatch):
    fixture = (Path(__file__).parent / "fixtures" / "speckledspace_home.html").read_bytes()

    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return self.url

        def read(self, _limit):
            return fixture

    monkeypatch.setattr(website_context, "urlopen", lambda request, timeout=8: Response(request.full_url))
    chunks, notice, report = fetch_website_evidence("https://speckledspace.test/", max_pages=5, return_report=True)
    assert chunks
    assert notice is None
    assert len(report.indexed_pages) >= 1
    assert any("free delivery" in chunk.text.lower() for chunk in chunks)
    assert report.character_counts


def test_redirected_website_ingestion_returns_chunks(monkeypatch):
    fixture = (Path(__file__).parent / "fixtures" / "speckledspace_home.html").read_bytes()

    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://www.speckledspace.test/" if self.url == "https://speckledspace.test/" else self.url

        def read(self, _limit):
            return fixture

    monkeypatch.setattr(website_context, "urlopen", lambda request, timeout=8: Response(request.full_url))
    chunks, notice, report = fetch_website_evidence("http://speckledspace.test", max_pages=5, return_report=True)
    assert chunks
    assert notice is None
    assert report.resolved_url == "https://www.speckledspace.test/"
    assert all(chunk.source == "Website: speckledspace.test" for chunk in chunks)


def test_requested_website_failure_prevents_deliverable_ready():
    spec = DocumentSpec(
        title="Speckled Space Business Strategy Report",
        client_brief="Create a business strategy report with market analysis and recommendations.",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
        company_website="https://speckledspace.test/",
    )
    trace = DocumentWorkflow(
        retriever=EmptyRetriever(),
        reasoner=LocalReasoner(),
        external_research=False,
    ).run(spec)
    assert not trace.final_qc.passed
    assert "Requested website evidence could not be acquired." in trace.final_qc.major_issues


def test_consulting_report_with_zero_total_evidence_fails_final_qc():
    spec = DocumentSpec(
        title="Client Strategy Report",
        client_brief="Create a strategy assessment with recommendations and a roadmap.",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
    )
    trace = DocumentWorkflow(retriever=EmptyRetriever(), reasoner=LocalReasoner(), external_research=False).run(spec)
    assert not trace.final_qc.passed
    assert "No factual client or public evidence was successfully acquired." in trace.final_qc.major_issues


class CapableSectionReasoner(LocalReasoner):
    def __init__(self, *, fail: bool = False):
        self.calls = []
        self.model = "test-model"
        self.fail = fail

    def _json(self, _system, _payload, schema):
        self.calls.append(schema.__name__)
        if self.fail:
            raise RuntimeError("synthetic synthesis outage")
        if schema is SectionAnalysis:
            return SectionAnalysis(
                section_id="analysis",
                section_mode="diagnostic",
                objective="Assess the workstream.",
                observable_context=["The supplied public material is limited."],
                analytical_inferences=["Compare the relevant decision criteria."],
                hypotheses=["The internal condition remains a hypothesis."],
                data_gaps=["Internal baseline and performance data are required."],
                recommended_analysis=["Test the hypothesis against the baseline."],
                planned_paragraphs=["context", "analysis", "data gaps"],
            )
        return SectionDraft(markdown="Assess the decision criteria and state what internal data is required before reaching a company-specific conclusion.")


class BatchConsultingReasoner(LocalReasoner):
    model = "test-consulting-model"

    def __init__(self):
        self.calls = []

    def _json(self, _system, payload, schema, **_kwargs):
        self.calls.append(schema.__name__)
        if schema is StrategyReportAnalysis:
            return StrategyReportAnalysis(
                client_profile=["Speckled Space is assessed from public evidence."],
                recommendations=["Validate the operating baseline and sequence growth work."],
                data_gaps=["Internal financial, workforce, and customer data are required."],
                evidence_map=[{"text": "The website describes the public proposition.", "evidence_ids": ["E1"]}],
                publicly_observable_facts=[{"text": "The website describes the public proposition.", "evidence_ids": ["E1"]}],
            )
        if schema is ReportBatchDraft:
            sections = [
                ReportSectionDraft(
                    section_id=item["section_id"],
                    title=item["title"],
                    markdown=(
                        f"Speckled Space should assess {item['title'].lower()} using public evidence and internal validation. [E1]\n\n"
                        "Internal data is required before treating operating, financial, workforce, or market hypotheses as facts. [E1]"
                    ),
                )
                for item in payload["batch_sections"]
            ]
            return ReportBatchDraft(sections=sections)
        return SectionDraft(markdown="")


def test_empty_section_evidence_still_uses_capable_synthesis_model():
    reasoner = CapableSectionReasoner()
    spec = DocumentSpec(
        client_brief="Create a strategy assessment.",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
    )
    trace = DocumentWorkflow(retriever=EmptyRetriever(), reasoner=reasoner, external_research=False).run(spec)
    assert "SectionAnalysis" in reasoner.calls
    assert "SectionDraft" in reasoner.calls
    assert any(section.analysis_model_used and section.synthesis_model_used for section in trace.sections if section.section_id != "evidence")


def test_consulting_smoke_mode_executes_only_selected_section_and_disables_external_research(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST_SECTIONS", "executive-summary")
    reasoner = CapableSectionReasoner()
    spec = DocumentSpec(
        client_brief="Create a business strategy report with market analysis and recommendations.",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
    )

    trace = DocumentWorkflow(
        retriever=DocumentFakeRetriever(),
        reasoner=reasoner,
        external_research=True,
    ).run(spec)

    executed = [section for section in trace.sections if section.section_id != "evidence"]
    assert trace.smoke_test_mode is True
    assert trace.smoke_test_sections == ["executive-summary"]
    assert [section.section_id for section in executed] == ["executive-summary"]
    assert len(trace.plan.sections) > len(trace.sections)
    assert trace.external_research_enabled is False
    assert executed[0].latency_ms is not None
    assert executed[0].analysis_model_used is True
    assert executed[0].synthesis_model_used is True


def test_consulting_smoke_mode_accepts_numbered_scope_selectors(monkeypatch):
    for selector, expected in (("1.1", "scope-1-1"), ("1.2", "scope-1-2")):
        monkeypatch.setenv("SMOKE_TEST_SECTIONS", selector)
        spec = DocumentSpec(
            client_brief=(
                "Create a business strategy report.\nStage 1 — Business Diagnostic\n"
                "1.1 Organisational Health Check\n1.2 Competitive Landscape"
            ),
            source_kind="uploaded",
            deliverable_type="Consulting Assessment",
        )
        trace = DocumentWorkflow(
            retriever=DocumentFakeRetriever(),
            reasoner=CapableSectionReasoner(),
            external_research=True,
        ).run(spec)
        assert [section.section_id for section in trace.sections if section.section_id != "evidence"] == [expected]
        assert trace.external_research_enabled is False


def test_uploaded_consulting_uses_dedicated_batched_path_without_legacy_revision():
    brief = """Create a business strategy report for Speckled Space.

Stage 1 - Business Diagnostic & Competitive Landscape
1.1 Organisational Health Check & Operating Model Review
1.2 Competitive Landscape & Market Positioning Assessment

Stage 2 - Strategic Planning & Business Development
2.1 Market & Customer Intelligence
2.2 Business Model & Financial Strategy
2.3 Workforce Planning
2.4 Brand & Marketing Strategy
2.5 Internationalisation
2.6 AI Integration Review & Adoption Roadmap
2.7 Strategic Roadmap & Comprehensive Action Plan

Stage 3 - Implementation Planning and Knowledge Transfer
3.1 Strategic Frameworks & Execution Training Workshop
3.2 AI-Enabled Process & Digital Tools Workshop
3.3 Playbooks, Change Readiness & Project Closure"""
    reasoner = BatchConsultingReasoner()
    spec = DocumentSpec(
        title="Evidence-Grounded Report",
        client_brief=brief,
        source_kind="uploaded",
        deliverable_type="Auto",
        target_depth="Standard",
        company_website="https://speckledspace.com/",
    )
    trace = DocumentWorkflow(
        retriever=DocumentFakeRetriever(),
        reasoner=reasoner,
        external_research=False,
    ).run(spec)

    assert trace.plan.deliverable_type == "Consulting Assessment"
    assert trace.analysis_llm_calls == 1
    assert trace.synthesis_llm_calls <= 4
    assert trace.total_llm_calls <= 6
    assert trace.total_research_iterations == 0
    assert "connect the requested focus" not in trace.final_markdown
    assert "Knowledge Base" not in trace.final_markdown
    assert "Client Brief" not in trace.final_markdown
    assert "E1" not in trace.final_markdown


def test_local_fallback_cannot_mark_long_consulting_report_ready():
    class EvidenceRetriever:
        def search(self, _query, k=6):
            return [EvidenceChunk(chunk_id="client-1", page=1, source="client.pdf", text="Market strategy analysis covers customer demand, financial context, capabilities, roadmap, and recommendations.", score=0.9)]

    spec = DocumentSpec(
        client_brief="Create a business strategy report with market analysis and recommendations.",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
        target_depth="Standard",
    )
    trace = DocumentWorkflow(retriever=EvidenceRetriever(), reasoner=LocalReasoner(), external_research=False).run(spec)
    assert not trace.final_qc.passed
    assert any("fallback" in issue.lower() for issue in trace.final_qc.major_issues)


def test_requirements_only_fallback_is_bounded_not_word_padded():
    spec = DocumentSpec(client_brief="Create a detailed strategy report.", source_kind="uploaded", deliverable_type="Consulting Assessment")
    section = DocumentSectionPlan(section_id="market", title="Market Analysis", objective="Assess customer segments and market sizing.", approximate_word_budget=5000)
    content = requirements_only_section(spec, section)
    assert len(content.split()) <= 180
    assert "The market analysis workstream is part of" not in content


def test_cross_section_duplication_checks_no_evidence_sections():
    sections = [
        GeneratedSection(section_id=f"s{i}", title=f"Section {i}", objective="Assess the workstream.", content_markdown="The market workstream is part of the requested engagement and should assess the available evidence.", evidence=[], research_trace=AgentTrace(question="q"), qc=SectionQC(passed=True))
        for i in range(1, 4)
    ]
    assert cross_section_duplication(sections)


def test_synthesis_failure_is_exposed_in_trace():
    spec = DocumentSpec(client_brief="Create a strategy assessment.", source_kind="uploaded", deliverable_type="Consulting Assessment")
    trace = DocumentWorkflow(retriever=EmptyRetriever(), reasoner=CapableSectionReasoner(fail=True), external_research=False).run(spec)
    assert any(section.synthesis_error for section in trace.sections if section.section_id != "evidence")


def test_external_research_limitation_appears_once():
    spec = DocumentSpec(
        client_brief="Create a business strategy report.\nStage 1 - Competitive Landscape and Market Analysis.\nStage 2 - Planning.\nStage 3 - Training.",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
    )
    trace = DocumentWorkflow(retriever=EmptyRetriever(), reasoner=LocalReasoner(), external_research=False).run(spec)
    marker = "Public competitor, market, and internationalisation sources were not available"
    applicable = [section for section in trace.sections if "market" in section.title.lower() or "competitive" in section.objective.lower()]
    assert applicable
    assert all(section.content_markdown.count(marker) <= 1 for section in applicable)


def test_stage_scope_is_preserved_in_strategy_plan():
    brief = """Create a new business strategy report.\n\nStage 1 — Business Diagnostic & Competitive Landscape\n1.1 Organisational Health Check & Operating Model Review — assess structure and workflows.\n1.2 Competitive Landscape & Market Positioning Assessment — benchmark positioning.\n\nStage 2 — Strategic Planning & Business Development\n2.1 Market & Customer Intelligence — segment and size opportunities.\n2.6 AI Integration Review & Adoption Roadmap — prioritize use cases.\n2.7 Strategic Roadmap & Comprehensive Action Plan — define KPIs.\n\nStage 3 — Training, Implementation Planning & Knowledge Transfer\n3.1 Strategic Frameworks & Execution Training Workshop\n3.2 AI-Enabled Process & Digital Tools Workshop\n3.3 Playbooks, Change Readiness & Project Closure"""
    spec = DocumentSpec(client_brief=brief, source_kind="uploaded", target_depth="Detailed", deliverable_type="Auto")
    plan = make_workflow().plan(spec, [])
    titles = " ".join(section.title for section in plan.sections)
    for marker in ("Stage 1", "1.1", "1.2", "Stage 2", "2.1", "2.6", "2.7", "Stage 3", "3.1", "3.2", "3.3"):
        assert marker in titles
    assert len(extract_scope_items(brief)) >= 10
    assert plan.deliverable_type == "Consulting Assessment"


def test_reference_profile_is_structure_only_and_old_client_name_is_checked():
    class Reference:
        chunks = [EvidenceChunk(chunk_id="ref-1", page=1, text="Executive Summary\nRecommendations\nRoadmap and KPI matrix", source="Core Biz Holdings - Business Strategy EDG Report V2.pdf")]

    reference = Reference()
    profile = analyze_reference_report(reference)
    assert profile.approximate_word_count >= 1
    assert profile.title
    clean_spec = DocumentSpec(
        client_brief="Create a new strategy report for Speckled Space.",
        source_kind="uploaded",
        reference_source_names=["Core Biz Holdings - Business Strategy EDG Report V2.pdf"],
    )
    assert reference_leakage_issues(clean_spec, []) == []
    leaked = GeneratedSection(
        section_id="findings",
        title="Findings",
        objective="Assess the new client.",
        content_markdown="Core Biz Holdings has established this operating model.",
        research_trace=AgentTrace(question="findings"),
        qc=SectionQC(passed=True),
    )
    assert reference_leakage_issues(clean_spec, [leaked]) == clean_spec.reference_source_names


def test_detailed_curriculum_plan_has_meaningful_sections_and_research_questions():
    spec = DocumentSpec(
        client_brief="Create a detailed 3,000-word study guide explaining the major topics and provide a final revision checklist.",
        source_kind="uploaded",
        deliverable_type="Curriculum / Teaching Material",
        target_depth="Detailed",
    )
    plan = make_workflow().plan(spec, [])
    content_sections = [section for section in plan.sections if section.section_id != "evidence"]
    assert len(content_sections) >= 6
    assert all(len(section.questions) >= 2 for section in content_sections)
    assert sum(section.approximate_word_budget or 0 for section in content_sections) == 3000


def test_detailed_uploaded_workflow_hits_requested_scale_and_tracks_metrics():
    spec = DocumentSpec(
        title="Course Study Guide",
        client_brief="Create a detailed 3,000-word study guide explaining the major mathematical topics covered by this course. Provide a final revision checklist.",
        source_kind="uploaded",
        deliverable_type="Curriculum / Teaching Material",
        target_depth="Detailed",
    )
    trace = make_workflow().run(spec)
    assert 2550 <= trace.final_word_count <= 3450
    assert trace.final_qc.passed
    assert trace.total_retrieved_evidence_chunks >= trace.total_unique_cited_pages
    assert trace.total_unique_cited_pages >= 1
    assert "Final Revision Checklist" in trace.final_markdown


def test_pdf_export_is_a_valid_nonempty_pdf():
    trace = make_workflow().run(make_spec())
    output = pdf_bytes(trace)
    assert output.startswith(b"%PDF-")
    assert len(output) > 1000


def test_source_survey_topics_are_semantic_not_retrieved_sentence_headings():
    survey = [
        EvidenceChunk(
            chunk_id="one",
            page=1,
            source="client.pdf",
            text="Manual prompting became the first major engineering bottleneck for the project.",
            score=0.9,
        ),
        EvidenceChunk(
            chunk_id="two",
            page=2,
            source="client.pdf",
            text="The methodology evaluates the study using a controlled data collection approach and reports findings.",
            score=0.8,
        ),
    ]
    topics = source_topic_labels(survey)
    assert topics
    assert "Manual prompting became the first major engineering bottleneck" not in topics
    assert any(topic in topics for topic in ("Methodology", "Key Findings"))


def test_section_relevance_gate_rejects_tangential_chunks_and_caps_pages():
    section = DocumentSectionPlan(
        section_id="methodology",
        title="Methodology",
        objective="Explain the research method and evaluation approach.",
        research_questions=["Which method and evaluation approach does the source document?"],
    )
    candidates = [
        EvidenceChunk(chunk_id="primary", page=1, source="client.pdf", text="The methodology uses a controlled evaluation approach.", score=0.8),
        EvidenceChunk(chunk_id="support", page=2, source="client.pdf", text="The study reports the evaluation method and sample design.", score=0.7),
        EvidenceChunk(chunk_id="tangent", page=3, source="client.pdf", text="The company inventory lists office locations and unrelated staffing totals.", score=0.95),
    ]
    selected = filter_section_evidence(section, candidates, requested_k=8)
    assert {chunk.chunk_id for chunk in selected} == {"primary", "support"}


def test_long_form_writer_avoids_known_filler_templates():
    spec = DocumentSpec(client_brief="Create a detailed report.", source_kind="uploaded", target_depth="Detailed")
    section = DocumentSectionPlan(
        section_id="findings",
        title="Key Findings",
        objective="Synthesize the source-backed findings.",
        approximate_word_budget=300,
        requirements=["findings"],
    )
    evidence = [EvidenceChunk(chunk_id="f1", page=4, source="client.pdf", text="The study reports a repeatable evaluation method and records measurable findings across the sample.", score=0.9)]
    content = build_long_form_section(spec, section, evidence, [])
    assert "the material also explains" not in content.lower()
    assert "this gives the reader a grounded way" not in content.lower()
    assert "taken with the other retrieved passages" not in content.lower()
    assert "this section addresses" not in content.lower()


def test_document_section_filter_does_not_inject_irrelevant_sparse_evidence():
    section = DocumentSectionPlan(
        section_id="market",
        title="Market Analysis",
        objective="Assess customer segments and market sizing.",
        requirements=["market"],
    )
    unrelated = [EvidenceChunk(chunk_id="unrelated", page=2, source="client.pdf", text="A private office seating arrangement and internal staff roster.", score=0.95)]
    assert filter_section_evidence(section, unrelated, requested_k=6) == []


def test_evidence_ids_are_replaced_deterministically_and_unknown_ids_rejected():
    evidence = [EvidenceChunk(chunk_id="one", page=12, source="Strategy_Report.pdf", text="A supported finding about the operating model.")]
    cleaned, unknown = replace_evidence_ids("A supported finding [E1]. Another claim [E9].", evidence)
    assert "[Strategy_Report.pdf p.12]" in cleaned
    assert "[E9]" not in cleaned
    assert unknown == ["E9"]


def test_grouped_evidence_ids_are_replaced_without_raw_tokens():
    evidence = [
        EvidenceChunk(chunk_id="E1", page=1, source="Website: speckledspace.com", text="The website describes home decor products."),
        EvidenceChunk(chunk_id="E2", page=2, source="Website: speckledspace.com", text="The website describes delivery and service."),
        EvidenceChunk(chunk_id="E3", page=3, source="Website: speckledspace.com", text="The website describes showroom contact details."),
    ]
    cleaned, unknown = replace_evidence_ids("The proposition is visible [E1, E2]. Contact details are visible [E3].", evidence)
    assert unknown == []
    assert "E1" not in cleaned and "E2" not in cleaned and "E3" not in cleaned
    assert "[Website: speckledspace.com p.1][Website: speckledspace.com p.2]" in cleaned


def test_duplicate_section_heading_is_removed_from_model_body():
    body = "## Stage 1 - Business Diagnostic\n\n### Stage 1 - Business Diagnostic\n\nUseful analysis [E1]."
    cleaned = sanitize_generated_section_body(body, "Stage 1 - Business Diagnostic")
    assert not cleaned.startswith("#")
    assert duplicate_heading_count("## Stage 1 - Business Diagnostic\n### Stage 1 - Business Diagnostic\n") == 1


def test_consulting_final_qc_rejects_legacy_revision_language_and_raw_e_ids():
    spec = DocumentSpec(
        client_brief="Create a business strategy report.\nStage 1 - Diagnostic\n1.1 Health\n1.2 Market\nStage 2 - Planning\n2.1 Market\n2.2 Finance\n2.3 Workforce\n2.4 Brand\n2.5 International\n2.6 AI\n2.7 Roadmap\nStage 3 - Implementation\n3.1 Workshop\n3.2 AI Tools\n3.3 Playbooks",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
    )
    plan = make_workflow().plan(spec, [])
    issues = consulting_final_output_issues("connect the requested focus [E1]\n\nCore Biz Holdings", spec, plan)
    assert any("Raw E" in issue for issue in issues)
    assert any("connect the requested focus" in issue for issue in issues)
    assert any("Core Biz" in issue for issue in issues)


def test_consulting_front_matter_excludes_debug_metadata():
    spec = DocumentSpec(
        title="Speckled Space Business Strategy Report",
        client_brief="Create a business strategy report.",
        source_kind="uploaded",
        deliverable_type="Consulting Assessment",
        reference_source_names=["Core Biz Holdings - Business Strategy EDG Report V2.pdf"],
        company_website="https://speckledspace.com/",
    )
    plan = make_workflow().plan(spec, [])
    section = GeneratedSection(
        section_id="executive-summary",
        title="Executive Summary",
        objective="Summarize.",
        content_markdown="Speckled Space has a public website. [Website: speckledspace.com p.1]",
        evidence=[],
        research_trace=AgentTrace(question="q"),
        qc=SectionQC(passed=True, citation_valid=True),
    )
    markdown = assemble_consulting_markdown(spec, plan, [section])
    assert markdown.startswith("# Speckled Space Business Strategy Report")
    assert "Knowledge Base" not in markdown
    assert "Reference Precedent" not in markdown
    assert "Client Brief" not in markdown


def test_citation_sentence_split_does_not_break_pdf_filename():
    sentences = cited_sentences("A supported statement uses the report evidence. [Strategy_Report.pdf p.12]")
    assert len(sentences) == 1
    assert "[Strategy_Report.pdf p.12]" in sentences[0]


def test_citation_tokens_do_not_trigger_repeated_stem_or_gap_support_failure():
    evidence = [
        EvidenceChunk(chunk_id="p1", page=1, source="client.pdf", text="The source describes the customer proposition and service model."),
        EvidenceChunk(chunk_id="p2", page=2, source="client.pdf", text="The source describes the product assortment and delivery experience."),
        EvidenceChunk(chunk_id="p3", page=3, source="client.pdf", text="The source describes customer reviews and showroom support."),
    ]
    content = (
        "The supplied material does not establish organisational structure or competitive market share. "
        "[client.pdf p.1][client.pdf p.2][client.pdf p.3]"
    )
    assert internal_pipeline_language_issues(content) == []
    assert citation_coverage_issues(content, evidence) == []


def test_citation_support_accepts_extracted_text_paraphrase():
    evidence = [
        EvidenceChunk(
            chunk_id="p4",
            page=4,
            source="client.pdf",
            text="Products are organized by residential space and customer use case, with seasonal promotions.",
        ),
        EvidenceChunk(chunk_id="p5", page=5, source="client.pdf", text="The source lists customer support and delivery information."),
        EvidenceChunk(chunk_id="p6", page=6, source="client.pdf", text="The source describes the product assortment."),
    ]
    content = (
        "The customer-facing proposition uses space-based merchandising and seasonal offers to guide selection. "
        "[client.pdf p.4][client.pdf p.5][client.pdf p.6]"
    )
    assert citation_coverage_issues(content, evidence) == []


class FailsOnceQC:
    def __init__(self):
        self.failed = False

    def check(self, section, content, evidence):
        if not self.failed:
            self.failed = True
            return SectionQC(
                passed=False,
                missing_requirements=["revision marker"],
                citation_valid=True,
                issues=["Needs one revision."],
                revision_instructions="Add revision marker.",
            )
        return SectionQC(passed=True, requirements_covered=["revision marker"], citation_valid=True)


class AlwaysPassQC:
    def check(self, section, content, evidence):
        return SectionQC(passed=True, requirements_covered=["ok"], citation_valid=True)

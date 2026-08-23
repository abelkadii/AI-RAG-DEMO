from document_export import docx_bytes, markdown_bytes, pdf_bytes
from document_models import DocumentSpec, SectionQC
from document_workflow import (
    DEFAULT_BRIEF,
    DocumentWorkflow,
    citation_coverage_issues,
    resolve_deliverable_type,
    serializable_evidence,
    DEPTH_PROFILES,
    explicit_word_count,
    target_word_count,
)
from llm import LocalReasoner
from models import EvidenceChunk
from uploaded_corpus import clean_uploaded_text, diversify_results, extract_pdf_bytes


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
    assert DEPTH_PROFILES["Detailed"][0] == 3000
    assert explicit_word_count(spec.client_brief) == 3000
    assert target_word_count(spec) == 3000


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

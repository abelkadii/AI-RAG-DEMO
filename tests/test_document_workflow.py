from document_export import docx_bytes, markdown_bytes
from document_models import DocumentSpec, SectionQC
from document_workflow import DEFAULT_BRIEF, DocumentWorkflow
from llm import LocalReasoner
from models import EvidenceChunk


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
    assert trace.total_research_iterations >= len(trace.sections)
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

from agent import Agent
from document_models import SectionAnalysis
from llm import LocalReasoner, OpenAIReasoner, configured_reasoner
from models import AgentState, EvidenceChunk, EvidenceAssessment, SearchDecision


class FakeRetriever:
    def search(self, query, k=6):
        return [EvidenceChunk(chunk_id="same", page=42, section="Cost Optimization", text="Cost optimization guidance.", score=0.8)]


class NeverEnough(LocalReasoner):
    def assess(self, state):
        result = super().assess(state)
        return result.model_copy(update={"sufficient": False})


class MultiPillarRetriever:
    def search(self, query, k=6):
        if "cost" in query.lower():
            return [EvidenceChunk(chunk_id="cost", page=20, section="Cost Optimization", text="Select resource type, size, and number based on data to minimize waste.", score=0.8)]
        if "security" in query.lower():
            return [EvidenceChunk(chunk_id="security", page=30, section="Security", text="Identity and access management, data protection, infrastructure security, logging, and monitoring protect workloads.", score=0.8)]
        return [EvidenceChunk(chunk_id="reliability", page=10, section="Reliability", text="Fault isolated boundaries limit failures. Test recovery and disaster recovery plans.", score=0.8)]


def test_configured_reasoner_exposes_safe_selection_diagnostics_and_model_alias(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-not-used")
    local = configured_reasoner()
    assert isinstance(local, LocalReasoner)
    assert "LLM_MODE=local" in local.selection_reason

    monkeypatch.setenv("LLM_MODE", "auto")
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "alias-model")
    openai_reasoner = configured_reasoner()
    assert isinstance(openai_reasoner, OpenAIReasoner)
    assert openai_reasoner.model == "alias-model"
    assert "OPENAI_API_KEY configured" in openai_reasoner.selection_reason


class EmptyRetriever:
    def search(self, query, k=6):
        return []


def test_trace_structure_is_valid():
    _, trace = Agent(FakeRetriever(), LocalReasoner()).run("How can I reduce cost?")
    assert trace.question
    assert trace.iterations[0].search_decision.search_query
    assert trace.stop_reason == "sufficient_evidence"


def test_max_iteration_limit_works():
    state, trace = Agent(FakeRetriever(), NeverEnough(), max_iterations=2).run("failure and cost")
    assert len(trace.iterations) == 2
    assert state.stop_reason == "no_new_evidence"
    assert trace.stop_reason == "no_new_evidence"


def test_retrieved_chunks_preserve_page_metadata():
    _, trace = Agent(FakeRetriever(), LocalReasoner()).run("cost")
    assert trace.iterations[0].retrieved[0].page == 42


def test_duplicate_evidence_is_deduplicated():
    state, _ = Agent(FakeRetriever(), NeverEnough(), max_iterations=3).run("failure")
    assert len(state.gathered_evidence) == 1


def test_final_answer_does_not_run_with_zero_evidence():
    try:
        LocalReasoner().answer(type("State", (), {"gathered_evidence": []})())
    except ValueError as error:
        assert "zero evidence" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_agent_zero_evidence_records_no_evidence_stop():
    _, trace = Agent(EmptyRetriever(), LocalReasoner(), max_iterations=2).run("security")
    assert trace.stop_reason == "no_evidence"
    assert trace.final_answer == "No evidence was retrieved, so an evidence-grounded answer cannot be generated."
    assert trace.citation_validation.valid is False


def test_successful_multi_iteration_flow_records_trace_metadata():
    question = "How can we improve reliability, security, and cost?"
    _, trace = Agent(MultiPillarRetriever(), LocalReasoner()).run(question)
    assert trace.stop_reason == "sufficient_evidence"
    assert trace.total_iterations == 3
    assert trace.total_unique_evidence_chunks == 3
    assert trace.completed_at
    assert trace.duration_ms is not None
    assert trace.iterations[-1].assessment.supported_information == [
        "reliability and failure preparation",
        "cost optimization and avoiding unnecessary spend",
        "security",
    ]


def test_citation_validation_removes_invalid_citations_and_flags_uncited_claims():
    answer = "Reliability\nUse backups. [AWS-WAF p.999]"
    cleaned, validation = Agent.validate_citations(
        answer,
        [EvidenceChunk(chunk_id="c1", page=10, text="backup evidence")],
    )
    assert "[AWS-WAF p.999]" not in cleaned
    assert validation.valid is False
    assert validation.uncited_claims == ["Use backups."]


def test_citation_validation_requires_matching_source_and_page():
    answer = "Finding\nUse the strategy. [Other.pdf p.10]"
    cleaned, validation = Agent.validate_citations(
        answer,
        [EvidenceChunk(chunk_id="c1", page=10, source="Strategy.pdf", text="strategy evidence")],
    )
    assert "[Other.pdf p.10]" not in cleaned
    assert validation.valid is False
    assert validation.retrieved_references == ["Strategy.pdf p.10"]


def test_duplicate_query_is_replanned_without_duplicate_retrieval():
    reasoner = DuplicateThenReformulatedReasoner()
    retriever = RecordingRetriever()
    _, trace = Agent(retriever, reasoner, max_iterations=2).run("What does this framework cover?")
    assert retriever.queries == ["framework overview purpose", "framework six pillars definition"]
    assert trace.stop_reason == "sufficient_evidence"


def test_duplicate_query_retries_stop_without_retrieving_again():
    reasoner = AlwaysDuplicateReasoner()
    retriever = RecordingRetriever()
    _, trace = Agent(retriever, reasoner, max_iterations=4).run("What is this about?")
    assert retriever.queries == ["framework overview purpose"]
    assert trace.stop_reason == "no_new_search_strategy"
    assert trace.total_iterations == 1


def test_no_progress_stops_when_distinct_queries_return_same_chunks_and_coverage():
    retriever = RecordingRetriever(always_same=True)
    _, trace = Agent(retriever, DistinctQueryReasoner(), max_iterations=4).run("overview")
    assert retriever.queries == ["framework overview purpose", "framework pillars definition"]
    assert trace.stop_reason == "no_new_evidence"
    assert trace.total_iterations == 2


def test_vague_question_triggers_overview_search_and_cited_answer():
    retriever = OverviewRetriever()
    _, trace = Agent(retriever, LocalReasoner()).run("What is this about?")
    query = trace.iterations[0].search_decision.search_query.lower()
    assert "overview" in query or "introduction" in query or "definition" in query
    assert trace.stop_reason == "sufficient_evidence"
    assert "six pillars" in trace.final_answer.lower()
    assert trace.citation_validation.valid is True


def test_standard_overview_questions_do_not_dead_end_on_duplicate_strategy():
    questions = [
        "what is this about?",
        "what is the main topic of this?",
        "what does the AWS Well-Architected Framework cover?",
    ]
    for question in questions:
        retriever = OverviewRetriever()
        _, trace = Agent(retriever, LocalReasoner(), max_iterations=4).run(question)
        first_query = trace.iterations[0].search_decision.search_query.lower()
        assert trace.stop_reason == "sufficient_evidence"
        assert trace.stop_reason != "no_new_search_strategy"
        assert trace.citation_validation.valid is True
        assert "[AWS-WAF p." in trace.final_answer
        assert "what is the main topic" not in first_query
        assert "what is this about" not in first_query
        assert "framework cover" not in first_query


def test_overview_retry_changes_information_target_after_partial_evidence():
    retriever = OverviewNeedsSecondStrategyRetriever()
    _, trace = Agent(retriever, LocalReasoner(), max_iterations=4).run("what is the main topic of this?")
    assert retriever.queries[0].lower() != retriever.queries[1].lower()
    assert "overview" in retriever.queries[0].lower() or "purpose" in retriever.queries[0].lower()
    assert "pillar" in retriever.queries[1].lower()
    assert trace.stop_reason == "sufficient_evidence"
    assert trace.citation_validation.valid is True


def test_status_only_fallback_does_not_require_a_citation():
    answer = "The evidence remained incomplete at the iteration limit, so no broader conclusion is asserted."
    _, validation = Agent.validate_citations(answer, [])
    assert validation.valid is True
    assert validation.uncited_claims == []


def test_empty_question_fails_cleanly():
    try:
        Agent(FakeRetriever(), LocalReasoner()).run("  ")
    except ValueError as error:
        assert "empty" in str(error).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_structured_model_output_retries_after_malformed_json():
    reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
    reasoner.model = "fake"
    reasoner.max_json_tokens = 500
    reasoner.client = FakeClient(["not json", '{"search_query":"AWS reliability","reason":"Need evidence."}'])
    result = reasoner._json("system", {"question": "q"}, SearchDecision)
    assert result.search_query == "AWS reliability"


def test_openai_section_analysis_normalizes_shape_variants_without_retry():
    reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
    reasoner.model = "fake"
    reasoner.max_json_tokens = 500
    reasoner.client = FakeClient([
        '{"section_id":"executive-summary","requirements":[{"id":"R1","requirement":"Assess organisational health"}],'
        '"known_facts":["A public rating is shown. [E1]"],"evidence_claims":[{"claim":"A roadmap is described.","source_ids":["E2"]}]}'
    ])

    result = reasoner._json("system", {}, SectionAnalysis, max_attempts=2)

    assert result.known_facts[0].text == "A public rating is shown."
    assert result.known_facts[0].evidence_ids == ["E1"]
    assert result.evidence_claims[0].text == "A roadmap is described."
    assert result.evidence_claims[0].evidence_ids == ["E2"]
    assert result.requirements == ["Assess organisational health"]
    assert reasoner.last_structured_normalized is True
    assert reasoner.last_structured_repair_retry is False


def test_openai_assessment_falls_back_after_empty_structured_output():
    reasoner = OpenAIReasoner.__new__(OpenAIReasoner)
    reasoner.model = "gpt-5.6-luna"
    reasoner.max_json_tokens = 500
    reasoner.client = FakeClient(["", "", ""])
    state = AgentState(
        original_question="what does this talk about, explain briefly",
        gathered_evidence=[
            EvidenceChunk(
                chunk_id="doc-p001-c00",
                page=1,
                source="final_report.pdf",
                text="The presentation explains the program goals, background, and expected outcomes for stakeholders.",
            )
        ],
    )
    assessment = reasoner.assess(state)
    assert assessment.sufficient is True
    assert assessment.supported_information == ["main topic and source-backed explanation"]


def test_evidence_assessment_normalizes_model_returned_objects():
    assessment = EvidenceAssessment.model_validate({
        "sufficient": False,
        "reason": "Need more.",
        "supported_information": [{"text": "The AWS Well-Architected Framework is relevant.", "id": "aws-waf-p007-c00"}],
        "partially_supported_information": [{"text": "The review process is mentioned.", "chunk_id": "aws-waf-p070-c00"}],
        "unsupported_information": [],
        "missing_information": [{"description": "Security evidence"}],
    })
    assert assessment.supported_information == [
        "The AWS Well-Architected Framework is relevant. (aws-waf-p007-c00)"
    ]
    assert assessment.partially_supported_information == [
        "The review process is mentioned. (aws-waf-p070-c00)"
    ]
    assert assessment.missing_information == ["Security evidence"]


class FakeClient:
    def __init__(self, contents):
        self.chat = FakeChat(contents)


class FakeChat:
    def __init__(self, contents):
        self.completions = FakeCompletions(contents)


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)

    def create(self, **kwargs):
        return FakeResponse(self.contents.pop(0))


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class RecordingRetriever:
    def __init__(self, always_same=False):
        self.queries = []
        self.always_same = always_same

    def search(self, query, k=6):
        self.queries.append(query)
        chunk_id = "same" if self.always_same else f"chunk-{len(self.queries)}"
        return [EvidenceChunk(chunk_id=chunk_id, page=7, text="Partial framework evidence.", score=0.8)]


class DuplicateThenReformulatedReasoner:
    def __init__(self):
        self.proposals = iter([
            "framework overview purpose",
            "framework overview purpose",
            "framework six pillars definition",
        ])

    def decide_search(self, state):
        return SearchDecision(search_query=next(self.proposals), reason="Need another framework aspect.")

    def assess(self, state):
        sufficient = len(state.searches) == 2
        return EvidenceAssessment(
            sufficient=sufficient,
            reason="Complete." if sufficient else "Pillars missing.",
            missing_information=[] if sufficient else ["framework pillars"],
            supported_information=["framework overview and purpose"],
        )

    def answer(self, state):
        return "The framework has an overview and pillars. [AWS-WAF p.7]"


class AlwaysDuplicateReasoner(DuplicateThenReformulatedReasoner):
    def __init__(self):
        pass

    def decide_search(self, state):
        return SearchDecision(search_query="framework overview purpose", reason="Overview.")


class DistinctQueryReasoner(DuplicateThenReformulatedReasoner):
    def __init__(self):
        self.proposals = iter(["framework overview purpose", "framework pillars definition"])

    def assess(self, state):
        return EvidenceAssessment(
            sufficient=False,
            reason="Purpose remains missing.",
            missing_information=["purpose"],
            supported_information=[],
        )


class OverviewRetriever:
    def search(self, query, k=6):
        return [EvidenceChunk(
            chunk_id="overview",
            page=8,
            text=(
                "The AWS Well-Architected Framework provides a consistent set of best practices "
                "to evaluate architectures and is based on six pillars: operational excellence, "
                "security, reliability, performance efficiency, cost optimization, and sustainability."
            ),
            score=0.9,
        )]


class OverviewNeedsSecondStrategyRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=6):
        self.queries.append(query)
        if "pillar" in query.lower():
            return [EvidenceChunk(
                chunk_id="pillars",
                page=9,
                text=(
                    "The AWS Well-Architected Framework is based on six pillars: operational "
                    "excellence, security, reliability, performance efficiency, cost optimization, "
                    "and sustainability."
                ),
                score=0.9,
            )]
        return [EvidenceChunk(
            chunk_id="toc",
            page=2,
            text="Resources and related documents for AWS architecture guidance.",
            score=0.4,
        )]

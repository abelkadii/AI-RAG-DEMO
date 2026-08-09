from agent import Agent
from llm import LocalReasoner, OpenAIReasoner
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
    assert state.stop_reason == "max_iterations"
    assert trace.stop_reason == "max_iterations"


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
    reasoner.client = FakeClient(["not json", '{"search_query":"AWS reliability","reason":"Need evidence."}'])
    result = reasoner._json("system", {"question": "q"}, SearchDecision)
    assert result.search_query == "AWS reliability"


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

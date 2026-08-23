"""OpenAI-compatible reasoning, plus an offline fallback for a zero-secret demo."""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from pydantic import ValidationError

from models import AgentState, EvidenceAssessment, EvidenceChunk, SearchDecision


class Reasoner(Protocol):
    def decide_search(self, state: AgentState) -> SearchDecision: ...
    def assess(self, state: AgentState) -> EvidenceAssessment: ...
    def answer(self, state: AgentState) -> str: ...


class OpenAIReasoner:
    def __init__(self) -> None:
        from openai import OpenAI

        self.model = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.selection_reason = "OPENAI_API_KEY configured and LLM_MODE permits OpenAIReasoner."
        self.section_analysis_max_attempts = 2
        self.max_json_tokens = int(os.getenv("OPENAI_JSON_MAX_TOKENS", "900"))
        self.max_answer_tokens = int(os.getenv("OPENAI_ANSWER_MAX_TOKENS", "900"))
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

    def _token_limit_arg(self, token_limit: int) -> dict:
        if self.model.startswith("gpt-5"):
            return {"max_completion_tokens": token_limit}
        return {"max_tokens": token_limit}

    def _sampling_args(self) -> dict:
        if self.model.startswith("gpt-5"):
            return {}
        return {"temperature": 0}

    def _json(self, system: str, payload: dict, schema: type, *, max_attempts: int = 3):
        last_error: Exception | None = None
        self.last_structured_normalized = False
        self.last_structured_repair_retry = False
        for attempt in range(max(1, max_attempts)):
            try:
                token_limit = getattr(self, "max_json_tokens", 900)
                if self.model.startswith("gpt-5"):
                    token_limit = max(token_limit, 1200) + (attempt * 600)
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    **self._sampling_args(),
                    **self._token_limit_arg(token_limit),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("Model returned empty structured output")
                raw_payload = json.loads(content)
                if schema.__name__ == "SectionAnalysis":
                    # Imported lazily to avoid a module cycle during startup.
                    from document_models import normalize_section_analysis_payload

                    raw_payload, normalized = normalize_section_analysis_payload(raw_payload)
                    self.last_structured_normalized = self.last_structured_normalized or normalized
                return schema.model_validate(raw_payload)
            except (ValidationError, json.JSONDecodeError, ValueError, TypeError, KeyError, IndexError) as error:
                last_error = error
                if attempt + 1 < max_attempts:
                    self.last_structured_repair_retry = True
                    system = (
                        system
                        + " Return only valid JSON matching the requested schema. "
                        + f"This is repair retry {attempt + 1}."
                    )
            except Exception as error:
                raise RuntimeError(f"Model request failed: {error}") from error
        raise RuntimeError(f"Model returned malformed structured output after retry: {last_error}") from last_error

    def decide_search(self, state: AgentState) -> SearchDecision:
        evidence = [_evidence_payload(item, 600) for item in state.gathered_evidence]
        overview = overview_context(state)
        try:
            return self._json(
                "You plan searches over the supplied evidence corpus. Return JSON with "
                "search_query and reason. Target the most important information not yet supported. "
                "Do not repeat or lightly paraphrase prior searches. If overview_context is present, "
                "ignore vague phrasing as a retrieval target and plan from normalized_information_need. "
                "Choose a meaningfully different unresolved search strategy. "
                "When evidence is partial, change strategy to target the explicitly missing information. "
                "If search_strategy_feedback is present, obey it and produce a meaningfully different query.",
                {
                    "question": state.original_question,
                    "overview_context": overview,
                    "searches": [s.model_dump() for s in state.searches],
                    "rejected_duplicate_queries": state.rejected_search_queries,
                    "evidence": evidence,
                    "latest_assessment": state.assessments[-1].model_dump() if state.assessments else None,
                    "search_strategy_feedback": state.search_strategy_feedback,
                },
                SearchDecision,
            )
        except RuntimeError:
            return LocalReasoner().decide_search(state)

    def assess(self, state: AgentState) -> EvidenceAssessment:
        overview = overview_context(state)
        try:
            return self._json(
                "Assess whether the supplied evidence fully supports a cited answer to the question. "
                "Break the original question into material parts and track each part as supported, "
                "partially_supported, or unsupported. Stop only when all material parts are supported by "
                "specific supplied evidence. For broad overview questions, sufficient evidence can establish "
                "what the source is about and/or what it broadly covers. Return JSON with sufficient, reason, "
                "missing_information, suggested_next_search (null if sufficient), supported_information, "
                "partially_supported_information, and unsupported_information.",
                {
                    "question": state.original_question,
                    "overview_context": overview,
                    "evidence": [_evidence_payload(e, 1200) for e in state.gathered_evidence],
                },
                EvidenceAssessment,
            )
        except RuntimeError:
            return local_assess_any_corpus(state)

    def answer(self, state: AgentState) -> str:
        if not state.gathered_evidence:
            raise ValueError("Cannot generate a final answer with zero evidence")
        prompt = {
            "question": state.original_question,
            "evidence": [_evidence_payload(e, 1600) for e in state.gathered_evidence],
            "evidence_sufficient": state.stop_reason == "sufficient_evidence",
        }
        response = self.client.chat.completions.create(
            model=self.model,
            **self._sampling_args(),
            **self._token_limit_arg(getattr(self, "max_answer_tokens", 900)),
            messages=[
                {"role": "system", "content": "Answer only from supplied evidence. Support every major claim with a page citation exactly like [AWS-WAF p.42]. If evidence_sufficient is false, explicitly identify the limitation. Never invent facts."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        return response.choices[0].message.content or ""


CONCEPTS = {
    "framework overview and purpose": ("best practices", "consistent approach", "evaluate architectures", "understand the pros and cons"),
    "framework pillars": ("six pillars", "operational excellence", "sustainability"),
    "reliability and failure preparation": ("failure", "failures", "reliable", "reliability", "resilien", "recover"),
    "cost optimization and avoiding unnecessary spend": ("cost", "expense", "spend", "unnecessary", "waste", "efficient", "idle", "oversized", "right size"),
    "security": ("security", "secure", "identity", "encrypt", "threat", "protect", "sensitive", "data protection"),
    "performance efficiency": ("performance", "latency", "throughput", "efficient"),
    "operational excellence": ("operation", "deploy", "observe", "monitor"),
    "sustainability": ("sustainab", "environment", "carbon"),
}

CONCEPT_SEARCH_TERMS = {
    "framework overview and purpose": "framework introduction overview purpose definition",
    "framework pillars": "six pillars definition",
    "reliability and failure preparation": "reliability failures recovery resilience fault isolation testing",
    "cost optimization and avoiding unnecessary spend": "cost optimization eliminate waste right size resources demand",
    "security": "security identity encryption threat protection",
    "performance efficiency": "performance efficiency latency throughput resources",
    "operational excellence": "operational excellence observe monitor improve operations",
    "sustainability": "sustainability environmental impact resource efficiency",
}

OVERVIEW_INFORMATION_NEED = (
    "AWS Well-Architected Framework overview: what the framework is for, what it covers, "
    "its core structure or pillars, and how it is used to evaluate architectures."
)

OVERVIEW_SEARCH_STRATEGIES = (
    {
        "target": "framework overview and purpose",
        "query_terms": "framework overview purpose introduction definition",
        "reason": "Establish what the framework is for.",
    },
    {
        "target": "framework pillars",
        "query_terms": "six pillars framework definition",
        "reason": "Establish what the framework broadly covers.",
    },
    {
        "target": "framework use and evaluation",
        "query_terms": "architecture evaluation best practices purpose",
        "reason": "Establish how the framework is used.",
    },
)

CONCEPT_SECTIONS = {
    "reliability and failure preparation": "Reliability",
    "cost optimization and avoiding unnecessary spend": "Cost Optimization",
    "security": "Security",
    "performance efficiency": "Performance Efficiency",
    "operational excellence": "Operational Excellence",
    "sustainability": "Sustainability",
}

CONCEPT_LABELS = {
    "framework overview and purpose": "AWS Well-Architected Framework",
    "framework pillars": "Six pillars",
    "reliability and failure preparation": "Reliability",
    "cost optimization and avoiding unnecessary spend": "Cost Optimization",
    "security": "Security",
    "performance efficiency": "Performance Efficiency",
    "operational excellence": "Operational Excellence",
    "sustainability": "Sustainability",
}


def _question_concepts(question: str) -> list[str]:
    if is_overview_intent(question):
        return ["framework overview and purpose", "framework pillars"]
    lower = question.lower()
    found = [name for name, terms in CONCEPTS.items() if any(term in lower for term in terms)]
    return found or [question]


def is_overview_intent(question: str) -> bool:
    lower = question.lower()
    overview_patterns = (
        r"\bwhat is (?:this|the document|the framework|aws well[- ]architected|the aws well[- ]architected framework)(?: about)?\b",
        r"\bwhat(?:'s| is) (?:this|the document|the framework) about\b",
        r"\bwhat is the main (?:topic|point|idea|subject) of (?:this|the document|the framework)\b",
        r"\bwhat does (?:this |the |aws well[- ]architected |the aws well[- ]architected )?framework cover\b",
        r"\bgive me an overview\b",
        r"\boverview of (?:the )?(?:aws )?well[- ]architected framework\b",
        r"\bwhat is (?:the )?aws well[- ]architected framework\b",
    )
    return any(re.search(pattern, lower) for pattern in overview_patterns)


def overview_context(state: AgentState) -> dict | None:
    if not is_overview_intent(state.original_question):
        return None
    supported = _normalized_items(state.assessments[-1].supported_information) if state.assessments else set()
    missing = _normalized_items(state.assessments[-1].missing_information) if state.assessments else set()
    previous_queries = " ".join(search.query.lower() for search in state.searches)
    target_markers = {
        "framework overview and purpose": ("overview", "purpose", "introduction"),
        "framework pillars": ("pillar", "pillars"),
        "framework use and evaluation": ("evaluation", "evaluate", "best practices"),
    }
    strategies = []
    for strategy in OVERVIEW_SEARCH_STRATEGIES:
        target = strategy["target"]
        target_normalized = target.lower()
        already_targeted = (
            target_normalized in supported
            or target_normalized in previous_queries
            or any(marker in previous_queries for marker in target_markers[target])
        )
        if missing and target_normalized not in missing and target_normalized in supported:
            already_targeted = True
        strategies.append({**strategy, "already_targeted": already_targeted})
    return {
        "intent": "overview",
        "normalized_information_need": OVERVIEW_INFORMATION_NEED,
        "search_strategies": strategies,
        "sufficiency_rule": (
            "Treat overview evidence as sufficient when it establishes what the framework is for "
            "and/or what it broadly covers, including its pillars."
        ),
    }


def _normalized_items(items: list[str]) -> set[str]:
    return {" ".join(item.lower().split()) for item in items}


def _concept_label(concept: str) -> str:
    return CONCEPT_LABELS.get(concept, concept.title())


class LocalReasoner:
    """Transparent heuristic fallback. It plans by uncovered question concepts."""

    def __init__(self, selection_reason: str | None = None) -> None:
        self.selection_reason = selection_reason or "OpenAIReasoner was not configured."

    def _evidence_hits(self, state: AgentState, concept: str) -> list[EvidenceChunk]:
        terms = CONCEPTS.get(concept, (concept,))
        expected_section = CONCEPT_SECTIONS.get(concept)
        hits = []
        for chunk in state.gathered_evidence:
            text = chunk.text.lower()
            section_matches = not expected_section or not chunk.section or chunk.section == expected_section
            if section_matches and any(term in text for term in terms):
                hits.append(chunk)
        return hits

    def _supported(self, state: AgentState) -> set[str]:
        return {concept for concept in _question_concepts(state.original_question) if self._evidence_hits(state, concept)}

    def decide_search(self, state: AgentState) -> SearchDecision:
        if is_overview_intent(state.original_question):
            strategy = self._overview_strategy(state)
            return SearchDecision(
                search_query=f"AWS Well-Architected {strategy['query_terms']}",
                reason=strategy["reason"],
            )
        concepts = _question_concepts(state.original_question)
        if state.assessments and state.assessments[-1].missing_information:
            target = state.assessments[-1].missing_information[0]
        else:
            supported = self._supported(state)
            target = next((concept for concept in concepts if concept not in supported), concepts[-1])
        search_terms = CONCEPT_SEARCH_TERMS.get(target, target)
        if state.search_strategy_feedback:
            alternatives = ("core principles scope", "architectural guidance outcomes")
            search_terms = f"{search_terms} {alternatives[len(state.rejected_search_queries) % len(alternatives)]}"
        return SearchDecision(
            search_query=f"AWS Well-Architected {search_terms}",
            reason=f"Need direct framework evidence about {target}."
        )

    def _overview_strategy(self, state: AgentState) -> dict:
        previous = " ".join(search.query.lower() for search in state.searches)
        missing = _normalized_items(state.assessments[-1].missing_information) if state.assessments else set()
        target_markers = {
            "framework overview and purpose": ("overview", "purpose", "introduction"),
            "framework pillars": ("pillar", "pillars"),
            "framework use and evaluation": ("evaluation", "evaluate", "best practices"),
        }
        for strategy in OVERVIEW_SEARCH_STRATEGIES:
            target = strategy["target"].lower()
            already_targeted = target in previous or any(marker in previous for marker in target_markers[target])
            if missing and target not in missing and target in _normalized_items(state.assessments[-1].supported_information):
                already_targeted = True
            if not already_targeted:
                return strategy
        return OVERVIEW_SEARCH_STRATEGIES[len(state.searches) % len(OVERVIEW_SEARCH_STRATEGIES)]

    def assess(self, state: AgentState) -> EvidenceAssessment:
        concepts = _question_concepts(state.original_question)
        supported = sorted(self._supported(state), key=concepts.index)
        missing = [concept for concept in concepts if concept not in supported]
        gathered_text = " ".join(chunk.text.lower() for chunk in state.gathered_evidence)
        partial = [
            concept for concept in missing
            if any(word in gathered_text for word in re.findall(r"[a-z]+", concept))
        ]
        unsupported = [concept for concept in missing if concept not in partial]
        if is_overview_intent(state.original_question):
            enough = bool(supported) and any(
                concept in supported for concept in ("framework overview and purpose", "framework pillars")
            )
        else:
            enough = bool(state.gathered_evidence) and not missing
        return EvidenceAssessment(
            sufficient=enough,
            reason=(
                f"Supported: {', '.join(supported)}."
                if enough else f"Supported: {', '.join(supported) or 'none'}. Still missing: {', '.join(missing)}."
            ),
            missing_information=[] if enough else missing,
            suggested_next_search=(
                None if enough or not missing else f"AWS Well-Architected {CONCEPT_SEARCH_TERMS.get(missing[0], missing[0])}"
            ),
            supported_information=supported,
            partially_supported_information=partial,
            unsupported_information=unsupported,
        )

    def answer(self, state: AgentState) -> str:
        if not state.gathered_evidence:
            raise ValueError("Cannot generate a final answer with zero evidence")
        lines = []
        for concept in _question_concepts(state.original_question):
            statement = self._statement_for_concept(state, concept)
            if statement:
                lines.extend([_concept_label(concept), statement, ""])
        if state.stop_reason != "sufficient_evidence":
            lines.append("The evidence remained incomplete at the iteration limit, so no broader conclusion is asserted.")
        return "\n".join(lines).strip()

    def _statement_for_concept(self, state: AgentState, concept: str) -> str | None:
        hits = self._evidence_hits(state, concept)
        if not hits:
            return None
        combined = " ".join(chunk.text for chunk in hits)
        clauses = self._evidence_clauses(concept, combined)
        if not clauses:
            sentence = self._best_sentence(concept, hits)
            clauses = [sentence] if sentence else []
        if not clauses:
            return None
        chunk = hits[0]
        return f"{'; '.join(clauses)}. [{chunk.source} p.{chunk.page}]"

    def _evidence_clauses(self, concept: str, text: str) -> list[str]:
        lower = text.lower()
        clauses = []
        if concept == "reliability and failure preparation":
            if "fault isolated boundaries" in lower:
                clauses.append("Use fault-isolated boundaries to limit the impact of failures")
            if "test" in lower and ("recovery" in lower or "resilien" in lower):
                clauses.append("test resilience and recovery processes regularly")
            if "disaster recovery" in lower or "rto" in lower or "rpo" in lower:
                clauses.append("plan disaster recovery around business recovery objectives")
        elif concept == "framework overview and purpose":
            if "understand the pros and cons" in lower:
                clauses.append("It helps teams understand architectural trade-offs when building systems on AWS")
            if "best practices" in lower and ("measure" in lower or "evaluate" in lower):
                clauses.append("it provides a consistent way to evaluate architectures against cloud best practices and identify improvements")
        elif concept == "framework pillars":
            if "six pillars" in lower:
                clauses.append("It is organized around six pillars: operational excellence, security, reliability, performance efficiency, cost optimization, and sustainability")
        elif concept == "cost optimization and avoiding unnecessary spend":
            if "resource type, size, and number" in lower or "right size" in lower:
                clauses.append("Select resource type, size, and count from workload data")
            if "cost modeling" in lower:
                clauses.append("use cost modeling or proof-of-concept measurements before committing capacity")
            if "minimize waste" in lower or "waste" in lower:
                clauses.append("minimize waste from unnecessary or oversized resources")
        elif concept == "security":
            if "identity and access management" in lower:
                clauses.append("Apply identity and access management controls")
            if "data protection" in lower:
                clauses.append("protect sensitive data at rest and in transit")
            if "infrastructure security" in lower or "infrastructure protection" in lower:
                clauses.append("include infrastructure protection")
            if "logging" in lower or "monitoring" in lower:
                clauses.append("use logging and monitoring for visibility")
        return clauses[:4]

    def _best_sentence(self, concept: str, hits: list[EvidenceChunk]) -> str | None:
        terms = set(re.findall(r"[a-z]+", CONCEPT_SEARCH_TERMS.get(concept, concept)))
        candidates = []
        for chunk in hits:
            for sentence in re.split(r"(?<=[.!?])\s+|\s*-\s+", chunk.text.strip()):
                sentence = sentence.strip()
                lower_sentence = sentence.lower()
                if not lower_sentence or lower_sentence.startswith(("resources", "related documents", "related videos", "related examples")):
                    continue
                score = len(terms & set(re.findall(r"[a-z]+", lower_sentence)))
                if score and 45 <= len(sentence) <= 240:
                    candidates.append((score, chunk.score, sentence.rstrip(".")))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]


def local_assess_any_corpus(state: AgentState) -> EvidenceAssessment:
    if not state.gathered_evidence:
        return EvidenceAssessment(
            sufficient=False,
            reason="No evidence has been retrieved yet.",
            missing_information=["source evidence"],
            suggested_next_search=state.original_question,
            supported_information=[],
            partially_supported_information=[],
            unsupported_information=["source evidence"],
        )
    question_terms = {
        term
        for term in re.findall(r"[\w\u0600-\u06FF]{4,}", state.original_question.lower())
        if term not in GENERIC_STOP_TERMS
    }
    evidence_terms = {
        term
        for chunk in state.gathered_evidence
        for term in re.findall(r"[\w\u0600-\u06FF]{4,}", chunk.text.lower())
        if term not in GENERIC_STOP_TERMS
    }
    overlap = sorted(question_terms & evidence_terms)
    broad_summary = bool(re.search(r"\b(what.*talk about|explain briefly|summary|summarize|overview|main topic)\b", state.original_question.lower()))
    enough = broad_summary or bool(overlap) or bool(evidence_terms)
    supported = ["main topic and source-backed explanation"] if broad_summary else (overlap or ["retrieved source evidence"])
    missing = [] if enough else ["stronger semantic match to the brief"]
    return EvidenceAssessment(
        sufficient=enough,
        reason="Retrieved evidence is sufficient for the requested brief." if enough else "Retrieved evidence does not yet match the brief.",
        missing_information=missing,
        suggested_next_search=None if enough else state.original_question,
        supported_information=supported,
        partially_supported_information=[],
        unsupported_information=[] if enough else missing,
    )


GENERIC_STOP_TERMS = {
    "what",
    "does",
    "this",
    "talk",
    "about",
    "briefly",
    "explain",
    "summary",
    "summarize",
    "document",
    "source",
    "report",
    "presentation",
    "uploaded",
    "with",
    "from",
    "that",
    "into",
    "your",
    "their",
}


def configured_reasoner() -> Reasoner:
    mode = os.getenv("LLM_MODE", "auto").strip().lower()
    key_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    model = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    if mode == "local":
        return LocalReasoner(f"LLM_MODE=local explicitly selected the deterministic fallback (model {model}).")
    if not key_configured:
        return LocalReasoner("OPENAI_API_KEY is not configured in the environment or Streamlit secrets.")
    try:
        return OpenAIReasoner()
    except Exception as error:
        # Keep the app usable, but retain a safe diagnostic for the live feed;
        # never include credential values in the message.
        return LocalReasoner(f"OpenAIReasoner initialization failed: {type(error).__name__}.")


def _evidence_payload(chunk: EvidenceChunk, limit: int) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "page": chunk.page,
        "section": chunk.section,
        "text": chunk.text[:limit],
    }

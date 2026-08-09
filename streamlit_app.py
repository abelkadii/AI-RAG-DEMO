"""Streamlit demo for the Milestone 1 agentic RAG loop."""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from agent import Agent
from llm import configured_reasoner
from models import AgentTrace, IterationTrace
from retriever import Retriever


DEFAULT_QUESTION = (
    "How should we design a customer-facing workload so it can recover from "
    "Availability Zone failures, protect sensitive customer data, and avoid "
    "paying for idle or oversized resources?"
)

MAX_QUESTION_LENGTH = 1000


def configure_credentials() -> None:
    load_dotenv()
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
        "OPENAI_JSON_MAX_TOKENS",
        "OPENAI_ANSWER_MAX_TOKENS",
        "LLM_MODE",
    ):
        if os.getenv(key):
            continue
        try:
            secret_value = st.secrets.get(key)
        except Exception:
            secret_value = None
        if secret_value:
            os.environ[key] = str(secret_value)


def run_agent(question: str) -> AgentTrace:
    iterations: list[IterationTrace] = []
    agent = Agent(Retriever(), configured_reasoner(), max_iterations=4, k=6)
    _, trace = agent.run(question, iterations.append)
    return trace


def render_iteration(item: IterationTrace) -> None:
    assessment = item.assessment
    state = "SUFFICIENT" if assessment.sufficient else "INSUFFICIENT"
    st.subheader(f"ITERATION {item.iteration}")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Search**")
        st.code(item.search_decision.search_query, language=None)
        st.markdown("**Reason**")
        st.write(item.search_decision.reason)
    with right:
        if assessment.sufficient:
            st.success(state)
        else:
            st.warning(state)

    st.markdown("**Evidence**")
    for section, page in evidence_pages(item):
        st.write(f"- {section} - AWS-WAF p.{page}")

    with st.expander("Full retrieved evidence"):
        for chunk in item.retrieved:
            st.markdown(f"**{chunk.section or 'Unknown'} - AWS-WAF p.{chunk.page}**")
            st.caption(f"score: {chunk.score}")
            st.write(chunk.text_preview)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Supported**")
        render_list(assessment.supported_information, "+")
    with c2:
        st.markdown("**Partially supported**")
        render_list(assessment.partially_supported_information, "~")
    with c3:
        st.markdown("**Missing**")
        render_list(assessment.missing_information, "-")

    if assessment.suggested_next_search:
        st.markdown("**Next search**")
        st.code(assessment.suggested_next_search, language=None)


def render_list(items: list[str], marker: str) -> None:
    if not items:
        st.caption("None")
        return
    for item in items:
        st.write(f"{marker} {short_label(item)}")


def evidence_pages(item: IterationTrace) -> list[tuple[str, int]]:
    seen = set()
    result = []
    for chunk in item.retrieved:
        key = (chunk.section or "Unknown", chunk.page)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def short_label(text: str) -> str:
    labels = {
        "reliability and failure preparation": "Reliability / failure preparation",
        "cost optimization and avoiding unnecessary spend": "Cost Optimization",
        "security": "Security",
    }
    return labels.get(text, text)


def render_final(trace: AgentTrace) -> None:
    st.header("FINAL ANSWER")
    st.markdown(trace.final_answer or "")

    validation = trace.citation_validation
    if validation.valid:
        st.success("Citation validation: all citations grounded in retrieved evidence")
    else:
        st.error("Citation validation found issues")
        st.write(validation.model_dump())

    st.markdown("**Stop reason**")
    st.code(trace.stop_reason or "unknown", language=None)

    trace_json = trace.model_dump_json(indent=2, by_alias=True)
    with st.expander("View full trace"):
        st.json(json.loads(trace_json))
    st.download_button(
        "Download trace JSON",
        data=trace_json,
        file_name="agentic_rag_trace.json",
        mime="application/json",
    )


def main() -> None:
    configure_credentials()
    st.set_page_config(
        page_title="AWS Well-Architected - Agentic RAG Demo",
        page_icon="🔎",
        layout="wide",
    )

    st.title("AWS Well-Architected — Agentic RAG Demo")
    st.caption("Evidence-driven iterative retrieval with full trajectory tracing.")
    st.markdown("Search -> Retrieve -> Assess -> Refine if needed -> Stop when supported -> Cite")

    question = st.text_area(
        "Ask a question about the AWS Well-Architected Framework",
        value=DEFAULT_QUESTION,
        max_chars=MAX_QUESTION_LENGTH,
        height=120,
    )

    if st.button("Run agent", type="primary"):
        cleaned = question.strip()
        if not cleaned:
            st.error("Please enter a question.")
            return
        if len(cleaned) > MAX_QUESTION_LENGTH:
            st.error(f"Question must be {MAX_QUESTION_LENGTH} characters or fewer.")
            return

        with st.spinner("Running search -> retrieve -> assess loop..."):
            try:
                trace = run_agent(cleaned)
            except Exception as error:
                st.error(f"Agent run failed: {error}")
                return

        for item in trace.iterations:
            render_iteration(item)
            st.divider()
        render_final(trace)


if __name__ == "__main__":
    main()

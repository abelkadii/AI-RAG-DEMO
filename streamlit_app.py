"""Streamlit demo for the Milestone 1 agentic RAG loop."""

from __future__ import annotations

import json
import os
from html import escape

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


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --aws-orange: #ff9900;
            --ink: #f5f7fb;
            --muted: #9aa4b2;
            --panel: rgba(22, 27, 34, 0.82);
            --panel-strong: rgba(31, 38, 50, 0.96);
            --line: rgba(255, 255, 255, 0.12);
            --green: #45d483;
            --cyan: #55d7ff;
            --amber: #ffbf4c;
        }
        .stApp {
            background:
                radial-gradient(circle at 15% -8%, rgba(25,74,112,.22), transparent 32%),
                linear-gradient(145deg, #07111c 0%, #09131f 48%, #06101a 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1320px;
            padding: 1rem 2.4rem 4rem;
        }
        header[data-testid="stHeader"] { background: transparent; }
        .hero {
            padding: 0 6px 22px;
        }
        .eyebrow {
            display: inline-block;
            color: var(--aws-orange);
            font-size: 0.78rem;
            letter-spacing: .08em;
            text-transform: uppercase;
            font-weight: 800;
            padding: 9px 14px 11px;
            background: linear-gradient(90deg, rgba(31,45,60,.72), rgba(31,45,60,.18));
            border-bottom: 2px solid rgba(255,153,0,.55);
        }
        .hero h1 {
            margin: 18px 0 14px;
            font-size: clamp(2.25rem, 4vw, 3.55rem);
            line-height: 1.08;
            letter-spacing: -.035em;
            font-weight: 850;
        }
        .hero p {
            color: var(--muted);
            font-size: 1.04rem;
            margin: 0;
        }
        .flow {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 22px;
        }
        .flow span {
            border: 1px solid rgba(255,153,0,.72);
            background: rgba(7,17,28,.56);
            color: #ffd777;
            border-radius: 999px;
            padding: 8px 13px;
            font-size: .9rem;
            font-weight: 650;
        }
        .flow .flow-icon {
            color: var(--aws-orange);
            font-size: 1rem;
            margin-right: 6px;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 20px 0 18px;
        }
        .metric {
            border: 1px solid var(--line);
            border-radius: 9px;
            padding: 19px 18px;
            background: linear-gradient(145deg, rgba(18,31,46,.9), rgba(10,20,32,.9));
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .metric-icon {
            color: #d98600;
            font-size: 2.2rem;
            line-height: 1;
        }
        .metric .label { color: #aab8ca; font-size: .82rem; }
        .metric .value { color: var(--ink); font-size: 1.3rem; font-weight: 800; margin-top: 2px; }
        .iteration-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin: 4px 0 14px;
        }
        .iteration-title h3 {
            margin: 0;
            color: var(--ink);
        }
        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 6px 11px;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .02em;
            border: 1px solid var(--line);
        }
        .badge.good { color: #d6ffe6; background: rgba(69,212,131,.16); border-color: rgba(69,212,131,.36); }
        .badge.warn { color: #fff0cc; background: rgba(255,191,76,.16); border-color: rgba(255,191,76,.36); }
        .section-label {
            color: #9fb1c7;
            text-transform: uppercase;
            font-size: .75rem;
            letter-spacing: .08em;
            font-weight: 800;
            margin: 10px 0 5px;
        }
        .evidence-pill {
            display: inline-block;
            margin: 4px 6px 4px 0;
            border-radius: 999px;
            border: 1px solid rgba(85,215,255,.38);
            background: rgba(16,55,72,.28);
            color: #9fe9ff;
            padding: 7px 11px;
            font-size: .88rem;
        }
        .coverage {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
            min-height: 128px;
            background: rgba(8,18,29,.46);
        }
        .coverage-title { color: #c4d1e1; font-weight: 750; margin-bottom: 12px; }
        .coverage-item { color: #c9d5e3; margin: 8px 0; }
        .coverage-item .status-icon {
            display: inline-flex;
            width: 18px;
            height: 18px;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            margin-right: 8px;
            font-size: .72rem;
            font-weight: 900;
        }
        .coverage-item.good .status-icon { color: #50e3b2; border: 1px solid #50e3b2; }
        .coverage-item.partial .status-icon { color: #ffc857; border: 0; font-size: 1rem; }
        .coverage-item.missing .status-icon { color: #ff746c; border: 1px solid #ff746c; }
        .coverage-empty { color: #718096; font-size: .9rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(151,170,192,.28);
            border-radius: 9px;
            background: rgba(6,15,25,.24);
        }
        div[data-testid="stCodeBlock"] pre {
            border-radius: 12px;
            border: 1px solid var(--line);
        }
        .stButton > button {
            border-radius: 8px;
            min-height: 54px;
            font-weight: 800;
        }
        .stButton > button[kind="primary"] {
            color: white;
            border: 1px solid #f5a000;
            background: linear-gradient(90deg, #d88400, #e19400 50%, #d88400);
            box-shadow: 0 8px 22px rgba(216,132,0,.16);
        }
        .stButton > button[kind="primary"]:hover {
            color: white;
            border-color: #ffb326;
            background: linear-gradient(90deg, #e29000, #f0a000 50%, #e29000);
        }
        hr { border-color: rgba(151,170,192,.24) !important; }
        @media (max-width: 900px) {
            .metric-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .block-container { padding-left: 1rem; padding-right: 1rem; }
        }
        @media (max-width: 600px) {
            .metric-row { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Milestone 1 Live Demo</div>
          <h1>AWS Well-Architected - Agentic RAG Demo</h1>
          <p>Evidence-driven iterative retrieval with full trajectory tracing.</p>
          <div class="flow">
            <span><b class="flow-icon">⌕</b>Search</span>
            <span><b class="flow-icon">▣</b>Retrieve</span>
            <span><b class="flow-icon">▤</b>Assess</span>
            <span><b class="flow-icon">↻</b>Refine if needed</span>
            <span><b class="flow-icon">✓</b>Stop when supported</span>
            <span><b class="flow-icon">“</b>Cite</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(trace: AgentTrace) -> None:
    valid = "Valid" if trace.citation_validation.valid else "Needs review"
    stop_reason = escape(trace.stop_reason or "unknown")
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric"><div class="metric-icon">↻</div><div><div class="label">Iterations</div><div class="value">{trace.total_iterations}</div></div></div>
          <div class="metric"><div class="metric-icon">◎</div><div><div class="label">Unique evidence chunks</div><div class="value">{trace.total_unique_evidence_chunks}</div></div></div>
          <div class="metric"><div class="metric-icon">⚑</div><div><div class="label">Stop reason</div><div class="value">{stop_reason}</div></div></div>
          <div class="metric"><div class="metric-icon">♢</div><div><div class="label">Citations</div><div class="value">{valid}</div></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_iteration(item: IterationTrace) -> None:
    assessment = item.assessment
    state = "SUFFICIENT" if assessment.sufficient else "INSUFFICIENT"
    badge_class = "good" if assessment.sufficient else "warn"
    st.divider()
    st.markdown(
        f"""
        <div class="iteration-title">
          <h3>Iteration {item.iteration}</h3>
          <span class="badge {badge_class}">{state}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown('<div class="section-label">Search</div>', unsafe_allow_html=True)
        st.code(item.search_decision.search_query, language=None)
        st.markdown('<div class="section-label">Reason</div>', unsafe_allow_html=True)
        st.write(item.search_decision.reason)
    with right:
        st.markdown('<div class="section-label">Evidence pages</div>', unsafe_allow_html=True)
        pills = "\n".join(
            f'<span class="evidence-pill">{escape(section)} - AWS-WAF p.{page}</span>'
            for section, page in evidence_pages(item)
        )
        st.markdown(pills or "No evidence retrieved", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_coverage("Supported", assessment.supported_information, "good")
    with c2:
        render_coverage("Partially supported", assessment.partially_supported_information, "partial")
    with c3:
        render_coverage("Missing", assessment.missing_information, "missing")

    if assessment.suggested_next_search:
        st.markdown('<div class="section-label">Next search</div>', unsafe_allow_html=True)
        st.code(assessment.suggested_next_search, language=None)

    with st.expander("Full retrieved evidence"):
        for chunk in item.retrieved:
            st.markdown(f"**{chunk.section or 'Unknown'} - AWS-WAF p.{chunk.page}**")
            st.caption(f"score: {chunk.score}")
            st.write(chunk.text_preview)


def render_coverage(title: str, items: list[str], status: str) -> None:
    symbols = {"good": "✓", "partial": "~", "missing": "×"}
    rows = "".join(
        f'<div class="coverage-item {status}"><span class="status-icon">{symbols[status]}</span>'
        f'{escape(short_label(item))}</div>'
        for item in items
    )
    if not rows:
        rows = '<div class="coverage-empty">None</div>'
    st.markdown(
        f'<div class="coverage"><div class="coverage-title">{escape(title)}</div>{rows}</div>',
        unsafe_allow_html=True,
    )


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
    st.divider()
    with st.container(border=True):
        st.header("Final Answer")
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
    st.set_page_config(
        page_title="AWS Well-Architected - Agentic RAG Demo",
        page_icon="AWS",
        layout="wide",
    )
    configure_credentials()
    apply_styles()
    render_hero()

    with st.container(border=True):
        question = st.text_area(
            "Ask a question about the AWS Well-Architected Framework",
            value=DEFAULT_QUESTION,
            max_chars=MAX_QUESTION_LENGTH,
            height=118,
        )
        run = st.button("▷  Run agent", type="primary", use_container_width=True)

    if run:
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

        render_metrics(trace)
        for item in trace.iterations:
            render_iteration(item)
        render_final(trace)


if __name__ == "__main__":
    main()

"""Streamlit demo for RAG exploration and document production."""

from __future__ import annotations

import json
import os
from html import escape

import streamlit as st
from dotenv import load_dotenv

from agent import Agent
from document_export import docx_bytes, markdown_bytes
from document_models import DocumentSpec, DocumentTrace
from document_workflow import DEFAULT_BRIEF, DocumentWorkflow
from llm import configured_reasoner
from models import AgentTrace, IterationTrace
from retriever import Retriever
from uploaded_corpus import MAX_UPLOAD_BYTES, MAX_UPLOAD_FILES, UploadedPDF, build_uploaded_retriever, corpus_hash


DEFAULT_QUESTION = (
    "How should we design a customer-facing workload so it can recover from "
    "Availability Zone failures, protect sensitive customer data, and avoid "
    "paying for idle or oversized resources?"
)

MAX_QUESTION_LENGTH = 1000
MAX_BRIEF_LENGTH = 1800
MAX_TITLE_LENGTH = 140
MAX_AUDIENCE_LENGTH = 140


def configure_credentials() -> None:
    load_dotenv(override=True)
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
        "OPENAI_JSON_MAX_TOKENS",
        "OPENAI_ANSWER_MAX_TOKENS",
        "LLM_MODE",
        "DEMO_ACCESS_CODE",
    ):
        if os.getenv(key):
            continue
        try:
            secret_value = st.secrets.get(key)
        except Exception:
            secret_value = None
        if secret_value:
            os.environ[key] = str(secret_value)


@st.cache_resource(show_spinner=False)
def shared_retriever() -> Retriever:
    return Retriever()


def run_agent(question: str) -> AgentTrace:
    agent = Agent(shared_retriever(), configured_reasoner(), max_iterations=4, k=6)
    _, trace = agent.run(question)
    return trace


def run_document(spec: DocumentSpec, retriever, status_box=None) -> DocumentTrace:
    def on_event(message: str) -> None:
        if status_box:
            status_box(message)

    workflow = DocumentWorkflow(retriever, configured_reasoner())
    return workflow.run(spec, on_event)


@st.cache_resource(show_spinner=False)
def cached_uploaded_retriever(cache_key: str, _files_payload: tuple[tuple[str, bytes], ...]):
    files = [UploadedPDF(name=name, content=content) for name, content in _files_payload]
    return build_uploaded_retriever(files)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --orange: #ff9900;
            --ink: #f7fafc;
            --muted: #a8b3c4;
            --line: rgba(255, 255, 255, .13);
        }
        .stApp {
            background:
                radial-gradient(circle at 12% -12%, rgba(255,153,0,.18), transparent 30%),
                radial-gradient(circle at 92% 2%, rgba(93,215,255,.12), transparent 32%),
                linear-gradient(140deg, #07111c 0%, #0b1420 54%, #060c14 100%);
        }
        .block-container {
            max-width: 1680px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
        }
        header[data-testid="stHeader"] { background: transparent; }
        .hero {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 28px 30px;
            margin-bottom: 22px;
            background: linear-gradient(145deg, rgba(22,33,49,.96), rgba(8,14,24,.86));
            box-shadow: 0 28px 80px rgba(0,0,0,.28);
        }
        .eyebrow {
            color: var(--orange);
            font-size: .78rem;
            letter-spacing: .09em;
            text-transform: uppercase;
            font-weight: 800;
        }
        .hero h1 {
            margin: 10px 0 8px;
            font-size: clamp(2.1rem, 4vw, 3.4rem);
            letter-spacing: -.03em;
            line-height: 1.06;
        }
        .hero p { color: var(--muted); font-size: 1.05rem; margin: 0; }
        .flow { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
        .flow span {
            border: 1px solid rgba(255,153,0,.34);
            background: rgba(255,153,0,.08);
            color: #ffe0a1;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: .88rem;
            font-weight: 650;
        }
        .input-help {
            color: var(--muted);
            font-size: .9rem;
            line-height: 1.45;
            margin: -2px 0 12px;
        }
        .source-card {
            border: 1px solid rgba(255,255,255,.14);
            border-radius: 12px;
            padding: 14px 15px;
            background: rgba(255,255,255,.045);
            margin-top: 10px;
        }
        .source-card strong { color: #f8fafc; }
        .source-card p { color: var(--muted); margin: 5px 0 0; font-size: .9rem; }
        .terminal-feed {
            height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(93,215,255,.28);
            border-radius: 12px;
            padding: 14px 16px;
            background:
                linear-gradient(180deg, rgba(2,6,12,.96), rgba(5,12,22,.96));
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.03), 0 20px 60px rgba(0,0,0,.22);
            font-family: Consolas, "SFMono-Regular", Menlo, monospace;
            color: #d9f99d;
            font-size: .88rem;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        .terminal-feed .dim { color: #8aa0b8; }
        .terminal-feed .ok { color: #86efac; }
        .terminal-feed .err { color: #fca5a5; }
        .terminal-feed .cursor {
            display: inline-block;
            width: 8px;
            height: 1em;
            margin-left: 3px;
            background: #86efac;
            vertical-align: -2px;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin: 18px 0 24px;
        }
        .metric {
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 15px 16px;
            background: linear-gradient(145deg, rgba(22,33,49,.88), rgba(9,17,28,.9));
        }
        .metric .label { color: var(--muted); font-size: .78rem; }
        .metric .value { color: var(--ink); font-size: 1.2rem; font-weight: 820; margin-top: 4px; }
        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: .76rem;
            font-weight: 800;
            border: 1px solid var(--line);
        }
        .badge.good { color: #d8ffe9; background: rgba(66,211,146,.15); border-color: rgba(66,211,146,.36); }
        .badge.warn { color: #fff0c9; background: rgba(255,200,87,.15); border-color: rgba(255,200,87,.36); }
        .section-label {
            color: #9fb1c7;
            text-transform: uppercase;
            font-size: .74rem;
            letter-spacing: .08em;
            font-weight: 800;
            margin: 8px 0 5px;
        }
        .evidence-pill {
            display: inline-block;
            margin: 4px 6px 4px 0;
            border-radius: 999px;
            border: 1px solid rgba(93,215,255,.36);
            background: rgba(93,215,255,.08);
            color: #b9f0ff;
            padding: 6px 10px;
            font-size: .86rem;
        }
        .coverage {
            border: 1px solid var(--line);
            border-radius: 9px;
            padding: 13px;
            min-height: 112px;
            background: rgba(255,255,255,.035);
        }
        .coverage-title { color: #d6dee9; font-weight: 750; margin-bottom: 8px; }
        .report-frame {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 24px 28px;
            background: rgba(248,250,252,.96);
            color: #172033;
        }
        .report-frame h1, .report-frame h2, .report-frame h3 { color: #111827; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(151,170,192,.26);
            border-radius: 10px;
            background: rgba(6,15,25,.24);
        }
        .stButton > button {
            border-radius: 9px;
            min-height: 48px;
            font-weight: 800;
        }
        @media (max-width: 900px) {
            .metric-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, stages: list[str], eyebrow: str) -> None:
    chips = "".join(f"<span>{escape(stage)}</span>" for stage in stages)
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(subtitle)}</p>
          <div class="flow">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_access_code() -> bool:
    configured = os.getenv("DEMO_ACCESS_CODE", "").strip()
    if not configured:
        return True
    entered = st.text_input("Demo access code", type="password", max_chars=80)
    if entered == configured:
        return True
    st.info("Enter the demo access code to run document generation.")
    return False


def render_terminal(events: list[str], *, state: str = "running") -> str:
    klass = "ok" if state == "complete" else "err" if state == "error" else "dim"
    prefix = "✓ complete" if state == "complete" else "✕ failed" if state == "error" else "● running"
    lines = [f'<span class="{klass}">{escape(prefix)}</span>', ""]
    lines.extend(f"$ {escape(event)}" for event in events[-80:])
    if state == "running":
        lines.append('<span class="cursor"></span>')
    return f'<div class="terminal-feed">{"<br>".join(lines)}</div>'


def document_studio() -> None:
    hero(
        "AI Document Studio",
        "Upload PDFs or use the AWS sample, then turn source knowledge and a client brief into a researched, reviewed, cited consulting deliverable.",
        ["Sources", "Requirements", "Plan", "Research", "Draft", "Review", "Deliver"],
        "Evidence-grounded document production",
    )
    source_col, settings_col = st.columns([1.05, 1.95], gap="large")
    with source_col:
        with st.container(border=True):
            st.subheader("1. Choose source knowledge")
            st.markdown(
                '<div class="input-help">Default mode is uploaded PDFs for client-specific demos. Use the AWS sample when you want a zero-setup walkthrough.</div>',
                unsafe_allow_html=True,
            )
            source_choice = st.radio(
                "Source",
                ["Upload documents", "Built-in AWS sample"],
                index=0,
                horizontal=True,
            )
            uploaded_files = []
            if source_choice == "Built-in AWS sample":
                st.markdown(
                    """
                    <div class="source-card">
                      <strong>AWS Well-Architected Framework</strong>
                      <p>Pre-indexed sample corpus with page and pillar metadata. No upload required.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                uploaded_files = st.file_uploader(
                    "Upload source PDFs",
                    type=["pdf"],
                    accept_multiple_files=True,
                    help=f"Upload 1-{MAX_UPLOAD_FILES} PDFs. Each file must be {MAX_UPLOAD_BYTES // (1024 * 1024)} MB or smaller.",
                ) or []
                st.markdown(
                    f'<div class="input-help">Upload 1–{MAX_UPLOAD_FILES} PDFs, up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB each. Files are parsed and indexed only for this demo session.</div>',
                    unsafe_allow_html=True,
                )
                if uploaded_files:
                    st.success(f"{len(uploaded_files)} PDF(s) ready to index: " + ", ".join(file.name for file in uploaded_files))
                else:
                    st.info("Drop PDFs here first, then describe the report you want.")
    with settings_col:
        with st.container(border=True):
            st.subheader("2. Describe the deliverable")
            st.markdown(
                '<div class="input-help">Tell the agent what to produce, who it is for, and what decisions the report should support.</div>',
                unsafe_allow_html=True,
            )
            default_title = (
                "AWS Well-Architected Architecture Assessment & Remediation Plan"
                if source_choice == "Built-in AWS sample"
                else "Concise Source Summary"
            )
            title = st.text_input(
                "Report title",
                value=default_title,
                max_chars=MAX_TITLE_LENGTH,
            )
            default_brief = DEFAULT_BRIEF if source_choice == "Built-in AWS sample" else (
                "what does this talk about, explain briefly"
            )
            brief = st.text_area(
                "Client brief / report requirements",
                value=default_brief,
                max_chars=MAX_BRIEF_LENGTH,
                height=190,
                help="No system prompts, URLs, or external files are accepted here. The report is grounded only in the selected corpus.",
            )
            c0, c1, c2 = st.columns([1.25, 2, 1])
            with c0:
                deliverable_type = st.selectbox(
                    "Deliverable type",
                    [
                        "Auto",
                        "Summary / Brief",
                        "Consulting Assessment",
                        "Research Report",
                        "Curriculum / Teaching Material",
                        "Custom",
                    ],
                    index=0,
                    help="Auto classifies the brief before planning. Pick a type to force the structure.",
                )
            with c1:
                audience = st.text_input(
                    "Audience",
                    value="Executive and technical stakeholders",
                    max_chars=MAX_AUDIENCE_LENGTH,
                )
            with c2:
                target_depth = st.selectbox("Target depth", ["Demo", "Detailed"], index=0)
            access_ok = require_access_code()
            run = st.button("Generate report from selected source", type="primary", use_container_width=True, disabled=not access_ok)

    st.markdown("### Live process")
    feed_placeholder = st.empty()
    feed_placeholder.markdown(
        render_terminal(["Waiting for input and source selection."], state="complete"),
        unsafe_allow_html=True,
    )

    if run:
        if not title.strip() or not brief.strip():
            st.error("Report title and brief are required.")
            return
        if source_choice == "Upload documents":
            if not uploaded_files:
                st.error("Upload at least one PDF.")
                return
            if len(uploaded_files) > MAX_UPLOAD_FILES:
                st.error(f"Upload at most {MAX_UPLOAD_FILES} PDFs.")
                return
            too_large = [file.name for file in uploaded_files if file.size > MAX_UPLOAD_BYTES]
            if too_large:
                st.error(f"These files exceed the demo size limit: {', '.join(too_large)}")
                return
            payload = tuple((file.name, file.getvalue()) for file in uploaded_files)
            uploaded = [UploadedPDF(name=name, content=content) for name, content in payload]
            key = corpus_hash(uploaded)
            retriever = cached_uploaded_retriever(key, payload)
            source_names = ", ".join(name for name, _ in payload)
            source_kind = "uploaded"
            knowledge_base = f"Uploaded documents: {source_names}"
        else:
            retriever = shared_retriever()
            source_kind = "aws_sample"
            knowledge_base = "AWS Well-Architected Framework"
        spec = DocumentSpec(
            title=title.strip(),
            client_brief=brief.strip(),
            audience=audience.strip() or "Stakeholders",
            target_depth=target_depth,
            knowledge_base=knowledge_base,
            source_kind=source_kind,
            deliverable_type=deliverable_type,
        )
        events = [
            f"Selected source: {knowledge_base}",
            f"Deliverable type: {deliverable_type}",
            "Starting evidence-grounded document workflow",
        ]
        feed_placeholder.markdown(render_terminal(events), unsafe_allow_html=True)

        def on_feed(message: str) -> None:
            events.append(message)
            feed_placeholder.markdown(render_terminal(events), unsafe_allow_html=True)

        with st.spinner("Generating report — watch the live process feed below."):
            try:
                trace = run_document(spec, retriever, on_feed)
            except Exception as error:
                events.append(f"Document generation failed: {error}")
                feed_placeholder.markdown(render_terminal(events, state="error"), unsafe_allow_html=True)
                st.error(f"Document generation failed: {error}")
                return
            events.append("Deliverable ready")
            feed_placeholder.markdown(render_terminal(events, state="complete"), unsafe_allow_html=True)
        st.session_state["document_trace"] = trace

    trace = st.session_state.get("document_trace")
    if trace:
        render_document_results(trace)


def render_document_results(trace: DocumentTrace) -> None:
    pages = sorted(trace.citation_validation.retrieved_pages)
    sections_passing = sum(1 for section in trace.sections if section.qc.passed)
    revised = sum(1 for section in trace.sections if section.revised)
    status = "Passed" if trace.final_qc.passed else "Needs review"
    metric_row(
        [
            ("Sections generated", str(len(trace.sections))),
            ("Evidence pages", str(len(pages))),
            ("Research iterations", str(trace.total_research_iterations)),
            ("Sections passing QC", str(sections_passing)),
            ("Final review", status),
        ]
    )
    st.caption(f"Sections revised once by QC: {revised}")

    preview, quality, evidence, downloads = st.tabs(["Report Preview", "Quality Review", "Research & Evidence", "Downloads"])
    with preview:
        st.markdown('<div class="report-frame">', unsafe_allow_html=True)
        st.markdown(trace.final_markdown)
        st.markdown("</div>", unsafe_allow_html=True)
    with quality:
        if trace.final_qc.passed:
            st.success(trace.final_qc.summary)
        else:
            st.warning(trace.final_qc.summary)
        st.write(f"Citation validation: {'passed' if trace.citation_validation.valid else 'needs review'}")
        for issue in trace.final_qc.major_issues:
            st.write(f"- {issue}")
        for section in trace.sections:
            with st.expander(section.title):
                st.write(f"QC passed: {section.qc.passed}")
                st.write(f"Revision count: {section.revision_count}")
                st.write(f"Citation valid: {section.qc.citation_valid}")
                if section.qc.issues:
                    st.write(section.qc.issues)
    with evidence:
        for section in trace.sections:
            with st.expander(section.title):
                st.write(f"Research objective: {section.objective}")
                for item in section.research_trace.iterations:
                    st.write(f"Search: {item.search_decision.search_query}")
                    st.caption(item.search_decision.reason)
                    pills = " ".join(
                        f'<span class="evidence-pill">{escape(source)} - AWS-WAF p.{page}</span>'
                        for source, page in evidence_pages(item)
                    )
                    st.markdown(pills or "No evidence", unsafe_allow_html=True)
                    st.write(f"Sufficiency: {item.assessment.sufficient}")
                st.write(f"Section QC: {'passed' if section.qc.passed else 'needs review'}")
    with downloads:
        trace_json = trace.model_dump_json(indent=2, by_alias=True)
        st.download_button("Download DOCX", docx_bytes(trace), "aws-architecture-assessment.docx")
        st.download_button("Download Markdown", markdown_bytes(trace), "aws-architecture-assessment.md", mime="text/markdown")
        st.download_button("Download JSON trace", trace_json, "document_trace.json", mime="application/json")
        with st.expander("View full document trace"):
            st.json(json.loads(trace_json))


def rag_explorer() -> None:
    hero(
        "AWS Well-Architected - Agentic RAG Explorer",
        "Ask a question and inspect the exact search, retrieval, assessment, refinement, and citation trail.",
        ["Search", "Retrieve", "Assess", "Refine", "Stop", "Cite"],
        "Existing evidence search demo",
    )
    with st.container(border=True):
        question = st.text_area(
            "Ask a question about the AWS Well-Architected Framework",
            value=DEFAULT_QUESTION,
            max_chars=MAX_QUESTION_LENGTH,
            height=118,
        )
        run = st.button("Run agent", type="primary", use_container_width=True)

    if run:
        cleaned = question.strip()
        if not cleaned:
            st.error("Please enter a question.")
            return
        with st.spinner("Running search -> retrieve -> assess loop..."):
            try:
                trace = run_agent(cleaned)
            except Exception as error:
                st.error(f"Agent run failed: {error}")
                return
        render_rag_results(trace)


def render_rag_results(trace: AgentTrace) -> None:
    metric_row(
        [
            ("Iterations", str(trace.total_iterations)),
            ("Unique evidence chunks", str(trace.total_unique_evidence_chunks)),
            ("Stop reason", trace.stop_reason or "unknown"),
            ("Citations", "Valid" if trace.citation_validation.valid else "Needs review"),
            ("Duration", f"{trace.duration_ms or 0} ms"),
        ]
    )
    for item in trace.iterations:
        render_iteration(item)
    st.divider()
    with st.container(border=True):
        st.header("Final Answer")
        st.markdown(trace.final_answer or "")
        if trace.citation_validation.valid:
            st.success("Citation validation: all citations grounded in retrieved evidence")
        else:
            st.error("Citation validation found issues")
        st.write(f"Stop reason: `{trace.stop_reason}`")
    trace_json = trace.model_dump_json(indent=2, by_alias=True)
    with st.expander("View full trace"):
        st.json(json.loads(trace_json))
    st.download_button("Download trace JSON", trace_json, "agentic_rag_trace.json", mime="application/json")


def metric_row(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="metric"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


def render_iteration(item: IterationTrace) -> None:
    assessment = item.assessment
    state = "SUFFICIENT" if assessment.sufficient else "INSUFFICIENT"
    badge_class = "good" if assessment.sufficient else "warn"
    st.divider()
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
          <h3 style="margin:0;">Iteration {item.iteration}</h3>
          <span class="badge {badge_class}">{state}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown('<div class="section-label">Search</div>', unsafe_allow_html=True)
        st.code(item.search_decision.search_query, language=None)
        st.markdown('<div class="section-label">Reason</div>', unsafe_allow_html=True)
        st.write(item.search_decision.reason)
    with right:
        st.markdown('<div class="section-label">Evidence</div>', unsafe_allow_html=True)
        pills = " ".join(
            f'<span class="evidence-pill">{escape(section)} - AWS-WAF p.{page}</span>'
            for section, page in evidence_pages(item)
        )
        st.markdown(pills or "No evidence retrieved", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_coverage("Supported", item.assessment.supported_information)
    with c2:
        render_coverage("Partially supported", item.assessment.partially_supported_information)
    with c3:
        render_coverage("Missing", item.assessment.missing_information)
    if item.assessment.suggested_next_search:
        st.markdown('<div class="section-label">Next search</div>', unsafe_allow_html=True)
        st.code(item.assessment.suggested_next_search, language=None)
    with st.expander("Full retrieved evidence"):
        for chunk in item.retrieved:
            st.markdown(f"**{chunk.section or 'Unknown'} - AWS-WAF p.{chunk.page}**")
            st.caption(f"score: {chunk.score}")
            st.write(chunk.text_preview)


def render_coverage(title: str, items: list[str]) -> None:
    st.markdown('<div class="coverage">', unsafe_allow_html=True)
    st.markdown(f'<div class="coverage-title">{escape(title)}</div>', unsafe_allow_html=True)
    if not items:
        st.caption("None")
    for item in items:
        st.write(short_label(item))
    st.markdown("</div>", unsafe_allow_html=True)


def evidence_pages(item: IterationTrace) -> list[tuple[str, int]]:
    seen = set()
    result = []
    for chunk in item.retrieved:
        key = (chunk.section or "Unknown", chunk.page)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
        if len(result) >= 6:
            break
    return result


def short_label(text: str) -> str:
    labels = {
        "framework overview and purpose": "Framework overview and purpose",
        "framework pillars": "Framework pillars",
        "reliability and failure preparation": "Reliability / failure preparation",
        "cost optimization and avoiding unnecessary spend": "Cost Optimization",
        "security": "Security",
    }
    return labels.get(text, text)


def main() -> None:
    st.set_page_config(
        page_title="AI Document Production Demo",
        page_icon=":page_facing_up:",
        layout="wide",
    )
    configure_credentials()
    apply_styles()
    studio, explorer = st.tabs(["Document Studio", "RAG Explorer"])
    with studio:
        document_studio()
    with explorer:
        rag_explorer()


if __name__ == "__main__":
    main()

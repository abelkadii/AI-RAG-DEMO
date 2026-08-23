# AI-RAG-DEMO

This repository demonstrates two connected capabilities over the AWS Well-Architected Framework corpus:

1. **Agentic RAG** - iterative evidence retrieval with sufficiency checks, refinement, citations, and full traces.
2. **Evidence-Grounded Document Production** - a consulting-style report pipeline that plans, researches, drafts, reviews, and exports a structured deliverable.

The AWS corpus is a representative consulting knowledge base. The same workflow architecture can be adapted to other company knowledge bases, templates, methodologies, lesson plans, or report formats.

## Document Studio Workflow

Source Knowledge + Client Brief
-> Document Plan / Outline
-> Section-by-Section Research
-> Evidence Sufficiency Check
-> Section Drafting
-> Automated QC
-> Revision if Required
-> Document-Level Review
-> Formatted Deliverable

The default Streamlit experience is **Document Studio**, which generates an AWS Well-Architected Architecture Assessment & Remediation Plan. It includes:

- editable report title, brief, audience, and target depth;
- AWS Well-Architected Framework as the source knowledge base;
- section-level research traces;
- cited report sections;
- automated section QC and document QC;
- downloads for DOCX, Markdown, and JSON trace.

## Agentic RAG

The RAG Explorer tab preserves the original Milestone 1 loop:

Question
-> Search Decision
-> Retrieval
-> Evidence Assessment
-> Refine if Needed
-> Evidence-Sufficient Stop
-> Cited Answer

Every run records search queries, reasons, retrieved page/section metadata, sufficiency decisions, supported/partial/unsupported coverage, final answer, citations, citation validation, and timing.

## Architecture

- `ingest.py`: downloads/extracts the AWS PDF, chunks it, detects pillar labels, and writes the persistent index.
- `vector_store.py`: persistent local cosine-search store with local hash embeddings or optional OpenAI-compatible embeddings.
- `retriever.py`: exposes `search(query, k) -> list[EvidenceChunk]`.
- `agent.py`: runs the explicit search -> retrieve -> assess -> refine -> stop loop.
- `llm.py`: OpenAI-compatible reasoning path with bounded structured-output retry and an offline deterministic fallback.
- `document_models.py`: serializable document-production trace models.
- `document_workflow.py`: plans reports, researches each section with the existing agent loop, drafts cited sections, and runs QC.
- `document_export.py`: exports DOCX and Markdown.
- `streamlit_app.py`: public demo UI with Document Studio and RAG Explorer tabs.

## Local Setup

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
python ingest.py
streamlit run streamlit_app.py
```

The repository includes a prebuilt `data/index` for the public demo. Re-run `python ingest.py` only when you want to rebuild the corpus index.

## Required Configuration

For local development, set values in `.env`. For Streamlit Community Cloud, set them in app secrets.

Required for live model-backed generation:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_CHAT_MODEL`
- `OPENAI_BASE_URL`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_JSON_MAX_TOKENS`
- `OPENAI_ANSWER_MAX_TOKENS`
- `DEMO_ACCESS_CODE`
- `LLM_MODE=local` to force the deterministic fallback

Credentials stay server-side and are never written to traces or UI output.

## Output Formats

Document Studio provides:

- DOCX report export;
- Markdown report export;
- complete JSON execution trace.

PDF export is intentionally not included because Streamlit Community Cloud does not guarantee LibreOffice or other OS-level converters.

## Testing

Run:

```powershell
pytest -q
```

Tests cover the RAG loop, duplicate-query/no-progress safety, citation validation, malformed structured-output retry, zero-evidence behavior, document workflow execution, section evidence association, section QC/revision behavior, DOCX export, and document trace serialization.

## Streamlit Deployment

To deploy on Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a Streamlit Community Cloud app.
3. Set the entrypoint to `streamlit_app.py`.
4. Configure Streamlit secrets:
   - `OPENAI_API_KEY`
   - optional `OPENAI_CHAT_MODEL`
   - optional `OPENAI_BASE_URL`
   - optional `DEMO_ACCESS_CODE`
5. Deploy.

Do not commit `.streamlit/secrets.toml`; it is ignored by git.

## Demo Notes

This is a prospect-facing demo, not a production multi-tenant product. It is designed to show how source knowledge, retrieval, evidence checks, QC, and formatted deliverables can work together. It does not claim to have produced real 100-200 page client reports.

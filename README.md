# AI-RAG-DEMO

This repository demonstrates two connected capabilities over a selectable evidence corpus:

1. **Agentic RAG** - iterative evidence retrieval with sufficiency checks, refinement, citations, and full traces.
2. **Evidence-Grounded Document Production** - a consulting-style report pipeline that plans, researches, drafts, reviews, and exports a structured deliverable.

The AWS Well-Architected Framework is included as the built-in zero-setup sample knowledge base. Document Studio can also ingest a small set of uploaded PDFs for a session-oriented prospect demo, using the same retrieval, evidence assessment, citation, and report-generation flow.

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

The default Streamlit experience is **Document Studio**, which can generate either an AWS Well-Architected assessment from the built-in sample corpus or a generic evidence-grounded report from uploaded PDFs. It includes:

- editable report title, brief, audience, and target depth;
- a source selector for the built-in AWS sample or uploaded PDF documents;
- section-level research traces;
- source/page cited report sections, such as `[AWS Well-Architected Framework p.303]` or `[Strategy_Report.pdf p.12]`;
- automated section QC and document QC;
- downloads for DOCX, Markdown, and JSON trace.

Uploaded PDFs are parsed page-by-page, chunked, indexed in memory, and cached by content hash for Streamlit reruns. Uploaded client documents are not committed to the repository and are not intended to become a persistent document library.

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
- `uploaded_corpus.py`: parses bounded PDF uploads into session-oriented searchable evidence chunks.
- `vector_store.py`: persistent and in-memory cosine-search stores with local hash embeddings or optional OpenAI-compatible embeddings.
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

Tests cover the RAG loop, duplicate-query/no-progress safety, source/page citation validation, malformed structured-output retry, zero-evidence behavior, document workflow execution, uploaded PDF chunking, generic uploaded-document planning, section evidence association, section QC/revision behavior, DOCX export, and document trace serialization.

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

This is a prospect-facing demo, not a production multi-tenant product. It is designed to show how source knowledge, retrieval, evidence checks, QC, and formatted deliverables can work together. It does not include persistent document libraries, user accounts, vector-database infrastructure, or multi-tenant storage, and it does not claim to have produced real 100-200 page client reports.

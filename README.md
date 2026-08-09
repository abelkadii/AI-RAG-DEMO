# Agentic RAG over AWS Well-Architected

Flow:

Question
-> Search Decision
-> Retrieval
-> Evidence Assessment
-> Refine if Needed
-> Evidence-Sufficient Stop
-> Cited Answer

This CLI POC ingests the AWS Well-Architected Framework PDF, searches page-aware chunks, runs an explicit agentic evidence loop, validates citations against retrieved pages, and saves the full trajectory as JSON.

## Example Agent Trajectory

Question:

How should we design a customer-facing workload so it can recover from Availability Zone failures, protect sensitive customer data, and avoid paying for idle or oversized resources?

Trajectory:

Question
-> Reliability search
-> Evidence assessment: incomplete
-> Cost Optimization search
-> Evidence assessment: still incomplete
-> Security search
-> Evidence assessment: sufficient
-> Cited synthesis

The stopping decision is evidence-driven rather than based on a fixed number of retrieval steps.

Artifacts:

- [Screenshot-friendly terminal output](examples/milestone1_demo.txt)
- [Full JSON trace](examples/milestone1_trace.json)
- [Previous client demo output](examples/client_demo.txt)
- [Previous client demo trace](examples/client_demo_trace.json)

## Milestone 1 Scope

Build the document workflow and agentic search loop, including citations and full trace capture.

Included: PDF ingestion, chunking, persistent retrieval, search decision, retrieval, evidence sufficiency assessment, query refinement, evidence-based stopping, cited final answer, citation validation, and full trace capture.

Not included: Plumloom, benchmark case suite, Milestone 2 work, or frontend UI.

## Architecture

- `ingest.py`: downloads/extracts the PDF page by page, chunks it with overlap, detects pillar labels, and writes the persistent index.
- `retriever.py`: exposes `search(query, k) -> list[EvidenceChunk]`.
- `agent.py`: runs the explicit search -> retrieve -> assess -> refine -> stop loop.
- `llm.py`: uses an OpenAI-compatible model when configured, with a local fallback for offline demos.
- `main.py`: renders terminal output and writes trace JSON.

## Quick Start

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
python ingest.py
python main.py "How should a workload prepare for failures while minimizing unnecessary cost?"
pytest -q
```

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_CHAT_MODEL` in `.env` for model-driven planning, assessment, and synthesis. `LLM_MODE=local` selects the offline fallback.

## Trace Format

Each saved trace includes the original request, timestamps, every iteration, search query, search reason, retrieved chunks with page/section/score metadata, evidence assessment, supported/partial/unsupported coverage, suggested next search, stop reason, final answer, citations, citation validation, total iterations, unique evidence count, and duration.

## Testing

Run:

```powershell
pytest -q
```

The tests cover multi-iteration flow, max-iteration stopping, evidence deduplication, citation validation, malformed structured-output retry, zero-evidence behavior, and trace completeness.

## Live Demo Deployment

To deploy the Streamlit demo on Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit Community Cloud app.
3. Set the entrypoint to `streamlit_app.py`.
4. Configure required secrets in the Streamlit app settings:
   - `OPENAI_API_KEY`
   - `OPENAI_CHAT_MODEL` if you do not want the default chat model
   - `OPENAI_BASE_URL` only when using an OpenAI-compatible proxy
5. Deploy the app.

Do not commit `.streamlit/secrets.toml`; it is ignored by git.
